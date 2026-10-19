"""Replay a recorded G2 LeRobot episode into serve_rtc_stream.py and measure the command stream.

Streams the episode's real state and camera frames at --hz (30 = recording rate),
receives the server's per-tick commands, and reports readiness time, command rate,
inter-arrival jitter, action age, and the distance of each command from the recorded
action of the same frame. Open loop: the state follows the recording, not the
commands, so this exercises the server, not task success.

Watchdog check: --pause-at S stops observations for --pause-s seconds (the server
must latch), then resumes and sends rearm (the server must reach ready again).

    python scripts/rtc_stream_client.py --uri ws://127.0.0.1:8001 \
        --dataset ~/g2sim/data/lerobot_v21_mp_ipw_train/folding_towels_infpose005 --episode 0 --seconds 40
"""

import dataclasses
import json
import pathlib
import threading
import time

import av
import cv2
import msgpack
import numpy as np
import pandas as pd
import tyro
import websockets.sync.client as ws_client

# Raw G2 LeRobot row (24): waist 0:5, head 5:8, arm_l 8:15, arm_r 15:22, grippers 22:24.
_ARMS = slice(8, 22)
_GRIPPERS = slice(22, 24)
_WAIST = slice(0, 5)
_HEAD = slice(5, 8)
_CAMERAS = {"head": "top_head", "hand_left": "hand_left", "hand_right": "hand_right"}


@dataclasses.dataclass
class Args:
    """Command-line arguments (parsed by tyro).

    Attributes:
        dataset: LeRobot v2.1 dataset root (``data/``, ``videos/``, ``meta/``).
        uri: Websocket URI of serve_rtc_stream.py.
        episode: Episode index to replay.
        hz: Observation send rate, Hz.
        seconds: Replay length, seconds (capped at the episode length).
        external_clock: Send tick=true with every observation (server --rtc.clock external).
        max_side: Longest image side sent, pixels.
        jpeg_quality: JPEG quality, 0-100.
        prompt: Task prompt; None = first task in meta/tasks.jsonl.
        robot_type: corobot robot_type sent with each observation.
        pause_at: Seconds into the replay at which to pause observations; None = never.
        pause_s: Pause length, seconds; rearm is sent after resuming.
        trace: Optional JSONL path for every received message.
    """

    dataset: str
    uri: str = "ws://127.0.0.1:8001"
    episode: int = 0
    hz: float = 30.0
    seconds: float = 40.0
    # Send tick=true with every observation (server --rtc.clock external).
    external_clock: bool = False
    # Longest image side sent to the server (the model resizes to 224 anyway).
    max_side: int = 640
    jpeg_quality: int = 90
    prompt: str | None = None
    robot_type: str = "G2_omnipicker"
    pause_at: float | None = None
    pause_s: float = 1.0
    trace: str | None = None


def _frames(path: pathlib.Path):
    container = av.open(str(path))
    for frame in container.decode(video=0):
        yield frame.to_ndarray(format="rgb24")


def _jpeg(img: np.ndarray, max_side: int, quality: int) -> dict:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return {"encoding": "JPEG", "image_data": buf.tobytes(), "height": img.shape[0], "width": img.shape[1]}


def _policy_layout(raw: np.ndarray) -> np.ndarray:
    return np.concatenate([raw[_ARMS], raw[_GRIPPERS], raw[_WAIST]])


def main(args: Args) -> None:
    """Replay one episode open loop into the server and print the command-stream report.

    Observations follow the recording (not the commands); a reader thread collects every
    server message, stamped with its arrival time and the frame being sent.

    Args:
        args: Parsed command-line arguments.
    """
    root = pathlib.Path(args.dataset).expanduser()
    episode = pd.read_parquet(root / f"data/chunk-000/episode_{args.episode:06d}.parquet")
    states = np.stack(episode["observation.state"].to_numpy()).astype(np.float64)
    actions = np.stack(episode["action"].to_numpy()).astype(np.float64)
    prompt = args.prompt or json.loads((root / "meta/tasks.jsonl").read_text().splitlines()[0])["task"]
    videos = {
        key: _frames(root / f"videos/chunk-000/observation.images.{name}/episode_{args.episode:06d}.mp4")
        for key, name in _CAMERAS.items()
    }
    # Initialise the AV1 decoders before connecting: their first frame is slow enough to
    # starve the server's 0.25 s observation watchdog right after it arms.
    first_frames = {key: next(gen) for key, gen in videos.items()}

    ws = ws_client.connect(args.uri, max_size=None, compression=None, open_timeout=30)
    received: list[dict] = []
    stop = threading.Event()
    lock = threading.Lock()
    current_frame = [0]

    def reader() -> None:
        while not stop.is_set():
            try:
                raw = ws.recv(timeout=0.5)
            except TimeoutError:
                continue
            except Exception:
                return
            msg = msgpack.unpackb(raw, raw=False)
            msg["_t"] = time.monotonic()
            with lock:
                msg["_frame"] = current_frame[0]
                received.append(msg)
            if msg.get("type") == "status":
                print(f"[status] {msg.get('state')} {msg.get('reason') or ''}", flush=True)
            elif msg.get("type") == "error":
                print(f"[error] {msg.get('error')}", flush=True)

    threading.Thread(target=reader, daemon=True).start()

    period = 1.0 / args.hz
    t0 = time.monotonic()
    next_t = t0
    sent = 0
    paused = resumed = False
    n = min(len(states), int(args.seconds * args.hz))
    for i in range(n):
        frames = first_frames if i == 0 else {key: next(gen) for key, gen in videos.items()}
        elapsed = time.monotonic() - t0
        if args.pause_at is not None and not paused and elapsed >= args.pause_at:
            paused = True
            print(f"[client] pausing observations for {args.pause_s:.1f}s", flush=True)
            time.sleep(args.pause_s)
            next_t = time.monotonic()
        raw = states[i]
        params = {
            "images": {key: _jpeg(img, args.max_side, args.jpeg_quality) for key, img in frames.items()},
            "states": {
                "arm_joint_states": raw[_ARMS].tolist(),
                "gripper_states": raw[_GRIPPERS].tolist(),
                "waist_joint_states": raw[_WAIST].tolist(),
                "head_joint_states": raw[_HEAD].tolist(),
            },
            "prompt": prompt,
            "robot_type": args.robot_type,
        }
        with lock:
            current_frame[0] = i
        ws.send(msgpack.packb({"type": "obs", "params": params, "tick": args.external_clock}))
        sent += 1
        if paused and not resumed:
            resumed = True
            time.sleep(0.2)
            ws.send(msgpack.packb({"type": "rearm"}))
            print("[client] resumed and sent rearm", flush=True)
        next_t += period
        slack = next_t - time.monotonic()
        if slack > 0:
            time.sleep(slack)
    stop.set()
    ws.close()
    _report(args, received, actions, t0, sent)


def _report(args: Args, received: list[dict], actions: np.ndarray, t0: float, sent: int) -> None:
    cmds = [m for m in received if m.get("type") == "cmd"]
    statuses = [m for m in received if m.get("type") == "status"]
    errors = [m for m in received if m.get("type") == "error"]
    ready_t = next((m["_t"] - t0 for m in statuses if m.get("state") == "ready"), None)
    latches = [m.get("reason") for m in statuses if m.get("state") == "latched"]
    print("\n==== rtc_stream_client report ====")
    print(f"observations sent: {sent}   commands: {len(cmds)}   errors: {len(errors)}")
    print(f"first ready at: {'never' if ready_t is None else f'{ready_t:.2f}s'}")
    print(f"latch reasons: {sorted(set(latches)) or 'none'}")
    if len(cmds) > 2:
        t = np.array([m["_t"] for m in cmds])
        gaps = np.diff(t) * 1000.0
        span = t[-1] - t[0]
        widths = {len(m["action"]) for m in cmds}
        ages = np.array([m["action_age_s"] for m in cmds]) * 1000.0
        print(f"command rate: {(len(cmds) - 1) / span:.2f} /s over {span:.1f}s   action widths: {sorted(widths)}")
        print(
            f"inter-arrival ms: p50 {np.percentile(gaps, 50):.1f}  p99 {np.percentile(gaps, 99):.1f}  "
            f"max {gaps.max():.1f}"
        )
        print(f"action age ms: p50 {np.percentile(ages, 50):.1f}  max {ages.max():.1f}")
        rows = np.array([m["action"] for m in cmds])
        rec = np.array([_policy_layout(actions[m["_frame"]]) for m in cmds])
        w = min(rows.shape[1], rec.shape[1])
        err = np.abs(rows[:, :w] - rec[:, :w])
        print(
            f"|cmd - recorded action| mean: arms {err[:, :14].mean():.4f}  grippers {err[:, 14:16].mean():.4f}"
            + (f"  waist {err[:, 16:21].mean():.4f}" if w >= 21 else "")
        )
        step = np.abs(np.diff(rows[:, :14], axis=0)).max(axis=1)
        print(
            f"per-tick arm step rad: p50 {np.percentile(step, 50):.4f}  p99 {np.percentile(step, 99):.4f}  max {step.max():.4f}"
        )
    if args.trace:
        with pathlib.Path(args.trace).open("w") as f:
            for m in received:
                f.write(json.dumps(m, default=float) + "\n")
        print(f"trace: {args.trace}")


if __name__ == "__main__":
    main(tyro.cli(Args))
