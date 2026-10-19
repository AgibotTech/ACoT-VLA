"""Genie Sim client for the server-owned RTC stream (scripts/serve_rtc_stream.py --rtc.clock external).

Counterpart of the harness rtc_policy.py with the RTC queue moved to the server,
the way anvil-embodied-ai runs it. Each act() is one control tick:

    act()                                      server (one connection)
    -----                                      ------
    obs + tick  ---------------------------->  store latest obs (worker refills the queue)
                                               pop one row, authorize its age
    JOINT_ABS row  <-------------------------  {"type": "cmd", "action": [21]}
      or hold the last command  <------------  {"type": "status", ...} (not ready / latched)

Everything else matches rtc_policy.py so the two can be compared: the same
CoRobotPolicy payload (images, states, prompt, robot_type), the same
_post_process_action, act() paced to 30 Hz wall clock, and the last command held
while no row is available. A new episode sends "reset" (readiness restarts); a
latch is followed by "rearm" once observations flow again, like Anvil's operator rearm.

Simulation time is not wall time (physics runs at ~0.33-0.5x), so the server must use
the external clock: it advances one row per act(), never by its own timer.

pipeline=False (default): act() waits for each reply, so every sim step costs one round
trip (~11-13 steps per wall second). Genie Sim's TimeOut counts wall seconds, so this
lock-step client gets about half the steps per episode budget of a client that answers
from a local queue. Most of that per-step cost is building the payload (three JPEGs)
in the sim thread, not the wait. pipeline=True: act() only copies the frames and hands
them to a sender thread, which encodes and sends them in order; a receiver thread keeps
the newest command and act() applies it. The sim steps at its own rate and each command
arrives a step or two late. If the sender falls behind, the oldest unsent observation is
dropped (counted as "dropped": that tick never reaches the server).
"""

from __future__ import annotations

import collections
import concurrent.futures
import copy
import dataclasses
import json
import logging
import os
import pathlib
import queue
import threading
import time

import msgpack
import numpy as np

log = logging.getLogger("rtc_stream_policy")


@dataclasses.dataclass
class StreamConfig:
    """Client settings, set once by run_rtc_stream.py through the module-level ``CFG``.

    Attributes:
        pace_hz: Wall-clock act() rate, Hz; 0 = run as fast as the sim goes.
        trace: JSONL path, one line per act(); empty = off.
        pipeline: Send without waiting for each reply; apply the newest command received.
    """

    pace_hz: float = 30.0  # wall-clock act() rate; 0 = run as fast as the sim goes
    trace: str = ""  # jsonl, one line per act()
    pipeline: bool = False  # send without waiting; apply the newest command received


CFG = StreamConfig()


def build(CoRobotPolicy):
    """Subclass the benchmark's CoRobotPolicy (importable only after SimulationApp boots).

    Args:
        CoRobotPolicy: The benchmark's ``CoRobotPolicy`` class.

    Returns:
        ``RtcStreamPolicy``, a CoRobotPolicy subclass bound to the current ``CFG``.
    """
    from geniesim_benchmark.utils.comm.retry import get_recv_timeout
    from geniesim_benchmark.utils.infer_post_process import get_arm_states

    class RtcStreamPolicy(CoRobotPolicy):
        """CoRobotPolicy whose act() is one tick of the server-owned RTC stream.

        Called from the sim thread only; pipeline mode adds a sender and a receiver
        thread that share state through ``_rx_lock``.
        """

        cfg: StreamConfig = CFG

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._last_out = None
            self._last_act_t = 0.0
            self._reset_pending = True
            self._episode = 0
            self._step = 0
            self._counts = collections.Counter()
            # pipeline mode: replies are read by a background thread
            self._rx: threading.Thread | None = None
            self._rx_lock = threading.Lock()
            self._rx_error: Exception | None = None
            self._latest_cmd: dict | None = None
            self._status: dict | None = None
            self._accept_epoch = 0
            self._await_reset = False
            self._rearmed_epoch = -1
            self._tx: threading.Thread | None = None
            self._tx_queue: queue.Queue = queue.Queue(maxsize=3)

        # -- benchmark hooks ------------------------------------------------------
        def need_infer(self):
            """Return True: render every step, the server needs a fresh observation each tick."""
            return True  # render every step: the server needs a fresh observation each tick

        def inference_due(self):
            """Return False: never settle-wait, the server decides when to infer."""
            return False  # never settle-wait; the server decides when to infer

        def reset(self):
            """Start a new episode; the next act() sends "reset" to the server first."""
            super().reset()
            if self._step:
                log.info("episode %d done: %s", self._episode, dict(self._counts))
            self._reset_pending = True
            self._last_out = None
            self._episode += 1
            self._step = 0
            self._counts.clear()

        # -- one control tick -----------------------------------------------------
        def _call(self, message: dict) -> dict:
            self._ensure_connection()
            self._ws.send(msgpack.packb(message))
            reply = self._ws.recv(timeout=get_recv_timeout())
            if isinstance(reply, str):
                raise RuntimeError(f"server sent text: {reply}")
            reply = msgpack.unpackb(reply, raw=False)
            if reply.get("type") == "error":
                raise RuntimeError(f"server error: {reply.get('error')}")
            return reply

        def act(self, observation, **kwargs):
            """Run one control tick: send the observation and apply the server's command.

            Paced to ``cfg.pace_hz`` wall clock. Without a command this tick (readiness,
            latch, error) the last target is held, or the current arm state before any.

            Args:
                observation: Genie Sim observation dict (``states``, images, ...).
                **kwargs: Benchmark extras; ``task_instruction`` and ``gen_config`` are used.

            Returns:
                The post-processed JOINT_ABS action dict, the held one, or None when the
                lock-step payload could not be built.
            """
            if self.cfg.pace_hz > 0:
                wait = self._last_act_t + 1.0 / self.cfg.pace_hz - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
            self._last_act_t = time.monotonic()

            instruction = os.environ.get("GENIESIM_PROMPT_OVERRIDE") or kwargs.get("task_instruction", "")
            if self.cfg.pipeline:
                self._step += 1
                reply = self._act_pipelined(observation, instruction, kwargs.get("gen_config"))
            else:
                # Stock CoRobotPolicy keeps sending depth until a server reply says otherwise; the
                # stream server never uses it, and with depth every observation is ~6.6 MB.
                self._send_depth = False
                payload = self.get_payload(observation, instruction, kwargs.get("gen_config"))
                if payload is None:
                    return None
                self._step += 1
                reply = self._act_lockstep(payload["params"])

            cur = get_arm_states(observation["states"], self._arm_dim)
            if reply is not None and reply.get("type") == "cmd":
                row = np.asarray(reply["action"], dtype=np.float64)
                entry = {"arm": row[0:14], "gripper": row[14:16], "kind": "JOINT_ABS", "eef_frame": "base_link"}
                if row.shape[0] >= 21:
                    entry["waist"] = row[16:21]
                out = self._post_process_action(entry, cur, arm_base_tf=observation.get("arm_base_transform"))
                self._counts["cmd"] += 1
                self._write_trace(reply, row)
                self._last_out = out
                return out

            # No row this tick (readiness proof, latch, error): keep the last target.
            self._counts["hold"] += 1
            self._write_trace(reply, None)
            if self._last_out is not None:
                return self._last_out
            return {"arm": [float(v) for v in cur[: self._arm_dim]]}

        def _act_lockstep(self, params: dict) -> dict | None:
            """Send this tick's observation and wait for the server's reply."""
            try:
                if self._reset_pending:
                    self._call({"type": "reset"})
                    self._reset_pending = False
                reply = self._call({"type": "obs", "params": params, "tick": True})
                if reply.get("type") == "status" and reply.get("state") == "latched":
                    self._counts["latched"] += 1
                    log.warning("server latched: %s; rearming", reply.get("reason"))
                    rearm = self._call({"type": "rearm"})
                    if not rearm.get("rearm_ok"):
                        log.warning("rearm refused: %s", rearm.get("rearm_reason"))
                return reply
            except Exception as exc:
                # A new connection starts a fresh server session; reset on the next tick too.
                self._counts["errors"] += 1
                log.warning("RTC stream error: %s: %s", type(exc).__name__, exc)
                self._drop_connection()
                self._reset_pending = True
                return None

        # -- pipeline mode ---------------------------------------------------------
        def _start_receiver(self) -> None:
            self._ensure_connection()
            ws = self._ws
            self._rx_error = None

            def loop() -> None:
                while self._ws is ws:
                    try:
                        raw = ws.recv(timeout=0.5)
                    except TimeoutError:
                        continue
                    except Exception as exc:
                        self._rx_error = exc
                        return
                    msg = {"type": "error", "error": raw} if isinstance(raw, str) else msgpack.unpackb(raw, raw=False)
                    with self._rx_lock:
                        kind = msg.get("type")
                        if kind == "cmd":
                            # Commands queued before a reset belong to the previous episode.
                            if not self._await_reset and msg.get("epoch", 0) >= self._accept_epoch:
                                self._latest_cmd = msg
                        elif kind == "status":
                            if msg.get("reset_ok"):
                                self._accept_epoch = msg.get("epoch", 0)
                                self._await_reset = False
                            self._status = msg
                        else:
                            self._rx_error = RuntimeError(f"server error: {msg.get('error')}")

            self._rx = threading.Thread(target=loop, name="rtc-stream-rx", daemon=True)
            self._rx.start()

        def _start_sender(self) -> None:
            ws = self._ws
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="rtc-jpeg")
            encode = type(self)._encode_image_jpeg  # stock staticmethod (image_rgb, quality=95)

            def build_payload(obs, instruction, gen_config):
                """get_payload with the three JPEG encodes run in parallel (cv2 releases the GIL)."""
                self._encode_image_jpeg = lambda image_rgb, quality=95: pool.submit(encode, image_rgb, quality)
                try:
                    payload = self.get_payload(obs, instruction, gen_config)
                finally:
                    del self._encode_image_jpeg
                if payload is not None:
                    images = payload["params"]["images"]
                    for key, value in images.items():
                        if isinstance(value, concurrent.futures.Future):
                            images[key] = value.result()
                return payload

            def loop() -> None:
                while self._ws is ws:
                    try:
                        item = self._tx_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    try:
                        if item[0] == "obs":
                            _, obs, instruction, gen_config = item
                            self._send_depth = False  # see the lock-step path
                            payload = build_payload(obs, instruction, gen_config)
                            if payload is None:
                                continue
                            ws.send(msgpack.packb({"type": "obs", "params": payload["params"], "tick": True}))
                        else:
                            ws.send(msgpack.packb({"type": item[0]}))
                    except Exception as exc:
                        self._rx_error = exc
                        return

            self._tx = threading.Thread(target=loop, name="rtc-stream-tx", daemon=True)
            self._tx.start()

        def _enqueue(self, item: tuple) -> None:
            """FIFO to the sender; when full, drop the oldest unsent observation."""
            while True:
                try:
                    self._tx_queue.put_nowait(item)
                    return
                except queue.Full:
                    try:
                        old = self._tx_queue.get_nowait()
                    except queue.Empty:
                        continue
                    if old[0] == "obs":
                        self._counts["dropped"] += 1
                    else:  # never drop a control message: put it back in front of this one
                        self._tx_queue.put_nowait(old)
                        self._tx_queue.put(item)
                        return

        def _act_pipelined(self, observation, instruction: str, gen_config) -> dict | None:
            """Queue this tick's observation, return the newest command not yet applied."""
            try:
                if self._rx_error is not None:
                    raise self._rx_error
                if self._rx is None or not self._rx.is_alive() or self._tx is None or not self._tx.is_alive():
                    self._start_receiver()
                    self._start_sender()
                if self._reset_pending:
                    with self._rx_lock:
                        self._await_reset = True
                        self._latest_cmd = None
                    self._enqueue(("reset",))
                    self._reset_pending = False
                # The sim may reuse its buffers: copy the frames, the encoding happens later.
                obs = {}
                for key, value in observation.items():
                    if key == "depth":
                        continue
                    try:
                        obs[key] = copy.deepcopy(value)
                    except Exception:  # uncopyable sim handles stay shared
                        obs[key] = value
                self._enqueue(("obs", obs, instruction, gen_config))
                with self._rx_lock:
                    cmd, self._latest_cmd = self._latest_cmd, None
                    status = self._status
                if status and status.get("state") == "latched" and status.get("epoch") != self._rearmed_epoch:
                    self._rearmed_epoch = status.get("epoch")
                    self._counts["latched"] += 1
                    log.warning("server latched: %s; rearming", status.get("reason"))
                    self._enqueue(("rearm",))
                return cmd if cmd is not None else status
            except Exception as exc:
                self._counts["errors"] += 1
                log.warning("RTC stream error: %s: %s", type(exc).__name__, exc)
                self._drop_connection()
                self._rx = self._tx = None
                self._rx_error = None
                self._tx_queue = queue.Queue(maxsize=3)
                self._reset_pending = True
                return None

        def _write_trace(self, reply: dict | None, row: np.ndarray | None) -> None:
            if not self.cfg.trace:
                return
            record = {
                "t": time.time(),
                "episode": self._episode,
                "step": self._step,
                "type": None if reply is None else reply.get("type"),
                "state": None if reply is None else reply.get("state"),
                "seq": None if reply is None else reply.get("seq"),
                "action_age_s": None if reply is None else reply.get("action_age_s"),
                "queue": None if reply is None else reply.get("queue"),
                "row": None if row is None else row.tolist(),
            }
            try:
                with pathlib.Path(self.cfg.trace).open("a") as f:
                    f.write(json.dumps(record) + "\n")
            except OSError as exc:
                log.debug("trace write failed: %s", exc)

    RtcStreamPolicy.cfg = CFG
    return RtcStreamPolicy
