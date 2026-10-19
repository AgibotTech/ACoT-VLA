# ACoT-VLA-PI05

π₀.₅ training and serving for the AgiBot G2, based on [ACoT-VLA](https://github.com/AgibotTech/ACoT-VLA)
and [OpenPI](https://github.com/Physical-Intelligence/openpi). The upstream ACoT-VLA material is kept
[at the end](#acot-vla).

- [Install](#install)
- [Checkpoints and training](#checkpoints-and-training)
- [Serving: server-side RTC (stream mode)](#serving-server-side-rtc-stream-mode)
- [Serving: request/response mode](#serving-requestresponse-mode)

## Install

We use **uv** to manage the Python environment.

```bash
git clone https://github.com/datamentors/ACoT-VLA-PI05.git
cd ACoT-VLA-PI05
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## Checkpoints and training

Released Genie Sim checkpoints and their training configs:

| Checkpoint (`checkpoints/<name>`) | `pi05` train config |
|---|---|
| `manipulation_pi05` | `pi05_genie_sim_manip_20260613` |
| `instruction_and_robust_pi05` | `pi05_genie_sim_instruction_and_robust_20260526` |
| `spatial_pi05` | `pi05_genie_sim_spatial_20260528` |

```bash
./scripts/download_checkpoint.sh manipulation_pi05      # -> ./checkpoints/manipulation_pi05/
```

G2 fine-tune configs (`pi05_g2_*`) are in `src/openpi/training/config.py`. To train:

```bash
CONFIG=pi05_g2_ipw_vision_action        # any config in config.py
uv run scripts/compute_norm_stats.py --config-name $CONFIG
bash scripts/train.sh $CONFIG my_run    # -> checkpoints/$CONFIG/my_run/<step>/
```

Datasets are LeRobot v2.1. Point the config's `repo_id` list at your local dataset directories.
`./scripts/download_dataset.sh [instruction|manipulation|sim2real]` fetches the Genie Sim suites from
ModelScope (`pip install modelscope` first).

## Serving: server-side RTC (stream mode)

### How it works

The server owns the action queue and the control clock. The client streams observations and gets
back **one absolute joint command per control tick**; it never sees chunks.

1. The client sends its latest observation (camera images, joint states, prompt) as often as it likes.
2. A background worker runs the policy whenever the queue is low, using the unexecuted rest of the
   previous chunk to guide the new one (Real-Time Chunking), and merges the new chunk into the queue.
3. Every control tick (30 Hz by default) the server pops one row and sends it as a command.
4. If the client stops sending observations, the server stops sending commands. The client is
   expected to hold its last target and keep its own limits and stop checks.

Only pi0 / pi0.5 configs are supported (not ACoT).

### Run

```bash
uv run scripts/serve_rtc_stream.py \
  --config pi05_genie_sim_manip_20260613 \
  --dir checkpoints/manipulation_pi05 \
  --port 8001 \
  --rtc.queue-threshold 15 --rtc.execution-horizon 20
```

The server compiles the model at start-up (~10 s) and then logs `server listening on 0.0.0.0:8001`.

| Flag | Default | Meaning |
|---|---|---|
| `--rtc.queue-threshold` | chunk length | Start the next inference when this many rows are left. 15 worked best in sim. |
| `--rtc.execution-horizon` | 20 | Rows of the new chunk guided towards the previous chunk. |
| `--rtc.prefix-attention-schedule` | `exp` | `exp` or `linear`: how quickly guidance fades over those rows. |
| `--rtc.max-guidance-weight` | 10 | Guidance strength; 0 turns guidance off. |
| `--no-fresh-noise` | fresh | Reuse one fixed noise key instead of new noise per chunk (worse in our tests). |
| `--rtc.clock` | `wall` | `wall`: server ticks at `--rtc.control-hz`. `external`: one tick per observation sent with `"tick": true` (simulators). |
| `--rtc.readiness-gate` / `--rtc.no-readiness-gate` | on | On: no commands until five guided refills prove the timing margins, and a violation latches. Off: start at once and hold when the queue runs dry. Turn it off when inference takes ~500 ms. |
| `--rtc.waist` | `auto` | `model`, `hold` (dims 16–19 held at the measured state) or `off` (all waist dims held). `auto` = `model` if the config trained the waist. |
| `--rtc.trace` | none | JSONL file, one line per inference (latency, rows consumed, seam). |

### Request

One websocket connection, msgpack-encoded messages. Send an observation:

```python
import cv2, msgpack
import websockets.sync.client as ws_client

def jpeg(rgb):  # HxWx3 uint8 RGB
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return {"encoding": "JPEG", "image_data": buf.tobytes(), "height": rgb.shape[0], "width": rgb.shape[1]}

ws = ws_client.connect("ws://<server>:8001", max_size=None, compression=None)

ws.send(msgpack.packb({
    "type": "obs",
    "params": {
        "images": {"head": jpeg(head_rgb), "hand_left": jpeg(left_rgb), "hand_right": jpeg(right_rgb)},
        "states": {
            "arm_joint_states": [...],    # 14 floats: left arm 7, right arm 7 (rad)
            "gripper_states": [...],      # 2 floats: left, right
            "waist_joint_states": [...],  # 5 floats
        },
        "prompt": "Grab the yellow package on the table with right arm. ...",
        "robot_type": "G2_omnipicker",
    },
}))
```

Send an observation every control tick, or at least every 0.25 s: older observations trip the
server's watchdog. Other messages: `{"type": "rearm"}` resumes after the watchdog latched (needs
fresh observations), and `{"type": "reset"}` starts a new episode (clears the queue).

### Response

Read replies on the same connection, for example in a separate thread:

```python
for raw in ws:
    msg = msgpack.unpackb(raw, raw=False)
    if msg["type"] == "cmd":
        target = msg["action"]      # apply this joint target now
    elif msg["type"] == "status":
        state = msg["state"]        # "starting" | "armed" | "ready" | "latched"
```

A command, sent once per control tick once the server is `ready`:

```python
{
    "type": "cmd",
    "seq": 1532,                 # increasing command counter
    "epoch": 3,                  # changes after every reset or latch; drop commands from an older epoch
    "t_server": 1791545000.12,   # server wall time
    "action_age_s": 0.11,        # age of the observation this command came from
    "queue": 12,                 # rows left in the server's queue
    "action": [...],             # 21 floats, absolute
}
```

`action` layout (21 floats; configs without a waist return 16):

| Index | Content |
|---|---|
| 0–6 | left arm joints (rad) |
| 7–13 | right arm joints (rad) |
| 14 | left gripper |
| 15 | right gripper |
| 16–20 | waist joints |

A status message, sent once per second and whenever the state changes:
`{"type": "status", "state": "ready", "reason": "", "epoch": 3}`. When `state` is `"latched"`,
`reason` says why (for example a stale observation) and no commands are sent until `rearm`.

### Genie Sim

`scripts/geniesim/` has the Genie Sim client (`run_rtc_stream.py`, use `--pipeline`) and the launcher
(`sim_rtc_stream.sh`). Run the server with
`--rtc.clock external` for Genie Sim, whose simulated time does not match wall time.

## Serving: request/response mode

The original mode (pause per inference): the client sends one observation, gets back a whole action
chunk, and plays it before asking again.

```bash
uv run scripts/serve_policy.py --port 8000 \
  policy:checkpoint --policy.config pi05_genie_sim_manip_20260613 --policy.dir checkpoints/manipulation_pi05
```

`serve_policy.py` forces cuBLAS matrix maths (~400–550 ms per inference on our GPU).
`OPENPI_KEEP_TRITON_GEMM=1` keeps XLA's Triton kernels instead (~60–100 ms, same actions).
`--fresh-noise` draws new noise per call; the default reuses one fixed key.

### Request

```python
ws = ws_client.connect("ws://<server>:8000", max_size=None, compression=None)
ws.send(msgpack.packb({
    "method": "infer",
    "params": {
        "images": {"head": jpeg(head_rgb), "hand_left": jpeg(left_rgb), "hand_right": jpeg(right_rgb)},
        "states": {"arm_joint_states": [...], "gripper_states": [...], "waist_joint_states": [...]},
        "prompt": "...",
        "robot_type": "G2_omnipicker",
        "task_name": "sorting_packages",
    },
}))
reply = msgpack.unpackb(ws.recv(), raw=False)
```

`task_name` decides which waist dims are returned: `sorting_packages*` returns all five (dims 16–19
held at the current state), tasks listed in the `G2SIM_FULL_WAIST_TASKS` environment variable return
all five from the model, any other task returns no waist.

### Response

```python
{
    "result": {
        "left_arm":  {"kind": "JOINT_ABS", "values": [[...7 floats], ...]},   # H rows (H = 30)
        "right_arm": {"kind": "JOINT_ABS", "values": [[...7 floats], ...]},
        "left_effector":  [[g], ...],                                       # H rows of 1
        "right_effector": [[g], ...],
        "waist": {"kind": "JOINT_ABS", "values": [[...5 floats], ...]},    # only when the waist is returned
    },
    "server_timing": {"infer_ms": 380.2, "model_ms": 371.9},
}
```

Row `j` is the target for `j + 1` control ticks after the observation (30 Hz for the G2 configs). On
failure the reply is `{"error": "<traceback>"}`.

---

<a id="acot-vla"></a>

## ACoT-VLA

This repository started from the **official implementation** of
[**ACoT-VLA**](https://arxiv.org/abs/2601.11404v2) and the AgiBot World Challenge π₀ / π₀.₅ baseline.

ACoT-VLA bridges the semantic-kinematic gap in robot policies by reasoning directly in the action
space:

* **Explicit Action Reasoner (EAR):** a light-weight Transformer that synthesizes coarse motion
  trajectories as direct motion cues.
* **Implicit Action Reasoner (IAR):** extracts latent action priors from the VLM backbone with
  cross-attention.
* **Action Chain-of-Thought (ACoT):** EAR and IAR together form structured action intents for
  grounded, long-horizon policy learning.

<p align="center">
  <img src="docs/framework.png" alt="ACoT-VLA framework" width="90%">
</p>

### Benchmarks

**LIBERO** (average success rate, %)

| Method | Spatial | Object | Goal | Long | **Avg.** |
| --- | --- | --- | --- | --- | --- |
| $\pi_0$ | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| $\pi_{0.5}$ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| **ACoT-VLA (Frozen)** | **99.4** | **99.6** | 98.8 | 96.0 | **98.5** |
| **ACoT-VLA** | 98.6 | 99.0 | **99.4** | **97.0** | **98.5** |

**LIBERO-Plus** (robustness, %)

| Setting | Method | Camera | Robot | Language | Light | Background | Noise | Layout | **Avg.** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Zero-Shot** | $\pi_0^*$ | 61.0 | 40.8 | 63.5 | 89.3 | 84.1 | 80.1 | 76.4 | 69.4 |
| | $\pi_{0.5}^*$ | **75.8** | 79.4 | 83.3 | 95.5 | 95.0 | **89.6** | 87.0 | 85.7 |
| | **ACoT-VLA (Frozen)** | 68.9 | 80.3 | 84.1 | 95.6 | 93.1 | 81.5 | **88.3** | 83.6 |
| | **ACoT-VLA** | 72.6 | **82.6** | **87.5** | **97.7** | **96.5** | 87.8 | 88.1 | **86.6** |
| **SFT** | $\pi_0$ (Frozen) | 79.6 | 21.1 | 72.5 | 84.7 | 86.2 | 68.3 | 69.4 | 67.4 |
| | $\pi_{0.5}$ (Frozen) | 70.3 | 41.7 | **81.1** | **97.3** | 94.6 | 71.8 | 84.9 | 75.7 |
| | **ACoT-VLA (Frozen)** | 91.2 | 62.5 | 80.3 | 95.1 | 91.5 | 88.3 | 84.9 | 84.1 |
| | **ACoT-VLA** | **96.6** | **70.4** | 79.7 | 95.1 | **97.1** | **95.9** | **85.0** | **88.0** |

**VLABench** (Intention Score / Progress Score)

| Method | In-dist. | Category | Commonsense | Instruction | Texture | **Avg.** |
| --- | --- | --- | --- | --- | --- | --- |
| $\pi_0$ (Frozen) | 67.8 / 62.7 | 44.0 / 33.6 | 54.9 / **43.0** | **58.0** / 38.7 | 50.6 / 42.5 | 55.0 / 44.1 |
| $\pi_{0.5}$ (Frozen) | 75.0 / 60.8 | 49.6 / 35.3 | **57.5** / 41.6 | 57.1 / 30.3 | 62.0 / 47.4 | 60.2 / 43.1 |
| **ACoT-VLA (Frozen)** | **79.8 / 66.1** | **54.1 / 38.9** | 52.3 / 37.8 | 56.8 / **39.6** | **74.6 / 54.6** | **63.5 / 47.4** |

"Frozen": LLM backbone frozen during training. \*: reproduced with the released checkpoints.

### Get started (LIBERO / VLABench)

```bash
python examples/libero/convert_libero_data_to_lerobot.py        # data to LeRobot format
uv run scripts/compute_norm_stats.py --config-name <CONFIG_NAME>
bash scripts/train.sh <CONFIG_NAME> <EXP_NAME>
bash scripts/server.sh <GPU_ID> <PORT>                           # request/response server
```

### Citation

```bibtex
@article{zhong2026acot,
  title={ACoT-VLA: Action Chain-of-Thought for Vision-Language-Action Models},
  author={Zhong, Linqing and Liu, Yi and Wei, Yifei and Xiong, Ziyu and Yao, Maoqing and Liu, Si and Ren, Guanghui},
  journal={arXiv preprint arXiv:2601.11404},
  year={2026}
}
```

Built on [OpenPI](https://github.com/Physical-Intelligence/openpi).
