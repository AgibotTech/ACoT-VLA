# ACoT Adapter Guide

This guide explains how to adapt ACoT-VLA to a new LeRobot-style dataset or a remote robot client without first downloading checkpoints or running full training. It is meant to help answer the practical questions behind custom dataset adaptation, mask choices, and WebSocket deployment.

The current paper version linked by this repository is arXiv v2, revised on March 30, 2026. The implementation exposes the ACoT idea through two action streams:

- `coarse_actions`: the coarse reference trajectory used by the explicit action reasoner.
- `actions`: the final executable action chunk used by the policy head.

## Inspect an Adapter Config

Run the adapter inspector before training or serving:

```bash
uv run python scripts/inspect_acot_adapter.py --config-name acot_icra_simulation_challenge_reasoning_to_action
```

The inspector does not load checkpoints. It reports:

- model type, action dimension, final action horizon, and coarse action horizon
- expected camera keys and image keys
- prompt source and whether a sample must provide `prompt`
- transform chain for repacking, robot-specific transforms, and model transforms
- `state_mask`, `action_mask`, and `delta_action_mask`
- whether config-local normalization stats were found

For machine-readable output:

```bash
uv run python scripts/inspect_acot_adapter.py \
  --config-name acot_icra_simulation_challenge_reasoning_to_action \
  --format json
```

To validate a tiny local sample:

```bash
uv run python scripts/inspect_acot_adapter.py \
  --config-name acot_icra_simulation_challenge_reasoning_to_action \
  --sample-npz sample_episode.npz
```

The `.npz` file can contain keys such as `state`, `actions`, `coarse_actions`, `prompt`, and camera arrays named after the expected cameras. Common aliases such as `action`, `observation/state`, and `observation.images.top_head` are also accepted for validation.

## Adapting a New Dataset

Start by finding the closest existing `DataConfigFactory` in `src/openpi/training/config.py`. For a new LeRobot dataset, the important pieces are:

- `repack_transforms`: maps dataset keys into the keys expected by the robot policy transform.
- `data_transforms`: converts dataset observations/actions into OpenPI model inputs and converts model outputs back into robot actions.
- `model_transforms`: handles prompt tokenization, image resize, and state/action padding.
- `action_sequence_keys`: tells the data loader which dataset key contains the action sequence.
- `prompt_from_task` or `prompt_from_hl_instruction`: controls whether prompts come from dataset metadata instead of a direct `prompt` key.

For ACoT configs, the robot input transform usually creates both `coarse_actions` and `actions` from the raw action stream. The `joint_action_shifts` setting controls how densely each stream samples the raw trajectory. A common pattern is a larger shift for coarse actions and a smaller shift for final actions.

## Choosing Masks

ACoT configs may use three mask types:

- `state_mask`: dimensions zeroed in the state before model input.
- `action_mask`: dimensions zeroed in action targets before later transforms.
- `delta_action_mask`: dimensions converted between absolute and delta action spaces.

The sign convention comes from `make_bool_mask`:

```python
make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
```

Positive numbers add masked dimensions. Negative numbers add unmasked dimensions. A zero value is ignored.

For gripper-like binary state, the maintainers have noted that hiding the gripper state can sometimes make training more stable because the model learns opening and closing from visual context and coarse action guidance instead of directly conditioning on the gripper state. Treat this as a tunable adapter choice, not a universal rule.

Use the inspector to check whether masks are longer than your sample state/action dimension:

```bash
uv run python scripts/inspect_acot_adapter.py --config-name <CONFIG_NAME> --sample-npz sample_episode.npz
```

## Remote Policy Serving

ACoT-VLA follows the OpenPI WebSocket serving flow. Start a trained policy server with either a default environment mode or an explicit checkpoint:

```bash
uv run python scripts/serve_policy.py --env G2SIM --port 8000
```

or:

```bash
uv run python scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=<CONFIG_NAME> \
  --policy.dir=<CHECKPOINT_DIR> \
  --port 8000
```

Then query it from the robot-side process with `openpi-client`:

```python
from openpi_client import image_tools
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

observation = {
    "observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(base_img, 224, 224)),
    "observation/wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, 224, 224)),
    "observation/state": state,
    "prompt": task_instruction,
}

action_chunk = client.infer(observation)["actions"]
```

The policy server returns an action chunk with shape `[action_horizon, action_dim]` after output transforms. Most robot loops execute part of the chunk open-loop and query the server again periodically.

## Out of Scope

This repository provides policy training, data transforms, and WebSocket inference. Robot-specific ROS nodes, low-level motor control, safety interlocks, calibration, and hardware execution policies remain outside this repository and must be implemented in the robot stack.
