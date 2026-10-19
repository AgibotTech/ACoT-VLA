# Held-out validation for fine-tuning

This repo keeps validation as a sidecar process instead of changing `scripts/train.py`.
Train writes checkpoints as usual; the validator watches those checkpoint folders,
computes loss on held-out LeRobot data, and logs `val/*` metrics to Weights & Biases.

## Prepare a validation set

Use full held-out episodes or tasks, not random frames from the same episodes used for
training. Put them in LeRobot format and make sure they use the same observation/action
contract as the training config.

For one held-out dataset:

```bash
VAL_REPO=/path/to/heldout_lerobot
```

For multiple held-out datasets, repeat `--val-repo-id`.

## Launch training

```bash
export HF_HOME=/tmp/hf-home-$USER
CONFIG=pi05_genie_sim_manip_20260613
EXP=my_pi05_run
WANDB_MODE=online bash scripts/train.sh "$CONFIG" "$EXP"
```

Training writes checkpoints to:

```text
checkpoints/<CONFIG>/<EXP>/<step>/
```

## Run validation once

```bash
.venv/bin/python scripts/validate_checkpoint.py \
  --config-name "$CONFIG" \
  --checkpoint-dir "checkpoints/$CONFIG/$EXP/5000" \
  --val-repo-id "$VAL_REPO" \
  --max-batches 20 \
  --batch-size 32 \
  --wandb-project ACoT-VLA \
  --wandb-run-name "${EXP}-val" \
  --wandb-group "$EXP" \
  --wandb-mode online
```

## Watch checkpoints while fine-tuning

Run this in a second shell on the workstation:

```bash
.venv/bin/python scripts/validate_checkpoint.py \
  --config-name "$CONFIG" \
  --checkpoint-root "checkpoints/$CONFIG/$EXP" \
  --val-repo-id "$VAL_REPO" \
  --watch \
  --poll-seconds 60 \
  --max-batches 20 \
  --batch-size 32 \
  --wandb-project ACoT-VLA \
  --wandb-run-name "${EXP}-val" \
  --wandb-group "$EXP" \
  --wandb-mode online
```

Equivalent with uv if your shell has it on PATH. The `HF_HOME` export gives Hugging Face a writable cache on this workstation:

```bash
/home/datamentors/.local/bin/uv run python scripts/validate_checkpoint.py --help
```

Logged metrics:

- `val/loss`
- `val/loss_std`
- `val/batches`
- `val/examples`
- `val/elapsed_s`
- `val/checkpoint_step`

The validator loads normalization stats from each checkpoint's `assets/norm_stats.json`,
so validation uses the same normalization as the checkpoint being scored.


## Action-chunk validation

`validate_actions.py` runs real policy inference on held-out observations, then
compares predicted action chunks against recorded future actions.

Modes:

- `all-frames`: score many frames from the held-out dataset. This is dense and is the default.
- `first-frame`: score only episode starts where `frame_index == 0` is available. If episode starts are not discoverable, it falls back to `--sample-stride`.

Dense validation:

```bash
.venv/bin/python scripts/validate_actions.py \
  --config-name "$CONFIG" \
  --checkpoint-root "checkpoints/$CONFIG/$EXP" \
  --val-repo-id "$VAL_REPO" \
  --mode all-frames \
  --max-samples 128 \
  --watch \
  --wandb-project ACoT-VLA \
  --wandb-run-name "${EXP}-action-val" \
  --wandb-group "$EXP" \
  --wandb-mode online
```

Episode-start validation:

```bash
.venv/bin/python scripts/validate_actions.py \
  --config-name "$CONFIG" \
  --checkpoint-root "checkpoints/$CONFIG/$EXP" \
  --val-repo-id "$VAL_REPO" \
  --mode first-frame \
  --max-samples 64 \
  --watch \
  --wandb-project ACoT-VLA \
  --wandb-run-name "${EXP}-action-val-first" \
  --wandb-group "$EXP" \
  --wandb-mode online
```

Action metrics logged:

- `val_actions/mse`
- `val_actions/mae`
- `val_actions/first_mse`
- `val_actions/first_mae`
- `val_actions/final_mse`
- `val_actions/final_mae`
- `val_actions/dim_XX_mae`
- `val_actions/sample_table`
