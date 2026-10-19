#!/usr/bin/env python3
"""Validate predicted action chunks against held-out demonstration actions.

Modes:
- all-frames: score many frames from the validation dataset.
- first-frame: score one starting frame per held-out episode when frame_index is available.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import dataclasses
import json
import logging
import pathlib
import time
from typing import Any

import numpy as np
import wandb

import openpi.policies.policy_config as _policy_config
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms


def _path(value: str | pathlib.Path) -> pathlib.Path:
    return pathlib.Path(value).expanduser().resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True, help="Training config name used for the checkpoint.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint-dir", help="One checkpoint step dir, e.g. checkpoints/<config>/<exp>/5000.")
    group.add_argument("--checkpoint-root", help="Checkpoint root containing numeric step dirs to validate/watch.")
    parser.add_argument(
        "--val-repo-id",
        action="append",
        default=None,
        help="Held-out LeRobot repo/path. Repeat for a MultiLeRobot validation set. Defaults to config repo_id.",
    )
    parser.add_argument(
        "--mode",
        choices=("all-frames", "first-frame"),
        default="all-frames",
        help="all-frames scores dense held-out frames; first-frame scores episode starts only.",
    )
    parser.add_argument("--max-samples", type=int, default=1024, help="Maximum validation samples per checkpoint.")
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=None,
        help="Stride used by all-frames mode and fallback first-frame mode. "
        "Default: len(dataset) // max_samples, so samples span the whole held-out set.",
    )
    parser.add_argument("--action-dim", type=int, default=None, help="Optional action dimension cap before scoring.")
    parser.add_argument("--max-dim-metrics", type=int, default=32, help="Maximum per-dimension MAE metrics to log.")
    parser.add_argument(
        "--moving-threshold",
        type=float,
        default=0.3,
        help="A sample counts as a motion chunk when some live dim of its ground-truth chunk moves more "
        "than this (rad / gripper units) away from the current state; the rest are static chunks. "
        "val_actions/motion_chunks/* and val_actions/static_chunks/* score each group separately. 0.3 is "
        "~the median motion on the G2 towel val set, so roughly 2/3 of samples are motion chunks.",
    )
    parser.add_argument(
        "--gripper-dims",
        default="14,15",
        help="Comma-separated model-order gripper dims for val_actions/gripper/*. Empty string disables.",
    )
    parser.add_argument("--watch", action="store_true", help="Keep polling checkpoint-root and validate new steps.")
    parser.add_argument("--poll-seconds", type=float, default=60.0, help="Seconds between checkpoint-root polls.")
    parser.add_argument(
        "--metrics-jsonl",
        default=None,
        help="Append one JSON line per scored checkpoint here. scripts/train.py relays these into the "
        "training wandb run when started with VAL_METRICS_JSONL pointing at the same file "
        "(pair with --wandb-mode disabled to avoid a second run).",
    )
    parser.add_argument("--wandb-project", default=None, help="W&B project. Defaults to TrainConfig.project_name.")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name for validation metrics.")
    parser.add_argument("--wandb-group", default=None, help="Optional W&B group, useful to pair train/action-val runs.")
    parser.add_argument("--wandb-tag", action="append", default=None, help="Optional W&B tag. Repeatable.")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=None,
        help="Override W&B mode. If omitted, wandb/env defaults apply.",
    )
    return parser.parse_args()


def _with_validation_repo(data_factory: Any, repo_ids: Sequence[str] | None) -> Any:
    if not repo_ids:
        return data_factory
    repo_id: str | list[str] = repo_ids[0] if len(repo_ids) == 1 else list(repo_ids)
    if not dataclasses.is_dataclass(data_factory) or not hasattr(data_factory, "repo_id"):
        raise TypeError(f"Config data factory {type(data_factory).__name__} does not expose repo_id")
    return dataclasses.replace(data_factory, repo_id=repo_id)


def _checkpoint_step(checkpoint_dir: pathlib.Path) -> int:
    try:
        return int(checkpoint_dir.name)
    except ValueError:
        return 0


def _checkpoint_ready(checkpoint_dir: pathlib.Path) -> bool:
    return (checkpoint_dir / "params").exists() and (checkpoint_dir / "assets").exists()


def _find_checkpoint_dirs(checkpoint_root: pathlib.Path) -> list[pathlib.Path]:
    if not checkpoint_root.exists():
        return []
    checkpoints = [p for p in checkpoint_root.iterdir() if p.is_dir() and p.name.isdigit() and _checkpoint_ready(p)]
    return sorted(checkpoints, key=_checkpoint_step)


def _scalar_from_item(item: dict[str, Any], key: str) -> int | None:
    if key not in item:
        return None
    value = item[key]
    try:
        return int(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None


def _select_indices(
    dataset: _data_loader.Dataset, *, mode: str, max_samples: int, sample_stride: int | None
) -> list[int]:
    if max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if sample_stride is None:
        # A stride of 1 would score only the first max_samples frames of the first episode.
        sample_stride = max(1, len(dataset) // max_samples)
    if sample_stride <= 0:
        raise ValueError("--sample-stride must be positive")

    if mode == "all-frames":
        return list(range(0, min(len(dataset), max_samples * sample_stride), sample_stride))[:max_samples]

    indices: list[int] = []
    seen_episodes: set[int] = set()
    for index in range(len(dataset)):
        item = dataset[index]
        episode_index = _scalar_from_item(item, "episode_index")
        frame_index = _scalar_from_item(item, "frame_index")
        if episode_index is None or frame_index is None:
            break
        if frame_index == 0 and episode_index not in seen_episodes:
            indices.append(index)
            seen_episodes.add(episode_index)
            if len(indices) >= max_samples:
                return indices

    if indices:
        return indices

    logging.warning("Could not identify episode starts; falling back to sample_stride=%d", sample_stride)
    return list(range(0, min(len(dataset), max_samples * sample_stride), sample_stride))[:max_samples]


def _action_layout(data_config: _config.DataConfig) -> tuple[list[int] | None, np.ndarray | None]:
    """Return (raw-action indices in model order, live model dims) from the policy input transform.

    The policy outputs actions in model order (e.g. Go1: arms, grippers, waist), while the
    raw dataset `action` is in recording order. Without this remap, targets and predictions
    are compared dimension-by-dimension across different joints. Masked dims (action_mask
    True) are not trained, so they are excluded from the aggregate metrics.
    """
    for transform in data_config.data_transforms.inputs:
        indices = getattr(transform, "action_indices", None)
        if indices is not None:
            mask = getattr(transform, "action_mask", None)
            live = None if mask is None else ~np.asarray(mask, dtype=bool)
            return list(indices), live
    logging.warning("No action_indices found in data transforms; comparing raw action order as-is")
    return None, None


def _state_layout(data_config: _config.DataConfig) -> list[int] | None:
    """Raw-state indices in model order, used to build the hold-still baseline."""
    for transform in data_config.data_transforms.inputs:
        indices = getattr(transform, "state_indices", None)
        if indices is not None:
            return list(indices)
    logging.warning("No state_indices found in data transforms; hold-still baseline uses raw state order")
    return None


def _prepare_raw_dataset(
    train_config: _config.TrainConfig, val_repo_ids: Sequence[str] | None
) -> tuple[_data_loader.Dataset, _transforms.DataTransformFn, list[int] | None, np.ndarray | None, list[int] | None]:
    data_factory = _with_validation_repo(train_config.data, val_repo_ids)
    data_config = data_factory.create(train_config.assets_dirs, train_config.model)
    dataset = _data_loader.create_torch_dataset(data_config, train_config.model)
    # With several val repos, create_torch_dataset returns a MultiLeRobotDataset whose
    # children are wrapped for prompt_from_task, which breaks its own __len__/__getitem__
    # (children lack num_frames). TransformedDataset walks the children itself, exactly as
    # the training loader does; with no transforms it is an identity wrapper.
    dataset = _data_loader.TransformedDataset(dataset, [])
    repack = _transforms.compose(data_config.repack_transforms.inputs)
    action_indices, live_dims = _action_layout(data_config)
    return dataset, repack, action_indices, live_dims, _state_layout(data_config)


def _align_actions(predicted: np.ndarray, target: np.ndarray, action_dim: int | None) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if predicted.ndim == 1:
        predicted = predicted[None, :]
    if target.ndim == 1:
        target = target[None, :]
    horizon = min(predicted.shape[0], target.shape[0])
    dims = min(predicted.shape[-1], target.shape[-1])
    if action_dim is not None:
        dims = min(dims, action_dim)
    if horizon <= 0 or dims <= 0:
        raise ValueError(f"Cannot align predicted {predicted.shape} and target {target.shape}")
    return predicted[:horizon, :dims], target[:horizon, :dims]


def _score_sample(
    policy: Any,
    sample: dict[str, Any],
    *,
    action_dim: int | None,
    action_indices: list[int] | None = None,
    live_dims: np.ndarray | None = None,
    state_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Run one policy inference and score its action chunk against the ground truth.

    All errors are in raw (unnormalized) action units: radians for joints, the
    dataset's gripper units for grippers. Targets are remapped to model order with
    ``action_indices``; predictions and targets are truncated to the common horizon H
    and dim count D (capped at ``action_dim``). "Live" dims are the dims whose
    ``action_mask`` entry is False (all dims when no mask is known). Aggregate scalars
    use live dims only; ``abs_err``, ``hold_abs_err`` and ``per_dim_mae`` keep all D dims.

    The hold-still baseline predicts ``state[:D]`` (state remapped to model order with
    ``state_indices`` when the state is long enough) for every step of the chunk.

    Args:
        policy: Trained policy exposing ``infer(sample) -> {"actions": ...}``.
        sample: Repacked dataset item with ``actions`` [H, A] and ``state`` [S].
        action_dim: Optional cap on the number of scored dims.
        action_indices: Raw-action indices in model order, or None to keep raw order.
        live_dims: Boolean [D'] mask of trained dims, or None for all dims live.
        state_indices: Raw-state indices in model order, or None to keep raw order.

    Returns:
        Dict with per-sample arrays ``abs_err``/``hold_abs_err`` [H, D], ``pred``/``target``
        [H, D], ``state`` [D], ``live`` [D] bool, ``per_dim_mae`` [D] (mean over H of
        |pred - target|), and float scalars over live dims: ``mse``/``mae`` (mean over H
        and live dims of squared / absolute error), ``first_*``/``final_*`` (the same at
        the first / last chunk step), ``hold_mae``/``hold_first_mae``/``hold_final_mae``
        (baseline absolute error), ``motion`` (max |state - target| over H and live dims),
        ``horizon`` (H), ``dims`` (live dim count) and ``infer_ms`` (wall-clock
        milliseconds of ``policy.infer``, including any JIT compile on the first call).

    Raises:
        ValueError: If the sample lacks actions or state, the target is too short for
            ``action_indices``, the policy returns no actions, shapes cannot be aligned,
            or the state has fewer dims than the target.
    """
    target_actions = sample.get("actions")
    if target_actions is None:
        raise ValueError("Validation sample does not contain actions")
    if sample.get("state") is None:
        raise ValueError("Validation sample does not contain state (needed for the hold-still baseline)")
    # Policy input transforms may mutate the sample in place; keep GT and state immutable.
    target_actions = np.asarray(target_actions).copy()
    state = np.asarray(sample["state"], dtype=np.float32).copy().reshape(-1)
    if action_indices is not None:
        if target_actions.shape[-1] <= max(action_indices):
            raise ValueError(f"Target action dim {target_actions.shape[-1]} too small for indices {action_indices}")
        target_actions = target_actions[..., action_indices]
    if state_indices is not None and state.shape[-1] > max(state_indices):
        state = state[state_indices]

    start = time.time()
    result = policy.infer(sample)
    infer_ms = (time.time() - start) * 1000.0
    predicted_actions = result.get("actions")
    if predicted_actions is None:
        raise ValueError("Policy output does not contain actions")

    pred, target = _align_actions(predicted_actions, target_actions, action_dim)
    err = pred - target
    live = np.ones(pred.shape[1], dtype=bool)
    if live_dims is not None:
        live[: min(len(live_dims), len(live))] = live_dims[: len(live)]
    # Hold-still baseline: "the robot stays at its current state for the whole chunk".
    # Any policy worth deploying must beat this; on slow tasks it scores deceptively well.
    if state.shape[-1] < target.shape[-1]:
        raise ValueError(f"State dim {state.shape[-1]} smaller than target dim {target.shape[-1]}")
    hold_err = np.broadcast_to(state[: target.shape[-1]], target.shape) - target
    # Aggregates over live (trained) dims only; per-dim MAE still reports every dim.
    squared = np.square(err[:, live])
    absolute = np.abs(err[:, live])
    hold_absolute = np.abs(hold_err[:, live])
    return {
        "abs_err": np.abs(err),
        "hold_abs_err": np.abs(hold_err),
        # Kept for the gripper open/close event metrics, which need the trajectories themselves.
        "pred": pred,
        "target": target,
        "state": state[: target.shape[-1]],
        "live": live,
        "motion": float(np.max(hold_absolute)),
        "hold_mae": float(np.mean(hold_absolute)),
        "hold_first_mae": float(np.mean(hold_absolute[0])),
        "hold_final_mae": float(np.mean(hold_absolute[-1])),
        "mse": float(np.mean(squared)),
        "mae": float(np.mean(absolute)),
        "first_mse": float(np.mean(squared[0])),
        "first_mae": float(np.mean(absolute[0])),
        "final_mse": float(np.mean(squared[-1])),
        "final_mae": float(np.mean(absolute[-1])),
        "per_dim_mae": np.mean(np.abs(err), axis=0),
        "horizon": float(pred.shape[0]),
        "dims": float(live.sum()),
        "infer_ms": float(infer_ms),
    }


def validate_actions(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path,
    val_repo_ids: Sequence[str] | None,
    *,
    mode: str,
    max_samples: int,
    sample_stride: int,
    action_dim: int | None,
    max_dim_metrics: int,
    moving_threshold: float = 0.3,
    gripper_dims: Sequence[int] = (),
) -> dict[str, Any]:
    """Score one checkpoint's action chunks on held-out frames.

    Loads the policy from ``checkpoint_dir``, picks validation frames with
    ``_select_indices``, scores each with ``_score_sample`` and averages over samples.
    Units are raw action units (radians / gripper units); aggregates use live dims only.

    Args:
        train_config: Training config the checkpoint was trained with.
        checkpoint_dir: One checkpoint step directory (contains ``params`` and ``assets``).
        val_repo_ids: Held-out LeRobot repo ids/paths, or None to use the config's repo_id.
        mode: ``"all-frames"`` (strided frames) or ``"first-frame"`` (one frame_index 0
            frame per episode, falling back to strided frames).
        max_samples: Maximum number of scored frames.
        sample_stride: Frame stride, or None for ``max(1, len(dataset) // max_samples)``.
        action_dim: Optional cap on the number of scored dims.
        max_dim_metrics: Number of leading dims that get a ``dim_XX_mae`` metric.
        moving_threshold: A sample is a motion chunk when its ``motion`` exceeds this.
        gripper_dims: Model-order gripper dims for the ``gripper/*`` metrics.

    Returns:
        Dict of ``val_actions/*`` metrics: sample means of the per-sample ``mse``,
        ``mae``, ``first_*`` and ``final_*`` scalars; ``samples``; ``horizon_mean``;
        ``dims_mean``; ``infer_ms_mean``; ``checkpoint_step`` (step dir name, 0 if not
        numeric); ``dim_XX_mae`` (mean over samples and H, all dims, model order); the
        ``_baseline_metrics`` and ``_gripper_event_metrics`` outputs; and a
        ``sample_table`` ``wandb.Table`` with one row per scored frame.

    Raises:
        RuntimeError: If no validation frames are selected.
        ValueError: If ``max_samples`` or ``sample_stride`` is not positive, or a sample
            cannot be scored (see ``_score_sample``).
    """
    logging.info("Loading policy from %s", checkpoint_dir)
    policy = _policy_config.create_trained_policy(train_config, checkpoint_dir)
    dataset, repack, action_indices, live_dims, state_indices = _prepare_raw_dataset(train_config, val_repo_ids)
    fps = _dataset_fps(dataset)
    indices = _select_indices(dataset, mode=mode, max_samples=max_samples, sample_stride=sample_stride)
    if not indices:
        raise RuntimeError("No validation samples selected")

    rows: list[list[Any]] = []
    sample_metrics: list[dict[str, Any]] = []
    for raw_index in indices:
        raw_item = dataset[raw_index]
        sample = repack(raw_item)
        metrics = _score_sample(
            policy,
            sample,
            action_dim=action_dim,
            action_indices=action_indices,
            live_dims=live_dims,
            state_indices=state_indices,
        )
        episode_index = _scalar_from_item(raw_item, "episode_index")
        frame_index = _scalar_from_item(raw_item, "frame_index")
        sample_metrics.append(metrics)
        rows.append(
            [
                raw_index,
                episode_index,
                frame_index,
                metrics["mse"],
                metrics["mae"],
                metrics["first_mse"],
                metrics["first_mae"],
                metrics["hold_mae"],
                metrics["motion"],
                metrics["horizon"],
                metrics["dims"],
                metrics["infer_ms"],
            ]
        )

    per_dim_stack = np.stack([m["per_dim_mae"] for m in sample_metrics], axis=0)
    metrics_out: dict[str, Any] = {
        "val_actions/mse": float(np.mean([m["mse"] for m in sample_metrics])),
        "val_actions/mae": float(np.mean([m["mae"] for m in sample_metrics])),
        "val_actions/first_mse": float(np.mean([m["first_mse"] for m in sample_metrics])),
        "val_actions/first_mae": float(np.mean([m["first_mae"] for m in sample_metrics])),
        "val_actions/final_mse": float(np.mean([m["final_mse"] for m in sample_metrics])),
        "val_actions/final_mae": float(np.mean([m["final_mae"] for m in sample_metrics])),
        "val_actions/samples": float(len(sample_metrics)),
        "val_actions/horizon_mean": float(np.mean([m["horizon"] for m in sample_metrics])),
        "val_actions/dims_mean": float(np.mean([m["dims"] for m in sample_metrics])),
        "val_actions/infer_ms_mean": float(np.mean([m["infer_ms"] for m in sample_metrics])),
        "val_actions/checkpoint_step": float(_checkpoint_step(checkpoint_dir)),
    }
    for dim_index, value in enumerate(np.mean(per_dim_stack, axis=0)[:max_dim_metrics]):
        metrics_out[f"val_actions/dim_{dim_index:02d}_mae"] = float(value)
    metrics_out.update(_baseline_metrics(sample_metrics, moving_threshold=moving_threshold, gripper_dims=gripper_dims))
    metrics_out.update(_gripper_event_metrics(sample_metrics, gripper_dims=gripper_dims, fps=fps))

    metrics_out["val_actions/sample_table"] = wandb.Table(
        columns=[
            "raw_index",
            "episode_index",
            "frame_index",
            "mse",
            "mae",
            "first_mse",
            "first_mae",
            "baseline_hold_mae",
            "motion",
            "horizon",
            "dims",
            "infer_ms",
        ],
        data=rows,
    )
    # prune_motor.sh / overnight_v2.sh grep this line's prefix; keep "Action validation <dir> " intact.
    logging.info(
        "Action validation %s mode=%s: mse=%.6f mae=%.6f samples=%d | baseline_hold_mae=%.6f gain_vs_hold=%.4f "
        "motion_chunks/gain_vs_hold=%.4f (n=%d) static_chunks/gain_vs_hold=%.4f (n=%d) gripper/event_f1=%.4f",
        checkpoint_dir,
        mode,
        metrics_out["val_actions/mse"],
        metrics_out["val_actions/mae"],
        len(sample_metrics),
        metrics_out["val_actions/baseline_hold_mae"],
        metrics_out["val_actions/gain_vs_hold"],
        metrics_out["val_actions/motion_chunks/gain_vs_hold"],
        int(metrics_out["val_actions/motion_chunks/n"]),
        metrics_out["val_actions/static_chunks/gain_vs_hold"],
        int(metrics_out["val_actions/static_chunks/n"]),
        metrics_out.get("val_actions/gripper/event_f1", float("nan")),
    )
    return metrics_out


def _dataset_fps(dataset: Any) -> float:
    """Recording fps of the val set, used to turn chunk steps into milliseconds."""
    inner = getattr(dataset, "_dataset", dataset)  # TransformedDataset wrapper
    try:
        return float(inner.fps)
    except Exception:  # MultiLeRobotDataset children may be wrapped and hide .fps
        pass
    try:
        return float(inner._datasets[0].fps)
    except Exception as e:
        logging.warning("Could not read dataset fps (%s); assuming 30 for gripper onset_error_ms", e)
        return 30.0


def _gain_vs_hold(model_mae: float, hold_mae: float) -> float:
    """1 - model/baseline_hold: 0 = no better than not moving, 1 = perfect, <0 = worse than not moving."""
    return float(1.0 - model_mae / hold_mae) if hold_mae > 0 else float("nan")


def _baseline_metrics(
    sample_metrics: list[dict[str, Any]], *, moving_threshold: float, gripper_dims: Sequence[int]
) -> dict[str, float]:
    """Compute hold-still baseline, gain scores, per-step error, motion/static and gripper breakdowns.

    Baseline "hold": a policy that outputs the current state for every step of the chunk,
    i.e. the robot does not move. gain_vs_hold measures how much better the model is.
    All means below are over samples N, chunk steps H and live dims (the live mask of the
    first sample), except the gripper metrics, which use ``gripper_dims`` regardless of
    the live mask.

    Args:
        sample_metrics: Per-sample outputs of ``_score_sample``; all must share [H, D].
        moving_threshold: A sample is a motion chunk when its ``motion`` (max |state -
            target| over H and live dims) exceeds this; otherwise it is a static chunk.
        gripper_dims: Model-order gripper dims; dims >= D are ignored.

    Returns:
        Dict of ``val_actions/*`` floats: ``baseline_hold_mae`` (and ``_first``/``_final``
        at step 0 / H-1); ``gain_vs_hold`` = 1 - model_mae / hold_mae (NaN when
        hold_mae is 0), and ``gain_vs_hold_last_step`` at step H-1; ``step_error/tXX_mae``
        and ``step_error/tXX_baseline_hold_mae`` per chunk step; for ``motion_chunks``
        and ``static_chunks``: ``n``, ``mae``, ``baseline_hold_mae`` and ``gain_vs_hold``
        (NaN when the group is empty); and ``gripper/mae`` / ``gripper/baseline_hold_mae``
        when any gripper dim is in range.
    """
    abs_err = np.stack([m["abs_err"] for m in sample_metrics])  # [N, H, D]
    hold_abs = np.stack([m["hold_abs_err"] for m in sample_metrics])
    live = sample_metrics[0]["live"]
    model_live, hold_live = abs_err[..., live], hold_abs[..., live]

    out: dict[str, float] = {
        "val_actions/baseline_hold_mae": float(np.mean(hold_live)),
        "val_actions/baseline_hold_first_mae": float(np.mean(hold_live[:, 0])),
        "val_actions/baseline_hold_final_mae": float(np.mean(hold_live[:, -1])),
        "val_actions/gain_vs_hold": _gain_vs_hold(float(np.mean(model_live)), float(np.mean(hold_live))),
        "val_actions/gain_vs_hold_last_step": _gain_vs_hold(
            float(np.mean(model_live[:, -1])), float(np.mean(hold_live[:, -1]))
        ),
    }

    # Error at each step of the chunk (t00 = first action, t29 = last for H=30), model and baseline.
    for t, (model_t, hold_t) in enumerate(
        zip(np.mean(model_live, axis=(0, 2)), np.mean(hold_live, axis=(0, 2)), strict=True)
    ):
        out[f"val_actions/step_error/t{t:02d}_mae"] = float(model_t)
        out[f"val_actions/step_error/t{t:02d}_baseline_hold_mae"] = float(hold_t)

    # motion chunks: ground truth travels far -- does the policy predict the motion?
    # static chunks: ground truth barely moves -- does the policy hold position instead of drifting?
    moving = np.array([m["motion"] > moving_threshold for m in sample_metrics])
    for name, subset in (("motion_chunks", moving), ("static_chunks", ~moving)):
        out[f"val_actions/{name}/n"] = float(subset.sum())
        if subset.any():
            model_mae, hold_mae = float(np.mean(model_live[subset])), float(np.mean(hold_live[subset]))
            out[f"val_actions/{name}/mae"] = model_mae
            out[f"val_actions/{name}/baseline_hold_mae"] = hold_mae
            out[f"val_actions/{name}/gain_vs_hold"] = _gain_vs_hold(model_mae, hold_mae)
        else:
            logging.warning("No %s at threshold %.3f; %s/* metrics are NaN", name, moving_threshold, name)
            for key in ("mae", "baseline_hold_mae", "gain_vs_hold"):
                out[f"val_actions/{name}/{key}"] = float("nan")

    dims = [d for d in gripper_dims if d < abs_err.shape[-1]]
    if dims:
        out["val_actions/gripper/mae"] = float(np.mean(abs_err[..., dims]))
        out["val_actions/gripper/baseline_hold_mae"] = float(np.mean(hold_abs[..., dims]))
    return out


# A predicted open/close counts as matching a ground-truth one when it has the same direction
# and happens within this many chunk steps (6 steps = 200 ms at 30 fps).
GRIPPER_EVENT_TOLERANCE_STEPS = 6
# Hysteresis around the open/close threshold, as a fraction of the gripper range, so jitter
# near the threshold is not counted as repeated toggles.
GRIPPER_EVENT_BAND = 0.1


def _gripper_events(seq: np.ndarray, mid: float, band: float) -> list[tuple[int, int]]:
    """(chunk step, direction) for each threshold crossing; seq[0] is the current state, seq[1:] the chunk."""
    above = bool(seq[0] > mid)
    events = []
    for i in range(1, len(seq)):
        if above and seq[i] < mid - band:
            above = False
            events.append((i - 1, -1))
        elif not above and seq[i] > mid + band:
            above = True
            events.append((i - 1, 1))
    return events


def _gripper_event_metrics(
    sample_metrics: list[dict[str, Any]], *, gripper_dims: Sequence[int], fps: float
) -> dict[str, float]:
    """Score gripper open/close timing: does the predicted chunk toggle when the ground truth does?

    Threshold per gripper dim = midpoint of its range over the val samples. Events are threshold
    crossings of [current state, chunk...]. The hold baseline never has events, so there is no
    baseline here: recall is how many real open/closes the model predicted, precision how many
    predicted ones were real, false_toggles the unmatched predicted ones per chunk.

    Per dim, the range [lo, hi] is taken over the state and target values of all samples;
    a crossing needs to pass mid +/- ``GRIPPER_EVENT_BAND * (hi - lo)``. Dims with
    hi - lo < 1e-6 are skipped. A predicted event matches a ground-truth one with the same
    direction within ``GRIPPER_EVENT_TOLERANCE_STEPS`` chunk steps (greedy, nearest first).

    Args:
        sample_metrics: Per-sample outputs of ``_score_sample`` (needs ``state``, ``target``,
            ``pred``).
        gripper_dims: Model-order gripper dims; dims >= D are ignored.
        fps: Dataset recording rate, used to convert chunk steps to milliseconds.

    Returns:
        Empty dict when no gripper dim is in range; otherwise ``val_actions/gripper/*``
        floats: ``gt_events``, ``pred_events``, ``event_recall`` (matched / gt),
        ``event_precision`` (matched / pred), ``event_f1`` (2 * matched / (gt + pred)),
        ``onset_error_ms`` (mean |step offset| of matches * 1000 / fps) and
        ``false_toggles`` ((pred - matched) / number of samples). Ratios with a zero
        denominator, and onset_error_ms with no matches, are NaN.
    """
    dims = [d for d in gripper_dims if d < sample_metrics[0]["target"].shape[-1]]
    if not dims:
        return {}
    n_gt = n_pred = n_matched = 0
    onset_errors: list[int] = []
    for d in dims:
        values = np.concatenate([np.concatenate([[m["state"][d]], m["target"][:, d]]) for m in sample_metrics])
        lo, hi = float(np.min(values)), float(np.max(values))
        if hi - lo < 1e-6:
            logging.warning("Gripper dim %d never moves in the val samples; skipped for event metrics", d)
            continue
        mid, band = (lo + hi) / 2.0, GRIPPER_EVENT_BAND * (hi - lo)
        for m in sample_metrics:
            gt = _gripper_events(np.concatenate([[m["state"][d]], m["target"][:, d]]), mid, band)
            pred = _gripper_events(np.concatenate([[m["state"][d]], m["pred"][:, d]]), mid, band)
            n_gt, n_pred = n_gt + len(gt), n_pred + len(pred)
            unmatched = list(pred)
            for t_gt, direction in gt:
                candidates = [
                    p for p in unmatched if p[1] == direction and abs(p[0] - t_gt) <= GRIPPER_EVENT_TOLERANCE_STEPS
                ]
                if candidates:
                    best = min(candidates, key=lambda p: abs(p[0] - t_gt))
                    unmatched.remove(best)
                    n_matched += 1
                    onset_errors.append(abs(best[0] - t_gt))

    recall = n_matched / n_gt if n_gt else float("nan")
    precision = n_matched / n_pred if n_pred else float("nan")
    f1 = 2 * n_matched / (n_gt + n_pred) if (n_gt + n_pred) else float("nan")
    return {
        "val_actions/gripper/gt_events": float(n_gt),
        "val_actions/gripper/pred_events": float(n_pred),
        "val_actions/gripper/event_recall": float(recall),
        "val_actions/gripper/event_precision": float(precision),
        "val_actions/gripper/event_f1": float(f1),
        "val_actions/gripper/onset_error_ms": float(np.mean(onset_errors) * 1000.0 / fps)
        if onset_errors
        else float("nan"),
        "val_actions/gripper/false_toggles": float((n_pred - n_matched) / len(sample_metrics)),
    }


def _append_metrics_jsonl(path: str, metrics: dict[str, Any]) -> None:
    """Append one complete line per checkpoint, written in a single call so readers never see half a record.

    Args:
        path: JSONL file to append to (created if missing).
        metrics: Output of ``validate_actions``; scalars go to ``"metrics"`` and the
            ``val_actions/sample_table`` (if any) to ``"sample_table"``.
    """
    scalars = {k: float(v) for k, v in metrics.items() if isinstance(v, int | float | np.floating | np.integer)}
    record: dict[str, Any] = {"metrics": scalars}
    table = metrics.get("val_actions/sample_table")
    if table is not None:
        record["sample_table"] = {"columns": list(table.columns), "data": table.data}
    line = json.dumps(record, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n"
    with pathlib.Path(path).open("a") as f:
        f.write(line)
        f.flush()


def _init_wandb(args: argparse.Namespace, train_config: _config.TrainConfig) -> None:
    init_kwargs: dict[str, Any] = {
        "project": args.wandb_project or train_config.project_name,
        "name": args.wandb_run_name or f"{args.config_name}-action-validation",
        "group": args.wandb_group,
        "tags": args.wandb_tag,
        "config": vars(args),
    }
    if args.wandb_mode is not None:
        init_kwargs["mode"] = args.wandb_mode
    wandb.init(**init_kwargs)


def main() -> None:
    """Parse CLI args and validate one checkpoint, or poll a checkpoint root with ``--watch``.

    Each scored checkpoint is logged to wandb at its step and, with ``--metrics-jsonl``,
    appended to that file. Steps already recorded in the file are skipped.
    """
    # force=True: openpi imports configure the root logger first, which made this a no-op
    # and silently dropped every INFO line (including the per-checkpoint summary).
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    args = _parse_args()
    train_config = _config.get_config(args.config_name)
    _init_wandb(args, train_config)

    validated_steps: set[int] = set()
    if args.metrics_jsonl and pathlib.Path(args.metrics_jsonl).exists():
        # Resumed run appending to the same file: do not re-score checkpoints already in it.
        with pathlib.Path(args.metrics_jsonl).open() as f:
            for line in f:
                try:
                    validated_steps.add(int(json.loads(line)["metrics"]["val_actions/checkpoint_step"]))
                except (ValueError, KeyError):
                    continue
        logging.info("Skipping %d checkpoint steps already in %s", len(validated_steps), args.metrics_jsonl)
    while True:
        if args.checkpoint_dir:
            checkpoint_dirs = [_path(args.checkpoint_dir)]
        else:
            checkpoint_dirs = _find_checkpoint_dirs(_path(args.checkpoint_root))

        new_dirs = [p for p in checkpoint_dirs if _checkpoint_step(p) not in validated_steps]
        for checkpoint_dir in new_dirs:
            step = _checkpoint_step(checkpoint_dir)
            metrics = validate_actions(
                train_config,
                checkpoint_dir,
                args.val_repo_id,
                mode=args.mode,
                max_samples=args.max_samples,
                sample_stride=args.sample_stride,
                action_dim=args.action_dim,
                max_dim_metrics=args.max_dim_metrics,
                moving_threshold=args.moving_threshold,
                gripper_dims=[int(d) for d in args.gripper_dims.split(",") if d.strip()],
            )
            metrics["val_actions/mode_id"] = 0.0 if args.mode == "all-frames" else 1.0
            if args.metrics_jsonl:
                _append_metrics_jsonl(args.metrics_jsonl, metrics)
            wandb.log(metrics, step=step if step > 0 else None)
            validated_steps.add(step)

        if not args.watch:
            break
        if not new_dirs:
            logging.info("No new checkpoints under %s; sleeping %.1fs", args.checkpoint_root, args.poll_seconds)
        time.sleep(args.poll_seconds)

    wandb.finish()


if __name__ == "__main__":
    main()
