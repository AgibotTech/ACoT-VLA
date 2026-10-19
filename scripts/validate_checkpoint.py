#!/usr/bin/env python3
"""Run held-out validation for a trained openpi checkpoint.

This is intentionally a sidecar to scripts/train.py: point it at a held-out
LeRobot dataset and either one checkpoint directory or a checkpoint root to
watch while fine-tuning is running. Metrics are logged to Weights & Biases.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import dataclasses
import logging
import pathlib
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import wandb

import openpi.models.model as _model
import openpi.shared.normalize as _normalize
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def _path(value: str | pathlib.Path) -> pathlib.Path:
    return pathlib.Path(value).expanduser().resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True, help="Training config name used for the fine-tuned model.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint-dir", help="One checkpoint step dir, e.g. checkpoints/<config>/<exp>/5000.")
    group.add_argument("--checkpoint-root", help="Checkpoint root containing numeric step dirs to validate/watch.")
    parser.add_argument(
        "--val-repo-id",
        action="append",
        default=None,
        help="Held-out LeRobot repo/path. Repeat for a MultiLeRobot validation set. Defaults to config repo_id.",
    )
    parser.add_argument("--max-batches", type=int, default=20, help="Validation batches per checkpoint.")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Validation batch size. Defaults to config batch_size."
    )
    parser.add_argument("--num-workers", type=int, default=2, help="Torch DataLoader workers for validation.")
    parser.add_argument("--seed", type=int, default=42, help="Validation data/RNG seed.")
    parser.add_argument("--watch", action="store_true", help="Keep polling checkpoint-root and validate new steps.")
    parser.add_argument("--poll-seconds", type=float, default=60.0, help="Seconds between checkpoint-root polls.")
    parser.add_argument("--wandb-project", default=None, help="W&B project. Defaults to TrainConfig.project_name.")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name for validation metrics.")
    parser.add_argument("--wandb-group", default=None, help="Optional W&B group, useful to pair train/val runs.")
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


def _load_checkpoint_norm_stats(
    checkpoint_dir: pathlib.Path, data_config: _config.DataConfig
) -> dict[str, _normalize.NormStats]:
    assets_dir = checkpoint_dir / "assets"
    if (assets_dir / "norm_stats.json").exists():
        logging.info("Loading validation norm stats from %s", assets_dir)
        return _normalize.load(assets_dir)
    if data_config.asset_id is None:
        raise ValueError(f"No norm_stats.json at {assets_dir} and data_config.asset_id is None")
    return _checkpoints.load_norm_stats(assets_dir, data_config.asset_id)


def _create_validation_loader(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path,
    val_repo_ids: Sequence[str] | None,
    *,
    batch_size: int,
    max_batches: int,
    num_workers: int,
    seed: int,
) -> _data_loader.DataLoader:
    data_factory = _with_validation_repo(train_config.data, val_repo_ids)
    data_config = data_factory.create(train_config.assets_dirs, train_config.model)
    data_config = dataclasses.replace(data_config, norm_stats=_load_checkpoint_norm_stats(checkpoint_dir, data_config))

    if data_config.rlds_data_dir is not None:
        return _data_loader.create_rlds_data_loader(
            data_config,
            action_horizon=train_config.model.action_horizon,
            batch_size=batch_size,
            shuffle=False,
            num_batches=max_batches,
            seed=seed,
        )

    return _data_loader.create_torch_data_loader(
        data_config,
        model_config=train_config.model,
        action_horizon=train_config.model.action_horizon,
        batch_size=batch_size,
        shuffle=False,
        num_batches=max_batches,
        num_workers=num_workers,
        seed=seed,
    )


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


def _batch_loss(model: _model.BaseModel, rng: jax.Array, batch: tuple[Any, ...]) -> jax.Array:
    """Compute the mean training loss of one batch with ``train=False``.

    Args:
        model: Loaded model in eval mode.
        rng: Key for the loss's noise / timestep sampling.
        batch: ``(observation, actions)`` or ``(observation, actions, coarse_actions)``.

    Returns:
        Scalar mean of ``model.compute_loss`` over all of its elements.

    Raises:
        ValueError: If the batch is not a 2- or 3-tuple.
    """
    if len(batch) == 3:
        observation, actions, coarse_actions = batch
        return jnp.mean(model.compute_loss(rng, observation, actions, coarse_actions, train=False))
    if len(batch) == 2:
        observation, actions = batch
        return jnp.mean(model.compute_loss(rng, observation, actions, train=False))
    raise ValueError(f"Unexpected validation batch shape: tuple length {len(batch)}")


def validate_checkpoint(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path,
    val_repo_ids: Sequence[str] | None,
    *,
    batch_size: int,
    max_batches: int,
    num_workers: int,
    seed: int,
) -> dict[str, float]:
    """Compute the held-out training loss of one checkpoint.

    Loads ``params`` in bfloat16, builds an unshuffled loader normalized with the
    checkpoint's own norm stats, and averages the per-batch loss of ``_batch_loss``
    over up to ``max_batches`` batches. The loss is the model's own training loss in
    normalized action space; the RNG is reseeded with ``seed`` for every checkpoint.

    Args:
        train_config: Training config the checkpoint was trained with.
        checkpoint_dir: One checkpoint step directory (contains ``params`` and ``assets``).
        val_repo_ids: Held-out LeRobot repo ids/paths, or None to use the config's repo_id.
        batch_size: Validation batch size.
        max_batches: Maximum number of batches to score.
        num_workers: Torch DataLoader workers.
        seed: Loader and loss RNG seed.

    Returns:
        Dict with ``val/loss`` (mean of batch losses), ``val/loss_std`` (std of batch
        losses), ``val/batches``, ``val/examples`` (batches * batch_size),
        ``val/elapsed_s`` (wall-clock seconds of the scoring loop) and
        ``val/checkpoint_step`` (step dir name, 0 if not numeric).

    Raises:
        RuntimeError: If the loader yields no batches.
        ValueError: If no norm stats can be found for the checkpoint.
    """
    logging.info("Loading checkpoint from %s", checkpoint_dir)
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    model.eval()

    loader = _create_validation_loader(
        train_config,
        checkpoint_dir,
        val_repo_ids,
        batch_size=batch_size,
        max_batches=max_batches,
        num_workers=num_workers,
        seed=seed,
    )

    losses: list[float] = []
    rng = jax.random.key(seed)
    start = time.time()
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        rng, loss_rng = jax.random.split(rng)
        loss = _batch_loss(model, loss_rng, batch)
        losses.append(float(jax.device_get(loss)))

    if not losses:
        raise RuntimeError("Validation loader produced zero batches")

    elapsed_s = time.time() - start
    metrics = {
        "val/loss": float(np.mean(losses)),
        "val/loss_std": float(np.std(losses)),
        "val/batches": float(len(losses)),
        "val/examples": float(len(losses) * batch_size),
        "val/elapsed_s": float(elapsed_s),
        "val/checkpoint_step": float(_checkpoint_step(checkpoint_dir)),
    }
    logging.info(
        "Validation %s: loss=%.6f loss_std=%.6f batches=%d elapsed_s=%.1f",
        checkpoint_dir,
        metrics["val/loss"],
        metrics["val/loss_std"],
        len(losses),
        elapsed_s,
    )
    return metrics


def _init_wandb(args: argparse.Namespace, train_config: _config.TrainConfig) -> None:
    init_kwargs: dict[str, Any] = {
        "project": args.wandb_project or train_config.project_name,
        "name": args.wandb_run_name or f"{args.config_name}-validation",
        "group": args.wandb_group,
        "tags": args.wandb_tag,
        "config": {
            "config_name": args.config_name,
            "checkpoint_dir": args.checkpoint_dir,
            "checkpoint_root": args.checkpoint_root,
            "val_repo_id": args.val_repo_id,
            "max_batches": args.max_batches,
            "batch_size": args.batch_size or train_config.batch_size,
            "num_workers": args.num_workers,
            "watch": args.watch,
        },
    }
    if args.wandb_mode is not None:
        init_kwargs["mode"] = args.wandb_mode
    wandb.init(**init_kwargs)


def main() -> None:
    """Parse CLI args and validate one checkpoint, or poll a checkpoint root with ``--watch``.

    Raises:
        ValueError: If ``--max-batches`` is not positive.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()
    if args.max_batches <= 0:
        raise ValueError("--max-batches must be positive")

    train_config = _config.get_config(args.config_name)
    batch_size = args.batch_size or train_config.batch_size
    _init_wandb(args, train_config)

    validated_steps: set[int] = set()
    while True:
        if args.checkpoint_dir:
            checkpoint_dirs = [_path(args.checkpoint_dir)]
        else:
            checkpoint_dirs = _find_checkpoint_dirs(_path(args.checkpoint_root))

        new_dirs = [p for p in checkpoint_dirs if _checkpoint_step(p) not in validated_steps]
        for checkpoint_dir in new_dirs:
            step = _checkpoint_step(checkpoint_dir)
            metrics = validate_checkpoint(
                train_config,
                checkpoint_dir,
                args.val_repo_id,
                batch_size=batch_size,
                max_batches=args.max_batches,
                num_workers=args.num_workers,
                seed=args.seed,
            )
            metrics["val/checkpoint_step"] = float(step)
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
