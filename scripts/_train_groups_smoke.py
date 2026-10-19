import dataclasses
import functools
import json
import logging
import platform
import random
import time
from typing import Any
import os
import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


# TensorBoard mirror. wandb is the upstream logger, but this org's wandb
# credentials lack write access, so runs go to wandb offline and TensorBoard
# provides the live view. Purely additive: it never changes training math.
_tb_writer = None


def init_tensorboard(config: _config.TrainConfig):
    global _tb_writer
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        logging.warning("tensorboardX not installed; TensorBoard logging disabled")
        return
    logdir = epath.Path(config.checkpoint_dir) / "tb"
    logdir.mkdir(parents=True, exist_ok=True)
    _tb_writer = SummaryWriter(logdir=str(logdir))
    logging.info(f"TensorBoard logging to {logdir}")


def tb_log(metrics: dict, step: int):
    if _tb_writer is None:
        return
    try:
        for k, v in metrics.items():
            _tb_writer.add_scalar(k, float(v), step)
        _tb_writer.flush()
    except Exception as e:  # never take training down over a log write
        logging.warning(f"tensorboard log failed: {e}")


# Held-out validation relay. scripts/validate_actions.py runs as a sidecar process (its
# own GPU memory, its own crash domain) and appends one JSON line per scored checkpoint
# to $VAL_METRICS_JSONL. The trainer relays those lines into ITS wandb run, so train and
# val metrics live in one run instead of two. val_actions/* is plotted against
# val_actions/checkpoint_step, so each point sits at the checkpoint it scored even though
# it is relayed a few hundred training steps later. Purely additive: unset -> no-op.
_val_path = None
_val_offset = 0
_val_relayed: set[int] = set()


def init_val_relay(resuming: bool = False):
    global _val_path, _val_offset
    path = os.getenv("VAL_METRICS_JSONL")
    if not path:
        return
    _val_path = epath.Path(path)
    wandb.define_metric("val_actions/checkpoint_step")
    wandb.define_metric("val_actions/*", step_metric="val_actions/checkpoint_step")
    if resuming and _val_path.exists():
        # Records already in the file were relayed by the previous leg of this run;
        # start after them so the resumed wandb run does not get them twice.
        _val_offset = os.path.getsize(_val_path)
        logging.info(f"Resuming: skipping {_val_offset} bytes of already-relayed validation records")
    logging.info(f"Relaying validation metrics from {_val_path} into this wandb run")


def relay_val_metrics(step: int) -> None:
    """Log at most ONE new validator record at training step `step`.

    One per call: two records logged at the same wandb step would overwrite each other.
    A trailing partial line (validator mid-write) is left for the next call.
    """
    global _val_offset
    if _val_path is None or not _val_path.exists():
        return
    try:
        with open(_val_path, "rb") as f:
            f.seek(_val_offset)
            chunk = f.read()
        if b"\n" not in chunk:
            return
        raw = chunk.split(b"\n", 1)[0]
        _val_offset += len(raw) + 1
        if not raw.strip():
            return
        record = json.loads(raw)
        metrics = dict(record["metrics"])
        table = record.get("sample_table")
        if table:
            metrics["val_actions/sample_table"] = wandb.Table(columns=table["columns"], data=table["data"])
        ckpt_step = int(metrics["val_actions/checkpoint_step"])
        wandb.log(metrics, step=step)
        tb_log({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, step=ckpt_step)
        _val_relayed.add(ckpt_step)
        logging.info(f"Relayed validation of checkpoint {ckpt_step} at train step {step}")
    except Exception as e:  # never take training down over a validation relay
        logging.warning(f"validation relay failed: {e}")


def drain_val_metrics(final_step: int, last_logged_step: int) -> None:
    """After training: wait for the final checkpoint's validation, relaying anything pending."""
    if _val_path is None:
        return
    timeout_s = float(os.getenv("VAL_FINAL_WAIT_S", "1800"))
    step, waited = last_logged_step, 0.0
    while final_step not in _val_relayed and waited < timeout_s:
        before = _val_offset
        step += 1
        relay_val_metrics(step)
        if _val_offset == before:  # nothing new yet
            time.sleep(30)
            waited += 30
    if final_step in _val_relayed:
        logging.info(f"Final checkpoint {final_step} validation relayed")
    else:
        logging.warning(f"Gave up waiting for validation of checkpoint {final_step} after {timeout_s:.0f}s")


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


# Per-module breakdown of grad_norm / param_norm for wandb. Same path regexes as
# Pi0Config.get_freeze_filter, so a frozen group simply reports grad_norm 0. Logging only:
# none of these values feed back into the update.
_PARAM_GROUPS = {
    "vision": nnx_utils.PathRegex(".*img.*"),
    "llm": nnx_utils.PathRegex(".*llm(?!.*_1|.*_2).*"),
    "action_expert": nnx_utils.PathRegex(".*llm.*_1.*"),
}


def _f32_norm(tree) -> at.Array:
    # Frozen groups are stored in bfloat16 (see init_train_state); square-and-sum per leaf in
    # float32 so the norm is exact and always a float32 scalar (XLA fuses the cast per leaf).
    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.float32(0.0)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x.astype(jnp.float32))) for x in leaves))


def _group_norms(tree: nnx.State, prefix: str) -> dict[str, at.Array]:
    out = {f"{prefix}/{name}": _f32_norm(tree.filter(f)) for name, f in _PARAM_GROUPS.items()}
    out[f"{prefix}/other"] = _f32_norm(tree.filter(nnx.Not(nnx.Any(*_PARAM_GROUPS.values()))))
    return out


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    info.update(_group_norms(grads, "grad_norm"))
    info.update(_group_norms(kernel_params, "param_norm"))
    return new_state, info

@at.typecheck
def acot_train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions, _model.CoarseActions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions,
        coarse_actions: _model.CoarseActions
    ):
        return model.compute_loss(rng, observation, actions, coarse_actions, train=True)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions, coarse_actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions, coarse_actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info

def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    # Seed Python and NumPy global RNGs alongside JAX so every randomness source
    # in data sampling / SafeDataset / FrameSampler is reproducible from config.seed.
    random.seed(config.seed)
    np.random.seed(config.seed)

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite if not os.getenv("DEBUG_MODE", default=False) == "true" else True,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    init_tensorboard(config)
    init_val_relay(resuming)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")
    num_params = training_utils.count_parameters(train_state.params)
    logging.info(f"Total number of parameters: {num_params:,}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    if config.model.model_type == _model.ModelType.ACOT_VLA_PI05 or config.model.model_type == _model.ModelType.ACOT_VLA_PI0:
        ptrain_step = jax.jit(
            functools.partial(acot_train_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )
    else:
        ptrain_step = jax.jit(
            functools.partial(train_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )

    start_step = int(train_state.step)
    print("\n--- Trainable Parameters ---")
    model = nnx.merge(train_state.model_def, train_state.params)
    trainable_state = nnx.state(model, config.trainable_filter)
    logging.info(f"{training_utils.array_tree_to_info(trainable_state)}")
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    last_logged_step = start_step
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            tb_log(reduced_info, step=step)
            relay_val_metrics(step)
            last_logged_step = step
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()
    drain_val_metrics(config.num_train_steps - 1, last_logged_step)


if __name__ == "__main__":
    main(_config.cli())
