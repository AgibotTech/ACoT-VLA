from collections.abc import Sequence
import logging
import os
import pathlib
import time
from typing import Any, TypeAlias
import copy
import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

logger = logging.getLogger("openpi.policy")

BasePolicy: TypeAlias = _base_policy.BasePolicy

# LeRobot-style RTC request, set by the RTC stream server (serving/rtc_stream_server.py):
#   {"prev_actions": [L, W] absolute leftover in the policy's output layout (or None),
#    "inference_delay": int, "execution_horizon": int, "max_guidance_weight": float}
_RTC_OPTIONS_KEY = "_rtc"


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        fresh_noise: bool = False,
    ):
        """Wrap a model with input/output transforms and a jitted sampler.

        Args:
            model: Model whose ``sample_actions`` is jitted.
            rng: PRNG key for diffusion noise; defaults to ``jax.random.key(0)``.
            transforms: Input transforms applied to each observation.
            output_transforms: Output transforms applied to each model result.
            sample_kwargs: Extra keyword arguments passed to ``sample_actions``.
            metadata: Policy metadata.
            fresh_noise: If True, split the key on every call (new noise per chunk). If False, reuse one fixed key so
                identical payloads give identical outputs.
        """
        # The RTC prefix schedule is a Python string, so it must be static under jit.
        self._sample_actions = nnx_utils.module_jit(
            model.sample_actions, static_argnames=("rtc_prefix_attention_schedule",)
        )
        # False: the same noise every call (reproducible).
        # True: split the key every call, like LeRobot/Anvil (new noise per chunk).
        self._fresh_noise = fresh_noise
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._action_horizon = getattr(model, "action_horizon", None)
        self._action_dim = getattr(model, "action_dim", None)

    def _rtc_prefix_rows(self, rtc: dict) -> tuple[np.ndarray, int, int]:
        """LeRobot ``_normalize_prev_actions_length`` + pad to the model horizon.

        Leftover is truncated / hold-padded to ``execution_horizon`` rows (LeRobot),
        then hold-padded to ``action_horizon`` so the jitted graph sees one shape
        (rows past the horizon get zero guidance weight). Returns (rows, H, width).
        No leftover -> H = 0, i.e. every guidance weight is zero and the chunk is
        exactly the unguided one, through the same compiled graph.
        """
        horizon = int(self._action_horizon)
        prev = rtc.get("prev_actions")
        prev = None if prev is None else np.asarray(prev, dtype=np.float64)
        if prev is None or prev.size == 0:
            return None, 0, 0
        if prev.ndim != 2:
            raise ValueError(f"rtc prev_actions must be 2D [rows, dims], got shape {prev.shape}")
        h = max(1, min(int(rtc.get("execution_horizon", 10)), horizon))
        prev = prev[:h]
        if prev.shape[0] < horizon:
            prev = np.concatenate([prev, np.repeat(prev[-1:], horizon - prev.shape[0], axis=0)], axis=0)
        return prev, h, prev.shape[1]

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        """Run one inference, optionally with RTC guidance.

        Args:
            obs: Unbatched observation. May carry an RTC request under ``"_rtc"``, a dict with:
                ``prev_actions``: [rows, dims] absolute leftover in the policy's output layout, or None/empty
                (no guidance, same compiled graph); ``inference_delay``: int, default 0; ``execution_horizon``:
                int, default 10, clamped to [1, action_horizon]; ``max_guidance_weight``: float, default 10.0;
                ``prefix_attention_schedule``: optional str (static under jit; each new value recompiles).

        Returns:
            Post-processed outputs with "actions", "state" and "policy_timing", plus "rtc"
            (``inference_delay``, ``execution_horizon``, ``prev_rows``) when an RTC request was given.

        Raises:
            ValueError: If ``_rtc["prev_actions"]`` is not 2D.
        """
        # Make a copy since transformations may modify the inputs in place.
        rtc = obs.get(_RTC_OPTIONS_KEY)
        inputs = jax.tree.map(lambda x: x, {k: v for k, v in obs.items() if k != _RTC_OPTIONS_KEY})
        rtc_kwargs = {}
        rtc_info = None
        if rtc is not None:
            prev_abs, horizon_rows, width = self._rtc_prefix_rows(rtc)
            if prev_abs is not None:
                # LeRobot reanchor_relative_rtc_prefix: the leftover is absolute, the model
                # works in normalized delta-from-state space. Run it through this policy's own
                # input transforms (remap/mask -> DeltaActions vs the CURRENT state ->
                # Normalize) so it is expressed exactly like a training target for this obs.
                inputs["actions"] = prev_abs.copy()
        inputs = self._input_transform(inputs)
        if rtc is not None:
            if prev_abs is not None:
                model_prev = np.asarray(inputs.pop("actions"), dtype=np.float32)
                dim_mask = np.zeros(model_prev.shape[-1], dtype=np.float32)
                dim_mask[:width] = 1.0
            else:
                model_prev = np.zeros((int(self._action_horizon), int(self._action_dim)), dtype=np.float32)
                dim_mask = np.zeros(int(self._action_dim), dtype=np.float32)
            delay = max(0, int(rtc.get("inference_delay", 0)))
            rtc_kwargs = {
                "prev_chunk_left_over": jnp.asarray(model_prev)[np.newaxis, ...],
                "inference_delay": jnp.asarray(delay, dtype=jnp.int32),
                "execution_horizon": jnp.asarray(horizon_rows, dtype=jnp.int32),
                "rtc_max_guidance_weight": jnp.asarray(float(rtc.get("max_guidance_weight", 10.0)), dtype=jnp.float32),
                "rtc_dim_mask": jnp.asarray(dim_mask),
            }
            # Absent keeps the model default schedule (linear).
            if rtc.get("prefix_attention_schedule") is not None:
                rtc_kwargs["rtc_prefix_attention_schedule"] = str(rtc["prefix_attention_schedule"])
            rtc_info = {"inference_delay": delay, "execution_horizon": horizon_rows, "prev_rows": int(horizon_rows)}
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        start_time = time.monotonic()
        if self._fresh_noise:
            self._rng, sample_rng = jax.random.split(self._rng)
        else:
            # Use a fixed RNG every call so identical payloads produce identical outputs.
            # We intentionally do NOT split self._rng across calls; advancing it would make
            # consecutive infers non-reproducible even with the same observation.
            sample_rng = self._rng
        outputs = {
            "state": inputs["state"]
        }
        sample_kwargs = {**self._sample_kwargs, **rtc_kwargs}
        result = self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **sample_kwargs)

        if isinstance(result, dict):
            outputs.update(result)    
        else:
            outputs["actions"] = result
        # outputs["actions"] = inputs["actions"]

        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        model_time = time.monotonic() - start_time

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        if rtc_info is not None:
            outputs["rtc"] = rtc_info
        return self.post_process(obs, outputs)

    # Task names whose policy output includes waist control (4 dims at [16:20]).
    # All other tasks only use the first 16 hand-action dims.
    TASK_NAME_REQUIRING_WAIST = (
        "sorting_packages",
        "sorting_packages_continuous",
        "sorting_packages_part_1",
        "sorting_packages_part_2",
        "sorting_packages_part_3",
    )

    def post_process(self, obs: dict, outputs: dict) -> dict:
        task_name = jax.tree.map(lambda x: x, obs).get("task_name", None)

        if task_name is None:
            return outputs

        logger.info(
            "Policy infering for task: %s, with inference time: %.3f ms",
            task_name, outputs['policy_timing']['infer_ms'],
        )

        # Opt-in: tasks listed in G2SIM_FULL_WAIST_TASKS (comma-separated) keep all 5 waist
        # dims (16..20) as the policy predicted them, for checkpoints trained with the waist live.
        full_waist_tasks = {t.strip() for t in os.environ.get("G2SIM_FULL_WAIST_TASKS", "").split(",") if t.strip()}

        if task_name in full_waist_tasks:
            if outputs["actions"].shape[-1] < 21:
                logger.warning(
                    "G2SIM_FULL_WAIST_TASKS includes %r but the policy outputs %d dims (< 21); no waist sent",
                    task_name, outputs["actions"].shape[-1],
                )
            outputs["actions"] = outputs["actions"][:, :21]
        elif task_name in self.TASK_NAME_REQUIRING_WAIST:
            raw_state = jax.tree.map(lambda x: x, obs).get("state", None)
            assert raw_state is not None, "State is required for post-processing waist actions"
            # Freeze waist dims 16..19 to the current state; dim 20 (last waist joint) is
            # left to the policy. The full waist (16..20) is returned so the client maps
            # it positionally; the frozen dims simply hold the current pose.
            outputs["actions"][:, 16:20] = raw_state[16:20]
        else:
            # Cut off waist (and any extra) action dims for tasks that only use the hands.
            outputs["actions"] = outputs["actions"][:, :16]

        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
