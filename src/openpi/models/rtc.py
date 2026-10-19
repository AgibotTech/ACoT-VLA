"""Real-Time Chunking (RTC) guidance for flow-matching action models.

JAX port of LeRobot's ``RTCProcessor.denoise_step`` / ``get_prefix_weights``
(lerobot/policies/rtc/modeling_rtc.py), which itself follows Physical
Intelligence's Kinetix implementation.

Matches LeRobot exactly, including the part that differs from the paper:
LeRobot evaluates ``v_t`` *before* enabling gradients on ``x_t``, so the
autograd "correction" is just the weighted error ``err`` (identity Jacobian).
No backward pass through the action expert -> an RTC step costs the same as a
plain denoising step.

Time convention is openpi/LeRobot's: ``time`` goes 1 (noise) -> 0 (actions),
``x0_hat = x_t - time * v_t``.

All per-call RTC values (prefix, delay, horizon, guidance weight) are traced
arrays, so one compiled graph serves every delay/horizon. The caller must pass
``prev_chunk_left_over`` already shaped ``[B, action_horizon, action_dim]``
(see ``Policy.infer``), otherwise every new leftover length recompiles.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def prefix_weights(
    total_steps: int,
    *,
    start: jax.Array,
    end: jax.Array,
    schedule: str,
    dtype: jnp.dtype,
) -> jax.Array:
    """Compute LeRobot ``get_prefix_weights(start, end, total)`` with traced start/end.

    Ones on [0, start), a ramp on [start, end), zeros on [end, total).
    LINEAR ramp is ``linspace(1, 0, n + 2)[1:-1]`` with ``n = end - start``,
    i.e. ``w_i = 1 - (i - start + 1) / (n + 1)``.

    Args:
        total_steps: Static chunk length ``ah`` (action horizon).
        start: Scalar int, inference delay; clamped to ``end``. Rows before it get weight 1.
        end: Scalar int, execution horizon; rows from it on get weight 0.
        schedule: Static ramp shape, one of "linear", "exp", "zeros", "ones".
        dtype: Output dtype.

    Returns:
        [ah] per-row prefix weights in [0, 1].

    Raises:
        ValueError: If ``schedule`` is not a known schedule.
    """
    idx = jnp.arange(total_steps, dtype=jnp.int32)
    end = jnp.asarray(end, dtype=jnp.int32)
    start = jnp.minimum(jnp.asarray(start, dtype=jnp.int32), end)

    if schedule == "zeros":
        return (idx < start).astype(dtype)
    if schedule == "ones":
        return (idx < end).astype(dtype)

    n = (end - start).astype(dtype)
    lin = 1.0 - (idx - start + 1).astype(dtype) / (n + 1.0)
    if schedule == "exp":
        lin = lin * jnp.expm1(lin) / (math.e - 1.0)
    elif schedule != "linear":
        raise ValueError(f"unknown RTC prefix_attention_schedule: {schedule!r}")
    w = jnp.where(idx < start, 1.0, jnp.where(idx < end, lin, 0.0))
    return w.astype(dtype)


def guidance_weight(time: jax.Array, max_guidance_weight: jax.Array, dtype: jnp.dtype) -> jax.Array:
    """Compute LeRobot's clamped RTC guidance weight.

    Uses ``tau = 1 - time`` (t=1 noise -> t=0 actions) and
    ``w = min((1 - tau) / tau * ((1 - tau)^2 + tau^2) / (1 - tau)^2, max_guidance_weight)``,
    with infinities (``tau = 0``) mapped to ``max_guidance_weight``.

    Args:
        time: Scalar flow-matching time in (0, 1].
        max_guidance_weight: Scalar upper bound on the weight.
        dtype: Compute and output dtype.

    Returns:
        Scalar guidance weight.
    """
    tau = 1.0 - jnp.asarray(time, dtype=dtype)
    max_w = jnp.asarray(max_guidance_weight, dtype=dtype)
    one_minus_tau_sq = (1.0 - tau) ** 2
    inv_r2 = (one_minus_tau_sq + tau**2) / one_minus_tau_sq
    c = jnp.nan_to_num((1.0 - tau) / tau, posinf=max_w)
    w = jnp.nan_to_num(c * inv_r2, posinf=max_w)
    return jnp.minimum(w, max_w)


def guided_velocity(
    x_t: jax.Array,
    *,
    time: jax.Array,
    denoise_fn,
    prev_chunk_left_over: jax.Array | None,
    inference_delay: jax.Array | int,
    execution_horizon: jax.Array | int,
    max_guidance_weight: jax.Array | float = 10.0,
    prefix_attention_schedule: str = "linear",
    dim_mask: jax.Array | None = None,
) -> jax.Array:
    """Compute the RTC-guided velocity (LeRobot ``denoise_step``).

    Time runs t=1 (noise) -> t=0 (actions); the caller integrates
    ``x_t + dt * v`` with ``dt < 0``.

    Args:
        x_t: [b, ah, ad] current noisy chunk.
        time: Scalar flow-matching time in (0, 1].
        denoise_fn: Callable mapping [b, ah, ad] ``x_t`` to the unguided velocity [b, ah, ad].
        prev_chunk_left_over: [b or 1, ah, ad] (or [ah, ad]) previous chunk's unexecuted rows
            in model space, already re-anchored to the current state and padded to ``ah``.
            None -> plain ``denoise_fn(x_t)``.
        inference_delay: Scalar int, rows that will execute before this chunk lands (hard prefix).
        execution_horizon: Scalar int, rows of the prefix that are guided at all.
        max_guidance_weight: Scalar upper bound on the guidance weight.
        prefix_attention_schedule: Static ramp shape, one of "linear", "exp", "zeros", "ones".
        dim_mask: Optional [ad] 0/1 mask of action dims the leftover is valid for.

    Returns:
        [b, ah, ad] guided velocity.
    """
    v_t = denoise_fn(x_t)
    if prev_chunk_left_over is None:
        return v_t

    prev = jnp.asarray(prev_chunk_left_over, dtype=x_t.dtype)
    if prev.ndim == 2:
        prev = prev[jnp.newaxis]
    prev = jnp.broadcast_to(prev, x_t.shape)

    weights = prefix_weights(
        x_t.shape[1],
        start=inference_delay,
        end=execution_horizon,
        schedule=prefix_attention_schedule,
        dtype=x_t.dtype,
    )[jnp.newaxis, :, jnp.newaxis]
    if dim_mask is not None:
        weights = weights * jnp.asarray(dim_mask, dtype=x_t.dtype)[jnp.newaxis, jnp.newaxis, :]

    x1_t = x_t - time * v_t
    # LeRobot: v_t carries no graph to x_t, so d(x1_t)/d(x_t) = I and correction == err.
    correction = (prev - x1_t) * weights
    return v_t - guidance_weight(time, max_guidance_weight, x_t.dtype) * correction
