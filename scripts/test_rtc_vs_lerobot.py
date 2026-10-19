"""JAX openpi rtc.py vs LeRobot RTCProcessor (torch logic copied verbatim)."""

import math

import jax.numpy as jnp
import numpy as np
import torch

from openpi.models import rtc


class Ref:  # lerobot/policies/rtc/modeling_rtc.py, trimmed to the math
    """Torch reference copied from LeRobot ``RTCProcessor``, trimmed to the guidance math."""

    def __init__(self, sched, maxw, h):
        """Store the reference RTC settings.

        Args:
            sched: Prefix attention schedule, one of "linear", "exp", "zeros", "ones".
            maxw: Maximum guidance weight.
            h: Execution horizon (number of guided prefix rows).
        """
        self.sched, self.maxw, self.H = sched, maxw, h

    def _lin(self, start, end, total):
        skip = max(total - end, 0)
        n = total - skip - start
        if end <= start or n <= 0:
            return torch.tensor([])
        return torch.linspace(1, 0, n + 2)[1:-1]

    def w(self, start, end, total):
        """Compute LeRobot ``get_prefix_weights``.

        Args:
            start: Inference delay; rows before it get weight 1.
            end: Execution horizon; rows from it on get weight 0.
            total: Chunk length (action horizon).

        Returns:
            [total] prefix weights.
        """
        start = min(start, end)
        if self.sched == "zeros":
            w = torch.zeros(total)
            w[:start] = 1.0
            return w
        if self.sched == "ones":
            w = torch.ones(total)
            w[end:] = 0.0
            return w
        lw = self._lin(start, end, total)
        if self.sched == "exp":
            lw = lw * torch.expm1(lw).div(math.e - 1)
        z = total - end
        if z > 0:
            lw = torch.cat([lw, torch.zeros(z)])
        if min(start, total) > 0:
            lw = torch.cat([torch.ones(min(start, total)), lw])
        return lw

    def step(self, x_t, prev, delay, time, fn):
        """Run one LeRobot ``denoise_step`` (t=1 noise -> t=0 actions).

        Args:
            x_t: [b, ah, ad] current noisy chunk.
            prev: [b, rows, dims] previous chunk leftover; zero-padded to [b, ah, ad].
            delay: Inference delay (hard prefix rows).
            time: Flow-matching time in [0, 1].
            fn: Velocity function mapping x_t to v_t.

        Returns:
            [b, ah, ad] guided velocity.
        """
        tau = 1 - time
        x_t = x_t.clone().detach()
        H = self.H
        H = min(H, prev.shape[1])
        B, T, A = x_t.shape
        if prev.shape[1] < T or prev.shape[2] < A:
            p = torch.zeros(B, T, A)
            p[:, : prev.shape[1], : prev.shape[2]] = prev
            prev = p
        w = self.w(delay, H, T).unsqueeze(0).unsqueeze(-1)
        with torch.enable_grad():
            v_t = fn(x_t)
            x_t.requires_grad_(True)
            x1 = x_t - time * v_t
            err = (prev - x1) * w
            corr = torch.autograd.grad(x1, x_t, err.clone().detach(), retain_graph=False)[0]
        mw = torch.as_tensor(self.maxw)
        tt = torch.as_tensor(tau)
        sq = (1 - tt) ** 2
        inv_r2 = (sq + tt**2) / sq
        c = torch.nan_to_num((1 - tt) / tt, posinf=mw)
        g = torch.minimum(torch.nan_to_num(c * inv_r2, posinf=mw), mw)
        return v_t - g * corr


rng = np.random.default_rng(0)
T, A = 30, 32
Wm = rng.standard_normal((A, A)).astype(np.float32) * 0.1


def fn_t(x):
    """Torch toy velocity field."""
    return torch.tanh(x @ torch.from_numpy(Wm))


def fn_j(x):
    """JAX toy velocity field (same weights as ``fn_t``)."""
    return jnp.tanh(x @ jnp.asarray(Wm))


worst = 0.0
for sched in ("linear", "exp", "zeros", "ones"):
    for H in (1, 4, 10, 30):
        for delay in (0, 3, 9, 12):
            for time in (1.0, 0.7, 0.3, 0.1):
                x = rng.standard_normal((1, T, A)).astype(np.float32)
                prev = rng.standard_normal((1, H, A)).astype(np.float32)  # already H rows (LeRobot normalize)
                ref = Ref(sched, 10.0, H).step(torch.from_numpy(x), torch.from_numpy(prev), delay, time, fn_t).numpy()
                prev_pad = np.concatenate([prev, np.repeat(prev[:, -1:], T - H, 1)], 1)  # our hold-pad, weights 0 there
                got = np.asarray(
                    rtc.guided_velocity(
                        jnp.asarray(x),
                        time=jnp.float32(time),
                        denoise_fn=fn_j,
                        prev_chunk_left_over=jnp.asarray(prev_pad),
                        inference_delay=jnp.int32(delay),
                        execution_horizon=jnp.int32(H),
                        max_guidance_weight=jnp.float32(10.0),
                        prefix_attention_schedule=sched,
                    )
                )
                worst = max(worst, float(np.abs(ref - got).max()))
print("max |jax - lerobot| over 256 cases:", worst)
assert worst < 1e-4
# no-leftover path (H=0) must equal the plain velocity exactly
x = rng.standard_normal((1, T, A)).astype(np.float32)
v0 = np.asarray(fn_j(jnp.asarray(x)))
v1 = np.asarray(
    rtc.guided_velocity(
        jnp.asarray(x),
        time=jnp.float32(0.5),
        denoise_fn=fn_j,
        prev_chunk_left_over=jnp.zeros((1, T, A)),
        inference_delay=jnp.int32(0),
        execution_horizon=jnp.int32(0),
        max_guidance_weight=jnp.float32(10.0),
        dim_mask=jnp.zeros(A),
    )
)
print("H=0 diff vs plain:", float(np.abs(v0 - v1).max()))
assert np.array_equal(v0, v1)
print("RTC_LEROBOT_PARITY_OK")
