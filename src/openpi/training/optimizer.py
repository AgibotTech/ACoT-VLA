import dataclasses
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
import optax

import openpi.shared.array_typing as at


@runtime_checkable
class LRScheduleConfig(Protocol):
    def create(self) -> optax.Schedule: ...


@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule(LRScheduleConfig):
    """Cosine decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6

    def create(self) -> optax.Schedule:
        return optax.warmup_cosine_decay_schedule(
            init_value=self.peak_lr / (self.warmup_steps + 1),
            peak_value=self.peak_lr,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            end_value=self.decay_lr,
        )


@dataclasses.dataclass(frozen=True)
class RsqrtDecaySchedule(LRScheduleConfig):
    """Inverse square root decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 5e-5
    timescale: float = 10_000

    def create(self) -> optax.Schedule:
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=self.peak_lr / (self.warmup_steps + 1),
                    end_value=self.peak_lr,
                    transition_steps=self.warmup_steps,
                ),
                lambda step: self.peak_lr / jnp.sqrt((self.timescale + step) / self.timescale),
            ],
            [self.warmup_steps],
        )


@dataclasses.dataclass(frozen=True)
class WSDSchedule(LRScheduleConfig):
    """Warmup-Stable-Decay: linear warmup, constant peak_lr, then decay to end_lr over the last decay_steps.

    The decay starts at total_steps - decay_steps, so total_steps should match the config's num_train_steps.
    decay_type: "sqrt" (1 - sqrt, Hagele et al. 2024), "linear", or "cosine".

    Phases by step s: [0, warmup_steps) linear from peak_lr / (warmup_steps + 1) to
    peak_lr; [warmup_steps, total_steps - decay_steps) constant peak_lr; from
    total_steps - decay_steps on, decay over decay_steps steps, then held at end_lr.

    Attributes:
        warmup_steps: Length of the linear warmup, in optimizer steps.
        peak_lr: Learning rate at the end of warmup and through the stable phase.
        total_steps: Step at which the decay reaches end_lr; set to num_train_steps.
        decay_steps: Length of the final decay phase, in optimizer steps.
        end_lr: Learning rate at the end of the decay and after it.
        decay_type: Shape of the decay with frac = decay step / decay_steps:
            "sqrt" -> end_lr + (peak_lr - end_lr) * (1 - sqrt(frac));
            "linear" -> optax.linear_schedule(peak_lr, end_lr, decay_steps);
            "cosine" -> optax.cosine_decay_schedule(peak_lr, decay_steps,
            alpha=end_lr / peak_lr).
    """

    warmup_steps: int = 1_000
    peak_lr: float = 1e-5
    total_steps: int = 100_000
    decay_steps: int = 10_000
    end_lr: float = 1e-6
    decay_type: str = "sqrt"

    def create(self) -> optax.Schedule:
        """Build the optax schedule.

        Returns:
            A schedule mapping optimizer step to learning rate.

        Raises:
            ValueError: If total_steps - decay_steps < warmup_steps, or decay_type is unknown.
        """
        decay_start = self.total_steps - self.decay_steps
        if not self.warmup_steps <= decay_start:
            raise ValueError(f"WSD decay_start {decay_start} must be >= warmup_steps {self.warmup_steps}")
        if self.decay_type == "linear":
            decay = optax.linear_schedule(self.peak_lr, self.end_lr, self.decay_steps)
        elif self.decay_type == "cosine":
            decay = optax.cosine_decay_schedule(self.peak_lr, self.decay_steps, alpha=self.end_lr / self.peak_lr)
        elif self.decay_type == "sqrt":

            def decay(step):
                frac = jnp.clip(step / self.decay_steps, 0.0, 1.0)
                return self.end_lr + (self.peak_lr - self.end_lr) * (1.0 - jnp.sqrt(frac))

        else:
            raise ValueError(f"unknown WSD decay_type: {self.decay_type!r}")
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=self.peak_lr / (self.warmup_steps + 1),
                    end_value=self.peak_lr,
                    transition_steps=self.warmup_steps,
                ),
                optax.constant_schedule(self.peak_lr),
                decay,
            ],
            [self.warmup_steps, decay_start],
        )


@runtime_checkable
class OptimizerConfig(Protocol):
    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation: ...


@dataclasses.dataclass(frozen=True)
class AdamW(OptimizerConfig):
    """AdamW optimizer."""

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        tx = optax.adamw(
            lr, b1=self.b1, b2=self.b2, eps=self.eps, weight_decay=self.weight_decay, mask=weight_decay_mask
        )

        return optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), tx)


@dataclasses.dataclass(frozen=True)
class SGD(OptimizerConfig):
    """SGD optimizer."""

    lr: float = 5e-5
    momentum: float = 0.9
    nesterov: bool = False

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        assert weight_decay_mask is None, "Weight decay is not supported for SGD"
        return optax.sgd(lr, momentum=self.momentum, nesterov=self.nesterov)


def create_optimizer(
    optimizer: OptimizerConfig, lr_schedule: LRScheduleConfig, weight_decay_mask: at.PyTree | None = None
) -> optax.GradientTransformation:
    lr = lr_schedule.create()
    return optimizer.create(lr, weight_decay_mask=weight_decay_mask)
