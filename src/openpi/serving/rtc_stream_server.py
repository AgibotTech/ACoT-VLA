"""Server-owned Real-Time Chunking for G2 pi0.5 (Anvil-style), over one websocket.

Port of anvil-embodied-ai ``lerobot_control/inference_node.py`` (LeRobot 0.5.1 RTC):
the server owns the action queue, the inference worker and the execution clock, and
sends ONE absolute command per control tick. The client only streams observations
and executes (or holds) the latest command; authority, limits and stop stay with it.

Same as Anvil
  * worker infers whenever ``qsize <= queue_threshold`` (threshold = chunk -> continuous),
    at most once per observation, with the full unexecuted leftover as the RTC prefix;
  * RTC delay = ceil(max(last 5 guided latencies) * f), merge delay = rows really
    consumed while the request was in flight (wall-clock bounded, fail-closed);
  * readiness: discard warm-up, seed unguided (no publish), then 5 consecutive guided
    refills must pass coverage / projected-age / useful-overlap before POLICY_READY;
  * watchdog: stale observations, empty queue after ready, stale or misaligned refill,
    unguided refill after ready -> LATCHED (epoch bump, queue replaced, publishing
    stops). ``rearm`` needs fresh observations and repeats the whole readiness proof;
  * fresh diffusion noise per call, EXP prefix schedule.

Different from Anvil, forced by the model
  * The G2 configs predict arms/waist as deltas from the current state (grippers
    absolute). The queue therefore holds ABSOLUTE rows and Policy.infer re-anchors the
    leftover to the current state through the config's own input transforms on every
    refill (LeRobot 0.6.1 ``reanchor_relative_rtc_prefix``). Anvil's checkpoint is
    absolute, so it feeds the normalized original rows back instead; with absolute
    rows one queue serves both roles.
  * Only Pi0/pi0.5 models: ACoT's coarse head cannot take an RTC prefix.

Wire protocol (msgpack; one client at a time)
  client -> server
    {"type": "obs", "params": {images, states, prompt, task_name, robot_type}, "tick": bool}
        ``params`` is the corobot ``infer`` params block. ``tick`` is only used with
        clock="external" (simulation): it advances the execution clock by one step and
        is always answered with that step's cmd or status.
    {"type": "rearm"}   leave LATCHED; needs fresh observations, restarts readiness
    {"type": "reset"}   new episode: drop the queue and restart readiness (no fault)
    Both are answered with a status message.
  server -> client
    {"type": "cmd", "seq", "epoch", "t_server", "action_age_s", "queue", "action": [W]}
        ``action`` is one absolute row in the policy layout (gseries: arms 0:14,
        grippers 14:16, waist 16:21).
    {"type": "status", "state": "starting"|"armed"|"ready"|"latched", "reason", "epoch"}
    {"type": "error", "error"}
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import json
import logging
import math
import pathlib
import threading
import time
import traceback

import numpy as np
from openpi_client import msgpack_numpy
import websockets
import websockets.asyncio.server as _server

from openpi.policies import policy as _policy
from openpi.serving import websocket_policy_server as _corobot

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RtcStreamConfig:
    """RTC and safety settings. Defaults are Anvil's (inference_node.py / docs/inference.md)."""

    # Execution clock: "wall" = server timer at control_hz (robot); "external" = one step
    # per obs with tick=true (simulation, whose time is not wall time).
    clock: str = "wall"
    # Row rate. 30 = the rate the G2 training data was recorded at.
    control_hz: float = 30.0
    # Re-infer when queued rows <= this. None = chunk length, i.e. continuous refills.
    queue_threshold: int | None = None
    # Guided prefix length H. Must exceed the delay by a useful margin.
    execution_horizon: int = 20
    max_guidance_weight: float = 10.0
    prefix_attention_schedule: str = "exp"
    # Delay (rows) used before any latency has been measured.
    inference_delay_fallback: int = 10
    readiness_latency_guard_steps: int = 2
    readiness_index_phase_tolerance_steps: int = 1
    readiness_scheduler_guard_steps: int = 1
    readiness_min_guided_overlap_steps: int = 3
    # Anvil's readiness gate: no publishing until 5 guided refills prove the timing
    # margins, and any later margin violation, empty queue or unguided refill latches.
    # Off: publish from the first chunk, hold the last target when the queue runs dry and
    # merge whatever arrives (the client-side RTC's behaviour). The input watchdog and the
    # action-age limit stay on.
    readiness_gate: bool = True
    # A row is never published more than this long after its observation arrived.
    max_action_age_s: float = 1.5
    # Observations older than this make the inputs unhealthy.
    max_obs_age_s: float = 0.25
    # Time allowed after connect for the first healthy observation.
    startup_grace_s: float = 10.0
    # Waist dims 16:21: "model" = straight from the policy; "hold" = dims 16:19 held at the
    # measured state, dim 20 from the policy (configs that masked 16:19 in training, as
    # Policy.post_process does for sorting_packages); "off" = all five held (tasks the
    # policy does not move the waist on); "auto" = model iff the config unmasked it, else hold.
    waist: str = "auto"
    # Optional JSONL trace, one line per committed inference.
    trace: str | None = None


# Anvil hard-requires exactly five guided refills before publication.
READINESS_GUIDED_FORWARDS = 5


@dataclasses.dataclass(frozen=True)
class _Observation:
    obs: dict
    seq: int
    receipt_t: float


@dataclasses.dataclass(frozen=True)
class _Dispatch:
    """Queue identity, depth, index and clock captured with one RTC leftover."""

    queue: ActionQueue
    queue_size: int
    action_index: int
    requested_t: float


@dataclasses.dataclass(frozen=True)
class _Alignment:
    runtime_s: float
    wall_delay_steps: int
    consumed_steps: int
    merge_delay_steps: int
    merge_t: float


@dataclasses.dataclass(frozen=True)
class _Assessment:
    sustainable: bool
    failures: tuple[str, ...]
    q_start: int
    q_required: int
    refill_delay_bound_steps: int
    useful_overlap_steps: int
    age_at_next_refill_s: float

    def fields(self) -> str:
        return (
            f"q_start={self.q_start} q_required={self.q_required} "
            f"D_bound={self.refill_delay_bound_steps} overlap={self.useful_overlap_steps} "
            f"age_next={self.age_at_next_refill_s:.3f}s"
        )


class ActionQueue:
    """LeRobot ActionQueue in RTC mode over absolute rows: merge replaces, never appends.

    Every method takes the queue's own lock, so it is safe to call from the clock and
    the inference threads at once.
    """

    def __init__(self) -> None:
        """Create an empty queue (no rows, index 0)."""
        self._lock = threading.Lock()
        self._rows: np.ndarray | None = None
        self._index = 0

    def get(self) -> np.ndarray | None:
        """Pop the next row.

        Returns:
            A copy of the next unexecuted row, shape [W] float64, or None when empty.
        """
        with self._lock:
            if self._rows is None or self._index >= len(self._rows):
                return None
            row = self._rows[self._index].copy()
            self._index += 1
            return row

    def qsize(self) -> int:
        """Return the number of unexecuted rows (steps) left in the queue."""
        with self._lock:
            return 0 if self._rows is None else len(self._rows) - self._index

    def snapshot(self) -> tuple[int, int, np.ndarray | None]:
        """Read depth, index and leftover rows coherently under one lock.

        Returns:
            ``(depth, index, leftover)``: unexecuted row count, rows popped since the last
            merge, and a copy of the unexecuted rows [depth, W] (None before any merge).
        """
        with self._lock:
            if self._rows is None:
                return 0, self._index, None
            leftover = self._rows[self._index :].copy()
            return len(leftover), self._index, leftover

    def merge(self, rows: np.ndarray, delay: int) -> int:
        """Replace the queue with ``rows[delay:]`` and reset the index to 0.

        Args:
            rows: New chunk of absolute rows, shape [T, W].
            delay: Leading rows to skip (already executed in wall time); clamped to [0, T].

        Returns:
            The new queue depth, ``T - delay``.
        """
        with self._lock:
            delay = min(max(0, delay), len(rows))
            self._rows = np.array(rows[delay:], dtype=np.float64)
            self._index = 0
            return len(self._rows)


def inference_delay_steps(
    guided_latencies_s: tuple[float, ...], tracked_max_latency_s: float, control_hz: float, fallback: int
) -> int:
    """Return the RTC inference delay in rows (Anvil).

    Uses the guided latency window once it exists, else the all-time max, else the fallback.

    Args:
        guided_latencies_s: Recent guided-refill latencies, seconds.
        tracked_max_latency_s: All-time max latency, seconds (0 = none measured).
        control_hz: Row rate, rows per second.
        fallback: Delay in rows when no latency has been measured.

    Returns:
        ``ceil(latency * control_hz)`` rows, or ``fallback``.
    """
    if guided_latencies_s:
        return math.ceil(max(guided_latencies_s) * control_hz)
    if tracked_max_latency_s > 0:
        return math.ceil(tracked_max_latency_s * control_hz)
    return fallback


def resolve_merge_alignment(
    *,
    queue_identity_matches: bool,
    queue_size_before: int,
    queue_size_at_merge: int,
    index_before: int,
    index_at_merge: int,
    requested_t: float,
    merge_t: float,
    control_hz: float,
    policy_ready: bool,
    index_phase_tolerance_steps: int,
    allow_empty_queue: bool = False,
) -> _Alignment:
    """Resolve the merge delay for a finished refill (Anvil ``_resolve_rtc_merge_alignment``).

    Once publishing, the rows really consumed during inference are the merge delay;
    before readiness nothing may be consumed and the wall-clock delay is used instead.
    Times are in engine seconds (see ``RtcEngine._now``); step counts are rows.

    Args:
        queue_identity_matches: Whether the queue at merge is the one captured at dispatch.
        queue_size_before: Queue depth (rows) at dispatch.
        queue_size_at_merge: Queue depth (rows) at merge.
        index_before: Queue index at dispatch.
        index_at_merge: Queue index at merge.
        requested_t: Dispatch time, seconds.
        merge_t: Merge time, seconds.
        control_hz: Row rate, rows per second.
        policy_ready: Whether publication is open (POLICY_READY).
        index_phase_tolerance_steps: Rows of consumption allowed beyond the wall-clock bound.
        allow_empty_queue: Accept a queue that ran dry before the merge (readiness gate off).

    Returns:
        The runtime, wall delay, consumed rows, merge delay and merge time.

    Raises:
        ValueError: On invalid timing, a replaced queue, regressed or inconsistent
            depth/index, pre-ready consumption, an emptied queue, or consumption above
            the wall-clock bound.
    """
    if not math.isfinite(requested_t) or not math.isfinite(merge_t) or merge_t < requested_t:
        raise ValueError("RTC merge alignment has invalid timing")
    if not queue_identity_matches:
        raise ValueError("RTC action queue changed while inference was running")
    if queue_size_before < 0 or queue_size_at_merge < 0 or index_before < 0 or index_at_merge < index_before:
        raise ValueError(
            f"RTC queue depth/index regressed: q_before={queue_size_before} q_merge={queue_size_at_merge} "
            f"i_before={index_before} i_merge={index_at_merge}"
        )
    runtime_s = merge_t - requested_t
    wall_delay_steps = math.ceil(runtime_s * control_hz)
    consumed = index_at_merge - index_before
    if queue_size_before - queue_size_at_merge != consumed:
        raise ValueError(
            f"RTC queue depth/index consumption mismatch: depth_delta={queue_size_before - queue_size_at_merge}, "
            f"index_delta={consumed}"
        )
    if not policy_ready:
        if consumed != 0:
            raise ValueError(f"RTC pre-ready queue was consumed while publication was closed: {consumed} steps")
        merge_delay = wall_delay_steps
    else:
        if queue_size_at_merge < 1 and not allow_empty_queue:
            raise ValueError("RTC action queue emptied before refill merge")
        maximum = math.ceil(runtime_s * control_hz) + index_phase_tolerance_steps
        if consumed > maximum:
            raise ValueError(
                f"RTC queue consumption exceeds wall-clock bound: consumed={consumed}, maximum={maximum}, "
                f"runtime={runtime_s:.6f}s"
            )
        merge_delay = consumed
    return _Alignment(runtime_s, wall_delay_steps, consumed, merge_delay, merge_t)


def assess_readiness(
    *,
    chunk_size: int,
    candidate_delay_steps: int,
    guided_latencies_s: tuple[float, ...],
    control_hz: float,
    queue_threshold: int,
    source_age_s: float,
    max_action_age_s: float,
    execution_horizon: int,
    latency_guard_steps: int,
    scheduler_guard_steps: int,
    min_guided_overlap_steps: int,
) -> _Assessment:
    """Decide whether a merged chunk can survive the next refill (Anvil ``_assess_rtc_readiness``).

    Args:
        chunk_size: Rows in the new chunk (model action horizon).
        candidate_delay_steps: Merge delay of this chunk, rows.
        guided_latencies_s: Guided latencies including this one, seconds; must be non-empty.
        control_hz: Row rate, rows per second.
        queue_threshold: Queue depth (rows) at or below which the worker re-infers.
        source_age_s: Age of the chunk's observation at merge, seconds.
        max_action_age_s: Limit on a published row's source age, seconds.
        execution_horizon: Guided prefix length H, rows.
        latency_guard_steps: Rows added to the latency bound.
        scheduler_guard_steps: Rows allowed for worker scheduling slack.
        min_guided_overlap_steps: Minimum guided rows beyond the delay bound.

    Returns:
        The verdict, the failure reasons, and the margins it was computed from.

    Raises:
        ValueError: If ``guided_latencies_s`` is empty.
    """
    if not guided_latencies_s:
        raise ValueError("RTC readiness requires guided latency samples")
    q_start = max(0, chunk_size - candidate_delay_steps)
    wait_steps = max(0, q_start - queue_threshold)
    q_trigger = q_start - wait_steps
    q_required = max(0, q_trigger - scheduler_guard_steps)
    max_latency = max(guided_latencies_s)
    latency_bound_s = max_latency + latency_guard_steps / control_hz
    delay_bound = math.ceil(max_latency * control_hz) + latency_guard_steps
    age_next = source_age_s + (wait_steps + scheduler_guard_steps) / control_hz + latency_bound_s
    useful_overlap = max(0, min(execution_horizon, q_required) - delay_bound)

    failures = []
    if q_start <= 0:
        failures.append("candidate leaves an empty queue")
    if q_required < delay_bound + 1:
        failures.append(f"refill coverage {q_required} < {delay_bound + 1} steps")
    if age_next >= max_action_age_s:
        failures.append(f"projected source age {age_next:.3f}s >= {max_action_age_s:.3f}s")
    if useful_overlap < min_guided_overlap_steps:
        failures.append(f"useful guided overlap {useful_overlap} < {min_guided_overlap_steps} steps")
    return _Assessment(not failures, tuple(failures), q_start, q_required, delay_bound, useful_overlap, age_next)


class RtcEngine:
    """Observation slot, RTC worker, action queue, readiness proof and watchdog.

    ``_safety_lock`` is always taken first; the queue's own lock is taken inside it.
    The model runs outside ``_safety_lock`` so ``tick`` never waits on the GPU.
    All public methods are thread-safe.
    """

    def __init__(self, policy: _policy.Policy, cfg: RtcStreamConfig, *, waist: str) -> None:
        """Compile the model and start the inference worker thread.

        Blocks for the JIT compile (seconds) before returning.

        Args:
            policy: Pi0/pi0.5 policy whose ``infer`` accepts the ``_rtc`` options block.
            cfg: RTC and safety settings.
            waist: Resolved waist mode: "model", "hold" or "off" ("auto" is not accepted).

        Raises:
            ValueError: If ``cfg.clock`` or ``waist`` is not a supported value.
        """
        if cfg.clock not in ("wall", "external"):
            raise ValueError(f"clock must be 'wall' or 'external', got {cfg.clock!r}")
        self._policy = policy
        self._cfg = cfg
        if waist not in ("model", "hold", "off"):
            raise ValueError(f"waist must be model, hold or off, got {waist!r}")
        self._waist = waist
        self._safety_lock = threading.RLock()
        self._stop = threading.Event()
        self._trace = pathlib.Path(cfg.trace).open("a") if cfg.trace else None  # noqa: SIM115

        self._queue = ActionQueue()
        self._epoch = 0
        self._obs: _Observation | None = None
        self._obs_seq = 0
        self._last_inferred_seq = 0
        self._session = False
        self._started_t = 0.0
        self._armed = False
        self._latched = False
        self._latch_reason = ""
        self._latch_obs_seq = 0
        self._warmup_pending = True  # the first forward of the process is a cold-compile warm-up
        self._ready = False
        self._seeded = False
        self._guided_streak = 0
        self._guided_latencies: collections.deque[float] = collections.deque(maxlen=READINESS_GUIDED_FORWARDS)
        self._max_latency = 0.0
        self._action_source_t = 0.0
        self._last_published: np.ndarray | None = None
        self._cmd_seq = 0
        self._n_infer = 0
        self._ticks = 0  # control steps taken; the clock itself when clock="external"

        self._compile()
        self._worker = threading.Thread(target=self._worker_loop, name="rtc-inference", daemon=True)
        self._worker.start()

    def _compile(self) -> None:
        """JIT-compile the RTC graph before any client connects.

        Tracing holds the GIL for seconds; done inside a session it starves the websocket
        thread and the observation watchdog latches. Guided and unguided refills share one
        graph (the policy pads a missing leftover to the same shape), so one call covers both.
        """
        t0 = time.monotonic()
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        self._policy.infer(
            {
                "state": np.zeros(32, dtype=np.float64),
                "images": {key: image for key in _corobot._POLICY_IMG_KEY_MAP.values()},  # noqa: SLF001
                "prompt": "",
                "_rtc": {
                    "prev_actions": None,
                    "inference_delay": 0,
                    "execution_horizon": self._cfg.execution_horizon,
                    "max_guidance_weight": self._cfg.max_guidance_weight,
                    "prefix_attention_schedule": self._cfg.prefix_attention_schedule,
                },
            }
        )
        logger.info("[RTC] model compiled in %.1fs", time.monotonic() - t0)

    # -- session / safety ---------------------------------------------------------------

    def start_session(self) -> None:
        """Begin a client session: new epoch, empty queue, cleared latch, startup grace restarts."""
        with self._safety_lock:
            self._invalidate_locked()
            self._session = True
            self._latched = False
            self._latch_reason = ""
            self._armed = False
            self._obs = None
            self._started_t = self._now()
            logger.info("[RTC] session started epoch=%d", self._epoch)

    def end_session(self) -> None:
        """End the client session, latching with "client disconnected" unless already latched."""
        with self._safety_lock:
            if self._session and not self._latched:
                self._trip_locked("client disconnected")
            self._session = False

    def rearm(self) -> tuple[bool, str]:
        """Leave LATCHED and restart the readiness proof (Anvil rearm).

        Inputs must be fresh (within ``max_obs_age_s``) and include an observation newer
        than the one current at the fault.

        Returns:
            ``(ok, reason)``; ``reason`` is empty on success.
        """
        with self._safety_lock:
            if not self._latched:
                return False, "not latched"
            now = self._now()
            if self._obs is None or now - self._obs.receipt_t > self._cfg.max_obs_age_s:
                return False, "observations are not fresh"
            if self._obs.seq <= self._latch_obs_seq:
                return False, "no new observation since the fault"
            self._invalidate_locked()
            self._latched = False
            self._latch_reason = ""
            self._armed = False
            self._started_t = now
            logger.info("[WATCHDOG] REARMED epoch=%d; readiness proof restarts", self._epoch)
            return True, ""

    def reset(self) -> None:
        """Start a new episode on the same connection.

        Drops every action and the last observation, re-runs the startup grace and the
        whole readiness proof (Anvil rearm + model.reset, without a fault). Also clears a latch.
        """
        with self._safety_lock:
            self._invalidate_locked()
            # Drop the previous episode's last frame: with clock="external" it still looks
            # fresh (no ticks between episodes), and inferring it would seed the new episode
            # with a plan for the old scene and arm pose.
            self._obs = None
            self._latched = False
            self._latch_reason = ""
            self._armed = False
            self._started_t = self._now()
            logger.info("[RTC] RESET epoch=%d; readiness proof restarts", self._epoch)

    def close(self) -> None:
        """Stop the worker thread (waits up to 5 s) and close the trace file."""
        self._stop.set()
        self._worker.join(timeout=5.0)
        if self._trace:
            self._trace.close()

    def status(self) -> dict:
        """Return the wire status message.

        Returns:
            ``{"type": "status", "state", "reason", "epoch"}`` with state one of
            "latched", "ready", "armed", "starting" (checked in that order).
        """
        with self._safety_lock:
            if self._latched:
                state = "latched"
            elif self._ready:
                state = "ready"
            elif self._armed:
                state = "armed"
            else:
                state = "starting"
            return {"type": "status", "state": state, "reason": self._latch_reason, "epoch": self._epoch}

    def _invalidate_locked(self) -> None:
        """Bump the epoch and drop every action and readiness fact (Anvil fault path)."""
        self._epoch += 1
        self._queue = ActionQueue()
        self._ready = False
        self._seeded = False
        self._guided_streak = 0
        self._guided_latencies.clear()
        self._max_latency = 0.0
        self._last_published = None

    def _trip_locked(self, reason: str) -> None:
        if self._latched:
            return
        self._latched = True
        self._latch_reason = reason
        self._latch_obs_seq = self._obs.seq if self._obs is not None else 0
        self._invalidate_locked()
        logger.error("[WATCHDOG] LATCHED epoch=%d: %s", self._epoch, reason)

    def _evaluate_locked(self, now: float) -> bool:
        """Input health. True only when the inputs are fresh and nothing is latched."""
        if not self._session or self._latched:
            return False
        fresh = self._obs is not None and now - self._obs.receipt_t <= self._cfg.max_obs_age_s
        if self._armed:
            if not fresh:
                age = math.inf if self._obs is None else now - self._obs.receipt_t
                self._trip_locked(f"observation stale: age={age:.3f}s > {self._cfg.max_obs_age_s:.3f}s")
            return fresh
        if fresh:
            self._armed = True
            logger.info("[WATCHDOG] ARMED epoch=%d", self._epoch)
            return True
        if now - self._started_t > self._cfg.startup_grace_s:
            self._trip_locked(f"no healthy observation within {self._cfg.startup_grace_s:.1f}s")
        return False

    # -- observation in, command out ----------------------------------------------------

    def _now(self) -> float:
        """Engine time in seconds: wall clock, or control steps / f with clock="external".

        A simulator runs slower than real time and stalls while rendering; measured in its
        own steps, observation freshness, action age, inference delay and merge alignment
        are what the simulated robot experiences, not how slow the host happens to be.
        """
        if self._cfg.clock == "external":
            return self._ticks / self._cfg.control_hz
        return time.monotonic()

    def accept_observation(self, obs: dict) -> None:
        """Store ``obs`` as the latest observation, stamped with a new seq and engine time.

        Args:
            obs: Policy input dict (``state``, ``images``, ``prompt``) from the corobot params.
        """
        with self._safety_lock:
            self._obs_seq += 1
            self._obs = _Observation(obs, self._obs_seq, self._now())

    def tick(self) -> dict | None:
        """Advance one control step: pop, authorize and return one command.

        Always advances the tick counter (the engine clock when clock="external"). May latch
        on stale inputs, an empty queue after POLICY_READY (gate on) or an over-age row.

        Returns:
            The wire ``cmd`` message, or None when not ready, latched or no row is due.
        """
        with self._safety_lock:
            self._ticks += 1
            now = self._now()
            if not self._evaluate_locked(now) or not self._ready:
                return None
            row = self._queue.get()
            if row is None:
                if self._cfg.readiness_gate:
                    self._trip_locked("RTC action queue emptied after POLICY_READY")
                return None  # gate off: the client holds its last target until a refill lands
            age = now - self._action_source_t
            if age > self._cfg.max_action_age_s:
                self._trip_locked(f"action age {age:.3f}s > {self._cfg.max_action_age_s:.3f}s")
                return None
            self._cmd_seq += 1
            self._last_published = row
            return {
                "type": "cmd",
                "seq": self._cmd_seq,
                "epoch": self._epoch,
                "t_server": time.time(),
                "action_age_s": age,
                "queue": self._queue.qsize(),
                "action": row.tolist(),
            }

    # -- inference worker ---------------------------------------------------------------

    def _queue_threshold(self, chunk_size: int) -> int:
        return chunk_size if self._cfg.queue_threshold is None else self._cfg.queue_threshold

    def _worker_loop(self) -> None:
        cfg = self._cfg
        chunk_size = int(self._policy._action_horizon)  # noqa: SLF001
        threshold = self._queue_threshold(chunk_size)
        while not self._stop.is_set():
            record = None
            with self._safety_lock:
                now = self._now()
                if (
                    self._evaluate_locked(now)
                    and not (self._ready and self._queue.qsize() > threshold)
                    and self._obs is not None
                    and self._obs.seq != self._last_inferred_seq
                ):
                    record = self._obs
                    self._last_inferred_seq = record.seq
                    queue = self._queue
                    queue_size, index, leftover = queue.snapshot()
                    dispatch = _Dispatch(queue, queue_size, index, now)
                    epoch = self._epoch
                    latencies = tuple(self._guided_latencies)
                    max_latency = self._max_latency
            if record is None:
                time.sleep(0.002)
                continue

            guided = leftover is not None and len(leftover) > 0
            delay = inference_delay_steps(latencies, max_latency, cfg.control_hz, cfg.inference_delay_fallback)
            obs = dict(record.obs)
            state = np.array(obs["state"], dtype=np.float64)
            obs["_rtc"] = {
                "prev_actions": leftover if guided else None,
                "inference_delay": delay,
                # LeRobot clamps H to the leftover length; the policy pads past it.
                "execution_horizon": min(cfg.execution_horizon, len(leftover)) if guided else cfg.execution_horizon,
                "max_guidance_weight": cfg.max_guidance_weight,
                "prefix_attention_schedule": cfg.prefix_attention_schedule,
            }
            try:
                out = self._policy.infer(obs)
                rows = np.array(out["actions"], dtype=np.float64)
                if rows.ndim != 2 or len(rows) == 0 or not np.isfinite(rows).all():
                    raise ValueError(f"policy returned an invalid chunk: shape={rows.shape}")
                if self._waist != "model" and rows.shape[1] >= 21:
                    held = slice(16, 20) if self._waist == "hold" else slice(16, 21)
                    rows[:, held] = state[held]
            except Exception as exc:
                logger.error("[RTC] inference failed: %s\n%s", exc, traceback.format_exc())
                with self._safety_lock:
                    if epoch == self._epoch:
                        self._trip_locked(f"RTC inference failed: {type(exc).__name__}: {exc}")
                continue
            completed_t = self._now()
            with self._safety_lock:
                self._commit_locked(rows, dispatch, record.receipt_t, epoch, guided, completed_t, delay, out)

    def _clear_provisional_locked(self) -> None:
        self._queue = ActionQueue()
        self._seeded = False

    def _reset_streak_locked(self) -> None:
        self._guided_streak = 0
        self._guided_latencies.clear()

    def _merge_locked(
        self, rows: np.ndarray, dispatch: _Dispatch, obs_t: float
    ) -> tuple[int, _Alignment, float | None]:
        """Validate alignment and merge atomically against the captured queue."""
        merge_t = self._now()
        queue_size, index, _ = self._queue.snapshot()
        alignment = resolve_merge_alignment(
            queue_identity_matches=self._queue is dispatch.queue,
            queue_size_before=dispatch.queue_size,
            queue_size_at_merge=queue_size,
            index_before=dispatch.action_index,
            index_at_merge=index,
            requested_t=dispatch.requested_t,
            merge_t=merge_t,
            control_hz=self._cfg.control_hz,
            policy_ready=self._ready,
            index_phase_tolerance_steps=self._cfg.readiness_index_phase_tolerance_steps,
            allow_empty_queue=not self._cfg.readiness_gate,
        )
        if merge_t - obs_t > self._cfg.max_action_age_s:
            raise ValueError(f"RTC result became stale before merge: source_age={merge_t - obs_t:.3f}s")
        delay = alignment.merge_delay_steps
        seam = None
        if self._ready and self._last_published is not None and delay < len(rows):
            width = min(14, rows.shape[1])
            seam = float(np.max(np.abs(rows[delay, :width] - self._last_published[:width])))
        merged = self._queue.merge(rows, delay)
        if merged != max(0, len(rows) - delay):
            raise ValueError(f"RTC merged queue depth is incoherent: queue={merged}")
        if merged > 0:
            self._action_source_t = obs_t
        return merged, alignment, seam

    def _commit_locked(
        self,
        rows: np.ndarray,
        dispatch: _Dispatch,
        obs_t: float,
        epoch: int,
        guided: bool,
        completed_t: float,
        delay_req: int,
        out: dict,
    ) -> None:
        """Anvil ``_commit_vla_result_locked``: commit only after proving real-time margins."""
        cfg = self._cfg
        if epoch != self._epoch or self._latched:
            logger.warning("[RTC] discarded in-flight result from epoch %d (current %d)", epoch, self._epoch)
            return
        self._n_infer += 1
        source_age = completed_t - obs_t
        latency = completed_t - dispatch.requested_t
        trace = {
            "t": time.time(),
            "infer": self._n_infer,
            "epoch": epoch,
            "guided": guided,
            "latency_ms": latency * 1000.0,
            "model_ms": (out.get("policy_timing") or {}).get("infer_ms"),
            "delay_req": delay_req,
            "prev_rows": dispatch.queue_size,
            "source_age_s": source_age,
        }

        if self._warmup_pending:
            self._warmup_pending = False
            self._ready = False
            self._seeded = False
            self._reset_streak_locked()
            self._max_latency = 0.0
            logger.info("[RTC] WARMUP_DISCARDED epoch=%d latency=%.1fms", epoch, latency * 1000.0)
            self._write_trace({**trace, "phase": "warmup"})
            return

        if not cfg.readiness_gate:
            self._commit_ungated_locked(rows, dispatch, obs_t, epoch, guided, source_age, trace)
            return

        if source_age > cfg.max_action_age_s:
            reason = f"source_age={source_age:.3f}s > {cfg.max_action_age_s:.3f}s"
            if self._ready:
                self._trip_locked(f"RTC sustainability lost: {reason}")
                return
            self._clear_provisional_locked()
            self._reset_streak_locked()
            logger.warning("[RTC] STALE_RESULT_DISCARDED epoch=%d %s; readiness remains closed", epoch, reason)
            return

        chunk_size = len(rows)
        if not guided:
            if self._ready:
                self._trip_locked("RTC sustainability lost: a refill ran without leftover-action guidance")
                return
            try:
                queue_size, alignment, _ = self._merge_locked(rows, dispatch, obs_t)
            except Exception as exc:
                self._clear_provisional_locked()
                self._reset_streak_locked()
                logger.warning("[RTC] EMPTY_RESULT_DISCARDED epoch=%d: merge failed: %s", epoch, exc)
                return
            self._reset_streak_locked()
            if queue_size <= 0:
                self._clear_provisional_locked()
                logger.warning("[RTC] EMPTY_RESULT_DISCARDED epoch=%d queue=%d", epoch, queue_size)
                return
            self._seeded = True
            self._max_latency = max(self._max_latency, alignment.runtime_s)
            logger.info(
                "[RTC] SEED_PROVISIONAL epoch=%d latency=%.1fms queue=%d; publication remains closed",
                epoch,
                alignment.runtime_s * 1000.0,
                queue_size,
            )
            self._write_trace(
                {**trace, "phase": "seed", "merge_delay": alignment.merge_delay_steps, "queue": queue_size}
            )
            return

        if not self._seeded:
            reason = "guided result arrived without a provisional seed"
            if self._ready:
                self._trip_locked(f"RTC sustainability lost: {reason}")
                return
            self._clear_provisional_locked()
            self._reset_streak_locked()
            logger.warning("[RTC] READINESS_REJECTED epoch=%d: %s", epoch, reason)
            return

        try:
            queue_size, alignment, seam = self._merge_locked(rows, dispatch, obs_t)
        except Exception as exc:
            reason = f"merge failed: {type(exc).__name__}: {exc}"
            if self._ready:
                self._trip_locked(f"RTC sustainability lost: {reason}")
                return
            self._clear_provisional_locked()
            self._reset_streak_locked()
            logger.warning("[RTC] READINESS_REJECTED epoch=%d: %s; publication remains closed", epoch, reason)
            return

        candidate_latencies = (*self._guided_latencies, alignment.runtime_s)[-READINESS_GUIDED_FORWARDS:]
        assessment = assess_readiness(
            chunk_size=chunk_size,
            candidate_delay_steps=alignment.merge_delay_steps,
            guided_latencies_s=candidate_latencies,
            control_hz=cfg.control_hz,
            queue_threshold=self._queue_threshold(chunk_size),
            source_age_s=alignment.merge_t - obs_t,
            max_action_age_s=cfg.max_action_age_s,
            execution_horizon=cfg.execution_horizon,
            latency_guard_steps=cfg.readiness_latency_guard_steps,
            scheduler_guard_steps=cfg.readiness_scheduler_guard_steps,
            min_guided_overlap_steps=cfg.readiness_min_guided_overlap_steps,
        )
        self._write_trace(
            {
                **trace,
                "phase": "steady" if self._ready else "readiness",
                "consumed": alignment.consumed_steps,
                "merge_delay": alignment.merge_delay_steps,
                "queue": queue_size,
                "seam_arm_max": seam,
                "sustainable": assessment.sustainable,
                "failures": list(assessment.failures),
            }
        )
        if not assessment.sustainable and self._ready:
            self._trip_locked(f"RTC sustainability lost: {'; '.join(assessment.failures)}; {assessment.fields()}")
            return

        self._max_latency = max(self._max_latency, alignment.runtime_s)
        if queue_size != assessment.q_start or queue_size <= 0:
            reason = f"merged queue depth {queue_size} != expected {assessment.q_start}"
            if self._ready:
                self._trip_locked(f"RTC sustainability lost: {reason}")
                return
            self._clear_provisional_locked()
            self._reset_streak_locked()
            logger.warning("[RTC] READINESS_REJECTED epoch=%d: %s", epoch, reason)
            return

        self._seeded = True
        if not assessment.sustainable:
            # A failed pre-ready proof restarts from a fresh unguided seed.
            self._clear_provisional_locked()
            self._reset_streak_locked()
            logger.warning(
                "[RTC] READINESS_REJECTED epoch=%d: %s; %s; streak reset",
                epoch,
                "; ".join(assessment.failures),
                assessment.fields(),
            )
            return

        self._guided_latencies.append(alignment.runtime_s)
        self._guided_streak += 1
        if not self._ready:
            if self._guided_streak < READINESS_GUIDED_FORWARDS:
                logger.info(
                    "[RTC] READINESS_PROGRESS epoch=%d guided=%d/%d %s",
                    epoch,
                    self._guided_streak,
                    READINESS_GUIDED_FORWARDS,
                    assessment.fields(),
                )
                return
            self._ready = True
            logger.info(
                "[RTC] POLICY_READY epoch=%d latency=%.1fms %s",
                epoch,
                alignment.runtime_s * 1000.0,
                assessment.fields(),
            )

    def _commit_ungated_locked(
        self,
        rows: np.ndarray,
        dispatch: _Dispatch,
        obs_t: float,
        epoch: int,
        guided: bool,
        source_age: float,
        trace: dict,
    ) -> None:
        """readiness_gate=False: merge every fresh result; never latch on timing margins."""
        if source_age > self._cfg.max_action_age_s:
            logger.warning("[RTC] STALE_RESULT_DISCARDED epoch=%d source_age=%.3fs", epoch, source_age)
            return
        try:
            queue_size, alignment, seam = self._merge_locked(rows, dispatch, obs_t)
        except Exception as exc:
            logger.warning("[RTC] RESULT_DISCARDED epoch=%d: merge failed: %s", epoch, exc)
            return
        self._max_latency = max(self._max_latency, alignment.runtime_s)
        if guided:
            self._guided_latencies.append(alignment.runtime_s)
        self._seeded = True
        if not self._ready and queue_size > 0:
            self._ready = True
            logger.info("[RTC] POLICY_READY epoch=%d (readiness gate off)", epoch)
        self._write_trace(
            {
                **trace,
                "phase": "ungated",
                "consumed": alignment.consumed_steps,
                "merge_delay": alignment.merge_delay_steps,
                "queue": queue_size,
                "seam_arm_max": seam,
            }
        )

    def _write_trace(self, record: dict) -> None:
        if self._trace is None:
            return
        try:
            self._trace.write(json.dumps(record, default=float) + "\n")
            self._trace.flush()
        except OSError as exc:
            logger.warning("trace write failed: %s", exc)


class RtcStreamServer:
    """Websocket front end: one client streams observations, the engine streams commands."""

    def __init__(self, engine: RtcEngine, cfg: RtcStreamConfig, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Bind the server settings; nothing listens until ``serve_forever``.

        Args:
            engine: The RTC engine that owns the queue and the inference worker.
            cfg: Same config the engine was built with (``clock``, ``control_hz``).
            host: Interface to listen on.
            port: TCP port to listen on.
        """
        self._engine = engine
        self._cfg = cfg
        self._host = host
        self._port = port
        self._busy = False

    def serve_forever(self) -> None:
        """Run the websocket server until interrupted, then close the engine (blocking)."""
        try:
            asyncio.run(self._run())
        finally:
            self._engine.close()

    async def _run(self) -> None:
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_corobot._health_check,  # noqa: SLF001
        ) as server:
            logger.info("RTC stream server on %s:%d clock=%s", self._host, self._port, self._cfg.clock)
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        if self._busy:
            await websocket.send(msgpack_numpy.packb({"type": "error", "error": "another client is connected"}))
            await websocket.close()
            return
        self._busy = True
        loop = asyncio.get_running_loop()
        stop_clock = threading.Event()
        clock = None
        logger.info("RTC stream client %s connected", websocket.remote_address)

        def send_threadsafe(message: dict) -> None:
            data = msgpack_numpy.packb(message)
            future = asyncio.run_coroutine_threadsafe(websocket.send(data), loop)
            # Retrieve the result so a send on a closing socket is not logged as unhandled.
            future.add_done_callback(lambda f: None if f.cancelled() else f.exception())

        try:
            self._engine.start_session()
            await websocket.send(msgpack_numpy.packb(self._engine.status()))
            if self._cfg.clock == "wall":
                clock = threading.Thread(
                    target=self._wall_clock, args=(send_threadsafe, stop_clock), name="rtc-clock", daemon=True
                )
                clock.start()
            async for raw in websocket:
                await self._handle_message(websocket, raw)
        except websockets.ConnectionClosed:
            pass
        finally:
            stop_clock.set()
            if clock is not None:
                clock.join(timeout=2.0)
            self._engine.end_session()
            self._busy = False
            logger.info("RTC stream client %s disconnected", websocket.remote_address)

    async def _handle_message(self, websocket: _server.ServerConnection, raw: bytes) -> None:
        try:
            msg = msgpack_numpy.unpackb(raw)
            kind = _corobot._get(msg, "type")  # noqa: SLF001
            if isinstance(kind, bytes):
                kind = kind.decode("utf-8")
            if kind == "obs":
                params = _corobot._get(msg, "params") or {}  # noqa: SLF001
                obs = _corobot._corobot_params_to_obs(params, _corobot._resolve_embodiment(params))  # noqa: SLF001
                # Without task_name Policy.post_process leaves the model's full output
                # untouched; the waist is handled by the engine.
                obs.pop("task_name", None)
                self._engine.accept_observation(obs)
                if self._cfg.clock == "external" and _corobot._get(msg, "tick"):  # noqa: SLF001
                    reply = self._engine.tick() or self._engine.status()
                    await websocket.send(msgpack_numpy.packb(reply))
            elif kind == "rearm":
                ok, reason = self._engine.rearm()
                await websocket.send(
                    msgpack_numpy.packb({**self._engine.status(), "rearm_ok": ok, "rearm_reason": reason})
                )
            elif kind == "reset":
                self._engine.reset()
                await websocket.send(msgpack_numpy.packb({**self._engine.status(), "reset_ok": True}))
            else:
                await websocket.send(msgpack_numpy.packb({"type": "error", "error": f"unknown message type {kind!r}"}))
        except websockets.ConnectionClosed:
            raise
        except Exception:
            logger.error("bad client message:\n%s", traceback.format_exc())
            await websocket.send(msgpack_numpy.packb({"type": "error", "error": traceback.format_exc()}))

    def _wall_clock(self, send, stop: threading.Event) -> None:
        """Server execution clock: one tick per period on an absolute schedule."""
        period = 1.0 / self._cfg.control_hz
        next_t = time.monotonic()
        last_status = None
        last_status_t = 0.0
        while not stop.is_set():
            message = self._engine.tick()
            if message is not None:
                send(message)
            status = self._engine.status()
            now = time.monotonic()
            if status["state"] != last_status or now - last_status_t >= 1.0:
                send(status)
                last_status, last_status_t = status["state"], now
            next_t += period
            slack = next_t - time.monotonic()
            if slack > 0:
                time.sleep(slack)
            elif slack < -period:
                next_t = time.monotonic()  # fell a whole tick behind: resync instead of bursting
