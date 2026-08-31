"""Physical-time contracts for simulated tactile histories and transitions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

TACTILE_HISTORY_SEC = 0.5
CONTACT_TRANSITION_HORIZON_SEC = 16.0 / 30.0


@dataclass(frozen=True)
class TimingContract:
    """Freeze simulator rates and derive sample counts from physical time."""

    physics_dt: float
    control_dt: float
    tactile_history_sec: float = TACTILE_HISTORY_SEC
    transition_horizon_sec: float = CONTACT_TRANSITION_HORIZON_SEC

    def __post_init__(self) -> None:
        if self.physics_dt <= 0 or self.control_dt <= 0:
            raise ValueError("physics_dt and control_dt must be positive")
        ratio = self.control_dt / self.physics_dt
        if not np.isclose(ratio, round(ratio), atol=1e-9):
            raise ValueError("control_dt must contain an integer number of physics steps")

    @property
    def physics_substeps(self) -> int:
        return int(round(self.control_dt / self.physics_dt))

    @property
    def control_hz(self) -> float:
        return 1.0 / self.control_dt

    @property
    def history_samples(self) -> int:
        # Both interval endpoints are samples.
        return int(round(self.tactile_history_sec / self.control_dt)) + 1

    @property
    def transition_control_steps(self) -> int:
        return int(round(self.transition_horizon_sec / self.control_dt))

    @property
    def actual_transition_horizon_sec(self) -> float:
        return self.transition_control_steps * self.control_dt


@dataclass(frozen=True)
class TransitionPair:
    anchor_index: int
    future_anchor_index: int
    current_history_indices: tuple[int, ...]
    future_history_indices: tuple[int, ...]
    target_horizon_sec: float
    actual_horizon_sec: float

    @property
    def raw_overlap(self) -> bool:
        return bool(set(self.current_history_indices) & set(self.future_history_indices))


def transition_pair_for_anchor(
    anchor_index: int, length: int, timing: TimingContract
) -> TransitionPair:
    """Return causal current/future raw-history indices for an anchor."""

    history = timing.history_samples
    future = anchor_index + timing.transition_control_steps
    if anchor_index < history - 1:
        raise IndexError("anchor does not have a complete current tactile history")
    if future >= length:
        raise IndexError("future anchor is outside the episode")
    future_start = future - history + 1
    current_indices = tuple(range(anchor_index - history + 1, anchor_index + 1))
    future_indices = tuple(range(future_start, future + 1))
    return TransitionPair(
        anchor_index=anchor_index,
        future_anchor_index=future,
        current_history_indices=current_indices,
        future_history_indices=future_indices,
        target_horizon_sec=timing.transition_horizon_sec,
        actual_horizon_sec=timing.actual_transition_horizon_sec,
    )


class SimTactileHistoryBuffer:
    """Timestamped buffer that resamples tactile histories in physical time."""

    def __init__(self, tactile_dim: int, max_duration_sec: float = 2.0):
        if tactile_dim <= 0 or max_duration_sec <= 0:
            raise ValueError("tactile_dim and max_duration_sec must be positive")
        self.tactile_dim = int(tactile_dim)
        self.max_duration_sec = float(max_duration_sec)
        self._samples: deque[tuple[float, np.ndarray]] = deque()

    def append(self, timestamp_sec: float, tactile: np.ndarray) -> None:
        value = np.asarray(tactile, dtype=np.float32)
        if value.shape != (self.tactile_dim,):
            raise ValueError(f"expected tactile shape {(self.tactile_dim,)}, got {value.shape}")
        if self._samples and timestamp_sec <= self._samples[-1][0]:
            raise ValueError("timestamps must be strictly increasing")
        self._samples.append((float(timestamp_sec), value.copy()))
        cutoff = timestamp_sec - self.max_duration_sec
        while len(self._samples) > 1 and self._samples[1][0] < cutoff:
            self._samples.popleft()

    def history(
        self,
        anchor_sec: float,
        duration_sec: float = TACTILE_HISTORY_SEC,
        sample_dt: float = 0.02,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._samples:
            raise ValueError("history buffer is empty")
        if anchor_sec > self._samples[-1][0] + 1e-9:
            raise ValueError("anchor would require future tactile data")
        count = int(round(duration_sec / sample_dt)) + 1
        target_times = anchor_sec - duration_sec + np.arange(count, dtype=np.float64) * sample_dt
        times = np.asarray([row[0] for row in self._samples], dtype=np.float64)
        values = np.stack([row[1] for row in self._samples], axis=0)
        if target_times[0] < times[0] - 1e-9:
            raise ValueError("insufficient past tactile data for requested history")
        columns = [
            np.interp(target_times, times, values[:, index]) for index in range(self.tactile_dim)
        ]
        return target_times, np.stack(columns, axis=1).astype(np.float32)
