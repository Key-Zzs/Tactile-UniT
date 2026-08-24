"""Physical-time tactile history/future sampling and deterministic resampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalWindow:
    history_sec: float
    future_sec: float
    history_steps: int
    future_steps: int

    def __post_init__(self) -> None:
        if self.history_sec <= 0 or self.future_sec <= 0:
            raise ValueError("physical durations must be positive")
        if self.history_steps < 2 or self.future_steps < 1:
            raise ValueError("history_steps >= 2 and future_steps >= 1 are required")

    def to_dict(self) -> dict:
        return {
            "history_sec": self.history_sec,
            "future_sec": self.future_sec,
            "history_steps": self.history_steps,
            "future_steps": self.future_steps,
            "resampling": "linear interpolation on real timestamps",
            "anchor_semantics": "history includes anchor; future strictly follows anchor",
        }


def _interp_columns(timestamps: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.interp(query, timestamps, values[:, i]) for i in range(values.shape[1])], axis=-1
    ).astype(np.float32)


def resample_window(
    timestamps: np.ndarray,
    values: np.ndarray,
    anchor_time: float,
    window: TemporalWindow,
    *,
    query_jitter: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return history/future values and their physical query timestamps.

    The input must cover the full requested range.  Extrapolation is rejected so
    that no boundary repetition can leak into scientific evaluation.
    """

    timestamps = np.asarray(timestamps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float32)
    if timestamps.ndim != 1 or values.ndim != 2 or len(timestamps) != len(values):
        raise ValueError("timestamps [N] and values [N,D] must align")
    if len(timestamps) < 2 or not np.all(np.diff(timestamps) > 0):
        raise ValueError("timestamps must be strictly increasing")
    history_query = np.linspace(
        anchor_time - window.history_sec, anchor_time, window.history_steps, dtype=np.float64
    )
    future_query = anchor_time + np.linspace(
        window.future_sec / window.future_steps,
        window.future_sec,
        window.future_steps,
        dtype=np.float64,
    )
    if query_jitter is not None:
        jitter = np.asarray(query_jitter, dtype=np.float64)
        if jitter.shape != (window.history_steps + window.future_steps,):
            raise ValueError("query_jitter has the wrong shape")
        history_query = history_query + jitter[: window.history_steps]
        future_query = future_query + jitter[window.history_steps :]
    all_query = np.concatenate([history_query, future_query])
    if all_query.min() < timestamps[0] or all_query.max() > timestamps[-1]:
        raise ValueError("episode does not cover requested history/future range")
    if not np.all(np.diff(history_query) > 0) or not np.all(np.diff(future_query) > 0):
        raise ValueError("query timestamps must remain strictly increasing")
    history = _interp_columns(timestamps, values, history_query)
    future = _interp_columns(timestamps, values, future_query)
    return history, future, history_query, future_query


def resample_windows(
    timestamps: np.ndarray,
    values: np.ndarray,
    anchor_times: np.ndarray,
    window: TemporalWindow,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized equivalent of :func:`resample_window` for many anchors."""

    timestamps = np.asarray(timestamps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float32)
    anchors = np.asarray(anchor_times, dtype=np.float64)
    if timestamps.ndim != 1 or values.ndim != 2 or len(timestamps) != len(values):
        raise ValueError("timestamps [N] and values [N,D] must align")
    if anchors.ndim != 1 or len(anchors) == 0:
        raise ValueError("anchor_times must be a non-empty vector")
    if len(timestamps) < 2 or not np.all(np.diff(timestamps) > 0):
        raise ValueError("timestamps must be strictly increasing")
    history_offsets = np.linspace(-window.history_sec, 0.0, window.history_steps)
    future_offsets = np.linspace(
        window.future_sec / window.future_steps, window.future_sec, window.future_steps
    )
    history_query = anchors[:, None] + history_offsets[None, :]
    future_query = anchors[:, None] + future_offsets[None, :]
    query = np.concatenate([history_query, future_query], axis=1)
    if query.min() < timestamps[0] or query.max() > timestamps[-1]:
        raise ValueError("episode does not cover requested history/future range")

    flat_query = query.reshape(-1)
    right = np.searchsorted(timestamps, flat_query, side="left")
    right = np.clip(right, 1, len(timestamps) - 1)
    left = right - 1
    denominator = timestamps[right] - timestamps[left]
    alpha = ((flat_query - timestamps[left]) / denominator).astype(np.float32)
    interpolated = values[left] + alpha[:, None] * (values[right] - values[left])
    interpolated = interpolated.reshape(len(anchors), query.shape[1], values.shape[1])
    return (
        interpolated[:, : window.history_steps].astype(np.float32, copy=False),
        interpolated[:, window.history_steps :].astype(np.float32, copy=False),
    )
