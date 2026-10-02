"""Pure helpers for the Track-A PI2B policy result analysis."""

from __future__ import annotations

import itertools
import math
from typing import Iterable

import numpy as np


def exact_sign_flip_p(deltas: Iterable[float]) -> float:
    """Return the exact two-sided sign-flip sensitivity p-value."""

    values = np.asarray(tuple(deltas), dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError("deltas must be a nonempty finite vector")
    observed = abs(float(values.mean()))
    tolerance = 1e-15
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=values.size):
        statistic = abs(float(np.mean(values * np.asarray(signs))))
        extreme += statistic + tolerance >= observed
        total += 1
    return extreme / total


def summarize_seed_deltas(deltas: Iterable[float]) -> dict[str, object]:
    values = np.asarray(tuple(deltas), dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError("deltas must be a nonempty finite vector")
    return {
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)) if values.size > 1 else None,
        "range": [float(values.min()), float(values.max())],
        "exact_two_sided_sign_flip_p": exact_sign_flip_p(values),
        "training_seed_count": int(values.size),
    }


def direction_label(
    deltas: Iterable[float],
    interval: Iterable[float],
    *,
    positive: str,
    negative: str,
    mixed: str,
    inconclusive: str,
) -> str:
    """Apply the frozen conservative direction-label rule."""

    values = np.asarray(tuple(deltas), dtype=np.float64)
    bounds = tuple(float(value) for value in interval)
    if values.ndim != 1 or not values.size or len(bounds) != 2:
        raise ValueError("invalid deltas or interval")
    if not np.all(np.isfinite(values)) or not all(math.isfinite(v) for v in bounds):
        raise ValueError("deltas and interval must be finite")
    if bounds[0] > bounds[1]:
        raise ValueError("interval bounds are reversed")
    if np.all(values > 0.0) and bounds[0] > 0.0:
        return positive
    if np.all(values < 0.0) and bounds[1] < 0.0:
        return negative
    if np.any(values > 0.0) and np.any(values < 0.0):
        return mixed
    return inconclusive


def percent(value: float) -> float:
    return 100.0 * float(value)


def percent_interval(interval: Iterable[float]) -> list[float]:
    return [percent(value) for value in interval]
