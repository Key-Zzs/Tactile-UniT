"""Pre-registered paired and crossed-seed statistics for Track A."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError((successes, total))
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    low = 0.0 if successes == 0 else max(0.0, center - half)
    high = 1.0 if successes == total else min(1.0, center + half)
    return low, high


def discordant_table(first: Iterable[bool], second: Iterable[bool]) -> dict[str, int]:
    a = np.asarray(tuple(first), dtype=np.bool_)
    b = np.asarray(tuple(second), dtype=np.bool_)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired outcomes must be same-length vectors")
    return {
        "both_fail": int(np.sum(~a & ~b)),
        "first_only": int(np.sum(a & ~b)),
        "second_only": int(np.sum(~a & b)),
        "both_success": int(np.sum(a & b)),
    }


def exact_mcnemar(first: Iterable[bool], second: Iterable[bool]) -> float:
    table = discordant_table(first, second)
    left = table["first_only"]
    right = table["second_only"]
    total = left + right
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(min(left, right) + 1)) / (2**total)
    return min(1.0, 2.0 * tail)


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = [float(value) for value in p_values]
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("p-values must be in [0,1]")
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.0] * len(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def paired_risk_difference(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("paired arrays must have the same nonempty shape")
    return float(np.mean(a - b))


def _quantile_interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [alpha, 1.0 - alpha])
    return float(low), float(high)


def conditional_reset_bootstrap(
    first: np.ndarray,
    second: np.ndarray,
    *,
    repetitions: int = 100_000,
    seed: int = 4317,
    confidence: float = 0.95,
    chunk_size: int = 2_000,
) -> dict[str, object]:
    """Resample the shared reset index while conditioning on trained seeds."""

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or not a.shape[0] or not a.shape[1]:
        raise ValueError("outcomes must be matching [training_seed, reset] arrays")
    differences = a - b
    generator = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    resets = a.shape[1]
    for start in range(0, repetitions, chunk_size):
        stop = min(repetitions, start + chunk_size)
        indices = generator.integers(0, resets, size=(stop - start, resets))
        samples[start:stop] = differences[:, indices].mean(axis=(0, 2))
    return {
        "estimate": float(differences.mean()),
        "interval": _quantile_interval(samples, confidence),
        "repetitions": repetitions,
        "seed": seed,
        "unit": "shared reset index across every model and training seed",
    }


def two_way_bootstrap(
    first: np.ndarray,
    second: np.ndarray,
    *,
    repetitions: int = 100_000,
    seed: int = 4317,
    confidence: float = 0.95,
    chunk_size: int = 2_000,
) -> dict[str, object]:
    """Resample shared training-seed and reset indices as a sensitivity analysis."""

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or not a.shape[0] or not a.shape[1]:
        raise ValueError("outcomes must be matching [training_seed, reset] arrays")
    differences = a - b
    generator = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    seeds, resets = differences.shape
    for start in range(0, repetitions, chunk_size):
        stop = min(repetitions, start + chunk_size)
        width = stop - start
        seed_indices = generator.integers(0, seeds, size=(width, seeds))
        reset_indices = generator.integers(0, resets, size=(width, resets))
        values = differences[seed_indices[:, :, None], reset_indices[:, None, :]]
        samples[start:stop] = values.mean(axis=(1, 2))
    return {
        "estimate": float(differences.mean()),
        "interval": _quantile_interval(samples, confidence),
        "repetitions": repetitions,
        "seed": seed,
        "unit": "shared training-seed index crossed with shared reset index",
        "limitation": "three training seeds provide weak outer-level uncertainty estimation",
    }
