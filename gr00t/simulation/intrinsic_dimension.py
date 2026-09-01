"""Deterministic intrinsic-dimension diagnostics for tactile representations."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors


REGIONS = 5
FEATURES = 6
COP_SLICE = slice(3, 6)


def fit_active_cop_means(tactile: np.ndarray) -> np.ndarray:
    """Fit per-region CoP means using contact-active TRAIN frames only."""

    values = np.asarray(tactile, dtype=np.float64)
    if values.shape[-1] != REGIONS * FEATURES:
        raise ValueError("tactile values must end in 30 features")
    matrix = values.reshape(-1, REGIONS, FEATURES)
    means = np.zeros((REGIONS, 3), dtype=np.float64)
    for region in range(REGIONS):
        active = matrix[:, region, 0] > 0.5
        if active.any():
            means[region] = matrix[active, region, COP_SLICE].mean(axis=0)
    return means


def mask_inactive_cop(tactile: np.ndarray, active_cop_means: np.ndarray) -> np.ndarray:
    """Neutralize inactive CoP entries without changing any training tensor.

    Replacing inactive CoP with the TRAIN active-contact mean makes those entries
    contribute zero centered CoP deviation. This avoids treating schema zeros as
    physical observations while preserving the original array shape.
    """

    values = np.asarray(tactile, dtype=np.float64)
    if values.shape[-1] != REGIONS * FEATURES:
        raise ValueError("tactile values must end in 30 features")
    means = np.asarray(active_cop_means, dtype=np.float64)
    if means.shape != (REGIONS, 3):
        raise ValueError("active CoP means must have shape [5,3]")
    output = values.copy().reshape(-1, REGIONS, FEATURES)
    active = output[..., 0] > 0.5
    cop = output[..., COP_SLICE]
    cop[:] = np.where(active[..., None], cop, means[None, ...])
    return output.reshape(values.shape)


def temporal_summary(history: np.ndarray) -> np.ndarray:
    """Return mean/std/last/max/min/first-to-last delta per tactile channel."""

    values = np.asarray(history, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (26, 30):
        raise ValueError("history must have shape [N,26,30]")
    return np.concatenate(
        [
            values.mean(axis=1),
            values.std(axis=1),
            values[:, -1],
            values.max(axis=1),
            values.min(axis=1),
            values[:, -1] - values[:, 0],
        ],
        axis=1,
    )


def fit_standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a deterministic feature standardizer, retaining constant columns as zero."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("standardizer requires a matrix")
    mean = array.mean(axis=0)
    std = array.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    return mean, std


def apply_standardizer(
    values: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return (array - np.asarray(mean)) / np.asarray(std)


def _twonn(values: np.ndarray, seed: int, maximum_samples: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) > maximum_samples:
        indices = np.random.default_rng(seed).choice(len(array), maximum_samples, replace=False)
        array = array[np.sort(indices)]
    if len(array) < 10:
        return {"status": "UNSTABLE", "reason": "fewer_than_10_samples", "value": None}
    neighbors = NearestNeighbors(n_neighbors=3, algorithm="auto", n_jobs=1).fit(array)
    distances = neighbors.kneighbors(array, return_distance=True)[0]
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    valid = (r1 > 1e-12) & (r2 > r1)
    valid_fraction = float(valid.mean())
    if valid_fraction < 0.8:
        return {
            "status": "UNSTABLE",
            "reason": "duplicate_or_tied_neighbors",
            "valid_fraction": valid_fraction,
            "samples": int(len(array)),
            "value": None,
        }
    logs = np.log(r2[valid] / r1[valid])
    mean_log = float(logs.mean())
    if not np.isfinite(mean_log) or mean_log <= 1e-12:
        return {
            "status": "UNSTABLE",
            "reason": "nonpositive_log_ratio",
            "valid_fraction": valid_fraction,
            "samples": int(len(array)),
            "value": None,
        }
    return {
        "status": "DIAGNOSTIC_ONLY",
        "estimator": "TwoNN_maximum_likelihood",
        "value": 1.0 / mean_log,
        "valid_fraction": valid_fraction,
        "samples": int(len(array)),
    }


def dimension_metrics(
    values: np.ndarray,
    *,
    twonn: bool = True,
    seed: int = 4242,
    maximum_twonn_samples: int = 1500,
    include_spectrum: bool = True,
) -> dict[str, Any]:
    """Compute covariance-spectrum dimension metrics without fitting a model."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("dimension metrics require [samples,features]")
    if len(array) < 2:
        raise ValueError("dimension metrics require at least two samples")
    if not np.isfinite(array).all():
        raise ValueError("dimension metrics require finite values")
    centered = array - array.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(eigenvalues.sum())
    if total <= 0:
        probabilities = np.zeros_like(eigenvalues)
        effective_rank = stable_rank = participation_ratio = 0.0
        cumulative = probabilities
    else:
        probabilities = eigenvalues / total
        positive = probabilities > 0
        effective_rank = float(np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive]))))
        stable_rank = float(total / max(float(eigenvalues[0]), 1e-30))
        participation_ratio = float(total**2 / max(float(np.square(eigenvalues).sum()), 1e-30))
        cumulative = np.cumsum(probabilities)

    def explained_dimension(target: float) -> int:
        if total <= 0:
            return 0
        return int(np.searchsorted(cumulative, target, side="left") + 1)

    variances = np.var(array, axis=0)
    positive_variances = variances[variances > 0]
    variance_reference = (
        float(np.median(positive_variances)) if len(positive_variances) else 0.0
    )
    near_zero_threshold = 1e-4 * variance_reference
    metrics: dict[str, Any] = {
        "samples": int(array.shape[0]),
        "features": int(array.shape[1]),
        "effective_rank": effective_rank,
        "stable_rank": stable_rank,
        "participation_ratio": participation_ratio,
        "d90": explained_dimension(0.90),
        "d95": explained_dimension(0.95),
        "d99": explained_dimension(0.99),
        "top_pc_explained_variance": float(probabilities[0]) if len(probabilities) else 0.0,
        "mean_variance": float(variances.mean()),
        "minimum_variance": float(variances.min()),
        "median_positive_variance": variance_reference,
        "near_zero_variance_threshold": near_zero_threshold,
        "near_zero_variance_fraction": float(np.mean(variances < near_zero_threshold)),
    }
    if include_spectrum:
        metrics["eigenvalues"] = eigenvalues.tolist()
        metrics["explained_variance_ratio"] = probabilities.tolist()
    metrics["twonn"] = (
        _twonn(array, seed, maximum_twonn_samples)
        if twonn
        else {"status": "NOT_COMPUTED", "value": None}
    )
    return metrics


def sampled_pairwise_diversity(
    values: np.ndarray, *, seed: int = 4242, maximum_samples: int = 4096
) -> dict[str, float]:
    """Compare exact duplicate distances with deterministic different-sample pairs."""

    array = np.asarray(values, dtype=np.float64)
    count = min(len(array), maximum_samples)
    rng = np.random.default_rng(seed)
    first = rng.choice(len(array), count, replace=False)
    second = np.roll(first, 1)
    different = np.linalg.norm(array[first] - array[second], axis=1)
    duplicate = np.linalg.norm(array[first] - array[first], axis=1)
    return {
        "same_sample_duplicate_distance_mean": float(duplicate.mean()),
        "different_sample_distance_mean": float(different.mean()),
        "different_sample_distance_p05": float(np.quantile(different, 0.05)),
    }
