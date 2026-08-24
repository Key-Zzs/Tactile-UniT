"""Deterministic metrics for the canonical UniT representation benchmark.

This module intentionally depends only on NumPy and SciPy, so metric tests and
analysis can run without loading the model or changing the ``unit`` runtime.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


PAIRS: tuple[tuple[int, int, str], ...] = (
    (0, 1, "vision-action"),
    (0, 2, "vision-multimodal"),
    (1, 2, "action-multimodal"),
)


def default_pair_specs(modality_names: Iterable[str]) -> tuple[tuple[int, int, str], ...]:
    names = tuple(modality_names)
    return tuple(
        (left, right, f"{names[left]}-{names[right]}")
        for left in range(len(names))
        for right in range(left + 1, len(names))
    )


def mean_query_pool(features: np.ndarray) -> np.ndarray:
    """Mean pool ``[N, M, Q, D]`` features and L2-normalize the result."""

    values = np.asarray(features, dtype=np.float64).mean(axis=-2)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def flatten_query_pool(features: np.ndarray) -> np.ndarray:
    """Flatten ``[N, M, Q, D]`` features and L2-normalize the result."""

    values = np.asarray(features, dtype=np.float64).reshape(features.shape[0], features.shape[1], -1)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def paired_cosine_statistics(source: np.ndarray, target: np.ndarray) -> dict[str, float]:
    values = np.sum(source * target, axis=1)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
    }


def shuffled_negative_cosine(source: np.ndarray, target: np.ndarray, seed: int = 42) -> dict[str, float]:
    permutation = np.random.default_rng(seed).permutation(len(source))
    negative = np.sum(source * target[permutation], axis=1)
    positive = np.sum(source * target, axis=1)
    return {
        "positive_mean": float(np.mean(positive)),
        "negative_mean": float(np.mean(negative)),
        "margin": float(np.mean(positive) - np.mean(negative)),
        "permutation_seed": int(seed),
    }


def retrieval_metrics(source: np.ndarray, target: np.ndarray, ks: Iterable[int] = (1, 5, 10)) -> dict[str, float]:
    similarities = source @ target.T
    positive = np.diag(similarities)
    ranks = 1 + np.sum(similarities > positive[:, None], axis=1)
    result: dict[str, float] = {
        f"recall_at_{k}": float(np.mean(ranks <= k)) for k in ks
    }
    result["median_rank"] = float(np.median(ranks))
    result["mrr"] = float(np.mean(1.0 / ranks))
    result["candidate_count"] = int(len(source))
    result["chance_recall_at_1"] = 1.0 / len(source)
    result["chance_recall_at_5"] = min(5, len(source)) / len(source)
    result["chance_recall_at_10"] = min(10, len(source)) / len(source)
    return result


def _pairwise_squared_distances(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    y = x if y is None else y
    result = np.sum(x * x, axis=1)[:, None] + np.sum(y * y, axis=1)[None, :] - 2 * (x @ y.T)
    return np.maximum(result, 0.0)


def rbf_bandwidth(x: np.ndarray, y: np.ndarray) -> float:
    distances = _pairwise_squared_distances(np.concatenate([x, y], axis=0))
    upper = distances[np.triu_indices_from(distances, k=1)]
    nonzero = upper[upper > 0]
    return float(np.sqrt(np.median(nonzero))) if nonzero.size else 1.0


def mmd_rbf(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    sigma = rbf_bandwidth(x, y)
    denom = max(2.0 * sigma * sigma, 1e-12)
    kxx = np.exp(-_pairwise_squared_distances(x) / denom)
    kyy = np.exp(-_pairwise_squared_distances(y) / denom)
    kxy = np.exp(-_pairwise_squared_distances(x, y) / denom)
    squared = float(np.mean(kxx) + np.mean(kyy) - 2.0 * np.mean(kxy))
    return {"mmd": float(np.sqrt(max(squared, 0.0))), "bandwidth": sigma}


def sliced_wasserstein(x: np.ndarray, y: np.ndarray, projections: int = 128, seed: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(projections, x.shape[1]))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    projected_x = np.sort(x @ directions.T, axis=0)
    projected_y = np.sort(y @ directions.T, axis=0)
    # Canonical samples have equal cardinality. Interpolate only for future callers.
    if len(projected_x) != len(projected_y):
        grid_x = np.linspace(0.0, 1.0, len(projected_x))
        grid_y = np.linspace(0.0, 1.0, len(projected_y))
        common = np.linspace(0.0, 1.0, max(len(projected_x), len(projected_y)))
        projected_x = np.stack([np.interp(common, grid_x, projected_x[:, i]) for i in range(projections)], axis=1)
        projected_y = np.stack([np.interp(common, grid_y, projected_y[:, i]) for i in range(projections)], axis=1)
    return {
        "swd": float(np.mean(np.abs(projected_x - projected_y))),
        "projections": int(projections),
        "seed": int(seed),
    }


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - y.mean(axis=0, keepdims=True)
    numerator = float(np.sum((x_centered @ x_centered.T) * (y_centered @ y_centered.T)))
    denominator = math.sqrt(
        float(np.sum((x_centered @ x_centered.T) ** 2))
        * float(np.sum((y_centered @ y_centered.T) ** 2))
    )
    return float(numerator / denominator) if denominator > 0 else 0.0


def nmi_score(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    a = np.asarray(labels_a, dtype=np.int64).ravel()
    b = np.asarray(labels_b, dtype=np.int64).ravel()
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("NMI inputs must have equal nonzero length")
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    contingency = np.zeros((ia.max() + 1, ib.max() + 1), dtype=np.int64)
    np.add.at(contingency, (ia, ib), 1)
    n = float(len(a))
    rows = contingency.sum(axis=1, keepdims=True)
    cols = contingency.sum(axis=0, keepdims=True)
    nz = contingency > 0
    mi = float(np.sum((contingency[nz] / n) * np.log((contingency[nz] * n) / (rows @ cols)[nz])))
    p_rows = rows[:, 0] / n
    p_cols = cols[0, :] / n
    h_a = float(-np.sum(p_rows[p_rows > 0] * np.log(p_rows[p_rows > 0])))
    h_b = float(-np.sum(p_cols[p_cols > 0] * np.log(p_cols[p_cols > 0])))
    denom = math.sqrt(h_a * h_b)
    return float(mi / denom) if denom > 0 else 1.0


def ami_score(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Adjusted mutual information using the fixed-margin hypergeometric expectation."""

    a = np.asarray(labels_a, dtype=np.int64).ravel()
    b = np.asarray(labels_b, dtype=np.int64).ravel()
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("AMI inputs must have equal nonzero length")
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    contingency = np.zeros((ia.max() + 1, ib.max() + 1), dtype=np.int64)
    np.add.at(contingency, (ia, ib), 1)
    n = int(len(a))
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)
    mi = 0.0
    for i, ai in enumerate(row_sums):
        for j, bj in enumerate(col_sums):
            nij = int(contingency[i, j])
            if nij:
                mi += (nij / n) * math.log((nij * n) / (ai * bj))

    def log_comb(n_value: int, k_value: int) -> float:
        if k_value < 0 or k_value > n_value:
            return -math.inf
        return math.lgamma(n_value + 1) - math.lgamma(k_value + 1) - math.lgamma(n_value - k_value + 1)

    expected_mi = 0.0
    for ai in row_sums:
        for bj in col_sums:
            lo = max(1, int(ai + bj - n))
            hi = min(int(ai), int(bj))
            for nij in range(lo, hi + 1):
                log_probability = log_comb(int(ai), nij) + log_comb(n - int(ai), int(bj) - nij) - log_comb(n, int(bj))
                expected_mi += (nij / n) * math.log((nij * n) / (ai * bj)) * math.exp(log_probability)

    def entropy(counts: np.ndarray) -> float:
        probabilities = counts[counts > 0] / n
        return float(-np.sum(probabilities * np.log(probabilities)))

    h_a = entropy(row_sums)
    h_b = entropy(col_sums)
    average_entropy = (h_a + h_b) / 2.0
    denominator = average_entropy - expected_mi
    if denominator <= 1e-15:
        return 1.0 if math.isclose(mi, expected_mi, abs_tol=1e-12) else 0.0
    return float((mi - expected_mi) / denominator)


def codebook_stats(
    codes: np.ndarray,
    codebook_size: int,
    modality_names: Iterable[str] | None = None,
) -> list[dict[str, float | int]]:
    """Return per-modality, per-stage usage statistics for ``[N, M, Q, S]`` codes."""

    rows: list[dict[str, float | int]] = []
    names = tuple(modality_names) if modality_names is not None else tuple(
        f"modality_{index}" for index in range(codes.shape[1])
    )
    if len(names) != codes.shape[1]:
        raise ValueError("modality_names must match the modality axis of codes")
    for modality_index, modality in enumerate(names):
        for stage in range(codes.shape[-1]):
            values = codes[:, modality_index, :, stage].astype(np.int64).ravel()
            counts = np.bincount(values, minlength=codebook_size)
            probabilities = counts[counts > 0] / len(values)
            entropy = float(-np.sum(probabilities * np.log(probabilities)))
            frequencies = counts / len(values)
            rows.append({
                "modality": modality,
                "stage": stage,
                "active_codes": int(np.count_nonzero(counts)),
                "codebook_size": int(codebook_size),
                "active_ratio": float(np.count_nonzero(counts) / codebook_size),
                "entropy": entropy,
                "perplexity": float(np.exp(entropy)),
                "top1_frequency": float(np.max(frequencies)),
                "top5_frequency": float(np.sort(frequencies)[-5:].sum()),
            })
    return rows


def code_agreement(
    codes: np.ndarray,
    modality_names: Iterable[str] | None = None,
    pair_specs: Iterable[tuple[int, int, str]] | None = None,
) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    names = tuple(modality_names) if modality_names is not None else tuple(
        f"modality_{index}" for index in range(codes.shape[1])
    )
    if len(names) != codes.shape[1]:
        raise ValueError("modality_names must match the modality axis of codes")
    specs = tuple(pair_specs) if pair_specs is not None else default_pair_specs(names)
    for left, right, pair_name in specs:
        if left >= codes.shape[1] or right >= codes.shape[1]:
            raise ValueError("pair_specs contain a modality index outside the codes tensor")
        for stage in range(codes.shape[-1]):
            rows.append({
                "pair": pair_name,
                "stage": stage,
                "stage_exact_match": float(np.mean(codes[:, left, :, stage] == codes[:, right, :, stage])),
            })
        rows.append({
            "pair": pair_name,
            "stage": "full_tuple",
            "stage_exact_match": float(np.mean(np.all(codes[:, left] == codes[:, right], axis=(-1, -2)))),
        })
        for stage in range(codes.shape[-1]):
            left_set = set(np.unique(codes[:, left, :, stage]).tolist())
            right_set = set(np.unique(codes[:, right, :, stage]).tolist())
            union = left_set | right_set
            rows.append({
                "pair": pair_name,
                "stage": stage,
                "active_set_jaccard": float(len(left_set & right_set) / len(union)) if union else 1.0,
            })
        for stage in range(codes.shape[-1]):
            left_values = codes[:, left, :, stage].ravel()
            right_values = codes[:, right, :, stage].ravel()
            rows.append({
                "pair": pair_name,
                "stage": stage,
                "nmi": nmi_score(left_values, right_values),
                "ami": ami_score(left_values, right_values),
            })
    return rows
