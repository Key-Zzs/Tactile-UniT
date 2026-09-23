"""Frozen evaluation utilities for the PI2B matched-teacher comparison."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def flatten_normalize(value: np.ndarray) -> np.ndarray:
    flat = np.asarray(value, dtype=np.float64).reshape(len(value), -1)
    return flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)


def retrieval_ranks(query: np.ndarray, candidate: np.ndarray, chunk: int = 256) -> np.ndarray:
    left = flatten_normalize(query).astype(np.float32)
    right = flatten_normalize(candidate).astype(np.float32)
    if len(left) != len(right):
        raise ValueError("paired retrieval requires equal row counts")
    ranks = np.empty(len(left), dtype=np.int64)
    for start in range(0, len(left), chunk):
        stop = min(start + chunk, len(left))
        similarity = left[start:stop] @ right.T
        positive = similarity[np.arange(stop - start), np.arange(start, stop)]
        ranks[start:stop] = 1 + np.sum(similarity > positive[:, None], axis=1)
    return ranks


def retrieval_metrics(query: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    ranks = retrieval_ranks(query, candidate)
    count = len(ranks)
    return {
        "candidate_count": count,
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "chance": {
            "recall_at_1": 1.0 / count,
            "recall_at_5": min(5, count) / count,
            "recall_at_10": min(10, count) / count,
            "mrr": float(np.sum(1.0 / np.arange(1, count + 1)) / count),
            "median_rank": float((count + 1) / 2.0),
        },
    }


def paired_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(flatten_normalize(left) * flatten_normalize(right), axis=1)


def recovery_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction64 = np.asarray(prediction, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    residual = prediction64 - target64
    total = np.square(target64 - target64.mean(axis=0, keepdims=True)).sum()
    return {
        "mse": float(np.square(residual).mean()),
        "r2": float(1.0 - np.square(residual).sum() / max(float(total), 1e-12)),
        "cosine": float(paired_cosine(prediction64, target64).mean()),
    }


def geometry_metrics(value: np.ndarray) -> dict[str, float]:
    flat = np.asarray(value, dtype=np.float64).reshape(len(value), -1)
    centered = flat - flat.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    probability = np.square(singular) / max(float(np.square(singular).sum()), 1e-12)
    rank = float(np.exp(-np.sum(probability * np.log(np.maximum(probability, 1e-15)))))
    variance = flat.var(axis=0)
    tokens = np.asarray(value, dtype=np.float64)
    token_norm = tokens / np.maximum(np.linalg.norm(tokens, axis=-1, keepdims=True), 1e-12)
    token_similarity = np.einsum("bqd,bkd->bqk", token_norm, token_norm)
    off_diagonal = ~np.eye(tokens.shape[1], dtype=bool)
    return {
        "effective_rank": rank,
        "near_zero_variance_fraction": float(np.mean(variance < 1e-8)),
        "mean_dimension_variance": float(variance.mean()),
        "mean_query_off_diagonal_cosine": float(token_similarity[:, off_diagonal].mean()),
    }


def source_relative_rank(value: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    ranks = [geometry_metrics(value[groups == group])["effective_rank"] for group in np.unique(groups)]
    return {"mean": float(np.mean(ranks)), "minimum": float(np.min(ranks)), "maximum": float(np.max(ranks))}


def continuity(value: np.ndarray, episode: np.ndarray, anchor: np.ndarray) -> float:
    distances = []
    flat = np.asarray(value, dtype=np.float64).reshape(len(value), -1)
    for current in np.unique(episode):
        rows = np.flatnonzero(episode == current)
        rows = rows[np.argsort(anchor[rows])]
        if len(rows) > 1:
            distances.extend(np.linalg.norm(np.diff(flat[rows], axis=0), axis=1).tolist())
    return float(np.mean(distances)) if distances else float("nan")


def deterministic_control_indices(
    task: np.ndarray, group: np.ndarray, episode: np.ndarray, *, same_task: bool
) -> np.ndarray:
    count = len(task)
    result = np.empty(count, dtype=np.int64)
    for index in range(count):
        if same_task:
            pool = np.flatnonzero((task == task[index]) & (episode != episode[index]))
        else:
            pool = np.flatnonzero(group != group[index])
        if not len(pool):
            raise ValueError("control candidate pool is empty")
        token = f"{task[index]}\0{group[index]}\0{episode[index]}\0{index}".encode("utf-8")
        offset = int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % len(pool)
        result[index] = pool[offset]
    return result


def classification_probe(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    test_y: np.ndarray,
    classes: Iterable[int],
) -> tuple[dict[str, Any], np.ndarray]:
    classes = list(classes)
    observed = np.unique(train_y)
    if len(observed) < 2:
        return ({"status": "N/A_SINGLE_CLASS", "train_classes": observed.tolist()}, np.full(len(test_y), observed[0] if len(observed) else -1))
    model = make_pipeline(
        StandardScaler(),
        RidgeClassifier(alpha=10.0, class_weight="balanced"),
    ).fit(np.asarray(train_x).reshape(len(train_x), -1), train_y)
    prediction = model.predict(np.asarray(test_x).reshape(len(test_x), -1))
    return (
        {
            "status": "OK",
            "probe": "StandardScaler+RidgeClassifier(alpha=10,class_weight=balanced)",
            "macro_f1": float(f1_score(test_y, prediction, labels=classes, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(test_y, prediction)),
            "accuracy": float(np.mean(prediction == test_y)),
            "classes": classes,
        },
        prediction,
    )


def regression_probe(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    test_y: np.ndarray,
) -> tuple[dict[str, float | str], np.ndarray]:
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(
        np.asarray(train_x).reshape(len(train_x), -1), train_y
    )
    prediction = model.predict(np.asarray(test_x).reshape(len(test_x), -1))
    return (
        {
            "probe": "StandardScaler+Ridge(alpha=10)",
            "r2": float(r2_score(test_y, prediction)),
            "mae": float(mean_absolute_error(test_y, prediction)),
        },
        prediction,
    )


def stratified_group_bootstrap(
    group_values: dict[str, float], group_tasks: dict[str, str], *, samples: int, seed: int
) -> dict[str, Any]:
    tasks = sorted(set(group_tasks.values()))
    by_task = {task: sorted(group for group, value in group_tasks.items() if value == task) for task in tasks}
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = []
        for task in tasks:
            groups = by_task[task]
            selected.extend(rng.choice(groups, size=len(groups), replace=True).tolist())
        draws[index] = np.mean([group_values[group] for group in selected])
    point = float(np.mean(list(group_values.values())))
    lower, upper = np.quantile(draws, (0.025, 0.975))
    p = float(min(1.0, 2.0 * min(np.mean(draws <= 0.0), np.mean(draws >= 0.0))))
    return {"estimate": point, "ci95": [float(lower), float(upper)], "two_sided_bootstrap_p": p, "samples": samples, "seed": seed}


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, name in enumerate(ordered):
        value = min(1.0, (count - rank) * p_values[name])
        running = max(running, value)
        adjusted[name] = running
    return adjusted
