"""Metrics and deterministic controls for S2 contact dynamics."""

from __future__ import annotations

import numpy as np


def transition_metrics(
    current: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    current = np.asarray(current, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        current, target, prediction = current[mask], target[mask], prediction[mask]
    error = prediction - target
    target_norm = np.linalg.norm(target, axis=1)
    prediction_norm = np.linalg.norm(prediction, axis=1)
    cosine = np.sum(target * prediction, axis=1) / np.maximum(
        target_norm * prediction_norm, 1e-12
    )
    delta_error = (prediction - current) - (target - current)
    return {
        "windows": int(len(target)),
        "future_mse": float(np.square(error).mean()),
        "future_cosine": float(cosine.mean()),
        "delta_mse": float(np.square(delta_error).mean()),
    }


def different_episode_permutation(episode_ids: np.ndarray, seed: int = 42) -> np.ndarray:
    """Return a deterministic permutation with no same-episode pairing."""

    episode_ids = np.asarray(episode_ids)
    n = len(episode_ids)
    if n < 2 or len(np.unique(episode_ids)) < 2:
        raise ValueError("negative control requires at least two episodes")
    # Sort into episode blocks, rotate by the largest block, then map the
    # rotated indices back to original row order.  This is a deterministic
    # derangement whenever no episode owns more than half the split.
    order = np.argsort(episode_ids, kind="stable")
    unique, counts = np.unique(episode_ids[order], return_counts=True)
    if len(unique) < 2 or counts.max() * 2 > n:
        raise ValueError("cannot construct an all-different episode permutation")
    candidate = np.roll(order, int(counts.max()))
    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = candidate
    # The seed is part of the public protocol identity even though the
    # block-rotation construction itself needs no randomness.
    _ = int(seed)
    if not np.all(episode_ids[inverse] != episode_ids):
        raise AssertionError("failed to construct negative-control permutation")
    return inverse


def query_diversity(code: np.ndarray) -> dict[str, float]:
    code = np.asarray(code, dtype=np.float64)
    if code.ndim != 3:
        raise ValueError("code must be [N,Q,D]")
    centered = code - code.mean(axis=1, keepdims=True)
    within_distance = np.linalg.norm(centered, axis=2)
    normalized = code / np.maximum(np.linalg.norm(code, axis=2, keepdims=True), 1e-12)
    cosine = np.einsum("nqd,nrd->nqr", normalized, normalized)
    off_diagonal = ~np.eye(code.shape[1], dtype=bool)
    return {
        "mean_distance_from_sample_token_mean": float(within_distance.mean()),
        "mean_off_diagonal_cosine": float(cosine[:, off_diagonal].mean()),
        "collapsed_sample_fraction": float(np.mean(within_distance.max(axis=1) < 1e-6)),
    }
