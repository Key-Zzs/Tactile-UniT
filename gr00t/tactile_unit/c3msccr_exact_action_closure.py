"""Exact raw-Action perturbations for C3-MS-CC-R closure.

Perturbations are constructed in raw 58-D joint space and only then passed
through the accepted normalization and complete frozen A-R/C2-R pipeline.
This module intentionally contains no Contact predictor or test-selection
logic.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .continuous_vac_shared_space import different_episode_permutation
from .trex_action_data import ACTION_HORIZON, RAW_ACTION_DIM


CANONICAL_DIM = 128
VARIANTS = ("correct", "reversed", "shuffled", "different")
ACTION_ORDERING = ("left arm 7", "left hand 22", "right arm 7", "right hand 22")


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def action_moments(feature_stats: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(feature_stats["action_mean"], dtype=np.float32)
    std = np.asarray(feature_stats["action_std"], dtype=np.float32)
    if mean.shape != (RAW_ACTION_DIM,) or std.shape != (RAW_ACTION_DIM,):
        raise ValueError("accepted Action moments must be 58-D")
    if np.any(std <= 0) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("accepted Action moments are invalid")
    return mean, std


def raw_action_from_canonical(
    canonical_action: np.ndarray, feature_stats: Mapping[str, Any]
) -> np.ndarray:
    value = np.asarray(canonical_action, dtype=np.float32)
    if value.ndim != 3 or value.shape[1:] != (ACTION_HORIZON, CANONICAL_DIM):
        raise ValueError("canonical Action must be [B,16,128]")
    if not np.array_equal(value[..., RAW_ACTION_DIM:], np.zeros_like(value[..., RAW_ACTION_DIM:])):
        raise ValueError("canonical Action padding changed")
    mean, std = action_moments(feature_stats)
    return value[..., :RAW_ACTION_DIM] * std + mean


def canonical_action_from_raw(
    raw_action: np.ndarray, feature_stats: Mapping[str, Any]
) -> np.ndarray:
    value = np.asarray(raw_action, dtype=np.float32)
    if value.ndim != 3 or value.shape[1:] != (ACTION_HORIZON, RAW_ACTION_DIM):
        raise ValueError("raw Action must be [B,16,58]")
    mean, std = action_moments(feature_stats)
    result = np.zeros((len(value), ACTION_HORIZON, CANONICAL_DIM), dtype=np.float32)
    result[..., :RAW_ACTION_DIM] = (value - mean) / std
    return result


def deterministic_temporal_orders(count: int, seed: int) -> np.ndarray:
    """Replay the accepted NumPy permutation protocol independently per row."""

    rng = np.random.default_rng(int(seed))
    orders = np.empty((int(count), ACTION_HORIZON), dtype=np.uint8)
    for index in range(int(count)):
        order = rng.permutation(ACTION_HORIZON).astype(np.uint8)
        if np.array_equal(order, np.arange(ACTION_HORIZON, dtype=np.uint8)):
            order = np.roll(order, 1)
        orders[index] = order
    return orders


def same_split_different_indices(episode_id: np.ndarray, seed: int) -> np.ndarray:
    episode = np.asarray(episode_id, dtype=np.int64)
    result = different_episode_permutation(episode, int(seed))
    if result.shape != (len(episode),):
        raise AssertionError("different-episode mapping has invalid shape")
    if np.any(episode[result] == episode):
        raise AssertionError("different-episode mapping retained an episode")
    return result


def perturb_raw_action(
    raw_action: np.ndarray,
    *,
    temporal_orders: np.ndarray,
    different_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construct all four sources before feature construction or A-R encoding."""

    raw = np.asarray(raw_action, dtype=np.float32)
    orders = np.asarray(temporal_orders, dtype=np.int64)
    different = np.asarray(different_indices, dtype=np.int64)
    if raw.ndim != 3 or raw.shape[1:] != (ACTION_HORIZON, RAW_ACTION_DIM):
        raise ValueError("raw Action must be [B,16,58]")
    if orders.shape != (len(raw), ACTION_HORIZON):
        raise ValueError("temporal shuffle orders must be [B,16]")
    if different.shape != (len(raw),):
        raise ValueError("different indices must be [B]")
    if np.any(np.sort(orders, axis=1) != np.arange(ACTION_HORIZON)):
        raise ValueError("every temporal shuffle order must be a permutation of 0..15")
    row = np.arange(len(raw), dtype=np.int64)[:, None]
    return {
        "correct": raw.copy(),
        "reversed": raw[:, ::-1].copy(),
        "shuffled": raw[row, orders].copy(),
        "different": raw[different].copy(),
    }


def row_mse(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.square(
        np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    ).reshape(len(left), -1).mean(axis=1)


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = np.asarray(left, dtype=np.float64).reshape(len(left), -1)
    y = np.asarray(right, dtype=np.float64).reshape(len(right), -1)
    return np.sum(x * y, axis=1) / np.maximum(
        np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-12
    )


def per_query_distance(left: np.ndarray, right: np.ndarray) -> list[float]:
    value = np.square(
        np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    ).mean(axis=(0, 2))
    return [float(item) for item in value]
