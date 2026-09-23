"""Reset identity and completeness helpers for the shared 200-reset cohort."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

import numpy as np


def sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def ordered_sequence_sha256(identities: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(identities).encode()).hexdigest()


def validate_reset_manifest(payload: dict[str, Any]) -> None:
    blocks = payload.get("blocks")
    specs = payload.get("reset_specs")
    identities = payload.get("ordered_reset_identities")
    if not isinstance(blocks, list) or len(blocks) != 4:
        raise ValueError("formal cohort must contain four blocks")
    if any(block.get("episodes") != 50 for block in blocks):
        raise ValueError("every formal block must contain 50 resets")
    if not isinstance(specs, list) or len(specs) != 200:
        raise ValueError("formal cohort must contain 200 reset specs")
    if not isinstance(identities, list) or len(identities) != 200:
        raise ValueError("formal cohort must contain 200 ordered identities")
    if len(set(identities)) != 200:
        raise ValueError("formal reset identities must be unique")
    if [row.get("reset_identity") for row in specs] != identities:
        raise ValueError("ordered identities and reset specs differ")
    if [row.get("global_index") for row in specs] != list(range(200)):
        raise ValueError("reset global indices are not exact")
    if payload.get("ordered_reset_sequence_sha256") != ordered_sequence_sha256(identities):
        raise ValueError("reset sequence hash differs")


def validate_rollout_tuples(rows: list[dict[str, Any]], expected_reset_ids: list[str]) -> None:
    expected_models = ("B0", "B_VA27", "B1", "B_HVA", "B2")
    expected = {
        (model, seed, reset_id)
        for model in expected_models
        for seed in (42, 43, 44)
        for reset_id in expected_reset_ids
    }
    actual = {
        (row.get("model_id"), row.get("training_seed"), row.get("reset_identity"))
        for row in rows
        if row.get("canonical") is True
    }
    if len(rows) != 3000 or len(actual) != 3000 or actual != expected:
        raise ValueError("formal rollout tuples are not the exact 15x200 canonical product")
