from __future__ import annotations

import pytest

from gr00t.simulation.pi2b_policy.resets import (
    ordered_sequence_sha256,
    validate_reset_manifest,
    validate_rollout_tuples,
)


def fake_manifest():
    identities = [f"reset-{index}" for index in range(200)]
    return {
        "blocks": [{"seed": seed, "episodes": 50} for seed in (16, 17, 18, 19)],
        "reset_specs": [
            {"global_index": index, "reset_identity": identity}
            for index, identity in enumerate(identities)
        ],
        "ordered_reset_identities": identities,
        "ordered_reset_sequence_sha256": ordered_sequence_sha256(identities),
    }


def test_reset_manifest_requires_exact_four_by_fifty():
    validate_reset_manifest(fake_manifest())
    broken = fake_manifest()
    broken["reset_specs"] = broken["reset_specs"][:-1]
    with pytest.raises(ValueError):
        validate_reset_manifest(broken)


def test_rollout_completeness_requires_exact_3000_shared_tuples():
    reset_ids = [f"reset-{index}" for index in range(200)]
    rows = [
        {"model_id": model, "training_seed": seed, "reset_identity": reset, "canonical": True}
        for model in ("B0", "B_VA27", "B1", "B_HVA", "B2")
        for seed in (42, 43, 44)
        for reset in reset_ids
    ]
    validate_rollout_tuples(rows, reset_ids)
    rows[-1]["canonical"] = False
    with pytest.raises(ValueError):
        validate_rollout_tuples(rows, reset_ids)
