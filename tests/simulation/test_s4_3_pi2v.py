from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from scripts.simulation.s4_3_pi2v_unit_adapter import (
    ACTION_RESAMPLE_INDICES,
    AppendDexJoCoSlot,
    dexjoco_action_view,
    dexjoco_state_view,
    normalize_and_pad,
    quaternion_wxyz_to_rotation_6d,
)


ROOT = Path(__file__).resolve().parents[2]


def test_protocol_is_frozen_before_formal_training() -> None:
    protocol = json.loads(
        (ROOT / "configs/simulation/s4_3_pi2v_unit_adapter_protocol.json").read_text()
    )
    assert protocol["status"] == "FROZEN_BEFORE_FORMAL_TRAINING"
    assert protocol["official_unit"]["commit"] == "0d762e32180bddd765694ef3846a3a5053f9d37f"
    assert protocol["adapter"]["new_category_id"] == 30
    assert protocol["adapter"]["trainable_parameter_count"] == 23_485_568
    assert protocol["optimization"]["effective_global_batch"] == 256
    assert protocol["optimization"]["optimizer_steps"] == 80_000
    assert protocol["formal_selection"] == "single preregistered step-80000 checkpoint; no DEV checkpoint selection"


def test_adapter_append_preserves_official_bank_and_only_trains_new_row() -> None:
    original = nn.Parameter(
        torch.arange(12, dtype=torch.float32).reshape(3, 4), requires_grad=False
    )
    adapter = AppendDexJoCoSlot(original, "W")
    value = adapter(original)
    assert value.shape == (4, 4)
    assert torch.equal(value[:3], original)
    assert adapter.adapter.requires_grad
    value[-1].sum().backward()
    assert original.grad is None
    assert torch.equal(adapter.adapter.grad, torch.ones_like(adapter.adapter))


def test_quaternion_wxyz_rotation6d_identity_and_state_layout() -> None:
    identity = quaternion_wxyz_to_rotation_6d(np.asarray([[1.0, 0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(identity, [[1, 0, 0, 0, 1, 0]], atol=1e-7)
    state = np.zeros((2, 23), dtype=np.float32)
    state[:, 3] = 1
    converted = dexjoco_state_view(state)
    assert converted.shape == (2, 25)
    np.testing.assert_allclose(converted[:, 3:9], identity.repeat(2, axis=0))


def test_action_resampling_and_masked_padding_are_exact() -> None:
    assert ACTION_RESAMPLE_INDICES.tolist() == [0, 2, 3, 5, 7, 8, 10, 12, 13, 15, 17, 18, 20, 22, 23, 25]
    action = np.broadcast_to(np.arange(27, dtype=np.float32)[:, None], (27, 22)).copy()
    selected = dexjoco_action_view(action)
    np.testing.assert_array_equal(selected[:, 0], ACTION_RESAMPLE_INDICES)
    padded, mask = normalize_and_pad(
        selected, np.zeros(22, dtype=np.float32), np.ones(22, dtype=np.float32), 128
    )
    assert padded.shape == mask.shape == (16, 128)
    assert np.all(mask[:, :22] == 1) and np.all(mask[:, 22:] == 0)
    assert np.all(padded[:, 22:] == 0)


def test_train_and_dev_pair_sources_remain_disjoint() -> None:
    pair_root = ROOT / ".local/cache/simulation/s4_2/pairs"
    if not pair_root.exists():
        pytest.skip("local frozen S4.2 pair cache is not materialized")
    with np.load(pair_root / "train.npz", allow_pickle=False) as train:
        train_groups = set(train["source_trajectory_id"].tolist())
    with np.load(pair_root / "validation.npz", allow_pickle=False) as dev:
        dev_groups = set(dev["source_trajectory_id"].tolist())
    assert len(train_groups) == 42 and len(dev_groups) == 9
    assert train_groups.isdisjoint(dev_groups)


def test_final_decision_enforces_preregistered_early_stop() -> None:
    decision = json.loads(
        (ROOT / "configs/simulation/s4_3_pi2v_final_decision.json").read_text()
    )
    assert decision["decision"] == "S4_3_PI2V_UNIT_ADAPTER_REPRESENTATION_FAIL"
    assert decision["failure_class"] == "REPRESENTATION_TRAINING_FAILURE"
    gates = decision["representation_validation"]["gates"]
    assert list(gates.values()).count("PASS") == 9
    assert [name for name, status in gates.items() if status == "FAIL"] == [
        "fused_vision_better_than_no_motion"
    ]
    reconstruction = decision["representation_validation"]["reconstruction"]
    assert reconstruction["fused_action_smooth_l1"] < reconstruction[
        "train_mean_action_smooth_l1"
    ]
    assert reconstruction["fused_vision_cosine_loss"] > reconstruction[
        "no_motion_vision_cosine_loss"
    ]
    assert not any(decision["downstream_guard"].values())
    assert decision["protocol_integrity"] == {
        "thresholds_changed_after_training": False,
        "checkpoint_shopping": False,
        "downstream_rollout_used": False,
    }
