from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gr00t.tactile_unit.trex_action_bootstrap import TREX_EMBODIMENT_ID
from gr00t.tactile_unit.trex_action_transition import (
    NativeTransitionActionModel,
    TransitionFeatureStats,
    TransitionFeatureTransform,
    _feature_segment,
    load_transition_checkpoint,
    save_transition_checkpoint,
)
from scripts.tactile_unit.train_action_transition_remediation import validation_selection_key
from scripts.tactile_unit.evaluate_action_transition_remediation import random_indices


ROOT = Path(__file__).resolve().parents[2]


def synthetic_stats() -> TransitionFeatureStats:
    zeros = [0.0] * 58
    ones = [1.0] * 58
    return TransitionFeatureStats(
        state_mean=zeros,
        state_std=ones,
        action_mean=zeros,
        action_std=ones,
        relative_mean=zeros,
        relative_std=ones,
        velocity_mean=zeros,
        velocity_std=ones,
    )


def canonical_batch(batch_size: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = torch.zeros(batch_size, 128)
    action = torch.zeros(batch_size, 16, 128)
    action[:, :, :58] = torch.arange(16, dtype=torch.float32)[None, :, None]
    embodiment = torch.full((batch_size,), TREX_EMBODIMENT_ID, dtype=torch.long)
    return state, action, embodiment


def test_transition_features_are_absolute_relative_and_velocity() -> None:
    transform = TransitionFeatureTransform(synthetic_stats())
    state, action, _ = canonical_batch(1)
    features = transform(state[:, :58], action[:, :, :58])
    assert features.shape == (1, 16, 174)
    torch.testing.assert_close(features[..., :58], action[..., :58])
    torch.testing.assert_close(features[..., 58:116], action[..., :58] - state[:, None, :58])
    assert torch.equal(features[:, 0, 116:], torch.zeros(1, 58))
    torch.testing.assert_close(features[:, 1:, 116:], torch.ones(1, 15, 58))


def test_grouped_feature_order_preserves_anatomical_slices() -> None:
    features = torch.arange(174, dtype=torch.float32).view(1, 1, 174)
    left_hand = _feature_segment(features, slice(7, 29))
    expected = torch.cat((features[..., 7:29], features[..., 65:87], features[..., 123:145]), dim=-1)
    assert torch.equal(left_hand, expected)


def test_native_interface_gradient_routing_and_temporal_sensitivity() -> None:
    torch.manual_seed(9)
    model = NativeTransitionActionModel(synthetic_stats(), hidden_size=32)
    state, action, embodiment = canonical_batch(2)
    output = model(state, action, embodiment)
    reversed_output = model(state, action.flip(1), embodiment)
    assert output["z_action"].shape == (2, 8, 32)
    assert output["prediction"].shape == (2, 16, 128)
    assert torch.isfinite(output["z_action"]).all()
    assert not torch.equal(output["z_action"], reversed_output["z_action"])
    output["prediction"][..., :58].square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_decoder_zero_and_shuffled_controls_are_well_defined() -> None:
    torch.manual_seed(12)
    model = NativeTransitionActionModel(synthetic_stats(), hidden_size=32).eval()
    state, action, embodiment = canonical_batch(2)
    z_action, state_features, _ = model.encode(state, action, embodiment)
    zero = model.decode(torch.zeros_like(z_action), state_features, embodiment)
    shuffled = model.decode(z_action[:, torch.tensor([7, 0, 6, 1, 5, 2, 4, 3])], state_features, embodiment)
    assert zero.shape == shuffled.shape == (2, 16, 128)
    assert torch.isfinite(zero).all() and torch.isfinite(shuffled).all()


def test_native_checkpoint_reload_is_exact(tmp_path: Path) -> None:
    torch.manual_seed(14)
    model = NativeTransitionActionModel(synthetic_stats(), hidden_size=32).eval()
    state, action, embodiment = canonical_batch(1)
    expected = model(state, action, embodiment)["z_action"]
    path = tmp_path / "native.pt"
    digest = save_transition_checkpoint(path, model, {"selection_split": "validation"})
    reloaded, metadata = load_transition_checkpoint(path)
    reloaded.eval()
    actual = reloaded(state, action, embodiment)["z_action"]
    assert len(digest) == 64
    assert torch.equal(expected, actual)
    assert metadata == {"selection_split": "validation"}


def test_native_rejects_non_trex_or_wrong_shapes() -> None:
    model = NativeTransitionActionModel(synthetic_stats(), hidden_size=32)
    state, action, embodiment = canonical_batch(1)
    with pytest.raises(ValueError, match="T-Rex"):
        model(state, action, torch.zeros_like(embodiment))
    with pytest.raises(ValueError, match="canonical state"):
        model(state[:, :58], action, embodiment)


def test_validation_selection_requires_temporal_token_and_noncollapse_gates() -> None:
    acceptance = {
        "normalized_mse_max": 1.0,
        "dynamic_reversed_ratio_min": 1.05,
        "dynamic_shuffled_ratio_min": 1.05,
        "zero_ratio_min": 1.1,
        "mean_ratio_min": 1.1,
        "effective_rank_min": 8.0,
        "collapsed_query_fraction_max": 0.05,
    }
    passing = {
        "normalized_mse": 0.2,
        "dynamic_reversed_ratio": 1.2,
        "dynamic_shuffled_ratio": 1.2,
        "all_different_episode_ratio": 2.0,
        "all_zero_ratio": 1.3,
        "all_mean_ratio": 1.2,
        "effective_rank": 10.0,
        "collapsed_query_fraction": 0.0,
        "finite": True,
        "z_action_shape_without_batch": [8, 32],
    }
    passing_key, passed, shortfall = validation_selection_key(passing, acceptance)
    failing = dict(passing, dynamic_shuffled_ratio=1.0)
    failing_key, failed, failing_shortfall = validation_selection_key(failing, acceptance)
    assert passed and shortfall == 0.0
    assert not failed and failing_shortfall > 0.0
    assert passing_key < failing_key


def test_random_evaluation_indices_are_deterministic_unique_and_in_bounds() -> None:
    first = random_indices(1000, 128, 41)
    second = random_indices(1000, 128, 41)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 128
    assert first.min() >= 0 and first.max() < 1000


def test_remediation_config_has_frozen_contract_and_no_private_paths() -> None:
    path = ROOT / "configs/tactile_unit/s3_3_r_action_transition_remediation.json"
    text = path.read_text()
    config = json.loads(text)
    assert config["data"]["raw_action_shape"] == [16, 58]
    assert config["data"]["action_interval"] == "a_t:t+15"
    assert config["gpu"]["allowed_physical"] == [1, 2, 3]
    assert config["gpu"]["forbidden_physical"] == [0]
    assert config["gpu"]["gpu1_authorization"] == "explicit user authorization on 2026-08-26"
    assert "/" + "home/" not in text
    assert "/" + "mnt/" not in text
    assert "Bear" + "er " not in text


def test_state_action_feature_names_support_relative_semantics() -> None:
    normalization = json.loads(
        (ROOT / ".local/artifacts/tactile_unit/s3_1/state_action_normalization.json").read_text()
    )
    state_names = normalization["features"]["observation.state"]
    action_names = normalization["features"]["action"]
    assert len(state_names) == len(action_names) == 58
    for state_name, action_name in zip(state_names, action_names):
        state_semantic = state_name.replace("left_arm_q_", "left_arm_").replace(
            "right_arm_q_", "right_arm_"
        ).replace("left_hand_q_", "left_hand_").replace("right_hand_q_", "right_hand_")
        action_semantic = action_name.replace("left_arm_target_dof_", "left_arm_").replace(
            "right_arm_target_dof_", "right_arm_"
        ).replace("left_hand_target_q_", "left_hand_").replace("right_hand_target_q_", "right_hand_")
        assert state_semantic == action_semantic
