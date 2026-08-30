import inspect
import json
from pathlib import Path

import pytest
import torch

from gr00t.tactile_unit.c5_planned_action import (
    ActionRepresentation,
    PlannedActionChunk,
    PlannedActionSource,
    TrainOnlyActionNormalizer,
    encode_planned_action,
)


ROOT = Path(__file__).resolve().parents[2]


def plan(source=PlannedActionSource.POLICY_GENERATED, representation=ActionRepresentation.NORMALIZED_PADDED_128, actions=None):
    dim = 58 if representation is ActionRepresentation.RAW_58 else 128
    actions = torch.randn(2, 16, dim) if actions is None else actions
    return PlannedActionChunk(
        actions=actions,
        source=source,
        start_time=torch.tensor([1.0, 1.0]),
        representation=representation,
        normalization_state="RAW_UNNORMALIZED" if dim == 58 else "TRAIN_ONLY_STANDARDIZED_PADDED_128",
        validity_mask=torch.ones(2, 16, dtype=torch.bool),
        horizon=16,
        embodiment=31,
        planner_policy_id="synthetic-test",
    )


class Encoder:
    def encode(self, state, actions, embodiment):
        assert state.shape == (2, 128) and actions.shape == (2, 16, 128)
        assert torch.equal(embodiment, torch.full((2,), 31))
        return actions[:, :8, :32], None


def test_source_tag_shape_horizon_and_embodiment_are_mandatory():
    signature = inspect.signature(PlannedActionChunk)
    assert signature.parameters["source"].default is inspect.Parameter.empty
    with pytest.raises(ValueError, match="exact shape"):
        plan(actions=torch.randn(2, 17, 128))
    with pytest.raises(ValueError, match="exactly 16"):
        PlannedActionChunk(torch.randn(2, 16, 128), PlannedActionSource.POLICY_GENERATED, 0.0, ActionRepresentation.NORMALIZED_PADDED_128, "x", torch.ones(2, 16, dtype=torch.bool), 17, 31)
    with pytest.raises(ValueError, match="ID 31"):
        PlannedActionChunk(torch.randn(2, 16, 128), PlannedActionSource.POLICY_GENERATED, 0.0, ActionRepresentation.NORMALIZED_PADDED_128, "x", torch.ones(2, 16, dtype=torch.bool), 16, 30)


def test_runtime_legality_is_provenance_typed_and_fails_closed():
    plan(PlannedActionSource.POLICY_GENERATED).assert_legal(runtime=True)
    with pytest.raises(PermissionError, match="POLICY_GENERATED"):
        plan(PlannedActionSource.DEMONSTRATION_TEACHER).assert_legal(runtime=True)
    plan(PlannedActionSource.DEMONSTRATION_TEACHER).assert_legal(runtime=False, offline_training=True)
    with pytest.raises(PermissionError, match="oracle_eval"):
        plan(PlannedActionSource.ORACLE_EVAL).assert_legal(runtime=False)
    plan(PlannedActionSource.ORACLE_EVAL).assert_legal(runtime=False, oracle_eval=True)


def test_identical_numeric_chunks_encode_identically_across_source_tags():
    actions = torch.randn(2, 16, 128)
    state = torch.randn(2, 128)
    encode = lambda source, **flags: encode_planned_action(
        plan(source, actions=actions.clone()), state, Encoder(), lambda value: value,
        runtime=source is PlannedActionSource.POLICY_GENERATED, **flags,
    )
    policy = encode(PlannedActionSource.POLICY_GENERATED)
    demo = encode(PlannedActionSource.DEMONSTRATION_TEACHER, offline_training=True)
    oracle = encode(PlannedActionSource.ORACLE_EVAL, oracle_eval=True)
    assert torch.equal(policy.z_a, demo.z_a) and torch.equal(policy.z_a, oracle.z_a)
    assert torch.equal(policy.u_a, demo.u_a) and torch.equal(policy.u_a, oracle.u_a)


def test_raw_58_uses_accepted_train_only_normalization_and_zero_padding():
    stats = {
        "mode": "mean_std", "fit_split": "frozen S1 train episodes only",
        "state": {"mean": [0.0] * 58, "std": [2.0] * 58},
        "action": {"mean": [1.0] * 58, "std": [4.0] * 58},
    }
    normalizer = TrainOnlyActionNormalizer(stats)
    raw = plan(PlannedActionSource.POLICY_GENERATED, representation=ActionRepresentation.RAW_58)
    encoded = encode_planned_action(raw, torch.randn(2, 58), Encoder(), lambda value: value, normalizer=normalizer, runtime=True)
    assert encoded.z_a.shape == (2, 8, 32)


def test_raw_58_accepts_frozen_transition_feature_stats_schema():
    stats = {
        "fit_split": "frozen train split only",
        "state_mean": [0.0] * 58, "state_std": [2.0] * 58,
        "action_mean": [1.0] * 58, "action_std": [4.0] * 58,
    }
    normalizer = TrainOnlyActionNormalizer(stats)
    state, actions = normalizer.transform(torch.zeros(2, 58), torch.ones(2, 16, 58))
    assert state.shape == (2, 128) and actions.shape == (2, 16, 128)
    assert torch.count_nonzero(actions) == 0
    assert torch.count_nonzero(state[..., 58:]) == 0


def test_config_declares_no_policy_fabrication_or_policy_domain_validation():
    config = json.loads((ROOT / "configs/tactile_unit/c5_causal_visual_planned_action.json").read_text())
    assert config["planned_action"]["policy_plan_domain_validated"] is False
    assert config["scope"]["policy_training"] is False
    assert config["scope"]["c6_m3_started"] is False
