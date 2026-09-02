import json
from pathlib import Path

import numpy as np
import pytest

from gr00t.simulation.s4_3_policy import (
    TensorProvenance,
    assert_inference_provenance,
    learning_sanity,
    macro_task_success,
    material_effect,
    timeout_statistics,
    valid_bc_windows,
    validate_policy_eval_resets,
)

ROOT = Path(__file__).resolve().parents[2]


def read_config(name: str) -> dict:
    return json.loads((ROOT / "configs/simulation" / name).read_text(encoding="utf-8"))


def test_frozen_restart_history_dataset_and_variants() -> None:
    policy = read_config("s4_3_restart_policy_protocol.json")
    assert policy["status"] == "FROZEN_BEFORE_ACT_IMPLEMENTATION_AND_TRAINING"
    assert policy["previous_s4_3_0"] == {
        "decision": "S4_3_0_DEMONSTRATION_CONTRACT_FAIL",
        "status": "FAILED_PRESERVED",
        "modified": False,
    }
    assert policy["restart_basis"] == "S4_3_PD_COMPLETE_POLICY_DATA_READY"
    assert policy["old_probing_data_used"] is False
    assert set(policy["variants"]) == {"P0", "P1", "P2", "P3"}
    assert policy["training_seeds"] == [0, 1, 2]
    assert (
        policy["mapped_tactile_inactive_control"]["warning"]
        == "HAMMER_NAIL_MAPPED_TACTILE_INACTIVE"
    )
    assert policy["preregistered_secondary"]["tasks"] == ["pinch_tongs", "click_mouse"]


def test_policy_eval_v1_is_exact_and_unique() -> None:
    evaluation = read_config("s4_3_policy_eval_v1.json")
    validate_policy_eval_resets(evaluation)
    assert len({row["evaluation_reset_id"] for row in evaluation["resets"]}) == 90
    assert len({row["reset_seed"] for row in evaluation["resets"]}) == 90


def test_timeout_and_window_contracts() -> None:
    assert valid_bc_windows(446) == 395
    assert valid_bc_windows(51) == 0
    assert valid_bc_windows(52) == 1
    assert timeout_statistics([100] * 10)["timeout_steps"] == 250
    assert timeout_statistics([1000] * 10)["timeout_steps"] == 1000
    policy = read_config("s4_3_restart_policy_protocol.json")
    assert policy["timeouts"]["tasks"]["pinch_tongs"]["timeout_steps"] == 823
    assert policy["timeouts"]["tasks"]["hammer_nail"]["timeout_steps"] == 635
    assert policy["timeouts"]["tasks"]["click_mouse"]["timeout_steps"] == 1000


def test_material_macro_and_learning_rules_are_frozen() -> None:
    assert material_effect(0.05, (0.001, 0.10)) == "MATERIAL_IMPROVEMENT"
    assert material_effect(-0.05, (-0.10, -0.001)) == "MATERIAL_HURT"
    assert material_effect(0.049, (0.001, 0.10)) == "NO_MATERIAL_DIFFERENCE"
    assert macro_task_success({"a": 0.0, "b": 0.5, "c": 1.0}, ("a", "b", "c")) == 0.5
    assert (
        learning_sanity({"pinch_tongs": 0.2, "hammer_nail": 0.0, "click_mouse": 0.3}) == "HEALTHY"
    )
    assert learning_sanity({"pinch_tongs": 0.1, "hammer_nail": 0.0, "click_mouse": 0.0}) == "WEAK"
    assert (
        learning_sanity({"pinch_tongs": 0.0, "hammer_nail": 0.0, "click_mouse": 0.0})
        == "ZERO_LEARNING"
    )


def test_inference_provenance_rejects_future_and_training_targets() -> None:
    valid = [
        TensorProvenance("episode", 25, 0, 25, "OBSERVATION"),
        TensorProvenance("episode", 25, 25, 51, "PLAN"),
        TensorProvenance("episode", 25, 0, 25, "DIAGNOSTIC"),
    ]
    assert_inference_provenance(valid)
    with pytest.raises(ValueError, match="future observation"):
        assert_inference_provenance([TensorProvenance("episode", 25, 25, 26, "OBSERVATION")])
    with pytest.raises(ValueError, match="training target"):
        assert_inference_provenance([TensorProvenance("episode", 25, 27, 52, "TRAINING_TARGET")])


def test_act_freeze_has_identical_p2_p3_inference_and_no_future_vision() -> None:
    act = read_config("s4_3_restart_act_protocol.json")
    assert act["status"] == "FROZEN_BEFORE_ACT_IMPLEMENTATION_AND_TRAINING"
    assert act["observation"]["vision"]["input"] == "I_t only"
    assert act["observation"]["vision"]["future_transition_latent"] is False
    assert act["p3_auxiliary"]["inference_inputs_identical_to"] == "P2"
    assert act["p3_auxiliary"]["lambda_contact"] == 0.1
    assert act["fairness"]["P2_P3_identical_inference_architecture_and_parameter_count"] is True
    assert act["jobs"]["total"] == 36
    assert act["act"]["action_chunk"] == 27
    assert act["act"]["action_dim"] == 22
    assert np.isclose(act["optimizer"]["learning_rate"], 1e-4)
