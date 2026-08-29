import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/tactile_unit/c5_causal_visual_planned_action.json"


def config():
    return json.loads(CONFIG.read_text())


def artifacts(required=None):
    root = ROOT / ".local/artifacts/tactile_unit/vac_c5"
    if required and not (root / required).is_file():
        pytest.skip(f"local C5 artifact unavailable: {required}")
    return root


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_branch_and_accepted_c1_through_c4_ancestry():
    assert subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip() == "develop/tactile-unit-vac"
    history = subprocess.run(["git", "log", "--format=%h", "-80"], cwd=ROOT, text=True, capture_output=True, check=True).stdout
    for prefix in ("f43b71c", "beaa831", "639b4a9", "808416a", "5fe4bdb", "6a271c1", "7e77f7e", "9e1b431"):
        assert prefix in history


def test_all_accepted_checkpoints_have_exact_frozen_hashes():
    value = config(); runtime, accepted = value["runtime"], value["accepted"]
    mapping = {
        "full_checkpoint": "full_checkpoint_sha256", "offline_va_checkpoint": "offline_va_checkpoint_sha256",
        "emergency_a_checkpoint": "emergency_a_checkpoint_sha256", "c4_uncertainty_checkpoint": "c4_uncertainty_checkpoint_sha256",
        "action_checkpoint": "action_checkpoint_sha256", "c2_checkpoint": "c2_checkpoint_sha256",
        "c2r_checkpoint": "c2r_checkpoint_sha256", "c3dp_checkpoint": "c3dp_checkpoint_sha256",
        "s1_checkpoint": "s1_checkpoint_sha256", "s2_checkpoint": "s2_checkpoint_sha256",
    }
    for path_key, hash_key in mapping.items():
        assert sha(ROOT / runtime[path_key]) == accepted[hash_key]
    assert sha(ROOT / runtime["c1_cache_root"] / "manifest.json") == accepted["c1_manifest_sha256"]


def test_config_freezes_exact_splits_six_trials_and_forbidden_scope():
    value = config()
    assert value["counts"] == {"train": 65536, "validation": 8192, "test": 17504}
    assert [row["id"] for row in value["training"]["trials"]] == [f"T{index}" for index in range(6)]
    assert value["training"]["maximum_trials"] == 6
    assert value["scope"]["policy_training"] is False
    assert value["scope"]["private_residual_prediction"] is False
    assert value["scope"]["shared_space_retuning"] is False
    assert value["scope"]["full_path_retuning"] is False
    assert value["scope"]["c6_m3_started"] is False


def test_pretest_contracts_are_hash_frozen_and_test_unloaded():
    root = artifacts("c5_contract.json")
    for name in ("c5_contract.json", "planned_action_contract.json"):
        path = root / name; digest = root / f"{name}.sha256"
        assert sha(path) == digest.read_text().split()[0]
        assert json.loads(path.read_text())["test_loaded"] is False
    for stem in ("causal_visual_selection", "uncertainty_selection", "runtime_router_contract"):
        path = root / f"{stem}.json"; digest = root / f"{stem}.sha256"
        if not path.is_file(): pytest.skip(f"{stem} not yet frozen")
        assert sha(path) == digest.read_text().split()[0]
        assert json.loads(path.read_text())["test_loaded"] is False


def test_planned_action_contract_is_continuous_pre_rq_and_policy_honest():
    value = json.loads((artifacts("planned_action_contract.json") / "planned_action_contract.json").read_text())
    assert value["continuous_pre_rq"] is True and value["rq_used"] is False
    assert value["interval"] == ["a_t", "a_t+15"] and value["a_t_plus_16"] is False
    assert value["embodiment"] == 31 and value["source_default"] is None
    assert value["raw_58_ordering"] == ["left arm 7", "left hand 22", "right arm 7", "right hand 22"]
    assert value["state_relative_features_recomputed_by_frozen_a_r"] is True
    assert value["first_differences_recomputed_by_frozen_a_r"] is True
    assert value["runtime_legal"] == ["POLICY_GENERATED"]
    assert set(value["runtime_rejected"]) == {"DEMONSTRATION_TEACHER", "ORACLE_EVAL"}
    assert value["policy_generated_plans_available"] is False
    assert value["policy_artifact_audit"]["candidates"] == []
    assert value["warning"] == "POLICY_PLAN_DOMAIN_WARNING"


@pytest.mark.parametrize("split,count", [("train", 65536), ("validation", 8192), ("test", 17504)])
def test_causal_cache_exact_current_history_and_no_future(split, count):
    root = ROOT / ".local/cache/tactile_unit/vac_c5" / split
    if not (root / "manifest.json").is_file() or not json.loads((root / "manifest.json").read_text()).get("complete"):
        pytest.skip(f"complete {split} causal cache unavailable")
    c1 = ROOT / ".local/cache/tactile_unit/vac_c1" / split
    episode = np.load(c1 / "episode_id.npy", mmap_mode="r"); anchor = np.load(c1 / "t.npy", mmap_mode="r")
    frame_episode = np.load(root / "frame_episode_id.npy", mmap_mode="r"); frame_index = np.load(root / "frame_index.npy", mmap_mode="r")
    current = np.load(root / "current_feature_index.npy", mmap_mode="r"); history = np.load(root / "history_feature_index.npy", mmap_mode="r")
    assert len(current) == count and history.shape == (count, 8)
    assert np.array_equal(frame_episode[current], episode) and np.array_equal(frame_index[current], anchor)
    assert np.all(frame_episode[history] == episode[:, None])
    assert np.array_equal(frame_index[history], anchor[:, None] + np.arange(-7, 1)[None])
    assert np.all(frame_index[history] <= anchor[:, None])
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["timestamp_alignment"]["enforced_during_every_new_frame_decode"] is True
    assert manifest["timestamp_alignment"]["maximum_allowed_error_seconds"] <= 1 / 30 + 1e-4


def test_visual_normalization_is_fit_on_train_only():
    path = ROOT / ".local/cache/tactile_unit/vac_c5/visual_feature_normalization.json"
    if not path.is_file():
        pytest.skip("C5 visual normalization unavailable")
    value = json.loads(path.read_text())
    assert value["fit_split"].startswith("frozen C1 train")
    assert value["validation_or_test_used_for_fit"] is False
    assert value["test_loaded"] is False
    assert len(value["mean"]) == len(value["std"]) == 32
    assert min(value["std"]) > 0


def test_selection_is_validation_only_bounded_and_obeys_simplicity_rules():
    root = artifacts("causal_visual_selection.json")
    selection = json.loads((root / "causal_visual_selection.json").read_text())
    training = json.loads((root / "training_summary.json").read_text())
    assert selection["selected_via"] == "VALIDATION ONLY" and selection["test_loaded"] is False
    assert training["total_trials"] == 6 <= training["maximum_trials"]
    assert "0.01" in selection["rationale"] or "simplicity" in selection["rationale"]
    assert set(selection["comparisons"]) == {"current_vs_history", "direct_vs_modular"}


def test_completed_trials_and_selected_checkpoint_are_identity_locked():
    root = artifacts("causal_visual_selection.json")
    selection = json.loads((root / "causal_visual_selection.json").read_text())
    training = json.loads((root / "training_summary.json").read_text())
    assert [row["trial"]["id"] for row in training["trials"]] == [f"T{i}" for i in range(6)]
    assert all(row["best"]["validation"]["test_loaded"] is False for row in training["trials"])
    assert sha(ROOT / selection["checkpoint"]) == selection["checkpoint_sha256"]
    metrics = selection["validation_metrics"]
    assert metrics["action_temporal"]["exact_raw_action_transform"] is True
    assert metrics["physics"]["teacher_side_h_only"] is True
    assert metrics["geometry"]["per_dimension_variance"]["near_zero_fraction"] < 1.0


def test_uncertainty_freeze_is_target_free_train_only_and_mode_aware():
    root = artifacts("uncertainty_selection.json")
    selection = json.loads((root / "uncertainty_selection.json").read_text())
    training = json.loads((root / "uncertainty_training.json").read_text())
    assert selection["parameters"] <= 50_000
    assert selection["common_scale_across_modes"] is True
    assert selection["selection_split"] == "validation only" and selection["test_loaded"] is False
    assert training["mean_predictors_frozen"] is True and training["plan_ood_fit_split"] == "train only"
    assert set(selection["validation"]) == {"FULL_AH", "FALLBACK_CAUSAL_VA", "FALLBACK_A"}
    assert sha(ROOT / selection["checkpoint"]) == selection["checkpoint_sha256"]


def test_raw_planned_action_domain_diagnostic_is_pre_ar_and_policy_honest():
    value = json.loads((artifacts("planned_action_domain_diagnostic.json") / "planned_action_domain_diagnostic.json").read_text())
    assert value["perturbation_space"].startswith("raw 58-D Action")
    assert value["actual_policy_available"] is False and value["actual_policy_domain_validated"] is False
    assert set(value["metrics"]) == {
        "oracle_demonstration_surrogate", "mild_raw_noise", "strong_raw_noise",
        "temporal_smoothing", "one_step_lag", "different_episode_plan",
    }
    assert value["accepted_oracle_u_a_reproduction_max_abs"] <= value["accepted_c3msccr_reproduction_tolerance"]
    assert value["warning"] == "POLICY_PLAN_DOMAIN_WARNING"


def test_router_has_no_offline_future_vision_route_and_abstains_without_action():
    value = json.loads((artifacts("runtime_router_contract.json") / "runtime_router_contract.json").read_text())
    assert value["pass"] is True and value["offline_oracle_va_runtime_routable"] is False
    assert value["demo_action_runtime_rejection"] is True
    assert value["planned_action_numeric_equivalence_max_abs"] == {"DEMONSTRATION_TEACHER": 0.0, "ORACLE_EVAL": 0.0, "POLICY_GENERATED": 0.0}
    no_action = [row for row in value["truth_table"] if not row["action_available"]]
    assert no_action and all(row["mode"] == "ABSTAIN_NO_ACTION" for row in no_action)


def test_locked_evaluation_is_deterministic_causal_and_stops_before_c6():
    locked = json.loads((artifacts("locked_test_evaluation.json") / "locked_test_evaluation.json").read_text())
    assert locked["rows"] == 17504 and locked["selection_frozen_before_test"] is True
    assert locked["label"] == "LOCKED POST-HOC C5 ENGINEERING EVALUATION"
    assert locked["first_look_untouched"] is False and locked["repeated_evaluation_exact"] is True
    assert locked["causal_leakage"]["pass"] is True
    assert locked["c6_m3"] == "NOT STARTED" and locked["m3"] == "NOT ESTABLISHED"
    assert locked["full_nonregression"]["pass"] is True
    assert locked["full_nonregression"]["h_context"]["gate"] is True
    assert locked["runtime_router"]["offline_oracle_va_runtime_routable"] is False


def test_final_decision_is_exactly_one_allowed_c5_state():
    value = json.loads((artifacts("final_decision.json") / "final_decision.json").read_text())
    allowed = {
        "C5_CAUSAL_SYSTEM_READY", "C5_CAUSAL_SYSTEM_READY_WITH_POLICY_PLAN_DOMAIN_WARNING",
        "C5_CAUSAL_SYSTEM_READY_A_ONLY_FALLBACK", "C5_CAUSAL_VISUAL_SUBSTITUTION_INSUFFICIENT",
        "C5_PLANNED_ACTION_INTERFACE_FAIL", "C5_ACTION_TEMPORAL_FAIL", "C5_UNCERTAINTY_UNCALIBRATED",
        "C5_CAUSAL_LEAKAGE_FAIL", "C5_FULL_PATH_REGRESSION", "STRUCTURAL_FAIL",
    }
    assert value["decision"] in allowed
    assert value["c6_m3"] == "NOT STARTED" and value["m3"] == "NOT ESTABLISHED"
