from __future__ import annotations

import json
from pathlib import Path

from gr00t.simulation.s4_2r_acceptance import evaluate_no_collapse_contract


ROOT = Path(__file__).resolve().parents[2]


def load(name: str) -> dict:
    return json.loads((ROOT / "configs/simulation" / name).read_text())


def test_original_s4_2_failure_is_unchanged_and_follow_up_is_separate() -> None:
    original = load("s4_2_final_decision.json")
    remediation = load("s4_2r_contact_state_rank_remediation.json")
    assert original["decision"] == "S4_2_CONTACT_STATE_FAIL"
    assert original["contact_state"]["effective_rank"] == 11.255626031431431
    assert original["contact_state"]["effective_rank_min"] == 16.0
    integrity = remediation["scientific_integrity"]
    assert integrity["original_s4_2_result"] == "FAIL"
    assert integrity["original_result_must_not_be_modified"] is True
    assert integrity["follow_up_is_separately_preregistered"] is True


def test_new_structural_contract_has_all_preregistered_gates() -> None:
    remediation = load("s4_2r_contact_state_rank_remediation.json")
    gates = remediation["new_structural_collapse_contract"]
    assert set(gates) == {
        "gate_a_variance",
        "gate_b_dominant_component",
        "gate_c_effective_rank_floor",
        "gate_d_source_relative_diversity",
        "gate_e_pairwise_diversity",
        "gate_f_semantic_functionality",
        "gate_g_perturbation_sensitivity",
    }
    assert gates["gate_a_variance"]["near_zero_variance_fraction_max"] == 0.01
    assert gates["gate_b_dominant_component"][
        "top_1_pca_explained_variance_max_exclusive"
    ] == 0.5
    assert gates["gate_c_effective_rank_floor"]["effective_rank_min"] == 8.0
    assert gates["gate_d_source_relative_diversity"]["ratio_min"] == 0.75


def test_source_reference_and_input_identities_are_frozen() -> None:
    remediation = load("s4_2r_contact_state_rank_remediation.json")
    source = remediation["r1_source_dimension_reference"]
    assert source["d_src"] == 3.121517446651421
    assert source["terms_canonical_sha256"] == (
        "62a33f61000606431bd3b34e6cb7dd3b6a903d3851ad56c0e152f6b6a47ddb7b"
    )
    frozen = remediation["frozen_inputs"]
    assert frozen["dataset_manifest_canonical_sha256"] == (
        "652181d45bc317a1b17debe5e9c58529ae7d096bb75441cd17d0165bf2d701d4"
    )
    assert frozen["teacher_checkpoint_sha256"] == (
        "3d4519195e7a0d9f5af63399840a48121d5f614ea587c80b9461fc696a372a19"
    )


def test_remediation_is_validation_only_and_bounded_to_three_trials() -> None:
    remediation = load("s4_2r_contact_state_rank_remediation.json")
    bounded = remediation["r5_bounded_remediation"]
    assert bounded["enabled_only_if_r4_fails"] is True
    assert bounded["maximum_new_trials_total"] == 3
    assert [trial["id"] for trial in bounded["trials_in_priority_order"]] == [
        "R0",
        "R1",
        "R2",
    ]
    assert bounded["selection_split"] == "validation"
    assert remediation["test_policy"] == {
        "test_loaded": False,
        "test_metrics_forbidden": True,
        "training_uses_test": False,
        "selection_uses_test": False,
        "locked_test_deferred_until_after_s4_2_6_and_s4_2_7": True,
    }
    assert remediation["training_started"] is False


def test_new_no_collapse_gate_evaluator_applies_all_frozen_thresholds() -> None:
    remediation = load("s4_2r_contact_state_rank_remediation.json")
    metrics = {
        "near_zero_variance_fraction": 0.0,
        "top_pc_explained_variance": 0.36,
        "effective_rank": 11.25,
    }
    pairwise = {
        "same_sample_duplicate_distance_mean": 0.0,
        "different_sample_distance_mean": 18.9,
        "different_sample_distance_p05": 1e-5,
    }
    required = remediation["new_structural_collapse_contract"][
        "gate_f_semantic_functionality"
    ]["require_all_original_validation_gates"]
    original = {name: True for name in required}
    controls = {
        name: {"correct_mse": 0.3, "control_mse": 0.8, "improvement_ci95": [0.1, 0.2]}
        for name in remediation["new_structural_collapse_contract"][
            "gate_g_perturbation_sensitivity"
        ]["correct_history_must_beat"]
    }
    result = evaluate_no_collapse_contract(
        remediation, metrics, pairwise, original, controls
    )
    assert result["overall_pass"] is True
    assert all(row["pass"] for row in result["gates"].values())
    metrics["top_pc_explained_variance"] = 0.5
    failed = evaluate_no_collapse_contract(
        remediation, metrics, pairwise, original, controls
    )
    assert failed["overall_pass"] is False
    assert failed["gates"]["B_dominant_component"]["pass"] is False


def test_contact_state_acceptance_promotes_existing_without_rewriting_original() -> None:
    original = load("s4_2_final_decision.json")
    acceptance = load("s4_2r_contact_state_acceptance.json")
    assert original["decision"] == "S4_2_CONTACT_STATE_FAIL"
    assert acceptance["final_state"] == "S4_2R_CONTACT_STATE_ACCEPTED_EXISTING"
    assert acceptance["classification"] == "LOW_INTRINSIC_DIMENSION_NOT_COLLAPSE"
    assert acceptance["original_s4_2"]["result_modified"] is False
    assert acceptance["remediation"] == {
        "required": False,
        "new_trials": 0,
        "remediation_trials_artifact": "NOT_APPLICABLE",
    }
    assert all(value == "PASS" for value in acceptance["validation"]["gates"].values())
    assert acceptance["test_loaded"] is False
    assert acceptance["s4_2_3_allowed"] is True
