from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/simulation/s4_2_tf_locked_test_v2.json"


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_v1_is_preserved_as_exposed_and_scientifically_ineligible() -> None:
    config = load_config()
    assert config["test_v1"]["classification"] == "TEST_V1_EXPOSED"
    assert config["test_v1"]["scientific_decision_eligible"] is False
    assert config["test_v1"]["structural_failure"] == "PRESERVED"
    assert config["test_v1"]["pairs"] == 4860
    assert config["test_v1"]["pair_identity_sha256"]


def test_v2_generation_identity_is_exact_and_fresh() -> None:
    protocol = load_config()["formal_test_v2"]
    assert protocol["tasks"] == ["pinch_tongs", "hammer_nail", "click_mouse"]
    assert protocol["source_group_indices"] == [20, 21, 22]
    assert protocol["episode_count"] == 45
    assert protocol["pair_count"] == 4860
    assert protocol["source_group_count"] == 9
    assert protocol["required_disjoint_from"] == [
        "TRAIN",
        "DS_DEV",
        "FORMAL_VALIDATION",
        "TEST_V1_EXPOSED",
    ]
    seeds = {
        protocol["seed_bases"][task] + 10 * group + perturbation
        for task in protocol["tasks"]
        for group in protocol["source_group_indices"]
        for perturbation in range(protocol["perturbations_per_group"])
    }
    assert len(seeds) == 45
    assert min(seeds) > 440194


def test_v2_runner_is_guarded_and_equality_only() -> None:
    source = (ROOT / "scripts/simulation/run_s4_2_tf_locked_test_v2.py").read_text()
    assert source.index("pretest = verify_pretest(repeat=repeat)") < source.index(
        "build_locked_cache("
    )
    assert 'test_name="FORMAL_TEST_V2"' in source
    assert 'result["metric_digest"] == first["metric_digest"]' in source
    assert "S4_2_TF_TEST_V2_DETERMINISTIC_REPEAT_FAIL" in source


def test_v2_integrity_phase_cannot_read_model_performance_cache() -> None:
    source = (ROOT / "scripts/simulation/manage_s4_2_tf.py").read_text()
    audit_body = source.split("def audit_dataset()", 1)[1].split("def finalize()", 1)[0]
    assert "evaluate_locked(" not in audit_body
    assert "TEST_V2 model-performance cache exists before pretest_v2_freeze" in audit_body
    assert '"model_performance_metrics_loaded": False' in audit_body


def test_tf_forbids_scientific_mutation() -> None:
    config = load_config()
    remediation = config["remediation"]
    assert all(
        remediation[name] is False
        for name in (
            "training_permitted",
            "checkpoint_selection_permitted",
            "threshold_change_permitted",
            "normalization_change_permitted",
            "split_change_permitted",
            "bridge_retraining_permitted",
            "predictor_retraining_permitted",
            "uncertainty_retraining_permitted",
        )
    )


def test_corrected_dataset_audit_uses_contractual_step_contiguity() -> None:
    source = (ROOT / "scripts/simulation/audit_s4_2_tf_test_v2_dataset.py").read_text()
    assert 'np.diff(values["control_step"])' in source
    assert 'np.ones(length - 1)' in source
    assert 'dataset_bytes_changed_after_generation": False' in source
    assert "model_performance_metrics_loaded" in source
