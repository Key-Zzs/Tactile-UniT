from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.simulation import audit_s4_2dr_baseline_ceiling as baseline
from scripts.simulation import evaluate_s4_2dr_regimes as regimes
from scripts.simulation import train_s4_2dr_dynamics as remediation
from scripts.simulation.s4_2dr_common import bootstrap_mean_ci, per_sample_mse


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/simulation/s4_2dr_contact_dynamics_remediation.json"
DECISION = ROOT / "configs/simulation/s4_2dr_final_decision.json"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_2dr"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_historical_failure_and_original_10_percent_gate_are_preserved() -> None:
    config = load(CONFIG)
    tracked = load(DECISION)
    assert config["historical_result"]["original_s4_2_3_result"] == "S4_2_3_CONTACT_DYNAMICS_FAIL"
    assert config["historical_result"]["original_dynamic_improvement"] == pytest.approx(0.05509144067764282)
    assert config["historical_result"]["original_required"] == 0.10
    assert config["historical_result"]["result_remains_unchanged"] is True
    assert tracked["historical_failure"]["modified"] is False


def test_routes_are_mutually_exclusive_and_frozen_before_training() -> None:
    config = load(CONFIG)
    assert config["follow_up"]["routes_mutually_exclusive"] is True
    assert config["route_A"]["existing_frozen_C3_only"] is True
    assert config["route_B"]["only_if_route_A_fails"] is True
    assert config["status"] == "FROZEN_AFTER_DR1_DR3_BEFORE_DR5_AND_BEFORE_NEW_DYNAMICS_TRAINING"
    assert config["locked_test"]["model_performance_test_loaded"] is False


def test_train_inner_split_is_deterministic_episode_disjoint_80_20() -> None:
    episode = np.repeat(np.asarray([f"e{i}" for i in range(20)]), 3)
    fit_a, inner_a = baseline.train_inner_split(episode)
    fit_b, inner_b = baseline.train_inner_split(episode)
    assert np.array_equal(fit_a, fit_b)
    assert np.array_equal(inner_a, inner_b)
    assert not set(episode[fit_a]).intersection(episode[inner_a])
    assert len(np.unique(episode[fit_a])) == 16
    assert len(np.unique(episode[inner_a])) == 4


def test_ridge_and_knn_candidates_are_preregistered_and_deterministic() -> None:
    assert baseline.RIDGE_ALPHAS == (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    assert baseline.KNN_K == (5, 10, 25, 50, 100)
    rng = np.random.default_rng(5)
    current = rng.normal(size=(30, 4)).astype(np.float32)
    future = rng.normal(size=(30, 4)).astype(np.float32)
    first = baseline.fit_residual_linear(current, future, 0.1)
    second = baseline.fit_residual_linear(current, future, 0.1)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_knn_queries_use_a_disjoint_fit_set_and_are_deterministic() -> None:
    fit_x = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    fit_y = fit_x.copy()
    query = np.asarray([[0.1, 0.1], [1.9, 1.9]], dtype=np.float32)
    first, indices_a, _ = baseline.knn_predict(fit_x, fit_y, query, (1, 2), torch.device("cpu"))
    second, indices_b, _ = baseline.knn_predict(fit_x, fit_y, query, (1, 2), torch.device("cpu"))
    assert np.array_equal(indices_a, indices_b)
    assert np.array_equal(first[1], second[1])
    assert query.ctypes.data != fit_x.ctypes.data


def test_regime_thresholds_are_train_only_and_bootstrap_is_deterministic() -> None:
    source = (ROOT / "scripts/simulation/evaluate_s4_2dr_regimes.py").read_text(encoding="utf-8")
    assert 'S42_PAIR_ROOT / "train.npz"' in source
    assert "FROZEN_BEFORE_VALIDATION_REGIME_EVALUATION" in source
    values = np.linspace(-1, 1, 40)
    assert bootstrap_mean_ci(values, samples=200, seed=7) == bootstrap_mean_ci(values, samples=200, seed=7)


def test_metric_parity_uses_per_sample_feature_mean() -> None:
    prediction = np.asarray([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    target = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    assert np.array_equal(per_sample_mse(prediction, target), np.asarray([2.5, 10.0]))


def test_remediation_budget_modalities_and_trials_are_bounded() -> None:
    config = load(CONFIG)["remediation"]
    assert config["maximum_trials"] == 3
    assert [row["id"] for row in config["trials"]] == ["R1", "R2", "R3"]
    assert config["input"] == ["h_current", "h_future"]
    assert config["decoder_may_use"] == ["h_current"]
    assert {"Action", "Vision", "raw_future_tactile"}.issubset(config["forbidden_modalities"])
    assert config["parameter_max"] == 2_000_000
    assert config["training"]["optimizer_steps_max_relative_to_original"] == 1.25


def test_remediation_model_shape_parameter_budget_and_frozen_contract() -> None:
    model = remediation.build_model().eval()
    current = torch.randn(3, 256)
    future = torch.randn(3, 256)
    with torch.inference_mode():
        output = model(current, future)
    assert output["code"].shape == (3, 8, 32)
    assert output["future"].shape == (3, 256)
    assert sum(parameter.numel() for parameter in model.parameters()) <= 2_000_000
    config = load(CONFIG)["remediation"]
    assert "Contact-State E_T^sim" in config["frozen"]
    assert "C2 baseline" in config["frozen"]


def test_no_model_performance_test_access_is_encoded_in_scripts() -> None:
    scripts = [
        "audit_s4_2dr_baseline_ceiling.py", "evaluate_s4_2dr_regimes.py",
        "freeze_s4_2dr_protocol.py", "audit_s4_2dr_existing.py",
        "train_s4_2dr_dynamics.py", "audit_s4_2dr_final.py",
    ]
    for name in scripts:
        source = (ROOT / "scripts/simulation" / name).read_text(encoding="utf-8")
        assert "test.npz" not in source
    assert load(DECISION)["locked_test"]["model_performance_test_loaded"] is False
    assert load(DECISION)["locked_test"]["selection_used_test"] is False


def test_local_artifacts_prove_reproduction_and_bounded_failure_when_available() -> None:
    if not (ARTIFACTS / "final_decision.json").exists():
        pytest.skip("local S4.2-DR artifact packet is unavailable")
    fairness = load(ARTIFACTS / "baseline_fairness.json")
    reproduction = load(ARTIFACTS / "baseline_reproduction.json")
    trials = load(ARTIFACTS / "remediation_trials.json")
    final = load(ARTIFACTS / "final_decision.json")
    assert fairness["gate"] == "PASS"
    assert reproduction["gate"] == "PASS"
    assert trials["total_trials"] <= 3
    assert all(row["failed_gates"] == ["A_dynamic_improvement_at_least_10_percent"] for row in trials["trials"])
    assert final["decision"] == "S4_2DR_DYNAMICS_REMEDIATION_FAIL"
    assert final["model_performance_test_loaded"] is False


def test_failure_blocks_s4_2_4_and_all_downstream_execution() -> None:
    decision = load(DECISION)
    assert decision["decision"] == "S4_2DR_DYNAMICS_REMEDIATION_FAIL"
    assert decision["selected_checkpoint"] is None
    assert decision["downstream"]["S4.2-4"] == "NOT_READY"
    assert decision["downstream"]["S4.2-5"] == "BLOCKED"
    assert decision["downstream"]["executed"] is False
    assert decision["stop_after"] == "S4.2-DR"
