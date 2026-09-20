from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/simulation/s4_3_pi2n_optimization_probe.json"
DECISION_PROTOCOL = ROOT / "configs/simulation/s4_3_pi2n_refinement_decision_protocol.json"
SCRIPT = ROOT / "scripts/simulation/diagnose_s4_3_pi2n_optimization.py"


def load_module():
    spec = importlib.util.spec_from_file_location("diagnose_s4_3_pi2n_optimization", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_common_reference_and_one_shot_route_are_frozen() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    decision = json.loads(DECISION_PROTOCOL.read_text())
    assert protocol["status"] == "FROZEN_BEFORE_NUMERICAL_DIAGNOSTIC"
    assert protocol["common_reference"]["optimizer_updates"] == 0
    assert protocol["common_reference"]["shared_parameter_scope"] == "all and only parameter paths matching .*lora.*"
    assert protocol["gradient_batches"]["count"] == 4
    assert protocol["gradient_batches"]["batch_size"] == 32
    assert protocol["gradient_batches"]["same_random_stream_for_g_pi_g_V_g_C"] is True
    assert len(protocol["gradient_batches"]["flow_time_bins"]) == 4
    assert protocol["route"]["priority"] == decision["one_shot_priority"]
    assert decision["candidate_checkpoint_or_rollout_results_may_influence_route"] is False
    assert decision["maximum_X_policy_runs"] == 1


def test_p4_probes_have_equal_frozen_budget() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    p4 = protocol["p4"]
    assert p4["rows"] == 500
    assert "modulo 5" in p4["fit_episodes"]
    assert p4["mean_features"].endswith("1024-to-648 projection")
    assert "1024-to-24" in p4["order_features"]
    assert p4["trainable_parameters_each"] == 166_144
    assert p4["optimizer_steps"] == 0


def test_standardized_ridge_counts_intercept_and_recovers_signal() -> None:
    module = load_module()
    rng = np.random.default_rng(7)
    x = rng.normal(size=(120, 12)).astype(np.float32)
    weights = rng.normal(size=(12, 5)).astype(np.float32)
    y = x @ weights + 0.25
    prediction, detail = module.standardized_ridge(x[:100], y[:100], x[100:], 0.001)
    assert prediction.shape == (20, 5)
    assert detail["trainable_parameters"] == 12 * 5 + 5
    assert np.mean(np.square(prediction - y[100:])) < 1e-3


def test_diagnostic_refuses_overwrite_and_does_not_read_candidate_results() -> None:
    source = SCRIPT.read_text()
    assert "refusing to overwrite N2-O output" in source
    assert "candidate_policy_checkpoints_read" in source
    assert "rollout_outcomes_read" in source
    assert "candidate_development_results" not in source
    assert "paired_statistics" not in source
    assert "B_VA27/pinch_tongs" not in source
    assert "B_VAC_V/pinch_tongs" not in source
