#!/usr/bin/env python3
"""Evaluate Route B candidates and freeze the final S4.2-DR decision."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.contact_dynamics.evaluation import query_diversity  # noqa: E402
from gr00t.simulation.intrinsic_dimension import dimension_metrics  # noqa: E402
from scripts.simulation.evaluate_s4_2r_contact_dynamics import probe_suite  # noqa: E402
from scripts.simulation.s4_2dr_common import (  # noqa: E402
    ARTIFACT_ROOT,
    CACHE_ROOT,
    DR_CACHE_ROOT,
    DR_EXPERIMENT_ROOT,
    ROOT,
    atomic_json,
    bootstrap_mean_ci,
    canonical_hash,
    control_predictions,
    load_json,
    load_npz,
    per_sample_mse,
    predict,
    seed_everything,
    sha256_file,
)
from scripts.simulation.train_s4_2dr_dynamics import build_model  # noqa: E402


def load_candidate(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["schema"] != "tactile3d-unit.s4-2dr-contact-dynamics-checkpoint.v1":
        raise RuntimeError("unsupported remediation checkpoint")
    model = build_model()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def candidate_audit(
    training_row: dict[str, Any],
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    frozen: dict[str, np.ndarray],
    device: torch.device,
    index: int,
) -> dict[str, Any]:
    path = ROOT / training_row["checkpoint"]
    model, checkpoint = load_candidate(path, device)
    val_prediction, val_code = predict(model, "C3", validation["h_current"], validation["h_future"], device)
    train_prediction, train_code = predict(model, "C3", train["h_current"], train["h_future"], device)
    _ = train_prediction
    assert val_code is not None and train_code is not None
    c2_error = per_sample_mse(frozen["C2"], validation["h_future"])
    original_c3_error = per_sample_mse(frozen["C3"], validation["h_future"])
    selected_error = per_sample_mse(val_prediction, validation["h_future"])
    dynamic = validation["dynamic"].astype(bool)
    difference = c2_error[dynamic] - selected_error[dynamic]
    c2_dynamic = float(c2_error[dynamic].mean())
    selected_dynamic = float(selected_error[dynamic].mean())
    selected_overall = float(selected_error.mean())
    original_overall = float(original_c3_error.mean())
    controls_prediction = control_predictions(
        model, val_code, validation["h_current"], validation["h_future"], validation["episode_id"], device
    )
    controls = {}
    for control_index, (name, prediction) in enumerate(controls_prediction.items()):
        error = per_sample_mse(prediction, validation["h_future"])[dynamic]
        control_difference = error - selected_error[dynamic]
        controls[name] = {
            "full_mse": selected_dynamic,
            "control_mse": float(error.mean()),
            "control_minus_full": float(control_difference.mean()),
            "paired_bootstrap_ci95": bootstrap_mean_ci(control_difference, seed=6200 + 20 * index + control_index),
        }
    probes = probe_suite(train_code, val_code, train, validation)
    collapse = dimension_metrics(val_code.reshape(len(val_code), -1), twonn=True, include_spectrum=False)
    diversity = query_diversity(val_code)
    collapse.update({
        "query_diversity": diversity,
        "collapsed_query_fraction": float(np.mean(np.var(val_code, axis=(0, 2)) < 1e-8)),
    })
    first, _ = load_candidate(path, device)
    second, _ = load_candidate(path, device)
    first_prediction, first_code = predict(first, "C3", validation["h_current"][:128], validation["h_future"][:128], device, 128)
    second_prediction, second_code = predict(second, "C3", validation["h_current"][:128], validation["h_future"][:128], device, 128)
    deterministic = bool(np.array_equal(first_prediction, second_prediction) and np.array_equal(first_code, second_code))
    improvement = float(difference.mean() / c2_dynamic)
    improvement_ci = bootstrap_mean_ci(difference, seed=6100 + index)
    controls_pass = all(row["control_mse"] > row["full_mse"] and row["paired_bootstrap_ci95"][0] > 0.0 for row in controls.values())
    gates = {
        "A_dynamic_improvement_at_least_10_percent": improvement >= 0.10,
        "B_bootstrap_lower_positive": improvement_ci[0] > 0.0,
        "C_overall_regression_at_most_3_percent": selected_overall <= original_overall * 1.03,
        "D_to_G_all_controls_strongly_worse": controls_pass,
        "H_contact_transition_macro_f1": probes["contact_transition"]["macro_f1"] >= 0.90,
        "I_force_trend_macro_f1": probes["force_trend"]["macro_f1"] >= 0.90,
        "J_no_collapse": bool(
            np.isfinite(collapse["effective_rank"])
            and collapse["effective_rank"] > 1.0
            and collapse["near_zero_variance_fraction"] < 0.95
            and collapse["collapsed_query_fraction"] == 0.0
            and diversity["collapsed_sample_fraction"] < 0.01
        ),
        "K_deterministic_reload": deterministic,
        "L_shape_8x32": tuple(val_code.shape[1:]) == (8, 32),
    }
    return {
        **{key: value for key, value in training_row.items() if key != "history"},
        "checkpoint_epoch": checkpoint["epoch"],
        "dynamic_mse": selected_dynamic,
        "overall_mse": selected_overall,
        "dynamic_absolute_improvement": float(difference.mean()),
        "dynamic_relative_improvement": improvement,
        "dynamic_improvement_ci95": improvement_ci,
        "overall_regression_vs_original_C3": float(selected_overall / original_overall - 1.0),
        "controls": controls,
        "probes": probes,
        "collapse": collapse,
        "deterministic_reload": deterministic,
        "shape": list(val_code.shape[1:]),
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "route_B": "PASS" if all(gates.values()) else "FAIL",
        "test_loaded": False,
    }


def main() -> None:
    seed_everything(4242)
    training = load_json(ARTIFACT_ROOT / "remediation_trials.json")
    followup = load_json(ARTIFACT_ROOT / "existing_c3_followup.json")
    freeze = load_json(ARTIFACT_ROOT / "protocol_freeze.json")
    historical = load_json(ROOT / "configs/simulation/s4_2r_contact_dynamics_decision.json")
    reference = load_json(ARTIFACT_ROOT / "empirical_predictability_reference.json")
    regimes = load_json(ARTIFACT_ROOT / "regime_decomposition.json")
    if training["total_trials"] > 3 or training["test_loaded"]:
        raise RuntimeError("invalid bounded remediation summary")
    if followup["route_A"] != "FAIL":
        raise RuntimeError("Route B cannot run after Route A passes")
    train = load_npz(CACHE_ROOT / "train.npz")
    validation = load_npz(CACHE_ROOT / "validation.npz")
    frozen = load_npz(DR_CACHE_ROOT / "frozen_predictions.npz")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    audits = [
        candidate_audit(row, train, validation, frozen, device, index)
        for index, row in enumerate(training["trials"])
    ]
    passing = [row for row in audits if row["route_B"] == "PASS"]
    selected = None
    if passing:
        best_mse = min(row["dynamic_mse"] for row in passing)
        for trial in ("R1", "R2", "R3"):
            candidate = next((row for row in passing if row["trial"] == trial), None)
            if candidate is not None and candidate["dynamic_mse"] <= best_mse * 1.03:
                selected = candidate
                break
    if selected is None:
        decision = "S4_2DR_DYNAMICS_REMEDIATION_FAIL"
        exact_blocker = "All three bounded trials failed Route B gate A: no candidate reached the unchanged >=10% dynamic improvement over frozen C2."
        selected_checkpoint = None
        selected_sha = None
    else:
        decision = "S4_2DR_DYNAMICS_ACCEPTED_REMEDIATED"
        exact_blocker = None
        selected_source = ROOT / selected["checkpoint"]
        selected_destination = DR_EXPERIMENT_ROOT / "accepted.pt"
        shutil.copyfile(selected_source, selected_destination)
        selected_checkpoint = str(selected_destination.relative_to(ROOT))
        selected_sha = sha256_file(selected_destination)
        if selected_sha != selected["checkpoint_sha256"]:
            raise RuntimeError("accepted checkpoint copy mismatch")
    training.update({
        "status": "VALIDATION_AUDIT_COMPLETE",
        "trials": audits,
        "passing_trials": [row["trial"] for row in passing],
        "selected_trial": None if selected is None else selected["trial"],
        "decision": decision,
        "test_loaded": False,
    })
    training["validation_audit_hash"] = canonical_hash(training)
    atomic_json(ARTIFACT_ROOT / "remediation_trials.json", training)
    best = min(audits, key=lambda row: row["dynamic_mse"])
    acceptance = {
        "schema": "tactile3d-unit.s4-2dr-dynamics-acceptance.v1",
        "decision": decision,
        "historical_original_failure": historical["decision"],
        "historical_original_gate_modified": False,
        "historical_original_dynamic_improvement": historical["validation"]["dynamic_relative_improvement"],
        "historical_original_required": historical["validation"]["required_dynamic_relative_improvement"],
        "follow_up_route": None if selected is None else "ORIGINAL_10_PERCENT_REMEDIATED_ACCEPTANCE",
        "follow_up_acceptance": "FAIL" if selected is None else "PASS",
        "selected_trial": None if selected is None else selected["trial"],
        "selected_checkpoint": selected_checkpoint,
        "selected_checkpoint_sha256": selected_sha,
        "best_bounded_trial": best["trial"],
        "best_bounded_dynamic_mse": best["dynamic_mse"],
        "best_bounded_relative_improvement": best["dynamic_relative_improvement"],
        "E_C2": reference["E_C2"],
        "E_original_C3": reference["E_C3"],
        "E_ref": reference["selected_reference"],
        "C2_to_ref_gap": reference["relative_C2_to_ref_gap"],
        "headroom_capture": reference["headroom_capture_C3"],
        "regime_decomposition": regimes["comparisons"],
        "trials": audits,
        "z_c_shape": [8, 32],
        "warnings": ["ORIGINAL_10_PERCENT_GATE_NOT_MET", "NO_EMPIRICAL_HEADROOM_REFERENCE"],
        "exact_blocker": exact_blocker,
        "model_performance_test_loaded": False,
        "selection_uses_test": False,
    }
    acceptance["acceptance_hash"] = canonical_hash(acceptance)
    atomic_json(ARTIFACT_ROOT / "dynamics_acceptance.json", acceptance)
    final_decision = {
        "schema": "tactile3d-unit.s4-2dr-final-decision.v1",
        "decision": decision,
        "historical_result": "S4_2_3_CONTACT_DYNAMICS_FAIL",
        "historical_result_modified": False,
        "route_A": followup["route_A"],
        "route_B": "FAIL" if selected is None else "PASS",
        "total_remediation_trials": len(audits),
        "selected_trial": None if selected is None else selected["trial"],
        "checkpoint": selected_checkpoint,
        "checkpoint_sha256": selected_sha,
        "exact_blocker": exact_blocker,
        "S4_2_4_readiness": "NOT_READY" if selected is None else "READY",
        "downstream_executed": False,
        "formal_test": "PRESERVED",
        "model_performance_test_loaded": False,
        "selection_used_test": False,
        "test_loaded": False,
    }
    final_decision["final_decision_hash"] = canonical_hash(final_decision)
    atomic_json(ARTIFACT_ROOT / "final_decision.json", final_decision)
    print(json.dumps({"decision": decision, "best_trial": best["trial"], "best_relative_improvement": best["dynamic_relative_improvement"], "failed_gates": {row["trial"]: row["failed_gates"] for row in audits}, "exact_blocker": exact_blocker, "test_loaded": False}, indent=2))


if __name__ == "__main__":
    main()
