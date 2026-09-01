#!/usr/bin/env python3
"""Evaluate the existing frozen C3 against the independent Route A gates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.contact_dynamics.evaluation import query_diversity  # noqa: E402
from gr00t.simulation.intrinsic_dimension import dimension_metrics  # noqa: E402
from scripts.simulation.evaluate_s4_2r_contact_dynamics import probe_suite  # noqa: E402
from scripts.simulation.s4_2dr_common import (  # noqa: E402
    ARTIFACT_ROOT,
    CACHE_ROOT,
    DR_CACHE_ROOT,
    ROOT,
    atomic_json,
    canonical_hash,
    load_json,
    load_npz,
    model_paths,
    sha256_file,
)

CODE_ROOT = ROOT / ".local/cache/simulation/s4_2r/contact_dynamics_codes"


def main() -> None:
    freeze = load_json(ARTIFACT_ROOT / "protocol_freeze.json")
    if freeze["status"] != "FROZEN_BEFORE_DR5_AND_BEFORE_NEW_DYNAMICS_TRAINING":
        raise RuntimeError("DR4 protocol is not frozen")
    reference = load_json(ARTIFACT_ROOT / "empirical_predictability_reference.json")
    regimes = load_json(ARTIFACT_ROOT / "regime_decomposition.json")
    reproduction = load_json(ARTIFACT_ROOT / "baseline_reproduction.json")
    train = load_npz(CACHE_ROOT / "train.npz")
    validation = load_npz(CACHE_ROOT / "validation.npz")
    train_code_cache = load_npz(CODE_ROOT / "train.npz")
    validation_code_cache = load_npz(CODE_ROOT / "validation.npz")
    frozen = load_npz(DR_CACHE_ROOT / "frozen_predictions.npz")
    if not np.array_equal(train_code_cache["pair_id"], train["pair_id"]):
        raise RuntimeError("train code pair identity mismatch")
    if not np.array_equal(validation_code_cache["pair_id"], validation["pair_id"]):
        raise RuntimeError("validation code pair identity mismatch")
    code_max_abs_difference = float(
        np.max(np.abs(validation_code_cache["z_c"] - frozen["C3_code"]))
    )
    if code_max_abs_difference > 4e-6:
        raise RuntimeError("reloaded C3 code differs from frozen code cache beyond tolerance")
    train_code = train_code_cache["z_c"]
    code = frozen["C3_code"]
    probes = probe_suite(train_code, code, train, validation)
    flat = code.reshape(len(code), -1)
    collapse = dimension_metrics(flat, twonn=True, include_spectrum=False)
    diversity = query_diversity(code)
    collapse.update({
        "query_diversity": diversity,
        "collapsed_query_fraction": float(np.mean(np.var(code, axis=(0, 2)) < 1e-8)),
    })
    dynamic = regimes["comparisons"]["dynamic"]
    boundary = regimes["comparisons"]["boundary"]
    high_force = regimes["comparisons"]["high_force"]
    high_tangential = regimes["comparisons"]["high_tangential"]
    controls = regimes["controls"]["dynamic"]
    gap = reference["relative_C2_to_ref_gap"]
    capture = reference["headroom_capture_C3"]
    gates = {
        "A_absolute_improvement_positive": dynamic["absolute_improvement"] > 0.0,
        "A_bootstrap_lower_positive": dynamic["paired_bootstrap_ci95"][0] > 0.0,
        "B_empirical_reference_exists": reference["empirical_headroom_reference_exists"],
        "C_C2_near_empirical_reference": gap is not None and gap <= 0.10,
        "D_C3_captures_half_remaining_headroom": capture is not None and capture >= 0.50,
        "E_boundary_gain": boundary["absolute_improvement"] > 0.0 and boundary["paired_bootstrap_ci95"][0] > 0.0,
        "E_high_force_or_tangential_gain": any(row["absolute_improvement"] > 0.0 and row["paired_bootstrap_ci95"][0] > 0.0 for row in (high_force, high_tangential)),
        "F_all_code_controls_worse": all(row["control_mse"] > row["full_C3_mse"] and row["paired_bootstrap_ci95"][0] > 0.0 for row in controls.values()),
        "G_contact_transition_probe": probes["contact_transition"]["macro_f1"] >= 0.90,
        "H_force_trend_probe": probes["force_trend"]["macro_f1"] >= 0.90,
        "I_no_collapse": bool(
            np.isfinite(collapse["effective_rank"])
            and collapse["effective_rank"] > 1.0
            and collapse["near_zero_variance_fraction"] < 0.95
            and collapse["collapsed_query_fraction"] == 0.0
            and diversity["collapsed_sample_fraction"] < 0.01
        ),
        "J_deterministic_reload": reproduction["deterministic_reload"]["C3"],
    }
    route_a = all(gates.values())
    result = {
        "schema": "tactile3d-unit.s4-2dr-existing-c3-followup.v1",
        "checkpoint": str(model_paths()["C3"].relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(model_paths()["C3"]),
        "E_C2": reference["E_C2"],
        "E_C3": reference["E_C3"],
        "absolute_improvement": dynamic["absolute_improvement"],
        "relative_improvement": dynamic["relative_improvement"],
        "bootstrap_ci95": dynamic["paired_bootstrap_ci95"],
        "E_ref": reference["selected_reference"],
        "relative_C2_to_ref_gap": gap,
        "headroom_capture_C3": capture,
        "critical_regimes": {"boundary": boundary, "high_force": high_force, "high_tangential": high_tangential},
        "controls": controls,
        "probes": probes,
        "collapse": collapse,
        "deterministic_reload": reproduction["deterministic_reload"]["C3"],
        "historical_code_cache_max_abs_difference": code_max_abs_difference,
        "historical_code_cache_tolerance": 4e-6,
        "gates": gates,
        "route_A": "PASS" if route_a else "FAIL",
        "classification": "S4_2DR_DYNAMICS_ACCEPTED_EXISTING_CEILING_LIMITED" if route_a else "DYNAMICS_REMEDIATION_REQUIRED",
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "historical_result": "S4_2_3_CONTACT_DYNAMICS_FAIL",
        "original_10_percent_gate": "FAIL",
        "test_loaded": False,
        "new_training_started": False,
    }
    result["result_hash"] = canonical_hash(result)
    atomic_json(ARTIFACT_ROOT / "existing_c3_followup.json", result)
    print(json.dumps({"route_A": result["route_A"], "classification": result["classification"], "failed_gates": result["failed_gates"], "test_loaded": False}, indent=2))


if __name__ == "__main__":
    main()
