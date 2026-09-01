#!/usr/bin/env python3
"""Freeze TRAIN thresholds and decompose frozen C2/C3 validation errors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.simulation.s4_2dr_common import (  # noqa: E402
    ARTIFACT_ROOT,
    CACHE_ROOT,
    DR_CACHE_ROOT,
    ROOT,
    S42_PAIR_ROOT,
    atomic_json,
    bootstrap_mean_ci,
    canonical_hash,
    control_predictions,
    ensure_local_roots,
    load_json,
    load_model,
    load_npz,
    model_paths,
    per_sample_mse,
    predict,
    sha256_file,
)

THRESHOLDS_PATH = ARTIFACT_ROOT / "regime_thresholds.json"
TRANSITION_NAMES = {
    0: "free_to_free",
    1: "free_to_contact",
    2: "contact_to_free",
    3: "contact_to_contact",
}


def endpoint_tangential(raw: dict[str, np.ndarray]) -> np.ndarray:
    current = raw["current_history"][:, -1].reshape(-1, 5, 6)[:, :, 2].sum(axis=1)
    future = raw["future_history"][:, -1].reshape(-1, 5, 6)[:, :, 2].sum(axis=1)
    return np.maximum(current, future)


def endpoint_normal(raw: dict[str, np.ndarray]) -> np.ndarray:
    return np.maximum(raw["current_total_force"], raw["future_total_force"])


def freeze_thresholds() -> dict[str, Any]:
    ensure_local_roots()
    train = load_npz(S42_PAIR_ROOT / "train.npz")
    value = {
        "schema": "tactile3d-unit.s4-2dr-regime-thresholds.v1",
        "status": "FROZEN_BEFORE_VALIDATION_REGIME_EVALUATION",
        "source_split": "train",
        "train_pair_cache_sha256": sha256_file(S42_PAIR_ROOT / "train.npz"),
        "high_force": {
            "definition": "max(current endpoint total normal force, future endpoint total normal force)",
            "quantile": 0.90,
            "threshold_newton": float(np.quantile(endpoint_normal(train), 0.90)),
        },
        "high_tangential": {
            "definition": "max(current endpoint sum of canonical regional tangential magnitudes, future endpoint sum)",
            "quantile": 0.90,
            "threshold_newton": float(np.quantile(endpoint_tangential(train), 0.90)),
        },
        "contact_transition_labels": TRANSITION_NAMES,
        "test_loaded": False,
        "validation_metrics_computed_before_freeze": False,
    }
    value["threshold_hash"] = canonical_hash(value)
    atomic_json(THRESHOLDS_PATH, value)
    return value


def masks(raw: dict[str, np.ndarray], dynamics: dict[str, np.ndarray], thresholds: dict[str, Any]) -> dict[str, np.ndarray]:
    result = {
        "overall": np.ones(len(dynamics["pair_id"]), dtype=bool),
        "static": ~dynamics["dynamic"].astype(bool),
        "dynamic": dynamics["dynamic"].astype(bool),
        "boundary": np.isin(dynamics["contact_transition"], [1, 2]),
        "high_force": endpoint_normal(raw) > thresholds["high_force"]["threshold_newton"],
        "high_tangential": endpoint_tangential(raw) > thresholds["high_tangential"]["threshold_newton"],
    }
    for code, name in TRANSITION_NAMES.items():
        result[name] = dynamics["contact_transition"] == code
    for task in np.unique(dynamics["task"]):
        result[str(task)] = dynamics["task"] == task
    return result


def comparison_row(c2_error: np.ndarray, c3_error: np.ndarray, mask: np.ndarray, seed: int) -> dict[str, Any]:
    absolute = c2_error[mask] - c3_error[mask]
    c2 = float(c2_error[mask].mean())
    c3 = float(c3_error[mask].mean())
    return {
        "samples": int(mask.sum()),
        "C2_mse": c2,
        "C3_mse": c3,
        "absolute_improvement": float(absolute.mean()),
        "relative_improvement": float(absolute.mean() / c2),
        "paired_bootstrap_ci95": bootstrap_mean_ci(absolute, seed=seed),
    }


def evaluate(device: torch.device) -> dict[str, Any]:
    if not THRESHOLDS_PATH.exists():
        raise RuntimeError("TRAIN-derived regime thresholds must be frozen first")
    thresholds = load_json(THRESHOLDS_PATH)
    if thresholds["status"] != "FROZEN_BEFORE_VALIDATION_REGIME_EVALUATION":
        raise RuntimeError("invalid regime threshold freeze")
    raw = load_npz(S42_PAIR_ROOT / "validation.npz")
    validation = load_npz(CACHE_ROOT / "validation.npz")
    if not np.array_equal(raw["pair_id"], validation["pair_id"]):
        raise RuntimeError("validation pair identity mismatch")
    frozen = load_npz(DR_CACHE_ROOT / "frozen_predictions.npz")
    if not np.array_equal(frozen["pair_id"], validation["pair_id"]):
        raise RuntimeError("frozen prediction pair identity mismatch")
    target = validation["h_future"]
    c2_error = per_sample_mse(frozen["C2"], target)
    c3_error = per_sample_mse(frozen["C3"], target)
    regimes = masks(raw, validation, thresholds)
    comparisons = {
        name: comparison_row(c2_error, c3_error, mask, 4600 + index)
        for index, (name, mask) in enumerate(regimes.items())
    }

    c3_model, _ = load_model(model_paths()["C3"], device)
    if not hasattr(c3_model, "decoder"):
        raise RuntimeError("selected C3 lacks transition decoder")
    controls_prediction = control_predictions(
        c3_model, frozen["C3_code"], validation["h_current"], target,
        validation["episode_id"], device,
    )
    full_error = c3_error
    control_regimes = {}
    control_masks = {
        name: regimes[name]
        for name in ("dynamic", "boundary", "high_force", "high_tangential", "pinch_tongs", "hammer_nail", "click_mouse")
    }
    for regime_index, (regime, mask) in enumerate(control_masks.items()):
        control_regimes[regime] = {}
        for control_index, (name, prediction) in enumerate(controls_prediction.items()):
            error = per_sample_mse(prediction, target)
            difference = error[mask] - full_error[mask]
            control_regimes[regime][name] = {
                "samples": int(mask.sum()),
                "full_C3_mse": float(full_error[mask].mean()),
                "control_mse": float(error[mask].mean()),
                "control_minus_full": float(difference.mean()),
                "paired_bootstrap_ci95": bootstrap_mean_ci(difference, seed=4800 + 20 * regime_index + control_index),
            }
    task_necessity = {
        task: all(row["control_mse"] > row["full_C3_mse"] and row["paired_bootstrap_ci95"][0] > 0.0 for row in control_regimes[task].values())
        for task in ("pinch_tongs", "hammer_nail", "click_mouse")
    }
    result = {
        "schema": "tactile3d-unit.s4-2dr-regime-decomposition.v1",
        "evaluation_split": "validation",
        "thresholds_artifact": ".local/artifacts/simulation/s4_2dr/regime_thresholds.json",
        "threshold_hash": thresholds["threshold_hash"],
        "comparisons": comparisons,
        "controls": control_regimes,
        "transition_code_necessity_by_task": task_necessity,
        "necessity_not_single_task_driven": all(task_necessity.values()),
        "test_loaded": False,
    }
    result["result_hash"] = canonical_hash(result)
    atomic_json(ARTIFACT_ROOT / "regime_decomposition.json", result)
    np.savez_compressed(
        DR_CACHE_ROOT / "control_predictions.npz", pair_id=validation["pair_id"],
        **controls_prediction,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-thresholds", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output: dict[str, Any] = {"test_loaded": False}
    if args.freeze_thresholds:
        threshold = freeze_thresholds()
        output["threshold_hash"] = threshold["threshold_hash"]
    if args.evaluate:
        result = evaluate(torch.device(args.device))
        output["DR3"] = "PASS"
        output["necessity_not_single_task_driven"] = result["necessity_not_single_task_driven"]
    if not args.freeze_thresholds and not args.evaluate:
        raise SystemExit("choose --freeze-thresholds and/or --evaluate")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

