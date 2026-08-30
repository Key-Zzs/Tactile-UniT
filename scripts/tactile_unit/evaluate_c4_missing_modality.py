#!/usr/bin/env python3
"""Run the single locked C4 benchmark after fallback and uncertainty freezes."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.c3r0_conditional_sufficiency import (  # noqa: E402
    bootstrap_f1_difference, evaluate_prediction,
)
from gr00t.tactile_unit.c4_availability_conditioning import (  # noqa: E402
    AvailabilityMode, load_fallback_checkpoint,
)
from gr00t.tactile_unit.continuous_vac_shared_space import (  # noqa: E402
    bootstrap_mean_ci, different_episode_permutation,
)
from scripts.tactile_unit.c4_runtime import (  # noqa: E402
    DEFAULT_CONFIG, atomic_json, identity_snapshot, load_config, load_full,
    load_parent_config, load_selected_fallback, load_selected_uncertainty,
    load_split, predict_fallback, validate_fallback_selection,
    validate_uncertainty_selection,
)
from scripts.tactile_unit.c3mscc_runtime import load_frozen_shared_space  # noqa: E402
from scripts.tactile_unit.continuous_contact_bridge_common import load_s2_model  # noqa: E402
from scripts.tactile_unit.evaluate_c3mscc_contact_prediction import (  # noqa: E402
    model_evaluation, oracle_probe, strip_arrays, wrong_time_indices,
)
from scripts.tactile_unit.evaluate_c3msccr_closure import exact_action_metrics, exact_split  # noqa: E402
from scripts.tactile_unit.train_c3mscc_contact_prediction import (  # noqa: E402
    fit_probe, majority, oracle_semantics, physics_prediction, predict_numpy,
    row_mse, semantic_evaluation,
)
from scripts.tactile_unit.train_c4_fallback import complete_validation  # noqa: E402
from scripts.tactile_unit.train_c4_uncertainty import (  # noqa: E402
    calibration_metrics, sources, uncertainty_numpy,
)
from scripts.tactile_unit.vac_runtime_common import resolve_device, set_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def spearman_ci(error, uncertainty, samples, seed):
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(error), size=len(error))
        values[index] = spearmanr(np.asarray(error)[selected], np.asarray(uncertainty)[selected]).statistic
    return np.quantile(values, (0.025, 0.975)).astype(float).tolist()


def boundary_physics(prediction, test, shared_space, decoder, device, batch_size):
    oracle_future = physics_prediction(shared_space, decoder, test["u_c"], test["h_current"], device, batch_size)
    predicted_future = physics_prediction(shared_space, decoder, prediction, test["h_current"], device, batch_size)
    error = row_mse(predicted_future, oracle_future)
    transition = np.asarray(test["contact_transition"])
    dynamic = np.asarray(test["dynamic"], dtype=bool)
    def mean(mask):
        return float(error[mask].mean()) if np.any(mask) else None
    return {
        "all_window_mse": float(error.mean()), "dynamic_mse": mean(dynamic),
        "free_to_contact_mse": mean(transition == 1),
        "contact_to_free_mse": mean(transition == 3),
        "teacher_side_true_h_only": True,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    artifacts = ROOT / config["runtime"]["artifact_root"]
    # Both hashes, pretest flags, and checkpoint identities are validated before test load.
    fallback_selection, fallback_selection_sha = validate_fallback_selection(config)
    uncertainty_selection, uncertainty_selection_sha = validate_uncertainty_selection(config)
    identities_before = identity_snapshot(config)
    if not identities_before["pass"]:
        raise RuntimeError("STRUCTURAL_FAIL: frozen identity mismatch before locked evaluation")
    device, lock_handle, gpu = resolve_device(args.device, allowed_physical=("0", "1", "2", "3"))
    try:
        set_seed(int(config["seed"]) + 20000)
        train = load_split(config, "train")
        validation = load_split(config, "validation")
        test = load_split(config, "test")
        if len(test["u_c"]) != 17504:
            raise RuntimeError("STRUCTURAL_FAIL: locked benchmark row count changed")
        parent = load_parent_config(config)
        shared_space, _, shared_digest = load_frozen_shared_space(parent, device)
        s2 = load_s2_model(ROOT / config["runtime"]["s2_checkpoint"], device)
        decoder = s2.decoder.eval().requires_grad_(False)
        full, _ = load_full(config, device)
        fallback, _, _ = load_selected_fallback(config, device)
        uncertainty_model, _, _ = load_selected_uncertainty(config, device)
        summary = json.loads((artifacts / "fallback_training_summary.json").read_text())
        best_a = max(
            (row for row in summary["trials"] if row["trial"]["source"] == "A"),
            key=lambda row: row["best"]["validation"]["utility"],
        )
        best_va = max(
            (row for row in summary["trials"] if row["trial"]["source"] == "VA"),
            key=lambda row: row["best"]["validation"]["utility"],
        )
        a_model, _ = load_fallback_checkpoint(ROOT / best_a["best"]["checkpoint"], device)
        va_model, _ = load_fallback_checkpoint(ROOT / best_va["best"]["checkpoint"], device)
        a_model.eval().requires_grad_(False)
        va_model.eval().requires_grad_(False)

        oracle = oracle_probe(train, test)
        full_raw = model_evaluation(
            full, train, test, oracle, shared_space, decoder, parent, device,
            args.batch_size, int(config["seed"]) + 21000,
        )
        closure_config = json.loads((ROOT / "configs/tactile_unit/c3msccr_exact_action_closure.json").read_text())
        exact = exact_split(closure_config, "test")
        full_raw["action_temporal"] = exact_action_metrics(
            full, train, test, exact, oracle, shared_space, decoder, parent,
            device, args.batch_size, int(config["seed"]) + 21100,
            bootstrap_samples=int(config["evaluation"]["bootstrap_samples"]),
        )
        full_prediction = np.asarray(full_raw["prediction"])
        full_metrics = strip_arrays(full_raw)

        evaluation_config = copy.deepcopy(config)
        evaluation_config["validation"]["bootstrap_samples"] = int(config["evaluation"]["bootstrap_samples"])
        fallback_oracle = oracle_semantics(train, test)
        a_metrics = complete_validation(
            a_model, train, test, fallback_oracle, evaluation_config, device,
            args.batch_size, shared_space, decoder, int(config["seed"]) + 22000,
        )
        va_metrics = complete_validation(
            va_model, train, test, fallback_oracle, evaluation_config, device,
            args.batch_size, shared_space, decoder, int(config["seed"]) + 23000,
        )
        a_prediction = predict_fallback(a_model, test, device, args.batch_size)
        va_prediction = predict_fallback(va_model, test, device, args.batch_size)
        a_metrics["physics"]["boundaries"] = boundary_physics(a_prediction, test, shared_space, decoder, device, args.batch_size)
        va_metrics["physics"]["boundaries"] = boundary_physics(va_prediction, test, shared_space, decoder, device, args.batch_size)

        # Locked zero/mean-H misuse comparison uses each predictor's train-fitted probe.
        h_mean = np.asarray(train["h_current"], dtype=np.float64).mean(0).astype(np.float32)
        full_train_prediction = predict_numpy(full, train, device, args.batch_size)
        full_contact_probe = fit_probe(full_train_prediction, train["contact_transition"])
        invalid_predictions = {
            "zero": predict_numpy(full, test, device, args.batch_size, h_current=np.zeros_like(test["h_current"])),
            "mean": predict_numpy(full, test, device, args.batch_size, h_current=np.broadcast_to(h_mean, np.asarray(test["h_current"]).shape)),
        }
        misuse = {}
        for name, prediction in invalid_predictions.items():
            labels = full_contact_probe.predict(prediction.reshape(len(prediction), -1))
            semantics = evaluate_prediction(test["contact_transition"], labels, majority(train["contact_transition"], len(labels), 4), 4)
            misuse[name] = {"shared_mse": float(row_mse(prediction, test["u_c"]).mean()), "contact_macro_f1": float(semantics["macro_f1"])}
        for model_metrics, prediction in ((a_metrics, a_prediction), (va_metrics, va_prediction)):
            gate = bool(
                model_metrics["shared_target"]["prediction_mse"] < min(value["shared_mse"] for value in misuse.values())
                and model_metrics["semantics"]["contact_transition"]["macro_f1"] > max(value["contact_macro_f1"] for value in misuse.values())
            )
            model_metrics["invalid_h_misuse"] = {"locked_controls": misuse, "gate": gate}
            model_metrics["gates"]["invalid_h_misuse"] = gate
            model_metrics["gates"]["all"] = all(value for name, value in model_metrics["gates"].items() if name != "all")

        # A-vs-VA semantic bootstrap uses independently train-fitted latent probes.
        a_train_prediction = predict_fallback(a_model, train, device, args.batch_size)
        va_train_prediction = predict_fallback(va_model, train, device, args.batch_size)
        a_semantic, a_labels = semantic_evaluation(a_train_prediction, a_prediction, train, test, fallback_oracle)
        va_semantic, va_labels = semantic_evaluation(va_train_prediction, va_prediction, train, test, fallback_oracle)
        vision_bootstrap = {}
        for metric in ("contact_transition", "force_trend_class"):
            vision_bootstrap[metric] = bootstrap_f1_difference(
                test[metric], va_labels if metric == "contact_transition" else fit_probe(va_train_prediction, train[metric]).predict(va_prediction.reshape(len(va_prediction), -1)),
                a_labels if metric == "contact_transition" else fit_probe(a_train_prediction, train[metric]).predict(a_prediction.reshape(len(a_prediction), -1)),
                samples=int(config["evaluation"]["bootstrap_samples"]), seed=int(config["seed"]) + 24000 + len(vision_bootstrap),
            )
        contact_gain = float(va_semantic["contact_transition"]["macro_f1"] - a_semantic["contact_transition"]["macro_f1"])
        if vision_bootstrap["contact_transition"][0] > 0 and vision_bootstrap["force_trend_class"][0] > 0:
            vision_classification = "VISION_MATERIALLY_IMPROVES_MISSING_H_FALLBACK"
        elif contact_gain > 0:
            vision_classification = "VISION_SMALL_POSITIVE_FALLBACK_GAIN"
        elif a_metrics["gates"]["all"] and a_metrics["utility"] >= va_metrics["utility"] - 0.01:
            vision_classification = "A_ONLY_FALLBACK_SUFFICIENT"
        else:
            vision_classification = "VISION_NO_FALLBACK_GAIN"

        # Validation-frozen common calibration is applied without test refitting.
        mode_predictions = {
            AvailabilityMode.FULL_AH: full_prediction,
            AvailabilityMode.FALLBACK_VA: va_prediction,
            AvailabilityMode.FALLBACK_A: a_prediction,
        }
        uncertainty_metrics = {}
        uncertainty_values = {}
        errors = {}
        scale = float(uncertainty_selection["calibration_scale"])
        threshold = float(uncertainty_selection["high_error_threshold"])
        constant = float(uncertainty_selection["constant_variance"])
        for offset, (mode, prediction) in enumerate(mode_predictions.items()):
            log_variance = uncertainty_numpy(
                uncertainty_model, mode, prediction, sources(test, mode), device,
                int(config["uncertainty"]["batch_size"]),
            )
            value = np.exp(log_variance) * scale
            error = row_mse(prediction, test["u_c"])
            uncertainty_values[mode] = value
            errors[mode] = error
            metrics = calibration_metrics(error, value, threshold, constant)
            metrics["spearman_ci95"] = spearman_ci(error, value, 500, int(config["seed"]) + 25000 + offset)
            uncertainty_metrics[mode.value] = metrics
        fallback_mode = AvailabilityMode(uncertainty_selection["canonical_fallback_mode"])
        mode_difference = uncertainty_values[fallback_mode] - uncertainty_values[AvailabilityMode.FULL_AH]
        mode_ci = bootstrap_mean_ci(mode_difference, samples=int(config["evaluation"]["bootstrap_samples"]), seed=int(config["seed"]) + 25100)
        transition = np.asarray(test["contact_transition"])
        dynamic = np.asarray(test["dynamic"], dtype=bool)
        boundary_uncertainty = {}
        for mode, value in uncertainty_values.items():
            boundary_uncertainty[mode.value] = {
                "static": float(value[~dynamic].mean()), "dynamic": float(value[dynamic].mean()),
                "free_to_contact": float(value[transition == 1].mean()),
                "contact_to_free": float(value[transition == 3].mean()),
            }

        # Diagnostic corrupt-H responses; no missingness detector is trained.
        different = different_episode_permutation(test["episode_id"], int(config["seed"]) + 26000)
        wrong = wrong_time_indices(test["episode_id"], test["t"])
        rng = np.random.default_rng(int(config["seed"]) + 26001)
        noisy_h = np.asarray(test["h_current"]) + rng.normal(0.0, float(np.asarray(train["h_current"]).std()) * 0.1, size=np.asarray(test["h_current"]).shape).astype(np.float32)
        corrupt_h = {
            "wrong_time": np.asarray(test["h_current"])[wrong],
            "different_episode": np.asarray(test["h_current"])[different],
            "noisy": noisy_h,
        }
        corruption = {}
        for name, h_value in corrupt_h.items():
            prediction = predict_numpy(full, test, device, args.batch_size, h_current=h_value)
            split_value = dict(test, h_current=h_value)
            log_variance = uncertainty_numpy(uncertainty_model, AvailabilityMode.FULL_AH, prediction, sources(split_value, AvailabilityMode.FULL_AH), device, int(config["uncertainty"]["batch_size"]))
            corruption[name] = {"shared_mse": float(row_mse(prediction, test["u_c"]).mean()), "mean_uncertainty": float((np.exp(log_variance) * scale).mean())}

        repeated = {
            "full": np.array_equal(full_prediction, predict_numpy(full, test, device, args.batch_size)),
            "fallback_va": np.array_equal(va_prediction, predict_fallback(va_model, test, device, args.batch_size)),
            "fallback_a": np.array_equal(a_prediction, predict_fallback(a_model, test, device, args.batch_size)),
        }
        canonical_metrics = va_metrics if fallback_selection["source"] == "VA" else a_metrics
        canonical_uncertainty = uncertainty_metrics[fallback_mode.value]
        fallback_gates = {
            "contact": canonical_metrics["semantics"]["contact_transition"]["semantic_ratio"] >= float(config["evaluation"]["contact_retention_min"]),
            "force": canonical_metrics["semantics"]["force_trend_class"]["semantic_ratio"] >= float(config["evaluation"]["force_retention_min"]),
            "future_change": bool(canonical_metrics["gates"]["future_change"]),
            "invalid_h_misuse": bool(canonical_metrics["invalid_h_misuse"]["gate"]),
            "latent": bool(canonical_metrics["gates"]["latent"]),
            "physics": bool(canonical_metrics["gates"]["physics"]),
            "action_temporal": bool(canonical_metrics["gates"]["action_temporal"]),
            "noncollapse": bool(canonical_metrics["noncollapse"]),
        }
        uncertainty_gates = {
            "correlation": canonical_uncertainty["spearman"] > 0 and canonical_uncertainty["spearman_ci95"][0] > 0,
            "auroc": canonical_uncertainty["auroc"] >= float(config["evaluation"]["high_error_auroc_min"]),
            "risk_coverage": canonical_uncertainty["risk_coverage"]["top20_removal_reduction"] >= float(config["evaluation"]["risk_reduction_min"]),
            "nll": canonical_uncertainty["nll"] < canonical_uncertainty["constant_variance_nll"],
            "mode_sensitivity": mode_ci[0] > 0,
            "within_mode_nonconstant": canonical_uncertainty["uncertainty_std"] > 0,
        }
        full_nonregression = {
            "contact_ratio": full_metrics["semantics"]["contact_transition"]["semantic_ratio"],
            "force_ratio": full_metrics["semantics"]["force_trend_class"]["semantic_ratio"],
            "exact_action": full_metrics["action_temporal"]["gate"],
            "h_context": full_metrics["h_context"]["gate"], "physics": full_metrics["physics"]["gate"],
            "checkpoint_identity": identities_before["equality"]["full"],
        }
        full_nonregression["pass"] = bool(
            abs(full_nonregression["contact_ratio"] - 0.900316) < 0.002
            and abs(full_nonregression["force_ratio"] - 0.794947) < 0.002
            and all(full_nonregression[name] for name in ("exact_action", "h_context", "physics", "checkpoint_identity"))
        )
        identities_after = identity_snapshot(config)
        if not full_nonregression["pass"]:
            decision = "C4_FULL_PATH_REGRESSION"
        elif not fallback_gates["action_temporal"]:
            decision = "C4_FALLBACK_ACTION_TEMPORAL_FAIL"
        elif not all(fallback_gates.values()):
            decision = "C4_FALLBACK_INSUFFICIENT"
        elif not all(uncertainty_gates.values()):
            decision = "C4_UNCERTAINTY_UNCALIBRATED"
        else:
            decision = "C4_READY_VA_FALLBACK" if fallback_selection["source"] == "VA" else "C4_READY_A_FALLBACK"
        c5_readiness = "READY" if decision in {"C4_READY_VA_FALLBACK", "C4_READY_A_FALLBACK"} else "NOT READY"
        locked = {
            "schema": "tactile3d-unit.vac-c4-locked-evaluation.v1",
            "evaluation": "LOCKED POST-HOC BENCHMARK RE-EVALUATION", "first_look_untouched": False,
            "rows": len(test["u_c"]), "test_loaded": True,
            "selection_frozen_before_test": True,
            "fallback_selection_sha256": fallback_selection_sha,
            "uncertainty_selection_sha256": uncertainty_selection_sha,
            "full": full_metrics, "fallbacks": {"A": a_metrics, "VA": va_metrics},
            "vision_incremental": {
                "contact_macro_f1": contact_gain,
                "force_macro_f1": float(va_semantic["force_trend_class"]["macro_f1"] - a_semantic["force_trend_class"]["macro_f1"]),
                "shared_mse_improvement": float(a_metrics["shared_target"]["prediction_mse"] - va_metrics["shared_target"]["prediction_mse"]),
                "bootstrap_ci95": vision_bootstrap, "classification": vision_classification,
            },
            "uncertainty": uncertainty_metrics,
            "availability_sensitivity": {
                "mean_full": uncertainty_metrics["FULL_AH"]["uncertainty_mean"],
                "mean_fallback": canonical_uncertainty["uncertainty_mean"],
                "difference": float(mode_difference.mean()), "difference_ci95": mode_ci,
                "gate": mode_ci[0] > 0,
            },
            "boundary_uncertainty": boundary_uncertainty,
            "availability_corruption": corruption, "invalid_h_misuse": misuse,
            "fallback_gates": fallback_gates, "uncertainty_gates": uncertainty_gates,
            "full_nonregression": full_nonregression,
            "repeated_evaluation_exact": all(repeated.values()), "repeat_checks": repeated,
            "identity_before": identities_before, "identity_after": identities_after,
            "shared_state_sha256": shared_digest,
            "gpu": {**gpu, "preferred_physical": 1, "fallback": gpu.get("actual_physical") != 1},
            "decision": decision, "c5_readiness": c5_readiness,
        }
        atomic_json(artifacts / "locked_test_evaluation.json", locked)
        atomic_json(artifacts / "availability_ablation.json", {
            "schema": "tactile3d-unit.vac-c4-availability-ablation.v1",
            "invalid_h_misuse": misuse, "corruption": corruption,
            "vision_incremental": locked["vision_incremental"], "test_loaded": True,
        })
        atomic_json(artifacts / "uncertainty_calibration.json", {
            "schema": "tactile3d-unit.vac-c4-uncertainty-calibration.v1",
            "validation_frozen_scale": scale, "validation_frozen_high_error_threshold": threshold,
            "common_scale_across_modes": True, "metrics": uncertainty_metrics,
            "availability_sensitivity": locked["availability_sensitivity"],
            "boundary": boundary_uncertainty, "test_loaded": True,
        })
        atomic_json(artifacts / "final_decision.json", {
            "schema": "tactile3d-unit.vac-c4-decision.v1", "decision": decision,
            "reasons": {"fallback_gates": fallback_gates, "uncertainty_gates": uncertainty_gates, "full_nonregression": full_nonregression["pass"]},
            "canonical_fallback": fallback_selection["source"], "vision_classification": vision_classification,
            "rank_warning": True, "c5_readiness": c5_readiness,
            "c5": "NOT STARTED", "c6_m3": "NOT STARTED", "m3": "NOT ESTABLISHED",
        })
        if identities_before["actual"] != identities_after["actual"]:
            raise RuntimeError("STRUCTURAL_FAIL: frozen identity changed during locked evaluation")
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    main()
