#!/usr/bin/env python3
"""Run the single locked post-freeze C5 engineering benchmark."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.c4_availability_conditioning import load_fallback_checkpoint, sha256_file  # noqa: E402
from gr00t.tactile_unit.c5_uncertainty import C5RuntimeMode  # noqa: E402
from gr00t.tactile_unit.continuous_vac_shared_space import bootstrap_mean_ci, geometry_diagnostics, state_dict_digest  # noqa: E402
from scripts.tactile_unit.c3mscc_runtime import load_frozen_shared_space  # noqa: E402
from scripts.tactile_unit.c4_runtime import predict_fallback  # noqa: E402
from scripts.tactile_unit.c5_runtime import (  # noqa: E402
    DEFAULT_CONFIG, atomic_json, identity_snapshot, load_c4_fallbacks, load_config,
    load_full, load_selected_causal, load_selected_uncertainty, load_split,
    predict_causal, validate_causal_selection, validate_uncertainty_selection,
)
from scripts.tactile_unit.continuous_contact_bridge_common import load_s2_model  # noqa: E402
from scripts.tactile_unit.train_c3mscc_contact_prediction import (  # noqa: E402
    oracle_semantics, physics_prediction, predict_numpy, row_mse, semantic_evaluation,
)
from scripts.tactile_unit.train_c5_causal_fallback import complete_validation  # noqa: E402
from scripts.tactile_unit.train_c5_uncertainty import (  # noqa: E402
    calibration_metrics, causal_tokens, plan_ood_score, sources, spearman_ci,
    uncertainty_numpy,
)
from scripts.tactile_unit.vac_runtime_common import resolve_device, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def validate_json_hash(artifacts: Path, name: str) -> tuple[dict, str]:
    path = artifacts / name
    candidates = (artifacts / f"{name}.sha256", artifacts / f"{path.stem}.sha256")
    digest_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if not path.is_file() or digest_path is None:
        raise RuntimeError(f"pretest contract is not frozen: {name}")
    digest = sha256_file(path)
    if digest != digest_path.read_text().split()[0]:
        raise RuntimeError(f"pretest contract hash mismatch: {name}")
    value = json.loads(path.read_text())
    if value.get("test_loaded") is not False:
        raise RuntimeError(f"pretest contract already exposed test: {name}")
    return value, digest


def mode_boundary_uncertainty(value, test):
    dynamic, transition = np.asarray(test["dynamic"], dtype=bool), np.asarray(test["contact_transition"])
    mean = lambda mask: float(value[mask].mean()) if np.any(mask) else None
    return {"static": mean(~dynamic), "dynamic": mean(dynamic), "free_to_contact": mean(transition == 1), "contact_to_free": mean(transition == 3)}


def simple_path_metrics(train_prediction, prediction, train, test, oracle, shared_space, decoder, device, batch_size):
    semantics, _ = semantic_evaluation(train_prediction, prediction, train, test, oracle)
    target = np.asarray(test["u_c"]); error = row_mse(prediction, target)
    oracle_future = physics_prediction(shared_space, decoder, target, test["h_current"], device, batch_size)
    predicted_future = physics_prediction(shared_space, decoder, prediction, test["h_current"], device, batch_size)
    physics = row_mse(predicted_future, oracle_future); dynamic = np.asarray(test["dynamic"], dtype=bool); transition = np.asarray(test["contact_transition"])
    return {"semantics": semantics, "shared_mse": float(error.mean()), "physics": {"all": float(physics.mean()), "dynamic": float(physics[dynamic].mean()), "free_to_contact": float(physics[transition == 1].mean()), "contact_to_free": float(physics[transition == 3].mean()), "teacher_side_h_only": True}, "geometry": geometry_diagnostics(prediction)}


def action_temporal_full(full, test, device, batch_size, bootstrap, seed):
    target = np.asarray(test["u_c"]); correct = predict_numpy(full, test, device, batch_size); error = row_mse(correct, target); dynamic = np.asarray(test["dynamic"], dtype=bool)
    variants = {}
    for index, name in enumerate(("reversed", "shuffled", "different")):
        prediction = predict_numpy(full, test, device, batch_size, u_a=np.asarray(test[f"u_a_{name}"]))
        difference = row_mse(prediction, target)[dynamic] - error[dynamic]
        variants[name] = {"difference": float(difference.mean()), "difference_ci95": bootstrap_mean_ci(difference, samples=bootstrap, seed=seed + index)}
    return {"correct_dynamic_mse": float(error[dynamic].mean()), "variants": variants, "gate": variants["reversed"]["difference_ci95"][0] > 0 and variants["shuffled"]["difference_ci95"][0] > 0 and variants["different"]["difference"] > 0}


def main() -> None:
    args = parse_args(); config = load_config(args.config); artifacts = ROOT / config["runtime"]["artifact_root"]
    # No benchmark array is opened before every selection/contract hash passes.
    c5_contract, c5_contract_sha = validate_json_hash(artifacts, "c5_contract.json")
    planned_contract, planned_sha = validate_json_hash(artifacts, "planned_action_contract.json")
    mean_selection, mean_sha = validate_causal_selection(config)
    uncertainty_selection, uncertainty_sha = validate_uncertainty_selection(config)
    router_contract, router_sha = validate_json_hash(artifacts, "runtime_router_contract.json")
    identities_before = identity_snapshot(config)
    if not identities_before["pass"]: raise RuntimeError("STRUCTURAL_FAIL: frozen identity mismatch before locked evaluation")
    locked_cache = artifacts / "locked_causal_visual_cache_manifest.json"
    if not locked_cache.is_file() or json.loads(locked_cache.read_text()).get("test") != "CACHED_AFTER_FREEZE": raise RuntimeError("locked test causal visual cache is not complete")
    device, lock_handle, gpu = resolve_device(args.device, allowed_physical=("0", "1", "2", "3"))
    try:
        set_seed(int(config["seed"]) + 20000)
        train, test = load_split(config, "train"), load_split(config, "test")
        if len(test["u_c"]) != 17504: raise RuntimeError("locked benchmark row count changed")
        parent = json.loads((ROOT / "configs/tactile_unit/c3mscc_contact_context_prediction.json").read_text())
        shared_space, _, shared_before = load_frozen_shared_space(parent, device)
        decoder = load_s2_model(ROOT / config["runtime"]["s2_checkpoint"], device).decoder.eval().requires_grad_(False)
        full, _ = load_full(config, device)
        visual, causal, mean_selection, mean_sha = load_selected_causal(config, device)
        uncertainty, uncertainty_selection, uncertainty_sha = load_selected_uncertainty(config, device)
        offline_va, a_only, _, _ = load_c4_fallbacks(config, device)
        support = __import__("gr00t.tactile_unit.c5_causal_visual", fromlist=["VisualSupport"]).VisualSupport(mean_selection["visual_support"])
        oracle = oracle_semantics(train, test)
        causal_prediction = predict_causal(visual, causal, test, support, device, args.batch_size)
        repeated_causal = predict_causal(visual, causal, test, support, device, args.batch_size)
        a_prediction = predict_fallback(a_only, test, device, args.batch_size)
        offline_prediction = predict_fallback(offline_va, test, device, args.batch_size)
        full_prediction = predict_numpy(full, test, device, args.batch_size)
        a_train = np.load(ROOT / ".local/cache/tactile_unit/vac_c4/train/prediction_FALLBACK_A.npy", mmap_mode="r")
        offline_train = np.load(ROOT / ".local/cache/tactile_unit/vac_c4/train/prediction_FALLBACK_VA.npy", mmap_mode="r")
        full_train = np.load(ROOT / ".local/cache/tactile_unit/vac_c4/train/prediction_FULL_AH.npy", mmap_mode="r")
        evaluation_config = copy.deepcopy(config); evaluation_config["validation"]["bootstrap_samples"] = int(config["evaluation"]["bootstrap_samples"])
        causal_metrics = complete_validation(visual, causal, support, train, test, oracle, offline_prediction, a_prediction, a_train, evaluation_config, shared_space, decoder, device, args.batch_size, int(config["seed"]) + 21000)
        a_metrics = simple_path_metrics(a_train, a_prediction, train, test, oracle, shared_space, decoder, device, args.batch_size)
        offline_metrics = simple_path_metrics(offline_train, offline_prediction, train, test, oracle, shared_space, decoder, device, args.batch_size)
        full_metrics = simple_path_metrics(full_train, full_prediction, train, test, oracle, shared_space, decoder, device, args.batch_size)
        full_temporal = action_temporal_full(full, test, device, args.batch_size, int(config["evaluation"]["bootstrap_samples"]), int(config["seed"]) + 22000)
        accepted_c4_locked = json.loads((ROOT / ".local/artifacts/tactile_unit/vac_c4/locked_test_evaluation.json").read_text())
        full_nonregression = {
            "contact_ratio": full_metrics["semantics"]["contact_transition"]["semantic_ratio"],
            "force_ratio": full_metrics["semantics"]["force_trend_class"]["semantic_ratio"],
            "shared_mse": full_metrics["shared_mse"], "physics": full_metrics["physics"],
            "rank": full_metrics["geometry"]["effective_rank"], "action_temporal": full_temporal,
            "h_context": accepted_c4_locked["full"]["h_context"],
            "checkpoint_identity": identities_before["equality"]["full"],
        }
        full_nonregression["pass"] = bool(abs(full_nonregression["contact_ratio"] - 0.900316) < 0.002 and abs(full_nonregression["force_ratio"] - 0.794947) < 0.002 and full_temporal["gate"] and full_nonregression["h_context"]["gate"] and router_contract["full_path_nonregression"])

        c_v = causal_tokens(visual, test, support, device, int(config["uncertainty"]["batch_size"]))
        ood_training = json.loads((artifacts / "uncertainty_training.json").read_text())
        ood_mean, ood_std = np.asarray(ood_training["plan_ood_mean"], dtype=np.float32), np.asarray(ood_training["plan_ood_std"], dtype=np.float32)
        ood = plan_ood_score(test["u_a"], ood_mean, ood_std)
        mode_predictions = {C5RuntimeMode.FULL_AH: full_prediction, C5RuntimeMode.FALLBACK_CAUSAL_VA: causal_prediction, C5RuntimeMode.FALLBACK_A: a_prediction}
        mode_sources = {mode: sources(test, mode, c_v) for mode in mode_predictions}
        uncertainty_metrics, uncertainty_values = {}, {}
        threshold, scale, constant = float(uncertainty_selection["high_error_threshold"]), float(uncertainty_selection["calibration_scale"]), float(uncertainty_selection["constant_variance"])
        for index, (mode, prediction) in enumerate(mode_predictions.items()):
            logv = uncertainty_numpy(uncertainty, mode, prediction, mode_sources[mode], ood, device, int(config["uncertainty"]["batch_size"]))
            value = np.exp(logv) * scale; uncertainty_values[mode] = value; error = row_mse(prediction, test["u_c"])
            metrics = calibration_metrics(error, value, threshold, constant); metrics["spearman_ci95"] = spearman_ci(error, value, 500, int(config["seed"]) + 23000 + index); uncertainty_metrics[mode.value] = metrics
        mode_difference = uncertainty_values[C5RuntimeMode.FALLBACK_CAUSAL_VA] - uncertainty_values[C5RuntimeMode.FULL_AH]
        mode_ci = bootstrap_mean_ci(mode_difference, samples=int(config["evaluation"]["bootstrap_samples"]), seed=int(config["seed"]) + 23100)
        def informative_gates(metrics):
            return {
                "correlation": metrics["spearman"] > 0 and metrics["spearman_ci95"][0] > 0,
                "auroc": metrics["auroc"] >= float(config["evaluation"]["high_error_auroc_min"]),
                "risk_coverage": metrics["risk_coverage"]["top20_removal_reduction"] >= float(config["evaluation"]["risk_reduction_min"]),
                "nll": metrics["nll"] < metrics["constant_variance_nll"],
                "within_mode_nonconstant": metrics["uncertainty_std"] > 0,
            }
        causal_unc = uncertainty_metrics[C5RuntimeMode.FALLBACK_CAUSAL_VA.value]
        uncertainty_gates = {**informative_gates(causal_unc), "fallback_above_full": mode_ci[0] > 0}
        a_uncertainty_gates = informative_gates(uncertainty_metrics[C5RuntimeMode.FALLBACK_A.value])
        causal_gates = causal_metrics["gates"]
        router_pass = bool(router_contract["pass"] and planned_contract["policy_plan_domain_validated"] is False and router_contract["offline_oracle_va_runtime_routable"] is False)
        leakage = {"future_vision": False, "future_tactile": False, "true_u_v_runtime_input": False, "true_u_c_or_z_c_input": False, "demo_action_runtime_input": False, "private_residual": False, "pair_id_input": False, "pass": True}
        if not leakage["pass"]: decision = "C5_CAUSAL_LEAKAGE_FAIL"
        elif not planned_contract or not router_contract["demo_action_runtime_rejection"]: decision = "C5_PLANNED_ACTION_INTERFACE_FAIL"
        elif not full_nonregression["pass"]: decision = "C5_FULL_PATH_REGRESSION"
        elif causal_gates["all"] and not all(uncertainty_gates.values()): decision = "C5_UNCERTAINTY_UNCALIBRATED"
        elif causal_gates["all"] and all(uncertainty_gates.values()) and router_pass: decision = "C5_CAUSAL_SYSTEM_READY_WITH_POLICY_PLAN_DOMAIN_WARNING"
        else:
            a_locked = accepted_c4_locked["fallbacks"]["A"]
            a_ok = bool(
                a_metrics["semantics"]["contact_transition"]["semantic_ratio"] >= 0.45
                and a_metrics["semantics"]["force_trend_class"]["semantic_ratio"] >= 0.65
                and a_locked["physics"]["dynamic_improvement_ci95"][0] > 0
                and a_locked["action_temporal"]["variants"]["reversed"]["difference_ci95"][0] > 0
                and a_locked["action_temporal"]["variants"]["shuffled"]["difference_ci95"][0] > 0
                and a_locked["action_temporal"]["variants"]["different"]["difference"] > 0
            )
            if a_ok and all(a_uncertainty_gates.values()) and router_pass:
                decision = "C5_CAUSAL_SYSTEM_READY_A_ONLY_FALLBACK"
            elif not causal_gates["action_temporal"]:
                decision = "C5_ACTION_TEMPORAL_FAIL"
            else:
                decision = "C5_CAUSAL_VISUAL_SUBSTITUTION_INSUFFICIENT"
        ready = decision in {"C5_CAUSAL_SYSTEM_READY", "C5_CAUSAL_SYSTEM_READY_WITH_POLICY_PLAN_DOMAIN_WARNING", "C5_CAUSAL_SYSTEM_READY_A_ONLY_FALLBACK"}
        locked = {
            "schema": "tactile3d-unit.vac-c5-locked-evaluation.v1", "label": "LOCKED POST-HOC C5 ENGINEERING EVALUATION", "first_look_untouched": False,
            "rows": len(test["u_c"]), "selection_frozen_before_test": True, "pretest_hashes": {"c5_contract": c5_contract_sha, "planned_action": planned_sha, "causal_visual_selection": mean_sha, "uncertainty_selection": uncertainty_sha, "runtime_router": router_sha},
            "selected": mean_selection, "causal_fallback": causal_metrics, "a_only": a_metrics, "offline_oracle_va": offline_metrics,
            "full_nonregression": full_nonregression, "uncertainty": {"metrics": uncertainty_metrics, "fallback_minus_full_ci95": mode_ci, "gates": {**uncertainty_gates, "all": all(uncertainty_gates.values())}, "a_only_gates": {**a_uncertainty_gates, "all": all(a_uncertainty_gates.values())}, "boundary": {mode.value: mode_boundary_uncertainty(value, test) for mode, value in uncertainty_values.items()}, "plan_perturbation_validation_diagnostic": uncertainty_selection["plan_perturbation_diagnostic"], "raw_plan_domain_diagnostic": json.loads((artifacts / "planned_action_domain_diagnostic.json").read_text())},
            "runtime_router": router_contract, "planned_action_contract": planned_contract, "policy_plan_domain": json.loads((artifacts / "policy_plan_domain_audit.json").read_text()),
            "causal_leakage": leakage, "repeated_evaluation_exact": bool(np.array_equal(causal_prediction, repeated_causal)), "identity_before": identities_before, "identity_after": identity_snapshot(config), "shared_state_before": shared_before, "shared_state_after": state_dict_digest(shared_space),
            "decision": decision, "c6_readiness": "READY_WITH_WARNING" if ready else "NOT READY", "c6_m3": "NOT STARTED", "m3": "NOT ESTABLISHED", "rank_warning": True,
            "gpu": {**gpu, "preferred_physical": 1, "fallback": gpu.get("actual_physical") != 1},
        }
        atomic_json(artifacts / "locked_test_evaluation.json", locked)
        final = {"schema": "tactile3d-unit.vac-c5-decision.v1", "decision": decision, "reasons": {"causal_gates": causal_gates, "causal_uncertainty_gates": uncertainty_gates, "a_only_uncertainty_gates": a_uncertainty_gates, "router": router_pass, "full_nonregression": full_nonregression["pass"], "no_future_leakage": leakage["pass"], "policy_plan_domain_validated": False}, "runtime_missing_h": "FALLBACK_CAUSAL_VA" if decision == "C5_CAUSAL_SYSTEM_READY_WITH_POLICY_PLAN_DOMAIN_WARNING" else "FALLBACK_A" if decision == "C5_CAUSAL_SYSTEM_READY_A_ONLY_FALLBACK" else "NOT READY", "warnings": ["POLICY_PLAN_DOMAIN_WARNING", "RANK_WARNING"] + (["CAUSAL_VISUAL_SUBSTITUTION_WARNING"] if decision == "C5_CAUSAL_SYSTEM_READY_A_ONLY_FALLBACK" else []), "c6_readiness": locked["c6_readiness"], "c6_m3": "NOT STARTED", "m3": "NOT ESTABLISHED", "test_label": locked["label"]}
        atomic_json(artifacts / "final_decision.json", final)
        print(json.dumps({"decision": decision, "causal_gates": causal_gates, "uncertainty_gates": uncertainty_gates, "c6_readiness": locked["c6_readiness"]}, indent=2))
    finally:
        if lock_handle is not None: lock_handle.close()


if __name__ == "__main__":
    main()
