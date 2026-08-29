#!/usr/bin/env python3
"""Train and validation-calibrate C5 uncertainty after freezing the causal mean."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.c4_availability_conditioning import sha256_file  # noqa: E402
from gr00t.tactile_unit.c5_uncertainty import (  # noqa: E402
    C5ContactUncertaintyEstimator, C5RuntimeMode, heteroscedastic_nll,
    load_c5_uncertainty_checkpoint, save_c5_uncertainty_checkpoint,
)
from gr00t.tactile_unit.continuous_vac_shared_space import bootstrap_mean_ci  # noqa: E402
from scripts.tactile_unit.c5_runtime import (  # noqa: E402
    DEFAULT_CONFIG, atomic_json, identity_snapshot, load_config, load_full,
    load_selected_causal, load_split, predict_causal, visual_batch,
)
from scripts.tactile_unit.train_c3mscc_contact_prediction import predict_numpy, row_mse  # noqa: E402
from scripts.tactile_unit.vac_runtime_common import resolve_device, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--epochs", type=int)
    return parser.parse_args()


def plan_ood_statistics(train_u_a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(train_u_a, dtype=np.float64).reshape(len(train_u_a), -1)
    mean, std = value.mean(0), value.std(0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def plan_ood_score(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    flat = np.asarray(value).reshape(len(value), -1)
    return np.sqrt(np.square((flat - mean) / std).mean(1)).astype(np.float32)


@torch.inference_mode()
def causal_tokens(visual, split, support, device, batch_size):
    output = np.empty((len(split["u_c"]), 8, 32), dtype=np.float32)
    visual.eval()
    for start in range(0, len(output), batch_size):
        stop = min(start + batch_size, len(output))
        features = torch.from_numpy(visual_batch(split, support, slice(start, stop))).to(device)
        output[start:stop] = visual(features).float().cpu().numpy()
    return output


def sources(split, mode, c_v=None):
    if mode is C5RuntimeMode.FULL_AH:
        return np.concatenate((np.asarray(split["u_a"]), np.asarray(split["h_current"]).reshape(-1, 8, 32)), axis=1)
    if mode is C5RuntimeMode.FALLBACK_CAUSAL_VA:
        if c_v is None: raise ValueError("causal source requires c_v")
        return np.concatenate((np.asarray(c_v), np.asarray(split["u_a"])), axis=1)
    return np.asarray(split["u_a"])


@torch.inference_mode()
def uncertainty_numpy(model, mode, prediction, source, ood, device, batch_size):
    output = np.empty(len(prediction), dtype=np.float32)
    model.eval()
    for start in range(0, len(output), batch_size):
        stop = min(start + batch_size, len(output))
        p = torch.from_numpy(np.array(prediction[start:stop], copy=True)).to(device)
        s = torch.from_numpy(np.array(source[start:stop], copy=True)).to(device)
        d = torch.from_numpy(np.array(ood[start:stop], copy=True)).to(device)
        output[start:stop] = model(mode, p, s, d).float().cpu().numpy()
    return output


def nll_rows(error, variance):
    return np.asarray(error) / (2.0 * np.asarray(variance)) + 0.5 * np.log(np.asarray(variance))


def risk_coverage(error, uncertainty):
    order = np.argsort(np.asarray(uncertainty)); result = {}
    for coverage in (1.0, 0.9, 0.8, 0.7, 0.5):
        count = max(1, int(math.floor(len(order) * coverage)))
        result[str(coverage)] = float(np.asarray(error)[order[:count]].mean())
    result["top20_removal_reduction"] = float((result["1.0"] - result["0.8"]) / max(result["1.0"], 1e-12))
    return result


def calibration_metrics(error, uncertainty, threshold, constant_variance):
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score, roc_auc_score
    label = np.asarray(error) >= threshold
    return {
        "rows": len(error), "spearman": float(spearmanr(error, uncertainty).statistic),
        "auroc": float(roc_auc_score(label, uncertainty)), "auprc": float(average_precision_score(label, uncertainty)),
        "nll": float(nll_rows(error, uncertainty).mean()),
        "constant_variance_nll": float(nll_rows(error, np.full(len(error), constant_variance)).mean()),
        "uncertainty_mean": float(np.mean(uncertainty)), "uncertainty_std": float(np.std(uncertainty)),
        "risk_coverage": risk_coverage(error, uncertainty),
    }


def spearman_ci(error, uncertainty, samples, seed):
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed); values = np.empty(samples)
    for index in range(samples):
        rows = rng.integers(0, len(error), len(error)); values[index] = spearmanr(np.asarray(error)[rows], np.asarray(uncertainty)[rows]).statistic
    return np.quantile(values, (0.025, 0.975)).astype(float).tolist()


def main() -> None:
    args = parse_args(); config = load_config(args.config)
    artifacts, experiments, cache_root = ROOT / config["runtime"]["artifact_root"], ROOT / config["runtime"]["experiment_root"], ROOT / config["runtime"]["cache_root"]
    identities_before = identity_snapshot(config)
    if not identities_before["pass"]: raise RuntimeError("STRUCTURAL_FAIL: frozen identity mismatch")
    _, _, selection, selection_sha = load_selected_causal(config, torch.device("cpu"))
    device, lock_handle, gpu = resolve_device(args.device, allowed_physical=("0", "1", "2", "3"))
    try:
        set_seed(int(config["seed"]) + 10000)
        train, validation = load_split(config, "train"), load_split(config, "validation")
        full, _ = load_full(config, device)
        visual, predictor, selection, selection_sha = load_selected_causal(config, device)
        support = __import__("gr00t.tactile_unit.c5_causal_visual", fromlist=["VisualSupport"]).VisualSupport(selection["visual_support"])
        c_v = {"train": causal_tokens(visual, train, support, device, int(config["uncertainty"]["batch_size"])), "validation": causal_tokens(visual, validation, support, device, int(config["uncertainty"]["batch_size"]))}
        predictions = {
            "train": {
                C5RuntimeMode.FULL_AH: np.load(ROOT / ".local/cache/tactile_unit/vac_c4/train/prediction_FULL_AH.npy", mmap_mode="r"),
                C5RuntimeMode.FALLBACK_CAUSAL_VA: predict_causal(visual, predictor, train, support, device, int(config["uncertainty"]["batch_size"])),
                C5RuntimeMode.FALLBACK_A: np.load(ROOT / ".local/cache/tactile_unit/vac_c4/train/prediction_FALLBACK_A.npy", mmap_mode="r"),
            },
            "validation": {
                C5RuntimeMode.FULL_AH: np.load(ROOT / ".local/cache/tactile_unit/vac_c4/validation/prediction_FULL_AH.npy", mmap_mode="r"),
                C5RuntimeMode.FALLBACK_CAUSAL_VA: predict_causal(visual, predictor, validation, support, device, int(config["uncertainty"]["batch_size"])),
                C5RuntimeMode.FALLBACK_A: np.load(ROOT / ".local/cache/tactile_unit/vac_c4/validation/prediction_FALLBACK_A.npy", mmap_mode="r"),
            },
        }
        for split_name in predictions:
            path = cache_root / split_name; path.mkdir(parents=True, exist_ok=True)
            np.save(path / "causal_visual_tokens.npy", c_v[split_name], allow_pickle=False)
            for mode, value in predictions[split_name].items(): np.save(path / f"prediction_{mode.value}.npy", value, allow_pickle=False)
        ood_mean, ood_std = plan_ood_statistics(train["u_a"])
        ood = {name: plan_ood_score(split["u_a"], ood_mean, ood_std) for name, split in (("train", train), ("validation", validation))}
        splits = {"train": train, "validation": validation}
        source = {name: {mode: sources(split, mode, c_v[name]) for mode in (C5RuntimeMode.FULL_AH, C5RuntimeMode.FALLBACK_CAUSAL_VA, C5RuntimeMode.FALLBACK_A)} for name, split in splits.items()}
        spec = config["uncertainty"]
        model = C5ContactUncertaintyEstimator(int(spec["hidden"])).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec["learning_rate"]), weight_decay=float(spec["weight_decay"]))
        best, history, stale = None, [], 0; checkpoint = experiments / "uncertainty" / "best.pt"
        for epoch in range(1, int(args.epochs or spec["epochs"]) + 1):
            model.train(); totals = []
            for mode in (C5RuntimeMode.FULL_AH, C5RuntimeMode.FALLBACK_CAUSAL_VA, C5RuntimeMode.FALLBACK_A):
                order = np.random.default_rng(int(config["seed"]) + epoch * 10 + len(mode.value)).permutation(len(train["u_c"]))
                for start in range(0, len(order), int(spec["batch_size"])):
                    rows = order[start:start + int(spec["batch_size"])]
                    p = torch.from_numpy(np.array(predictions["train"][mode][rows], copy=True)).to(device)
                    s = torch.from_numpy(np.array(source["train"][mode][rows], copy=True)).to(device)
                    d = torch.from_numpy(np.array(ood["train"][rows], copy=True)).to(device)
                    target = torch.from_numpy(np.array(train["u_c"][rows], copy=True)).to(device)
                    optimizer.zero_grad(set_to_none=True); loss = heteroscedastic_nll(model(mode, p, s, d), p, target)
                    if not torch.isfinite(loss): raise FloatingPointError("non-finite C5 uncertainty loss")
                    loss.backward()
                    if any(parameter.grad is not None for mean in (full, visual, predictor) for parameter in mean.parameters()): raise RuntimeError("uncertainty gradient reached frozen mean predictor")
                    optimizer.step(); totals.append(float(loss.detach()))
            validation_nll = []
            for mode in (C5RuntimeMode.FULL_AH, C5RuntimeMode.FALLBACK_CAUSAL_VA, C5RuntimeMode.FALLBACK_A):
                prediction = predictions["validation"][mode]; error = row_mse(prediction, validation["u_c"])
                logv = uncertainty_numpy(model, mode, prediction, source["validation"][mode], ood["validation"], device, int(spec["batch_size"]))
                validation_nll.append(float(nll_rows(error, np.exp(logv)).mean()))
            score = float(np.mean(validation_nll)); history.append({"epoch": epoch, "train_nll": float(np.mean(totals)), "validation_nll": dict(zip((mode.value for mode in (C5RuntimeMode.FULL_AH, C5RuntimeMode.FALLBACK_CAUSAL_VA, C5RuntimeMode.FALLBACK_A)), validation_nll)), "mean_validation_nll": score})
            if best is None or score < best["mean_validation_nll"]:
                digest = save_c5_uncertainty_checkpoint(checkpoint, model, {"epoch": epoch, "mean_validation_nll": score, "selection_split": "validation only", "test_loaded": False, "mean_predictors_frozen": True})
                best = {"epoch": epoch, "mean_validation_nll": score, "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": digest}; stale = 0
            else: stale += 1
            atomic_json(experiments / "uncertainty" / "history.json", {"history": history, "best": best, "test_loaded": False})
            if stale >= int(spec["patience"]): break
        model, metadata = load_c5_uncertainty_checkpoint(ROOT / best["checkpoint"], device); model.eval().requires_grad_(False)
        raw, errors = {}, {}
        for mode in (C5RuntimeMode.FULL_AH, C5RuntimeMode.FALLBACK_CAUSAL_VA, C5RuntimeMode.FALLBACK_A):
            prediction = predictions["validation"][mode]; errors[mode] = row_mse(prediction, validation["u_c"])
            raw[mode] = uncertainty_numpy(model, mode, prediction, source["validation"][mode], ood["validation"], device, int(spec["batch_size"]))
        calibration_scale = float(np.mean(np.concatenate([errors[mode] / np.exp(raw[mode]) for mode in errors])))
        threshold = float(np.quantile(errors[C5RuntimeMode.FALLBACK_CAUSAL_VA], float(spec["high_error_quantile"])))
        constant = float(np.mean(np.concatenate(list(errors.values())))); metrics, values = {}, {}
        for index, mode in enumerate(errors):
            values[mode] = np.exp(raw[mode]) * calibration_scale
            metrics[mode.value] = calibration_metrics(errors[mode], values[mode], threshold, constant)
            metrics[mode.value]["spearman_ci95"] = spearman_ci(errors[mode], values[mode], 500, int(config["seed"]) + 12000 + index)
        mode_difference = values[C5RuntimeMode.FALLBACK_CAUSAL_VA] - values[C5RuntimeMode.FULL_AH]
        mode_ci = bootstrap_mean_ci(mode_difference, samples=int(config["validation"]["bootstrap_samples"]), seed=int(config["seed"]) + 12100)
        causal = metrics[C5RuntimeMode.FALLBACK_CAUSAL_VA.value]
        gates = {"correlation": causal["spearman"] > 0 and causal["spearman_ci95"][0] > 0, "auroc": causal["auroc"] >= float(config["evaluation"]["high_error_auroc_min"]), "risk_coverage": causal["risk_coverage"]["top20_removal_reduction"] >= float(config["evaluation"]["risk_reduction_min"]), "nll": causal["nll"] < causal["constant_variance_nll"], "fallback_above_full": mode_ci[0] > 0, "within_mode_nonconstant": causal["uncertainty_std"] > 0}
        rng = np.random.default_rng(int(config["seed"]) + 12200); train_std = np.asarray(train["u_a"], dtype=np.float64).std(0).astype(np.float32)
        perturbations = {"oracle_demo_surrogate": np.asarray(validation["u_a"]), "mild": np.asarray(validation["u_a"]) + rng.normal(0, 0.05, np.asarray(validation["u_a"]).shape).astype(np.float32) * train_std, "strong": np.asarray(validation["u_a"]) + rng.normal(0, 0.20, np.asarray(validation["u_a"]).shape).astype(np.float32) * train_std}
        perturbation_metrics = {}
        for name, action in perturbations.items():
            pred = predict_causal(visual, predictor, validation, support, device, int(spec["batch_size"]), u_a=action)
            source_value = np.concatenate((c_v["validation"], action), axis=1); score = plan_ood_score(action, ood_mean, ood_std)
            u = np.exp(uncertainty_numpy(model, C5RuntimeMode.FALLBACK_CAUSAL_VA, pred, source_value, score, device, int(spec["batch_size"]))) * calibration_scale
            perturbation_metrics[name] = {"representation_distance": float(score.mean()), "prediction_mse": float(row_mse(pred, validation["u_c"]).mean()), "mean_uncertainty": float(u.mean())}
        training_artifact = {"schema": "tactile3d-unit.vac-c5-uncertainty-training.v1", "history": history, "best": best, "mean_predictors_frozen": True, "plan_ood_fit_split": "train only", "plan_ood_mean": ood_mean.tolist(), "plan_ood_std": ood_std.tolist(), "test_loaded": False}
        atomic_json(artifacts / "uncertainty_training.json", training_artifact)
        selection_value = {"schema": "tactile3d-unit.vac-c5-uncertainty-selection.v1", "model": "C5ContactUncertaintyEstimator", "parameters": model.parameter_count(), "checkpoint": best["checkpoint"], "checkpoint_sha256": best["checkpoint_sha256"], "epoch": best["epoch"], "calibration_scale": calibration_scale, "common_scale_across_modes": True, "high_error_threshold": threshold, "high_error_threshold_definition": "causal fallback validation shared-error 75th percentile", "constant_variance": constant, "validation": metrics, "fallback_minus_full_uncertainty_ci95": mode_ci, "gates": {**gates, "all": all(gates.values())}, "plan_perturbation_diagnostic": perturbation_metrics, "actual_policy_domain_validated": False, "warning": "POLICY_PLAN_DOMAIN_WARNING", "mean_predictors_frozen": True, "causal_visual_selection_sha256": selection_sha, "selected_via": "VALIDATION ONLY", "selection_split": "validation only", "test_loaded": False, "identity": identity_snapshot(config), "gpu": {**gpu, "preferred_physical": 1, "fallback": gpu.get("actual_physical") != 1}}
        path = artifacts / "uncertainty_selection.json"; atomic_json(path, selection_value); digest = sha256_file(path); (artifacts / "uncertainty_selection.sha256").write_text(digest + "  uncertainty_selection.json\n")
        if not identity_snapshot(config)["pass"]: raise RuntimeError("frozen identity changed during C5 uncertainty")
        print(json.dumps({"uncertainty_gates": selection_value["gates"], "sha256": digest}, indent=2))
    finally:
        if lock_handle is not None: lock_handle.close()


if __name__ == "__main__":
    main()
