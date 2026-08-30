#!/usr/bin/env python3
"""Train and validation-calibrate C4 uncertainty after freezing mean predictors."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.c4_availability_conditioning import (  # noqa: E402
    AvailabilityMode, load_fallback_checkpoint, sha256_file,
)
from gr00t.tactile_unit.c4_uncertainty import (  # noqa: E402
    ContactUncertaintyEstimator, heteroscedastic_nll, load_uncertainty_checkpoint,
    save_uncertainty_checkpoint,
)
from scripts.tactile_unit.c4_runtime import (  # noqa: E402
    DEFAULT_CONFIG, atomic_json, identity_snapshot, load_config, load_full,
    load_selected_fallback, load_split, predict_fallback,
)
from scripts.tactile_unit.train_c3mscc_contact_prediction import predict_numpy, row_mse  # noqa: E402
from scripts.tactile_unit.vac_runtime_common import resolve_device, set_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--epochs", type=int)
    return parser.parse_args()


def sources(split, mode):
    if mode is AvailabilityMode.FULL_AH:
        return np.concatenate((np.asarray(split["u_a"]), np.asarray(split["h_current"]).reshape(-1, 8, 32)), axis=1)
    if mode is AvailabilityMode.FALLBACK_VA:
        return np.concatenate((np.asarray(split["u_v"]), np.asarray(split["u_a"])), axis=1)
    return np.asarray(split["u_a"])


def uncertainty_numpy(model, mode, prediction, source, device, batch_size):
    output = np.empty(len(prediction), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(prediction), batch_size):
            stop = min(start + batch_size, len(prediction))
            p = torch.from_numpy(np.array(prediction[start:stop], copy=True)).to(device)
            s = torch.from_numpy(np.array(source[start:stop], copy=True)).to(device)
            output[start:stop] = model(mode, p, s).float().cpu().numpy()
    return output


def nll_rows(error, variance):
    return np.asarray(error) / (2.0 * np.asarray(variance)) + 0.5 * np.log(np.asarray(variance))


def risk_coverage(error, uncertainty):
    order = np.argsort(np.asarray(uncertainty))
    result = {}
    for coverage in (1.0, 0.9, 0.8, 0.7, 0.5):
        count = max(1, int(math.floor(len(order) * coverage)))
        result[str(coverage)] = float(np.asarray(error)[order[:count]].mean())
    result["top20_removal_reduction"] = float(
        (result["1.0"] - result["0.8"]) / max(result["1.0"], 1e-12)
    )
    return result


def calibration_metrics(error, uncertainty, threshold, constant_variance):
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score, roc_auc_score
    label = np.asarray(error) >= float(threshold)
    return {
        "rows": len(error), "spearman": float(spearmanr(error, uncertainty).statistic),
        "auroc": float(roc_auc_score(label, uncertainty)) if np.unique(label).size == 2 else float("nan"),
        "auprc": float(average_precision_score(label, uncertainty)) if np.unique(label).size == 2 else float("nan"),
        "nll": float(nll_rows(error, uncertainty).mean()),
        "constant_variance_nll": float(nll_rows(error, np.full(len(error), constant_variance)).mean()),
        "uncertainty_mean": float(np.mean(uncertainty)),
        "uncertainty_std": float(np.std(uncertainty)),
        "risk_coverage": risk_coverage(error, uncertainty),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    artifacts = ROOT / config["runtime"]["artifact_root"]
    experiments = ROOT / config["runtime"]["experiment_root"]
    identities_before = identity_snapshot(config)
    if not identities_before["pass"]:
        raise RuntimeError("STRUCTURAL_FAIL: frozen identity mismatch")
    fallback, fallback_selection, fallback_selection_sha = load_selected_fallback(config, torch.device("cpu"))
    training_summary = json.loads((artifacts / "fallback_training_summary.json").read_text())
    a_row = max(
        (row for row in training_summary["trials"] if row["trial"]["source"] == "A"),
        key=lambda row: row["best"]["validation"]["utility"],
    )
    device, lock_handle, gpu = resolve_device(args.device, allowed_physical=("0", "1", "2", "3"))
    try:
        set_seed(int(config["seed"]) + 10000)
        train = load_split(config, "train")
        validation = load_split(config, "validation")
        full, _ = load_full(config, device)
        fallback, _, _ = load_selected_fallback(config, device)
        a_model, a_metadata = load_fallback_checkpoint(ROOT / a_row["best"]["checkpoint"], device)
        a_model.eval().requires_grad_(False)
        if a_metadata.get("test_loaded") is not False:
            raise RuntimeError("STRUCTURAL_FAIL: emergency A checkpoint saw test")
        models = {
            AvailabilityMode.FULL_AH: full,
            AvailabilityMode.FALLBACK_VA if fallback.source == "VA" else AvailabilityMode.FALLBACK_A: fallback,
            AvailabilityMode.FALLBACK_A: a_model,
        }
        if AvailabilityMode.FALLBACK_VA not in models:
            va_row = max(
                (row for row in training_summary["trials"] if row["trial"]["source"] == "VA"),
                key=lambda row: row["best"]["validation"]["utility"],
            )
            va_model, _ = load_fallback_checkpoint(ROOT / va_row["best"]["checkpoint"], device)
            models[AvailabilityMode.FALLBACK_VA] = va_model.eval().requires_grad_(False)
        predictions: dict[str, dict[AvailabilityMode, np.ndarray]] = {"train": {}, "validation": {}}
        splits = {"train": train, "validation": validation}
        cache_root = ROOT / config["runtime"]["cache_root"]
        for split_name, split in splits.items():
            for mode, mean_model in models.items():
                if mode is AvailabilityMode.FULL_AH:
                    prediction = predict_numpy(mean_model, split, device, int(config["uncertainty"]["batch_size"]))
                else:
                    prediction = predict_fallback(mean_model, split, device, int(config["uncertainty"]["batch_size"]))
                predictions[split_name][mode] = prediction
                path = cache_root / split_name / f"prediction_{mode.value}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, prediction, allow_pickle=False)

        spec = config["uncertainty"]
        model = ContactUncertaintyEstimator(
            hidden=int(spec["hidden"]), log_variance_min=float(spec["log_variance_min"]),
            log_variance_max=float(spec["log_variance_max"]),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec["learning_rate"]), weight_decay=float(spec["weight_decay"]))
        best = None
        history = []
        stale = 0
        checkpoint = experiments / "uncertainty" / "best.pt"
        epochs = int(args.epochs or spec["epochs"])
        for epoch in range(1, epochs + 1):
            model.train()
            totals = []
            for mode in (AvailabilityMode.FULL_AH, AvailabilityMode.FALLBACK_VA, AvailabilityMode.FALLBACK_A):
                order = np.random.default_rng(int(config["seed"]) + epoch * 10 + len(mode.value)).permutation(len(train["u_c"]))
                source = sources(train, mode)
                prediction = predictions["train"][mode]
                for start in range(0, len(order), int(spec["batch_size"])):
                    index = order[start:start + int(spec["batch_size"])]
                    p = torch.from_numpy(np.array(prediction[index], copy=True)).to(device)
                    s = torch.from_numpy(np.array(source[index], copy=True)).to(device)
                    target = torch.from_numpy(np.array(train["u_c"][index], copy=True)).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    log_variance = model(mode, p, s)
                    loss = heteroscedastic_nll(log_variance, p, target)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite C4 uncertainty loss")
                    loss.backward()
                    if any(parameter.grad is not None for mean in models.values() for parameter in mean.parameters()):
                        raise RuntimeError("STRUCTURAL_FAIL: uncertainty gradient reached mean predictor")
                    optimizer.step()
                    totals.append(float(loss.detach()))
            validation_nll = []
            for mode in (AvailabilityMode.FULL_AH, AvailabilityMode.FALLBACK_VA, AvailabilityMode.FALLBACK_A):
                prediction = predictions["validation"][mode]
                error = row_mse(prediction, validation["u_c"])
                log_variance = uncertainty_numpy(model, mode, prediction, sources(validation, mode), device, int(spec["batch_size"]))
                validation_nll.append(float(nll_rows(error, np.exp(log_variance)).mean()))
            score = float(np.mean(validation_nll))
            history.append({"epoch": epoch, "train_nll": float(np.mean(totals)), "validation_nll": dict(zip(("FULL_AH", "FALLBACK_VA", "FALLBACK_A"), validation_nll)), "mean_validation_nll": score})
            if best is None or score < best["mean_validation_nll"]:
                digest = save_uncertainty_checkpoint(checkpoint, model, {
                    "epoch": epoch, "mean_validation_nll": score,
                    "selection_split": "validation only", "test_loaded": False,
                    "mean_predictors_frozen": True,
                })
                best = {"epoch": epoch, "mean_validation_nll": score, "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": digest}
                stale = 0
            else:
                stale += 1
            atomic_json(experiments / "uncertainty" / "history.json", {"history": history, "best": best, "test_loaded": False})
            if stale >= int(spec["patience"]):
                break
        model, metadata = load_uncertainty_checkpoint(ROOT / best["checkpoint"], device)
        model.eval().requires_grad_(False)
        raw_log_variance = {}
        errors = {}
        for mode in (AvailabilityMode.FULL_AH, AvailabilityMode.FALLBACK_VA, AvailabilityMode.FALLBACK_A):
            prediction = predictions["validation"][mode]
            errors[mode] = row_mse(prediction, validation["u_c"])
            raw_log_variance[mode] = uncertainty_numpy(model, mode, prediction, sources(validation, mode), device, int(spec["batch_size"]))
        numerator = np.concatenate([errors[mode] / np.exp(raw_log_variance[mode]) for mode in errors])
        calibration_scale = float(np.mean(numerator))
        fallback_mode = AvailabilityMode.FALLBACK_VA if fallback.source == "VA" else AvailabilityMode.FALLBACK_A
        high_error_threshold = float(np.quantile(errors[fallback_mode], float(spec["high_error_quantile"])))
        constant_variance = float(np.mean(np.concatenate(list(errors.values()))))
        metrics = {}
        for mode in errors:
            variance = np.exp(raw_log_variance[mode]) * calibration_scale
            metrics[mode.value] = calibration_metrics(errors[mode], variance, high_error_threshold, constant_variance)
        selection = {
            "schema": "tactile3d-unit.vac-c4-uncertainty-selection.v1",
            "model": "ContactUncertaintyEstimator", "parameters": model.parameter_count(),
            "checkpoint": best["checkpoint"], "checkpoint_sha256": best["checkpoint_sha256"],
            "epoch": best["epoch"], "calibration_scale": calibration_scale,
            "common_scale_across_modes": True, "validation": metrics,
            "high_error_threshold": high_error_threshold,
            "high_error_threshold_definition": "selected canonical fallback validation shared-error 75th percentile",
            "constant_variance": constant_variance,
            "canonical_fallback_mode": fallback_mode.value,
            "emergency_a_checkpoint": a_row["best"]["checkpoint"],
            "emergency_a_checkpoint_sha256": a_row["best"]["checkpoint_sha256"],
            "mean_predictors_frozen": True, "fallback_selection_sha256": fallback_selection_sha,
            "selected_via": "VALIDATION ONLY", "selection_split": "validation only",
            "test_loaded": False, "identity": identity_snapshot(config),
            "gpu": {**gpu, "preferred_physical": 1, "fallback": gpu.get("actual_physical") != 1},
        }
        atomic_json(artifacts / "uncertainty_training_summary.json", {
            "schema": "tactile3d-unit.vac-c4-uncertainty-training.v1",
            "history": history, "best": best, "selection_split": "validation only",
            "test_loaded": False, "mean_predictors_frozen": True,
        })
        path = artifacts / "uncertainty_selection.json"
        atomic_json(path, selection)
        digest = sha256_file(path)
        (artifacts / "uncertainty_selection.sha256").write_text(digest + "  uncertainty_selection.json\n")
        if not identity_snapshot(config)["pass"]:
            raise RuntimeError("STRUCTURAL_FAIL: frozen identity changed during uncertainty training")
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    main()
