#!/usr/bin/env python3
"""Evaluate conditional sufficiency, missing modalities, and uncertainty."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_2_formal import (  # noqa: E402
    ConditionalContactPredictor,
    ScalarLogVarianceHead,
)
from scripts.simulation.train_s4_2ds_bridge_pilot import classification_probe  # noqa: E402

CONFIG_PATH = ROOT / "configs/simulation/s4_2_formal_downstream.json"
PROTOCOL_PATH = ROOT / ".local/artifacts/simulation/s4_2_formal/protocol_freeze.json"
S4_2_6_PATH = ROOT / ".local/artifacts/simulation/s4_2_formal/s4_2_6_final.json"
PAIR_ROOT = ROOT / ".local/cache/simulation/s4_2_formal"
EXPERIMENT_ROOT = ROOT / ".local/experiments/simulation/s4_2_formal/s4_2_7"
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_2_formal"

PREDICTORS = {
    "A_plus_H": ("action", "history"),
    "V_plus_A_plus_H": ("vision", "action", "history"),
    "V_plus_A_missing_H": ("vision", "action"),
    "A_only_missing_H": ("action",),
}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def conditioning(
    pairs: dict[str, np.ndarray], shared: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    history = np.asarray(pairs["h_current"], dtype=np.float32).reshape(-1, 8, 32)
    return {
        "vision": shared["u_v"],
        "action": shared["u_a"],
        "history": history,
    }


@torch.inference_mode()
def predict(
    model: ConditionalContactPredictor,
    arrays: dict[str, np.ndarray],
    device: torch.device,
) -> np.ndarray:
    result = np.empty((len(next(iter(arrays.values()))), 8, 32), dtype=np.float32)
    model.eval()
    for start in range(0, len(result), 1024):
        stop = min(start + 1024, len(result))
        values = {
            name: torch.from_numpy(arrays[name][start:stop]).to(device) for name in model.modalities
        }
        result[start:stop] = model(values).float().cpu().numpy()
    return result


def train_predictor(
    name: str,
    train_values: dict[str, np.ndarray],
    validation_values: dict[str, np.ndarray],
    target_train: np.ndarray,
    target_validation: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], ConditionalContactPredictor, np.ndarray, np.ndarray]:
    seed_everything(int(config["seed"]))
    modalities = PREDICTORS[name]
    model = ConditionalContactPredictor(modalities).to(device)
    training = config["s4_2_7"]["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    rng = np.random.default_rng(int(config["seed"]))
    history = []
    model.train()
    for step in range(1, int(training["steps"]) + 1):
        rows = rng.choice(len(target_train), size=int(training["batch_size"]), replace=False)
        values = {
            modality: torch.from_numpy(train_values[modality][rows]).to(device)
            for modality in modalities
        }
        output = model(values)
        loss = F.mse_loss(output, torch.from_numpy(target_train[rows]).to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == int(training["steps"]):
            history.append({"step": step, "mse": float(loss.detach())})
            print(json.dumps({"predictor": name, **history[-1]}), flush=True)
    train_prediction = predict(model, train_values, device)
    validation_prediction = predict(model, validation_values, device)
    checkpoint = EXPERIMENT_ROOT / f"{name}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "tactile3d-unit.s4-2-7-conditional-mean.v1",
            "name": name,
            "modalities": modalities,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "formal_test_loaded": False,
        },
        checkpoint,
    )
    error = np.square(
        validation_prediction.astype(np.float64) - target_validation.astype(np.float64)
    ).mean(axis=(1, 2))
    result = {
        "name": name,
        "conditioning": list(modalities),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "validation_mse": float(error.mean()),
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_history": history,
        "formal_test_loaded": False,
    }
    return result, model, train_prediction, validation_prediction


@torch.inference_mode()
def infer_log_variance(
    model: ScalarLogVarianceHead,
    arrays: dict[str, np.ndarray],
    device: torch.device,
) -> np.ndarray:
    result = np.empty(len(next(iter(arrays.values()))), dtype=np.float32)
    model.eval()
    for start in range(0, len(result), 2048):
        stop = min(start + 2048, len(result))
        values = {
            name: torch.from_numpy(arrays[name][start:stop]).to(device) for name in model.modalities
        }
        result[start:stop] = model(values).float().cpu().numpy()
    return result


def gaussian_nll(error: np.ndarray, log_variance: np.ndarray) -> np.ndarray:
    return 0.5 * (log_variance + error / np.exp(log_variance))


def train_uncertainty(
    mode: str,
    modalities: tuple[str, ...],
    train_values: dict[str, np.ndarray],
    validation_values: dict[str, np.ndarray],
    train_error: np.ndarray,
    validation_error: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    seed_everything(int(config["seed"]))
    model = ScalarLogVarianceHead(modalities).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(int(config["seed"]))
    for step in range(1, 801):
        rows = rng.choice(len(train_error), size=512, replace=False)
        values = {
            name: torch.from_numpy(train_values[name][rows]).to(device) for name in modalities
        }
        log_variance = model(values)
        error = torch.from_numpy(train_error[rows].astype(np.float32)).to(device)
        loss = 0.5 * (log_variance + error / torch.exp(log_variance)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    raw_log_variance = infer_log_variance(model, validation_values, device).astype(np.float64)
    candidates = np.asarray([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0])
    calibration = []
    for scale in candidates:
        calibrated = raw_log_variance + np.log(scale)
        width = 1.6448536269514722 * np.sqrt(np.exp(calibrated))
        coverage = float(np.mean(np.sqrt(validation_error) <= width))
        calibration.append(
            {
                "variance_scale": float(scale),
                "nll": float(gaussian_nll(validation_error, calibrated).mean()),
                "coverage_90": coverage,
            }
        )
    selected = min(calibration, key=lambda value: (abs(value["coverage_90"] - 0.90), value["nll"]))
    constant_log_variance = np.log(max(float(train_error.mean()), 1e-12))
    constant_nll = float(
        gaussian_nll(validation_error, np.full_like(validation_error, constant_log_variance)).mean()
    )
    gates_config = config["s4_2_7"]["gates"]
    improvement = constant_nll - selected["nll"]
    gates = {
        "nll": improvement > float(gates_config["uncertainty_nll_improvement_over_constant_min"]),
        "coverage": float(gates_config["interval_90_coverage_min"])
        <= selected["coverage_90"]
        <= float(gates_config["interval_90_coverage_max"]),
    }
    checkpoint = EXPERIMENT_ROOT / f"uncertainty_{mode}.pt"
    torch.save(
        {
            "schema": "tactile3d-unit.s4-2-7-scalar-log-variance.v1",
            "mode": mode,
            "modalities": modalities,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "variance_scale": selected["variance_scale"],
            "formal_test_loaded": False,
        },
        checkpoint,
    )
    return {
        "mode": mode,
        "mean_predictor_frozen": True,
        "training_split": "train",
        "calibration_split": "validation",
        "calibration_candidates": calibration,
        "selected": selected,
        "constant_baseline_nll": constant_nll,
        "nll_improvement": improvement,
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "overall": "PASS" if all(gates.values()) else "FAIL",
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "formal_test_loaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config, protocol, prior = (
        load_json(CONFIG_PATH),
        load_json(PROTOCOL_PATH),
        load_json(S4_2_6_PATH),
    )
    if protocol["config_sha256"] != sha256_file(CONFIG_PATH):
        raise RuntimeError("formal protocol changed after freeze")
    if prior.get("S4_2_7_readiness") != "READY":
        raise RuntimeError("S4.2-6 dependency failed")
    train, validation = (
        load_npz(PAIR_ROOT / "paired_train.npz"),
        load_npz(PAIR_ROOT / "paired_validation.npz"),
    )
    shared_train, shared_validation = (
        load_npz(PAIR_ROOT / "shared_train.npz"),
        load_npz(PAIR_ROOT / "shared_validation.npz"),
    )
    if not np.array_equal(train["pair_id"], shared_train["pair_id"]) or not np.array_equal(
        validation["pair_id"], shared_validation["pair_id"]
    ):
        raise RuntimeError("S4.2-7 paired identity mismatch")
    train_values, validation_values = conditioning(train, shared_train), conditioning(
        validation, shared_validation
    )
    target_train, target_validation = shared_train["u_c"], shared_validation["u_c"]
    device = torch.device(args.device)
    results, models, train_predictions, validation_predictions = {}, {}, {}, {}
    for name in config["s4_2_7"]["predictors"]:
        (
            results[name],
            models[name],
            train_predictions[name],
            validation_predictions[name],
        ) = train_predictor(
            name,
            train_values,
            validation_values,
            target_train,
            target_validation,
            config,
            device,
        )
    full_improvement = (
        results["A_plus_H"]["validation_mse"] - results["V_plus_A_plus_H"]["validation_mse"]
    ) / max(results["A_plus_H"]["validation_mse"], 1e-12)
    missing_improvement = (
        results["A_only_missing_H"]["validation_mse"]
        - results["V_plus_A_missing_H"]["validation_mse"]
    ) / max(results["A_only_missing_H"]["validation_mse"], 1e-12)
    native_semantic = classification_probe(
        target_train,
        target_validation,
        train["contact_transition"],
        validation["contact_transition"],
        [0, 1, 2, 3],
    )
    missing_semantic = classification_probe(
        train_predictions["V_plus_A_missing_H"],
        validation_predictions["V_plus_A_missing_H"],
        train["contact_transition"],
        validation["contact_transition"],
        [0, 1, 2, 3],
    )
    semantic_retention = missing_semantic["macro_f1"] / max(native_semantic["macro_f1"], 1e-12)
    gates_config = config["s4_2_7"]["gates"]
    conditional_gates = {
        "full_conditioning": full_improvement
        >= float(gates_config["full_over_A_plus_H_relative_mse_improvement_min"]),
        "missing_modality_vision_utility": missing_improvement
        >= float(gates_config["missing_VA_over_A_relative_mse_improvement_min"]),
        "missing_contact_semantics": semantic_retention
        >= float(gates_config["missing_contact_macro_f1_retention_min"]),
        "finite": all(np.isfinite(value).all() for value in validation_predictions.values()),
    }
    conditional = {
        "schema": "tactile3d-unit.s4-2-7-conditional-sufficiency.v1",
        "predictors": results,
        "full_over_A_plus_H_relative_mse_improvement": full_improvement,
        "missing_VA_over_A_relative_mse_improvement": missing_improvement,
        "missing_contact_semantics": {
            "native": native_semantic,
            "predicted": missing_semantic,
            "retention": semantic_retention,
        },
        "gates": {name: "PASS" if value else "FAIL" for name, value in conditional_gates.items()},
        "overall": "PASS" if all(conditional_gates.values()) else "FAIL",
        "formal_test_loaded": False,
    }
    atomic_json(ARTIFACT_ROOT / "conditional_sufficiency.json", conditional)
    uncertainty = {}
    for mode, predictor_name in (
        ("full", "V_plus_A_plus_H"),
        ("missing_H", "V_plus_A_missing_H"),
    ):
        train_error = np.square(
            train_predictions[predictor_name].astype(np.float64) - target_train.astype(np.float64)
        ).mean(axis=(1, 2))
        validation_error = np.square(
            validation_predictions[predictor_name].astype(np.float64)
            - target_validation.astype(np.float64)
        ).mean(axis=(1, 2))
        uncertainty[mode] = train_uncertainty(
            mode,
            PREDICTORS[predictor_name],
            train_values,
            validation_values,
            train_error,
            validation_error,
            config,
            device,
        )
    uncertainty_result = {
        "schema": "tactile3d-unit.s4-2-7-uncertainty-evaluation.v1",
        "modes": uncertainty,
        "overall": (
            "PASS" if all(value["overall"] == "PASS" for value in uncertainty.values()) else "FAIL"
        ),
        "formal_test_loaded": False,
    }
    atomic_json(ARTIFACT_ROOT / "uncertainty_evaluation.json", uncertainty_result)
    if conditional["overall"] != "PASS" or uncertainty_result["overall"] != "PASS":
        final = {
            "schema": "tactile3d-unit.s4-2-7-final.v1",
            "decision": "S4_2_7_CONDITIONAL_OR_UNCERTAINTY_FAIL",
            "conditional": conditional["overall"],
            "uncertainty": uncertainty_result["overall"],
            "formal_test_loaded": False,
        }
        atomic_json(ARTIFACT_ROOT / "s4_2_7_final.json", final)
        raise SystemExit(final["decision"])
    np.savez_compressed(
        PAIR_ROOT / "conditional_validation.npz",
        pair_id=validation["pair_id"],
        target=target_validation,
        full=validation_predictions["V_plus_A_plus_H"],
        missing_H=validation_predictions["V_plus_A_missing_H"],
    )
    final = {
        "schema": "tactile3d-unit.s4-2-7-final.v1",
        "decision": "S4_2_7_CONDITIONAL_MISSING_MODALITY_UNCERTAINTY_ACCEPTED",
        "conditional": "PASS",
        "missing_modality": "PASS",
        "uncertainty": "PASS",
        "historical_10_percent_gate": "FAIL_UNCHANGED",
        "canonical_contact": "C3",
        "formal_test_loaded": False,
        "S4_2_8_readiness": "READY_FOR_PRETEST_FREEZE",
    }
    atomic_json(ARTIFACT_ROOT / "s4_2_7_final.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
