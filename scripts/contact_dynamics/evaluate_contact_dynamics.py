#!/usr/bin/env python3
"""Evaluate S2/M2 reconstruction, controls, probes, and collapse on test data."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap

from gr00t.contact_dynamics.evaluation import (
    different_episode_permutation,
    query_diversity,
    transition_metrics,
)
from gr00t.contact_dynamics.models import (
    ContactDynamicsEncoder,
    ContactDynamicsModel,
    CurrentOnlyPredictor,
    DeltaMLPEncoder,
    LatentTransitionDecoder,
)
from gr00t.tactile_teacher.evaluation import classification_metrics, collapse_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".local/cache/contact_dynamics/s2_transition_pairs"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(".local/experiments/contact_dynamics/s2_models"),
    )
    parser.add_argument(
        "--code-cache-dir",
        type=Path,
        default=Path(".local/cache/contact_dynamics/s2_codes"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".local/artifacts/contact_dynamics/s2_4"),
    )
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(cache_dir: Path, split: str) -> dict[str, np.ndarray]:
    names = (
        "current",
        "future",
        "episode_id",
        "primitive_id",
        "object_id",
        "current_force",
        "future_force",
        "contact_transition",
        "force_trend_class",
        "finger_change",
        "dynamic",
    )
    return {
        name: np.load(cache_dir / split / f"{name}.npy", mmap_mode="r")
        for name in names
    }


def build_model(name: str) -> torch.nn.Module:
    if name == "C1":
        return CurrentOnlyPredictor()
    if name == "C2":
        return ContactDynamicsModel(DeltaMLPEncoder(), LatentTransitionDecoder())
    if name == "proposed":
        return ContactDynamicsModel(ContactDynamicsEncoder(), LatentTransitionDecoder())
    raise ValueError(name)


def load_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(str(checkpoint["model"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().requires_grad_(False).to(device)
    return model, checkpoint


@torch.inference_mode()
def predict_model(
    model: torch.nn.Module,
    name: str,
    current: np.ndarray,
    future: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    result = np.empty_like(np.asarray(future), dtype=np.float32)
    for start in range(0, len(current), batch_size):
        stop = min(start + batch_size, len(current))
        c = torch.from_numpy(np.array(current[start:stop], copy=True)).to(device)
        f = torch.from_numpy(np.array(future[start:stop], copy=True)).to(device)
        if name == "C1":
            prediction = model(c)
        else:
            prediction = model(c, f)["future"]
        result[start:stop] = prediction.float().cpu().numpy()
    return result


@torch.inference_mode()
def extract_codes(
    model: ContactDynamicsModel,
    arrays: dict[str, np.ndarray],
    path: Path,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    code = open_memmap(path, mode="w+", dtype=np.float32, shape=(len(arrays["current"]), 8, 32))
    for start in range(0, len(code), batch_size):
        stop = min(start + batch_size, len(code))
        current = torch.from_numpy(np.array(arrays["current"][start:stop], copy=True)).to(device)
        future = torch.from_numpy(np.array(arrays["future"][start:stop], copy=True)).to(device)
        code[start:stop] = model.encoder(current, future).float().cpu().numpy()
    code.flush()
    del code
    return np.load(path, mmap_mode="r")


@torch.inference_mode()
def decode_codes(
    model: ContactDynamicsModel,
    code: np.ndarray,
    current: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    result = np.empty((len(current), 256), dtype=np.float32)
    for start in range(0, len(current), batch_size):
        stop = min(start + batch_size, len(current))
        z = torch.from_numpy(np.array(code[start:stop], copy=True)).to(device)
        c = torch.from_numpy(np.array(current[start:stop], copy=True)).to(device)
        result[start:stop] = model.decoder(z, c).float().cpu().numpy()
    return result


@torch.inference_mode()
def encode_control(
    model: ContactDynamicsModel,
    current: np.ndarray,
    future: np.ndarray,
    device: torch.device,
    batch_size: int,
    *,
    reversed_pair: bool = False,
) -> np.ndarray:
    result = np.empty((len(current), 8, 32), dtype=np.float32)
    for start in range(0, len(current), batch_size):
        stop = min(start + batch_size, len(current))
        c = torch.from_numpy(np.array(current[start:stop], copy=True)).to(device)
        f = torch.from_numpy(np.array(future[start:stop], copy=True)).to(device)
        result[start:stop] = (
            model.encoder(f, c) if reversed_pair else model.encoder(c, f)
        ).float().cpu().numpy()
    return result


def metric_bundle(
    current: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    dynamic: np.ndarray,
) -> dict:
    return {
        "all": transition_metrics(current, target, prediction),
        "dynamic": transition_metrics(current, target, prediction, dynamic),
    }


def fit_ridge(
    feature: np.ndarray,
    labels: np.ndarray,
    output_dim: int,
    device: torch.device,
    batch_size: int,
    ridge: float,
    *,
    classes: int | None = None,
) -> dict[str, np.ndarray]:
    feature = feature.reshape(len(feature), -1)
    mean = np.asarray(feature).mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.asarray(feature).std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-5)
    dim = feature.shape[1] + 1
    gram = torch.zeros((dim, dim), device=device)
    cross = torch.zeros((dim, output_dim), device=device)
    mean_t = torch.from_numpy(mean).to(device)
    std_t = torch.from_numpy(std).to(device)
    for start in range(0, len(feature), batch_size):
        stop = min(start + batch_size, len(feature))
        value = torch.from_numpy(np.array(feature[start:stop], copy=True)).to(device)
        value = (value - mean_t) / std_t
        value = torch.cat([value, torch.ones((len(value), 1), device=device)], dim=1)
        target = torch.from_numpy(np.array(labels[start:stop], copy=True)).long().to(device)
        if classes is not None:
            target = F.one_hot(target, classes).reshape(len(target), -1)
        target = target.float().reshape(len(target), output_dim)
        gram.add_(value.T @ value)
        cross.add_(value.T @ target)
    penalty = torch.eye(dim, device=device) * ridge
    penalty[-1, -1] = 0
    weights = torch.linalg.solve(gram + penalty, cross)
    return {"mean": mean, "std": std, "weights": weights.cpu().numpy()}


def apply_ridge(
    probe: dict[str, np.ndarray],
    feature: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    feature = feature.reshape(len(feature), -1)
    result = np.empty((len(feature), probe["weights"].shape[1]), dtype=np.float32)
    mean = torch.from_numpy(probe["mean"]).to(device)
    std = torch.from_numpy(probe["std"]).to(device)
    weights = torch.from_numpy(probe["weights"]).to(device)
    for start in range(0, len(feature), batch_size):
        stop = min(start + batch_size, len(feature))
        value = torch.from_numpy(np.array(feature[start:stop], copy=True)).to(device)
        value = (value - mean) / std
        value = torch.cat([value, torch.ones((len(value), 1), device=device)], dim=1)
        result[start:stop] = (value @ weights).cpu().numpy()
    return result


def probe_suite(
    train_feature: np.ndarray,
    test_feature: np.ndarray,
    train: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    manifest: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    definitions = {
        "contact_transition": ("contact_transition", 4, 1),
        "force_trend": ("force_trend_class", 3, 1),
        "per_finger_change": ("finger_change", 3, 10),
        "primitive": ("primitive_id", len(manifest["primitive_labels"]), 1),
        "object": ("object_id", len(manifest["object_labels"]), 1),
    }
    result = {}
    for name, (key, classes, fields) in definitions.items():
        train_labels = np.asarray(train[key])
        test_labels = np.asarray(test[key])
        probe = fit_ridge(
            train_feature,
            train_labels,
            classes * fields,
            device,
            args.batch_size,
            args.ridge,
            classes=classes,
        )
        scores = apply_ridge(probe, test_feature, device, args.batch_size)
        if fields == 1:
            prediction = scores.reshape(len(scores), classes).argmax(axis=1)
            metrics = classification_metrics(test_labels, prediction)
            majority = int(np.bincount(train_labels, minlength=classes).argmax())
            majority_metrics = classification_metrics(
                test_labels, np.full(len(test_labels), majority, dtype=np.int64)
            )
        else:
            prediction = scores.reshape(len(scores), fields, classes).argmax(axis=2)
            metrics = {
                "micro_accuracy": float(np.mean(prediction == test_labels)),
                "change_only_accuracy": float(
                    np.mean(prediction[test_labels != 1] == test_labels[test_labels != 1])
                ),
            }
            majority = np.stack(
                [
                    np.bincount(train_labels[:, index], minlength=classes).argmax()
                    for index in range(fields)
                ]
            )
            majority_prediction = np.broadcast_to(majority, test_labels.shape)
            metrics["majority_micro_accuracy"] = float(
                np.mean(majority_prediction == test_labels)
            )
            majority_metrics = {"micro_accuracy": metrics["majority_micro_accuracy"]}
        label_type = "ACTUAL METADATA" if name in ("primitive", "object") else "DERIVED"
        result[name] = {
            **metrics,
            "majority": majority_metrics,
            "label_type": label_type,
        }
    return result


def main() -> int:
    args = parse_args()
    start_time = time.monotonic()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.code_cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    training = json.loads((args.model_dir / "s2_training_summary.json").read_text())
    train = load_split(args.cache_dir, "train")
    test = load_split(args.cache_dir, "test")
    models = {}
    checkpoints = {}
    checkpoint_identity = {}
    for name in ("C1", "C2", "proposed"):
        path = args.model_dir / f"{name.lower()}_best.pt"
        models[name], checkpoints[name] = load_model(path, device)
        checkpoint_identity[name] = {
            "sha256": sha256(path),
            "epoch": int(checkpoints[name]["epoch"]),
            "validation": checkpoints[name]["validation"],
            "parameters": sum(parameter.numel() for parameter in models[name].parameters()),
        }
    current = np.asarray(test["current"])
    target = np.asarray(test["future"])
    dynamic = np.asarray(test["dynamic"], dtype=bool)
    baseline_predictions = {"C0": current.copy()}
    for name in ("C1", "C2", "proposed"):
        baseline_predictions[name] = predict_model(
            models[name], name, test["current"], test["future"], device, args.batch_size
        )
    baselines = {
        name: metric_bundle(current, target, prediction, dynamic)
        for name, prediction in baseline_predictions.items()
    }
    proposed = models["proposed"]
    if not isinstance(proposed, ContactDynamicsModel):
        raise AssertionError("proposed checkpoint has wrong model type")
    train_code = extract_codes(
        proposed, train, args.code_cache_dir / "train.npy", device, args.batch_size
    )
    test_code = extract_codes(
        proposed, test, args.code_cache_dir / "test.npy", device, args.batch_size
    )
    permutation = different_episode_permutation(test["episode_id"], seed=args.seed)
    zero_prediction = decode_codes(
        proposed, np.zeros_like(test_code), test["current"], device, args.batch_size
    )
    shuffled_prediction = decode_codes(
        proposed, np.asarray(test_code)[permutation], test["current"], device, args.batch_size
    )
    reversed_code = encode_control(
        proposed, test["current"], test["future"], device, args.batch_size, reversed_pair=True
    )
    reversed_prediction = decode_codes(
        proposed, reversed_code, test["current"], device, args.batch_size
    )
    shuffled_future_code = encode_control(
        proposed,
        test["current"],
        np.asarray(test["future"])[permutation],
        device,
        args.batch_size,
    )
    shuffled_future_prediction = decode_codes(
        proposed, shuffled_future_code, test["current"], device, args.batch_size
    )
    ablation_predictions = {
        "full": baseline_predictions["proposed"],
        "zero": zero_prediction,
        "shuffled_code": shuffled_prediction,
        "reversed_transition": reversed_prediction,
        "shuffled_future": shuffled_future_prediction,
    }
    ablations = {
        name: metric_bundle(current, target, prediction, dynamic)
        for name, prediction in ablation_predictions.items()
    }
    probes = {
        "transition_code": probe_suite(
            train_code, test_code, train, test, manifest, args, device
        ),
        "current_state": probe_suite(
            train["current"], test["current"], train, test, manifest, args, device
        ),
    }
    flattened_code = np.asarray(test_code).reshape(len(test_code), -1)
    collapse = {
        "flattened_8x32": collapse_diagnostics(flattened_code, seed=args.seed),
        "pooled_32": collapse_diagnostics(np.asarray(test_code).mean(axis=1), seed=args.seed),
        "token_level_32": collapse_diagnostics(
            np.asarray(test_code).reshape(-1, 32), seed=args.seed
        ),
        "query_diversity": query_diversity(test_code),
        "query_norm_mean": np.linalg.norm(np.asarray(test_code), axis=2).mean(axis=0).tolist(),
    }
    with torch.inference_mode():
        fixed_current = torch.from_numpy(np.array(test["current"][:32], copy=True)).to(device)
        fixed_future = torch.from_numpy(np.array(test["future"][:32], copy=True)).to(device)
        first_code = proposed.encoder(fixed_current, fixed_future)
        second_code = proposed.encoder(fixed_current, fixed_future)
        first_decoded = proposed.decoder(first_code, fixed_current)
        second_decoded = proposed.decoder(second_code, fixed_current)
    determinism = {
        "code_exact_equal": bool(torch.equal(first_code, second_code)),
        "decoded_exact_equal": bool(torch.equal(first_decoded, second_decoded)),
        "code_allclose": bool(torch.allclose(first_code, second_code)),
        "decoded_allclose": bool(torch.allclose(first_decoded, second_decoded)),
    }
    sample_count = min(5000, len(test_code))
    sample_rng = np.random.default_rng(args.seed)
    sample_index = np.sort(sample_rng.choice(len(test_code), sample_count, replace=False))
    np.savez_compressed(
        args.output_dir / "visualization_data.npz",
        index=sample_index,
        code=np.asarray(test_code)[sample_index],
        current=current[sample_index],
        target=target[sample_index],
        full=ablation_predictions["full"][sample_index],
        zero=zero_prediction[sample_index],
        shuffled=shuffled_prediction[sample_index],
        reversed=reversed_prediction[sample_index],
        dynamic=dynamic[sample_index],
        contact_transition=np.asarray(test["contact_transition"])[sample_index],
        force_delta=(
            np.asarray(test["future_force"])[sample_index]
            - np.asarray(test["current_force"])[sample_index]
        ),
        primitive_id=np.asarray(test["primitive_id"])[sample_index],
        object_id=np.asarray(test["object_id"])[sample_index],
    )
    full_dynamic = ablations["full"]["dynamic"]["future_mse"]
    semantic_candidates = (
        probes["transition_code"]["contact_transition"]["macro_f1"],
        probes["transition_code"]["force_trend"]["macro_f1"],
        probes["transition_code"]["per_finger_change"]["change_only_accuracy"],
    )
    majority_candidates = (
        probes["transition_code"]["contact_transition"]["majority"]["macro_f1"],
        probes["transition_code"]["force_trend"]["majority"]["macro_f1"],
        0.0,
    )
    gates = {
        "future_reconstruction_finite": bool(
            np.isfinite(ablations["full"]["all"]["future_mse"])
        ),
        "full_beats_zero_dynamic": bool(
            full_dynamic < ablations["zero"]["dynamic"]["future_mse"]
        ),
        "full_beats_shuffled_code_dynamic": bool(
            full_dynamic < ablations["shuffled_code"]["dynamic"]["future_mse"]
        ),
        "correct_beats_reversed_or_shuffled_future_dynamic": bool(
            full_dynamic < ablations["reversed_transition"]["dynamic"]["future_mse"]
            or full_dynamic < ablations["shuffled_future"]["dynamic"]["future_mse"]
        ),
        "dynamic_semantics_above_majority": bool(
            any(value > baseline for value, baseline in zip(semantic_candidates, majority_candidates))
        ),
        "non_collapse": bool(
            collapse["flattened_8x32"]["per_dimension_variance"]["near_zero_fraction"] < 0.95
            and collapse["flattened_8x32"]["effective_rank"] > 1
            and collapse["query_diversity"]["collapsed_sample_fraction"] < 0.01
        ),
        "interface_8x32": tuple(test_code.shape[1:]) == (8, 32),
        "deterministic": all(determinism.values()),
    }
    summary = {
        "schema": "tactile3d-unit.s2-contact-dynamics-evaluation.v1",
        "seed": args.seed,
        "test_windows": len(test_code),
        "dynamic_test_windows": int(dynamic.sum()),
        "checkpoint_identity": checkpoint_identity,
        "selected_lambda_delta": training["selected_lambda_delta"],
        "baselines": baselines,
        "ablations": ablations,
        "probes": probes,
        "collapse": collapse,
        "determinism": determinism,
        "horizon_audit": manifest["horizon_audit"]["test"],
        "gates": gates,
        "proposed_vs_delta": {
            "proposed_dynamic_mse": baselines["proposed"]["dynamic"]["future_mse"],
            "C2_dynamic_mse": baselines["C2"]["dynamic"]["future_mse"],
            "proposed_better": bool(
                baselines["proposed"]["dynamic"]["future_mse"]
                < baselines["C2"]["dynamic"]["future_mse"]
            ),
        },
        "visualization_data": str(args.output_dir / "visualization_data.npz"),
        "evaluation_seconds": time.monotonic() - start_time,
        "status": "PASS" if all(gates.values()) else "FAIL",
    }
    write_json(args.output_dir / "s2_evaluation_summary.json", summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "baselines": baselines,
                "ablations": ablations,
                "proposed_vs_delta": summary["proposed_vs_delta"],
                "gates": gates,
            },
            indent=2,
        )
    )
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
