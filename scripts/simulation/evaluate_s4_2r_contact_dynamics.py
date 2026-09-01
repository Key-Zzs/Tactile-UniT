#!/usr/bin/env python3
"""Validate S4.2-3 Contact-Dynamics on validation only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.contact_dynamics.evaluation import (  # noqa: E402
    different_episode_permutation,
    query_diversity,
)
from gr00t.contact_dynamics.models import (  # noqa: E402
    ContactDynamicsEncoder,
    ContactDynamicsModel,
    CurrentOnlyPredictor,
    DeltaMLPEncoder,
    LatentTransitionDecoder,
)
from gr00t.simulation.intrinsic_dimension import dimension_metrics  # noqa: E402
from gr00t.simulation.s4_2_dataset import sha256_file  # noqa: E402

CACHE_ROOT = ROOT / ".local/cache/simulation/s4_2r/contact_dynamics"
EXPERIMENT_ROOT = ROOT / ".local/experiments/simulation/s4_2r/contact_dynamics"
CODE_ROOT = ROOT / ".local/cache/simulation/s4_2r/contact_dynamics_codes"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_2r"
CONTRACT = ROOT / "configs/simulation/s4_2_representation_contract.json"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def build_model(name: str) -> torch.nn.Module:
    if name == "C1":
        return CurrentOnlyPredictor()
    if name == "C2":
        return ContactDynamicsModel(DeltaMLPEncoder(), LatentTransitionDecoder())
    if name == "C3":
        return ContactDynamicsModel(ContactDynamicsEncoder(), LatentTransitionDecoder())
    raise ValueError(name)


def load_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["schema"] != "tactile3d-unit.s4-2r-contact-dynamics-checkpoint.v1":
        raise ValueError("unsupported S4.2-R Contact-Dynamics checkpoint")
    model = build_model(checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    name: str,
    current: np.ndarray,
    future: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    predictions = []
    codes = []
    for start in range(0, len(current), batch_size):
        stop = min(start + batch_size, len(current))
        current_t = torch.from_numpy(current[start:stop]).to(device)
        future_t = torch.from_numpy(future[start:stop]).to(device)
        if name == "C1":
            predictions.append(model(current_t).cpu().numpy())
        else:
            output = model(current_t, future_t)
            predictions.append(output["future"].cpu().numpy())
            codes.append(output["code"].cpu().numpy())
    return np.concatenate(predictions), None if not codes else np.concatenate(codes)


@torch.inference_mode()
def decode(
    model: ContactDynamicsModel,
    code: np.ndarray,
    current: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    result = []
    for start in range(0, len(code), batch_size):
        stop = min(start + batch_size, len(code))
        code_t = torch.from_numpy(code[start:stop]).to(device)
        current_t = torch.from_numpy(current[start:stop]).to(device)
        result.append(model.decoder(code_t, current_t).cpu().numpy())
    return np.concatenate(result)


def per_sample_mse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.mean(np.square(prediction - target), axis=1)


def bootstrap_mean_ci(values: np.ndarray, samples: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 100):
        stop = min(samples, start + 100)
        index = rng.integers(0, len(array), size=(stop - start, len(array)))
        means[start:stop] = array[index].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).astype(float).tolist()


def metric_bundle(
    prediction: np.ndarray, target: np.ndarray, dynamic: np.ndarray
) -> dict[str, Any]:
    error = per_sample_mse(prediction, target)
    return {
        "overall_future_mse": float(error.mean()),
        "dynamic_future_mse": float(error[dynamic].mean()),
        "dynamic_windows": int(dynamic.sum()),
    }


def probe_metrics(
    train_x: np.ndarray,
    val_x: np.ndarray,
    train_y: np.ndarray,
    val_y: np.ndarray,
) -> dict[str, Any]:
    train_classes = np.unique(train_y)
    validation_classes = np.unique(val_y)
    if len(train_classes) < 2:
        return {
            "usable": False,
            "reason": "single_class_train_label",
            "train_classes": train_classes.astype(int).tolist(),
            "validation_classes": validation_classes.astype(int).tolist(),
        }
    model = RidgeClassifier(alpha=10.0).fit(train_x, train_y)
    prediction = model.predict(val_x)
    return {
        "usable": True,
        "macro_f1": float(f1_score(val_y, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(val_y, prediction)),
        "train_classes": train_classes.astype(int).tolist(),
        "validation_classes": validation_classes.astype(int).tolist(),
    }


def probe_suite(
    train_code: np.ndarray,
    validation_code: np.ndarray,
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
) -> dict[str, Any]:
    train_x = train_code.reshape(len(train_code), -1)
    val_x = validation_code.reshape(len(validation_code), -1)
    result = {
        "contact_transition": probe_metrics(
            train_x,
            val_x,
            train["contact_transition"],
            validation["contact_transition"],
        ),
        "force_trend": probe_metrics(
            train_x, val_x, train["force_trend"], validation["force_trend"]
        ),
        "per_region_contact_change": [],
        "by_contact_regime": {},
    }
    for region in range(5):
        result["per_region_contact_change"].append(
            {
                "region": region,
                **probe_metrics(
                    train_x,
                    val_x,
                    train["region_contact_change"][:, region],
                    validation["region_contact_change"][:, region],
                ),
            }
        )
    for code, name in enumerate(
        ("free_to_free", "free_to_contact", "contact_to_free", "contact_to_contact")
    ):
        mask = validation["contact_transition"] == code
        result["by_contact_regime"][name] = {
            "windows": int(mask.sum()),
            "code_effective_rank": dimension_metrics(
                validation_code[mask].reshape(mask.sum(), -1),
                twonn=False,
                include_spectrum=False,
            )["effective_rank"],
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--code-root", type=Path, default=CODE_ROOT)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    selection = json.loads(
        (args.artifacts / "contact_dynamics_selection.json").read_text(encoding="utf-8")
    )
    train = load_npz(args.cache_root / "train.npz")
    validation = load_npz(args.cache_root / "validation.npz")
    device = torch.device(args.device)
    paths = {
        "C1": ROOT / next(row["checkpoint"] for row in selection["trials"] if row["model"] == "C1"),
        "C2": ROOT / next(row["checkpoint"] for row in selection["trials"] if row["model"] == "C2"),
        "C3": args.experiment_root / "selected.pt",
    }
    models = {}
    checkpoints = {}
    for name, path in paths.items():
        models[name], checkpoints[name] = load_model(path, device)
    current = validation["h_current"]
    target = validation["h_future"]
    dynamic = validation["dynamic"].astype(bool)
    predictions = {"C0": current.copy()}
    codes = {}
    for name in ("C1", "C2", "C3"):
        predictions[name], code = predict(
            models[name], name, current, target, device, args.batch_size
        )
        if code is not None:
            codes[name] = code
    baselines = {
        name: metric_bundle(prediction, target, dynamic)
        for name, prediction in predictions.items()
    }
    strongest_name = min(("C0", "C1", "C2"), key=lambda name: baselines[name]["dynamic_future_mse"])
    strongest_error = per_sample_mse(predictions[strongest_name], target)[dynamic]
    proposed_error = per_sample_mse(predictions["C3"], target)[dynamic]
    baseline_difference = strongest_error - proposed_error
    improvement = 1.0 - float(proposed_error.mean() / strongest_error.mean())
    improvement_ci = bootstrap_mean_ci(
        baseline_difference, int(contract["bootstrap_samples"]), int(contract["seed"]) + 300
    )

    proposed = models["C3"]
    if not isinstance(proposed, ContactDynamicsModel):
        raise AssertionError("selected model is not ContactDynamicsModel")
    code = codes["C3"]
    permutation = different_episode_permutation(validation["episode_id"], seed=4242)
    zero_prediction = decode(proposed, np.zeros_like(code), current, device, args.batch_size)
    different_prediction = decode(proposed, code[permutation], current, device, args.batch_size)
    reversed_prediction, reversed_code = predict(
        proposed, "C3", target, current, device, args.batch_size
    )
    reversed_prediction = decode(
        proposed, reversed_code, current, device, args.batch_size
    )
    _, mismatch_code = predict(
        proposed, "C3", current, target[permutation], device, args.batch_size
    )
    mismatch_prediction = decode(
        proposed, mismatch_code, current, device, args.batch_size
    )
    controls_predictions = {
        "zero": zero_prediction,
        "different_episode": different_prediction,
        "reversed": reversed_prediction,
        "mismatched_future": mismatch_prediction,
    }
    controls = {}
    full_dynamic_error = per_sample_mse(predictions["C3"], target)[dynamic]
    for index, (name, prediction) in enumerate(controls_predictions.items()):
        control_error = per_sample_mse(prediction, target)[dynamic]
        difference = control_error - full_dynamic_error
        controls[name] = {
            "full_mse": float(full_dynamic_error.mean()),
            "control_mse": float(control_error.mean()),
            "improvement_ci95": bootstrap_mean_ci(
                difference,
                int(contract["bootstrap_samples"]),
                int(contract["seed"]) + 320 + index,
            ),
        }
    flat_code = code.reshape(len(code), -1)
    collapse = dimension_metrics(flat_code, twonn=True)
    diversity = query_diversity(code)
    token_norms = np.linalg.norm(code, axis=2)
    collapse.update(
        {
            "query_diversity": diversity,
            "query_token_norm_mean": token_norms.mean(axis=0).astype(float).tolist(),
            "query_token_norm_min": token_norms.min(axis=0).astype(float).tolist(),
            "collapsed_query_fraction": float(
                np.mean(np.var(code, axis=(0, 2)) < 1e-8)
            ),
        }
    )
    train_prediction, train_code = predict(
        proposed,
        "C3",
        train["h_current"],
        train["h_future"],
        device,
        args.batch_size,
    )
    _ = train_prediction
    probes = probe_suite(train_code, code, train, validation)
    first_model, _ = load_model(paths["C3"], device)
    second_model, _ = load_model(paths["C3"], device)
    first_prediction, first_code = predict(
        first_model, "C3", current[:128], target[:128], device, 128
    )
    second_prediction, second_code = predict(
        second_model, "C3", current[:128], target[:128], device, 128
    )
    deterministic = bool(
        np.array_equal(first_prediction, second_prediction)
        and np.array_equal(first_code, second_code)
    )
    gates = {
        "dynamic_improvement_at_least_10_percent": improvement
        >= float(contract["contact_dynamics"]["dynamic_improvement"]),
        "dynamic_improvement_bootstrap_lower_positive": improvement_ci[0] > 0.0,
        "full_beats_all_controls": all(
            row["control_mse"] > row["full_mse"] and row["improvement_ci95"][0] > 0.0
            for row in controls.values()
        ),
        "finite": bool(
            np.isfinite(predictions["C3"]).all() and np.isfinite(code).all()
        ),
        "deterministic_reload": deterministic,
        "shape_8x32": tuple(code.shape[1:]) == (8, 32),
        "no_structural_collapse": bool(
            collapse["effective_rank"] > 1.0
            and collapse["near_zero_variance_fraction"] < 0.95
            and diversity["collapsed_sample_fraction"] < 0.01
            and collapse["collapsed_query_fraction"] < 0.01
        ),
    }
    decision = (
        "S4_2_3_CONTACT_DYNAMICS_READY"
        if all(gates.values())
        else "S4_2_3_CONTACT_DYNAMICS_FAIL"
    )
    args.code_root.mkdir(parents=True, exist_ok=True)
    code_manifest = {}
    for split, arrays, split_code in (
        ("train", train, train_code),
        ("validation", validation, code),
    ):
        destination = args.code_root / f"{split}.npz"
        np.savez_compressed(
            destination,
            pair_id=arrays["pair_id"],
            episode_id=arrays["episode_id"],
            task=arrays["task"],
            source_trajectory_id=arrays["source_trajectory_id"],
            anchor_step=arrays["anchor_step"],
            future_step=arrays["future_step"],
            z_c=split_code,
            h_current=arrays["h_current"],
            contact_transition=arrays["contact_transition"],
            dynamic=arrays["dynamic"],
        )
        code_manifest[split] = {
            "path": str(destination.relative_to(ROOT)),
            "sha256": sha256_file(destination),
            "shape": list(split_code.shape),
        }
    validation_result = {
        "schema": "tactile3d-unit.s4-2r-contact-dynamics-validation.v1",
        "decision": decision,
        "checkpoint": str(paths["C3"].relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(paths["C3"]),
        "selected": checkpoints["C3"]["candidate"],
        "parameters": sum(parameter.numel() for parameter in proposed.parameters()),
        "h_t_shape": [256],
        "z_c_shape": [8, 32],
        "baselines": baselines,
        "strongest_baseline": strongest_name,
        "dynamic_relative_improvement": improvement,
        "dynamic_improvement_ci95": improvement_ci,
        "controls": controls,
        "collapse": collapse,
        "probes": probes,
        "deterministic_reload": deterministic,
        "gates": gates,
        "code_cache": code_manifest,
        "test_loaded": False,
        "selection_uses_test": False,
    }
    (args.artifacts / "contact_dynamics_validation.json").write_text(
        json.dumps(validation_result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    labels = ["full", *controls]
    values = [
        baselines["C3"]["dynamic_future_mse"],
        *[controls[name]["control_mse"] for name in controls],
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values)
    ax.set(ylabel="Dynamic validation MSE", title="S4.2-3 Contact-Dynamics controls")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(args.artifacts / "contact_dynamics_controls.png", dpi=180)
    plt.close(fig)
    print(
        json.dumps(
            {
                "decision": decision,
                "strongest_baseline": strongest_name,
                "dynamic_relative_improvement": improvement,
                "dynamic_improvement_ci95": improvement_ci,
                "gates": gates,
                "test_loaded": False,
            },
            indent=2,
        )
    )
    if decision != "S4_2_3_CONTACT_DYNAMICS_READY":
        raise SystemExit(decision)


if __name__ == "__main__":
    main()
