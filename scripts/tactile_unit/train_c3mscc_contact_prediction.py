#!/usr/bin/env python3
"""Run the bounded validation-only C3-MS-CC predictor trials and freeze selection."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_teacher.evaluation import classification_metrics  # noqa: E402
from gr00t.tactile_unit.c3mscc_contact_context import (  # noqa: E402
    C3MSCCLossWeights, ContactContextPredictor, contact_prediction_loss,
    load_checkpoint, save_checkpoint, sha256_file,
)
from gr00t.tactile_unit.c3r0_conditional_sufficiency import (  # noqa: E402
    evaluate_prediction, semantic_ratio,
)
from gr00t.tactile_unit.continuous_vac_shared_space import (  # noqa: E402
    bootstrap_mean_ci, different_episode_permutation, geometry_diagnostics,
    state_dict_digest,
)
from scripts.tactile_unit.c3mscc_runtime import (  # noqa: E402
    DEFAULT_CONFIG, atomic_json, identity_snapshot, load_aligned_split,
    load_config, load_frozen_shared_space,
)
from scripts.tactile_unit.continuous_contact_bridge_common import load_s2_model  # noqa: E402
from scripts.tactile_unit.vac_runtime_common import resolve_device, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-trials", type=int)
    return parser.parse_args()


def fit_probe(x: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(), RidgeClassifier(alpha=10.0, class_weight="balanced")
    )
    model.fit(np.asarray(x).reshape(len(x), -1), np.asarray(y, dtype=np.int64))
    return model


def majority(train_y: np.ndarray, rows: int, classes: int) -> np.ndarray:
    label = int(np.bincount(np.asarray(train_y, dtype=np.int64), minlength=classes).argmax())
    return np.full(rows, label, dtype=np.int64)


def predict_numpy(
    model: ContactContextPredictor,
    split: Mapping[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    *,
    u_a: np.ndarray | None = None,
    h_current: np.ndarray | None = None,
    u_v: np.ndarray | None = None,
) -> np.ndarray:
    action = np.asarray(split["u_a"] if u_a is None else u_a)
    context = np.asarray(split["h_current"] if h_current is None else h_current)
    vision = (
        np.asarray(split["u_v"] if u_v is None else u_v)
        if model.source == "VAH" else None
    )
    output = np.empty((len(action), 8, 32), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(action), batch_size):
            stop = min(start + batch_size, len(action))
            a = torch.from_numpy(np.array(action[start:stop], copy=True)).to(device)
            h = torch.from_numpy(np.array(context[start:stop], copy=True)).to(device)
            v = None if vision is None else torch.from_numpy(
                np.array(vision[start:stop], copy=True)
            ).to(device)
            output[start:stop] = model(a, h, v).float().cpu().numpy()
    return output


def row_mse(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.square(
        np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    ).reshape(len(left), -1).mean(1)


def physics_prediction(
    shared_space,
    decoder,
    shared: np.ndarray,
    h_current: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    output = np.empty((len(shared), 256), dtype=np.float32)
    shared_space.eval().requires_grad_(False)
    decoder.eval().requires_grad_(False)
    with torch.inference_mode():
        for start in range(0, len(shared), batch_size):
            stop = min(start + batch_size, len(shared))
            value = torch.from_numpy(np.array(shared[start:stop], copy=True)).to(device)
            current = torch.from_numpy(np.array(h_current[start:stop], copy=True)).to(device)
            output[start:stop] = decoder(
                shared_space.recover("contact", value), current
            ).float().cpu().numpy()
    return output


def semantic_evaluation(
    train_prediction: np.ndarray,
    validation_prediction: np.ndarray,
    train: Mapping[str, np.ndarray],
    validation: Mapping[str, np.ndarray],
    oracle: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    result = {}
    contact_prediction = np.empty(len(validation_prediction), dtype=np.int64)
    for metric, classes in (("contact_transition", 4), ("force_trend_class", 3)):
        probe = fit_probe(train_prediction, train[metric])
        prediction = np.asarray(
            probe.predict(validation_prediction.reshape(len(validation_prediction), -1)),
            dtype=np.int64,
        )
        if metric == "contact_transition":
            contact_prediction = prediction
        evaluated = evaluate_prediction(
            validation[metric], prediction,
            majority(train[metric], len(prediction), classes), classes,
        )
        evaluated["semantic_ratio"] = semantic_ratio(
            evaluated["macro_f1"], oracle[metric]["majority"]["macro_f1"],
            oracle[metric]["macro_f1"],
        )
        result[metric] = evaluated
    return result, contact_prediction


def oracle_semantics(train: Mapping[str, np.ndarray], validation: Mapping[str, np.ndarray]):
    result = {}
    for metric, classes in (("contact_transition", 4), ("force_trend_class", 3)):
        probe = fit_probe(train["u_c"], train[metric])
        prediction = probe.predict(np.asarray(validation["u_c"]).reshape(len(validation["u_c"]), -1))
        result[metric] = evaluate_prediction(
            validation[metric], prediction,
            majority(train[metric], len(prediction), classes), classes,
        )
    return result


def validation_metrics(
    model,
    train,
    validation,
    oracle,
    shared_space,
    decoder,
    config,
    device,
    batch_size,
    seed,
) -> dict[str, Any]:
    train_prediction = predict_numpy(model, train, device, batch_size)
    prediction = predict_numpy(model, validation, device, batch_size)
    semantic, contact_labels = semantic_evaluation(
        train_prediction, prediction, train, validation, oracle
    )
    target = np.asarray(validation["u_c"])
    mean_target = np.asarray(train["u_c"], dtype=np.float64).mean(0).astype(np.float32)
    mean_control = np.broadcast_to(mean_target, target.shape)
    prediction_error = row_mse(prediction, target)
    mean_error = row_mse(mean_control, target)
    shared_improvement = (mean_error - prediction_error) / max(float(mean_error.mean()), 1e-12)

    episode = np.asarray(validation["episode_id"])
    different = different_episode_permutation(episode, seed + 1)
    h_mean = np.asarray(train["h_current"], dtype=np.float64).mean(0).astype(np.float32)
    h_variants = {
        "zero": np.zeros_like(validation["h_current"]),
        "mean": np.broadcast_to(h_mean, np.asarray(validation["h_current"]).shape),
        "different_episode": np.asarray(validation["h_current"])[different],
        "time_shuffled": np.asarray(validation["h_current"])[np.random.default_rng(seed + 2).permutation(len(target))],
    }
    h_controls = {
        name: predict_numpy(model, validation, device, batch_size, h_current=value)
        for name, value in h_variants.items()
    }
    h_errors = {name: row_mse(value, target) for name, value in h_controls.items()}
    strongest_h = min(h_errors, key=lambda name: h_errors[name].mean())
    h_gain = (h_errors[strongest_h] - prediction_error) / max(
        float(h_errors[strongest_h].mean()), 1e-12
    )
    correct_contact_f1 = float(semantic["contact_transition"]["macro_f1"])
    invalid_h_contact = {}
    for name, value in h_controls.items():
        probe = fit_probe(train_prediction, train["contact_transition"])
        labels = probe.predict(value.reshape(len(value), -1))
        invalid_h_contact[name] = float(
            classification_metrics(validation["contact_transition"], labels)["macro_f1"]
        )

    # A-R raw reversal requires the immutable Original UniT tokenizer.  The local
    # surrogate is kept explicit and is never reported as satisfying the exact gate.
    reverse_surrogate = np.asarray(validation["u_a"])[:, ::-1].copy()
    shuffled_index = np.random.default_rng(seed + 3).permutation(len(target))
    shuffled_action = np.asarray(validation["u_a"])[shuffled_index]
    reversed_prediction = predict_numpy(
        model, validation, device, batch_size, u_a=reverse_surrogate
    )
    shuffled_prediction = predict_numpy(
        model, validation, device, batch_size, u_a=shuffled_action
    )
    dynamic = np.asarray(validation["dynamic"], dtype=bool)
    reversed_error = row_mse(reversed_prediction, target)
    shuffled_error = row_mse(shuffled_prediction, target)
    action_gain_rows = np.minimum(reversed_error, shuffled_error) - prediction_error
    action_gain = action_gain_rows[dynamic] / max(
        float(np.minimum(reversed_error, shuffled_error)[dynamic].mean()), 1e-12
    )

    oracle_future = physics_prediction(
        shared_space, decoder, target, validation["h_current"], device, batch_size
    )
    predicted_future = physics_prediction(
        shared_space, decoder, prediction, validation["h_current"], device, batch_size
    )
    mean_future = physics_prediction(
        shared_space, decoder, mean_control, validation["h_current"], device, batch_size
    )
    physics_error = row_mse(predicted_future, oracle_future)
    physics_control = row_mse(mean_future, oracle_future)
    physics_gain = (physics_control - physics_error) / max(float(physics_control.mean()), 1e-12)

    geometry = geometry_diagnostics(prediction)
    noncollapse = bool(
        geometry["per_dimension_variance"]["near_zero_fraction"] < 0.5
        and geometry["query_diversity"]["collapsed_pair_fraction"] < 0.5
    )
    weights = config["validation"]["utility"]
    future_change = float(semantic["contact_transition"]["future_change"]["macro_f1"])
    utility = (
        float(weights["contact_retention"]) * float(semantic["contact_transition"]["semantic_ratio"])
        + float(weights["force_retention"]) * float(semantic["force_trend_class"]["semantic_ratio"])
        + float(weights["future_change_macro_f1"]) * future_change
        + float(weights["shared_target_control_improvement"]) * float(shared_improvement.mean())
        + float(weights["shared_physics_control_improvement"]) * float(physics_gain.mean())
        + float(weights["action_temporal_sensitivity"]) * float(action_gain.mean())
        + float(weights["h_context_sensitivity"]) * float(h_gain.mean())
        + float(weights["noncollapse"]) * float(noncollapse)
        - float(weights["parameter_penalty_per_log10"]) * math.log10(model.parameter_summary()["total"])
        - (float(weights["vision_source_penalty"]) if model.source == "VAH" else 0.0)
    )
    bootstrap_samples = int(config["validation"]["bootstrap_samples"])
    contact_min = float(config["evaluation"]["contact_retention_min"])
    force_min = float(config["evaluation"]["force_retention_min"])
    gates = {
        "contact": semantic["contact_transition"]["semantic_ratio"] >= contact_min,
        "force": semantic["force_trend_class"]["semantic_ratio"] >= force_min,
        "shared_target": bootstrap_mean_ci(shared_improvement, samples=bootstrap_samples, seed=seed + 10)[0] > 0,
        "physics": bootstrap_mean_ci(physics_gain, samples=bootstrap_samples, seed=seed + 11)[0] > 0,
        "h_context_mse": bootstrap_mean_ci(h_gain, samples=bootstrap_samples, seed=seed + 12)[0] > 0,
        "h_context_contact": correct_contact_f1 > max(invalid_h_contact.values()),
        "action_surrogate": bootstrap_mean_ci(action_gain, samples=bootstrap_samples, seed=seed + 13)[0] > 0,
        "action_exact_ar": False,
        "noncollapse": noncollapse,
    }
    return {
        "selection_split": "validation only", "test_loaded": False,
        "rows": len(validation["u_c"]), "utility": float(utility),
        "semantic": semantic, "shared_target": {
            "prediction_mse": float(prediction_error.mean()),
            "mean_control_mse": float(mean_error.mean()),
            "normalized_improvement": float(shared_improvement.mean()),
            "improvement_ci95": bootstrap_mean_ci(shared_improvement, samples=bootstrap_samples, seed=seed + 10),
        },
        "physics": {
            "prediction_mse": float(physics_error.mean()),
            "mean_control_mse": float(physics_control.mean()),
            "normalized_improvement": float(physics_gain.mean()),
            "improvement_ci95": bootstrap_mean_ci(physics_gain, samples=bootstrap_samples, seed=seed + 11),
        },
        "h_context": {
            "strongest_control": strongest_h,
            "correct_mse": float(prediction_error.mean()),
            "controls_mse": {name: float(value.mean()) for name, value in h_errors.items()},
            "normalized_improvement": float(h_gain.mean()),
            "improvement_ci95": bootstrap_mean_ci(h_gain, samples=bootstrap_samples, seed=seed + 12),
            "correct_contact_f1": correct_contact_f1,
            "invalid_contact_f1": invalid_h_contact,
        },
        "action_temporal": {
            "method": "shared-token reversal surrogate; exact frozen A-R raw-action reversal unavailable",
            "exact_ar_transform": False,
            "correct_dynamic_mse": float(prediction_error[dynamic].mean()),
            "reversed_dynamic_mse": float(reversed_error[dynamic].mean()),
            "shuffled_dynamic_mse": float(shuffled_error[dynamic].mean()),
            "normalized_improvement": float(action_gain.mean()),
            "improvement_ci95": bootstrap_mean_ci(action_gain, samples=bootstrap_samples, seed=seed + 13),
        },
        "geometry": geometry, "noncollapse": noncollapse,
        "gates": {**gates, "all": all(gates.values())},
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    artifact_root = ROOT / config["runtime"]["artifact_root"]
    experiment_root = ROOT / config["runtime"]["experiment_root"]
    audit_path = artifact_root / "contract_audit.json"
    if not audit_path.is_file() or not json.loads(audit_path.read_text()).get("pass"):
        raise RuntimeError("STRUCTURAL_FAIL: run C3-MS-CC contract audit first")
    identities_before = identity_snapshot(config)
    if not identities_before["pass"]:
        raise RuntimeError("STRUCTURAL_FAIL: frozen identity mismatch")
    device, lock_handle, gpu = resolve_device(
        args.device, allowed_physical=tuple(str(value) for value in config["gpu"]["allowed_physical"])
    )
    gpu.update(
        {
            "preferred_physical": int(config["gpu"]["preferred_physical"]),
            "fallback": gpu["actual_physical"] != int(config["gpu"]["preferred_physical"]),
            "gpu1_authorization": config["gpu"].get("gpu1_authorization"),
        }
    )
    try:
        set_seed(int(config["seed"]))
        train = load_aligned_split(config, "train")
        validation = load_aligned_split(config, "validation")
        shared_space, _, shared_before = load_frozen_shared_space(config, device)
        s2 = load_s2_model(ROOT / config["runtime"]["s2_checkpoint"], device)
        decoder = s2.decoder.eval().requires_grad_(False)
        oracle = oracle_semantics(train, validation)
        trials = list(config["training"]["trials"])
        trials = trials[: min(len(trials), int(args.max_trials or len(trials)), 6)]
        artifact_root.mkdir(parents=True, exist_ok=True)
        experiment_root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema": "tactile3d-unit.vac-c3mscc-trials.v1",
            "test_loaded": False, "selection_split": "validation only",
            "maximum_allowed": 6, "planned": trials, "gpu": gpu,
            "config_sha256": sha256_file(args.config),
        }
        atomic_json(artifact_root / "trial_manifest.json", manifest)
        training = config["training"]
        architecture = config["architecture"]
        weights = C3MSCCLossWeights(**training["loss_weights"])
        batch_size = int(training["batch_size"])
        epochs = int(args.epochs or training["epochs"])
        results = []
        for trial_index, trial in enumerate(trials):
            set_seed(int(config["seed"]) + trial_index * 1000)
            model = ContactContextPredictor(
                str(trial["source"]), h_tokens=int(config["representations"]["h_tokens"]),
                blocks=int(architecture["blocks"]), heads=int(architecture["heads"]),
                mlp_width=int(architecture["mlp_width"]),
            ).to(device)
            if model.parameter_summary()["total"] >= int(architecture["maximum_parameters"]):
                raise RuntimeError("STRUCTURAL_FAIL: predictor exceeds parameter bound")
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
            )
            trial_root = experiment_root / f"trial_{trial_index:02d}_{trial['id']}"
            history = []
            best = None
            stale = 0
            started = time.monotonic()
            for epoch in range(1, epochs + 1):
                model.train()
                order = np.random.default_rng(
                    int(config["seed"]) + trial_index * 1000 + epoch
                ).permutation(len(train["u_c"]))
                totals: dict[str, float] = {}
                batches = 0
                for start in range(0, len(order), batch_size):
                    indices = order[start:start + batch_size]
                    if len(indices) < 2:
                        continue
                    u_a = torch.from_numpy(np.array(train["u_a"][indices], copy=True)).to(device)
                    u_c = torch.from_numpy(np.array(train["u_c"][indices], copy=True)).to(device)
                    h = torch.from_numpy(np.array(train["h_current"][indices], copy=True)).to(device)
                    dynamic = torch.from_numpy(np.array(train["dynamic"][indices], copy=True)).to(device)
                    u_v = None if model.source == "AH" else torch.from_numpy(
                        np.array(train["u_v"][indices], copy=True)
                    ).to(device)
                    invalid = ()
                    if bool(trial["temporal_order"]):
                        invalid = (u_a.flip(1), u_a.roll(1, 0))
                    optimizer.zero_grad(set_to_none=True)
                    loss, terms = contact_prediction_loss(
                        model, shared_space, decoder, u_a=u_a, h_current=h, u_c=u_c,
                        dynamic=dynamic, u_v=u_v, invalid_u_a=invalid,
                        enhanced=bool(trial["physics"] or trial["temporal_order"] or trial["rank_covariance"]),
                        dynamic_weight=float(training["dynamic_weight"]),
                        order_margin=float(training["order_margin"]),
                        variance_floor=float(training["variance_floor"]), weights=weights,
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite C3-MS-CC loss")
                    loss.backward()
                    if any(parameter.grad is not None for parameter in shared_space.parameters()):
                        raise RuntimeError("STRUCTURAL_FAIL: gradient reached C2-R")
                    if any(parameter.grad is not None for parameter in decoder.parameters()):
                        raise RuntimeError("STRUCTURAL_FAIL: gradient stored on frozen D_c")
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
                    optimizer.step()
                    for name, value in terms.items():
                        totals[name] = totals.get(name, 0.0) + float(value)
                    batches += 1
                if state_dict_digest(shared_space) != shared_before:
                    raise RuntimeError("STRUCTURAL_FAIL: C2-R changed during training")
                metrics = validation_metrics(
                    model, train, validation, oracle, shared_space, decoder,
                    config, device, batch_size, int(config["seed"]) + trial_index * 100 + epoch,
                )
                row = {
                    "epoch": epoch,
                    "train": {name: value / max(batches, 1) for name, value in totals.items()},
                    "validation": metrics,
                }
                history.append(row)
                score = float(metrics["utility"])
                if best is None or score > float(best["utility"]):
                    checkpoint = trial_root / "best.pt"
                    digest = save_checkpoint(
                        checkpoint, model,
                        {"trial_index": trial_index, "trial": trial, "epoch": epoch,
                         "validation": metrics, "selection_split": "validation only",
                         "test_loaded": False, "action_exact_ar_transform": False},
                    )
                    best = {
                        "epoch": epoch, "utility": score,
                        "checkpoint": str(checkpoint.relative_to(ROOT)),
                        "checkpoint_sha256": digest, "validation": metrics,
                    }
                    stale = 0
                else:
                    stale += 1
                atomic_json(trial_root / "history.json", {
                    "trial": trial, "history": history, "best": best, "test_loaded": False,
                })
                if stale >= int(training["patience"]):
                    break
            results.append({
                "trial_index": trial_index, "trial": trial,
                "parameter_summary": model.parameter_summary(), "epochs": len(history),
                "seconds": time.monotonic() - started, "best": best,
            })
        best_utility = max(float(row["best"]["utility"]) for row in results)
        ah = [row for row in results if row["trial"]["source"] == "AH"]
        best_ah = max(ah, key=lambda row: float(row["best"]["utility"]))
        tolerance = float(config["validation"]["simplicity_tolerance"])
        if best_ah["best"]["validation"]["gates"]["all"] and float(best_ah["best"]["utility"]) >= best_utility - tolerance:
            selected = best_ah
            rationale = "A+H passes all validation gates and is within 0.01 of best utility"
        else:
            selected = max(
                (row for row in results if row["trial"]["source"] == "VAH"),
                key=lambda row: float(row["best"]["utility"]),
            )
            rationale = "best validation-only V+A+H trial; A+H all-gates simplicity condition not met"
        selected_path = experiment_root / "selected.pt"
        shutil.copyfile(ROOT / selected["best"]["checkpoint"], selected_path)
        reloaded, metadata = load_checkpoint(selected_path, device)
        if metadata.get("test_loaded") is not False:
            raise RuntimeError("STRUCTURAL_FAIL: selected checkpoint permits test")
        selection = {
            "schema": "tactile3d-unit.vac-c3mscc-selection.v1",
            "source": selected["trial"]["source"], "trial": selected["trial"]["id"],
            "architecture": architecture, "loss_weights": training["loss_weights"],
            "epoch": selected["best"]["epoch"],
            "validation_metrics": selected["best"]["validation"],
            "checkpoint": str(selected_path.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(selected_path),
            "parameter_summary": reloaded.parameter_summary(),
            "selection_rationale": rationale,
            "vision_incremental_comparison": {
                row["trial"]["id"]: row["best"]["validation"]["utility"] for row in results
            },
            "selected_via": "VALIDATION ONLY", "selection_split": "validation only",
            "test_loaded": False, "action_exact_ar_transform": False,
            "frozen_shared_state_sha256": shared_before,
        }
        selection_path = artifact_root / "selection.json"
        atomic_json(selection_path, selection)
        selection_digest = sha256_file(selection_path)
        (artifact_root / "selection.sha256").write_text(selection_digest + "  selection.json\n")
        atomic_json(artifact_root / "training_summary.json", {
            "schema": "tactile3d-unit.vac-c3mscc-training.v1",
            "test_loaded": False, "trials": results, "selection": selection,
            "identity_before": identities_before, "identity_after": identity_snapshot(config),
            "gpu": gpu,
        })
        print(selection_path)
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    main()
