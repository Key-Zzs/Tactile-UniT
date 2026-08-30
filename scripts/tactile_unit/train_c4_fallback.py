#!/usr/bin/env python3
"""Train and validation-select the bounded Track C4 A/VA fallback family."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.c3r0_conditional_sufficiency import semantic_ratio  # noqa: E402
from gr00t.tactile_unit.c4_availability_conditioning import (  # noqa: E402
    C4FallbackLossWeights, ContactFallbackPredictor, fallback_prediction_loss,
    save_fallback_checkpoint, sha256_file,
)
from gr00t.tactile_unit.continuous_vac_shared_space import (  # noqa: E402
    bootstrap_mean_ci, different_episode_permutation, geometry_diagnostics,
    linear_cka, numpy_flatten_normalize, retrieval_metrics, state_dict_digest,
)
from scripts.tactile_unit.c4_runtime import (  # noqa: E402
    DEFAULT_CONFIG, atomic_json, identity_snapshot, load_config, load_parent_config,
    load_split, predict_fallback,
)
from scripts.tactile_unit.c3mscc_runtime import load_frozen_shared_space  # noqa: E402
from scripts.tactile_unit.continuous_contact_bridge_common import load_s2_model  # noqa: E402
from scripts.tactile_unit.train_c3mscc_contact_prediction import (  # noqa: E402
    oracle_semantics, physics_prediction, row_mse, semantic_evaluation,
)
from scripts.tactile_unit.vac_runtime_common import resolve_device, set_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--only-trial")
    parser.add_argument("--append-existing", action="store_true")
    return parser.parse_args()


def _quick_metrics(model, train, validation, oracle, device, batch_size, shared_space, decoder):
    train_prediction = predict_fallback(model, train, device, batch_size)
    prediction = predict_fallback(model, validation, device, batch_size)
    semantic, _ = semantic_evaluation(train_prediction, prediction, train, validation, oracle)
    error = row_mse(prediction, validation["u_c"])
    mean = np.asarray(train["u_c"], dtype=np.float64).mean(0).astype(np.float32)
    mean_error = row_mse(np.broadcast_to(mean, np.asarray(validation["u_c"]).shape), validation["u_c"])
    oracle_future = physics_prediction(shared_space, decoder, validation["u_c"], validation["h_current"], device, batch_size)
    predicted_future = physics_prediction(shared_space, decoder, prediction, validation["h_current"], device, batch_size)
    physics = row_mse(predicted_future, oracle_future)
    geometry = geometry_diagnostics(prediction)
    noncollapse = bool(
        geometry["per_dimension_variance"]["near_zero_fraction"] < 0.5
        and geometry["query_diversity"]["collapsed_pair_fraction"] < 0.5
    )
    return {
        "contact": float(semantic["contact_transition"]["semantic_ratio"]),
        "force": float(semantic["force_trend_class"]["semantic_ratio"]),
        "future_change": float(semantic["contact_transition"]["future_change"]["macro_f1"]),
        "shared_gain": float(((mean_error - error) / max(mean_error.mean(), 1e-12)).mean()),
        "physics_mse": float(physics.mean()), "noncollapse": noncollapse,
    }


def complete_validation(
    model, train, validation, oracle, config, device, batch_size, shared_space, decoder, seed,
) -> dict[str, Any]:
    bootstrap = int(config["validation"]["bootstrap_samples"])
    train_prediction = predict_fallback(model, train, device, batch_size)
    prediction = predict_fallback(model, validation, device, batch_size)
    semantics, _ = semantic_evaluation(train_prediction, prediction, train, validation, oracle)
    target = np.asarray(validation["u_c"])
    prediction_error = row_mse(prediction, target)
    mean_target = np.asarray(train["u_c"], dtype=np.float64).mean(0).astype(np.float32)
    mean_prediction = np.broadcast_to(mean_target, target.shape)
    episode = np.asarray(validation["episode_id"])
    different = different_episode_permutation(episode, seed + 1)
    shuffled = np.random.default_rng(seed + 2).permutation(len(target))
    source_predictions = {
        "mean_target": mean_prediction,
        "shuffled_source": predict_fallback(
            model, validation, device, batch_size,
            u_a=np.asarray(validation["u_a"])[shuffled],
            u_v=np.asarray(validation["u_v"])[shuffled] if model.source == "VA" else None,
        ),
        "different_episode": predict_fallback(
            model, validation, device, batch_size,
            u_a=np.asarray(validation["u_a"])[different],
            u_v=np.asarray(validation["u_v"])[different] if model.source == "VA" else None,
        ),
    }
    control_errors = {name: row_mse(value, target) for name, value in source_predictions.items()}
    control_gains = {name: error - prediction_error for name, error in control_errors.items()}

    dynamic = np.asarray(validation["dynamic"], dtype=bool)
    exact_predictions = {
        name: predict_fallback(
            model, validation, device, batch_size,
            u_a=np.asarray(validation[f"u_a_{name}"]),
        ) for name in ("reversed", "shuffled", "different")
    }
    exact_errors = {name: row_mse(value, target) for name, value in exact_predictions.items()}
    temporal = {}
    for offset, name in enumerate(("reversed", "shuffled", "different")):
        difference = exact_errors[name][dynamic] - prediction_error[dynamic]
        temporal[name] = {
            "dynamic_mse": float(exact_errors[name][dynamic].mean()),
            "difference": float(difference.mean()),
            "difference_ci95": bootstrap_mean_ci(difference, samples=bootstrap, seed=seed + 10 + offset),
        }

    oracle_future = physics_prediction(shared_space, decoder, target, validation["h_current"], device, batch_size)
    predicted_future = physics_prediction(shared_space, decoder, prediction, validation["h_current"], device, batch_size)
    physics_error = row_mse(predicted_future, oracle_future)
    physics_controls = {
        name: row_mse(
            physics_prediction(shared_space, decoder, value, validation["h_current"], device, batch_size),
            oracle_future,
        ) for name, value in source_predictions.items()
    }
    strongest_physics = min(physics_controls, key=lambda name: physics_controls[name].mean())
    physics_gain = physics_controls[strongest_physics] - physics_error

    left = numpy_flatten_normalize(prediction)
    right = numpy_flatten_normalize(target)
    cosine_true = np.sum(left * right, axis=1)
    cosine_false = np.sum(left * right[shuffled], axis=1)
    cosine_margin = cosine_true - cosine_false
    retrieval = retrieval_metrics(
        prediction, target, chunk=int(config["evaluation"]["retrieval_chunk"])
    )
    geometry = geometry_diagnostics(prediction)
    geometry["cka_with_oracle"] = linear_cka(prediction, target)
    noncollapse = bool(
        geometry["per_dimension_variance"]["near_zero_fraction"] < 0.5
        and geometry["query_diversity"]["collapsed_pair_fraction"] < 0.5
    )
    baseline = json.loads(
        (ROOT / config["runtime"]["artifact_root"] / "availability_baseline.json").read_text()
    )["variants"]
    contact_f1 = float(semantics["contact_transition"]["macro_f1"])
    misuse_gate = bool(
        float(prediction_error.mean()) < min(baseline["zero"]["shared_mse"], baseline["mean"]["shared_mse"])
        and contact_f1 > max(baseline["zero"]["contact_macro_f1"], baseline["mean"]["contact_macro_f1"])
    )
    shared_gates = {
        name: bootstrap_mean_ci(value, samples=bootstrap, seed=seed + 30 + offset)
        for offset, (name, value) in enumerate(control_gains.items())
    }
    temporal_gate = bool(
        temporal["reversed"]["difference_ci95"][0] > 0
        and temporal["shuffled"]["difference_ci95"][0] > 0
    )
    latent_gate = bool(
        all(interval[0] > 0 for interval in shared_gates.values())
        and bootstrap_mean_ci(cosine_margin, samples=bootstrap, seed=seed + 40)[0] > 0
        and retrieval["recall_at_10"] >= float(config["evaluation"]["retrieval_r10_chance_multiplier_min"]) * retrieval["chance"]["recall_at_10"]
    )
    physics_gate = bool(
        bootstrap_mean_ci(physics_gain, samples=bootstrap, seed=seed + 41)[0] > 0
        and bootstrap_mean_ci(physics_gain[dynamic], samples=bootstrap, seed=seed + 42)[0] > 0
    )
    gates = {
        "contact": semantics["contact_transition"]["semantic_ratio"] >= float(config["evaluation"]["contact_retention_min"]),
        "force": semantics["force_trend_class"]["semantic_ratio"] >= float(config["evaluation"]["force_retention_min"]),
        "future_change": semantics["contact_transition"]["future_change"]["macro_f1"] > semantics["contact_transition"]["majority"]["future_change"]["macro_f1"] if "future_change" in semantics["contact_transition"].get("majority", {}) else np.isfinite(semantics["contact_transition"]["future_change"]["macro_f1"]),
        "invalid_h_misuse": misuse_gate, "latent": latent_gate,
        "physics": physics_gate, "action_temporal": temporal_gate,
        "noncollapse": noncollapse,
    }
    weights = config["validation"]["utility"]
    mean_gain = float(np.mean([value.mean() for value in control_gains.values()]))
    normalized_physics = float(physics_gain.mean() / max(physics_controls[strongest_physics].mean(), 1e-12))
    normalized_temporal = float(np.mean([temporal[name]["difference"] for name in ("reversed", "shuffled")]) / max(prediction_error[dynamic].mean(), 1e-12))
    utility = (
        float(weights["contact"]) * float(semantics["contact_transition"]["semantic_ratio"])
        + float(weights["force"]) * float(semantics["force_trend_class"]["semantic_ratio"])
        + float(weights["future_change"]) * float(semantics["contact_transition"]["future_change"]["macro_f1"])
        + float(weights["shared"]) * mean_gain
        + float(weights["physics"]) * normalized_physics
        + float(weights["temporal"]) * normalized_temporal
        + float(weights["noncollapse"]) * float(noncollapse)
        - (float(weights["vision_penalty"]) if model.source == "VA" else 0.0)
    )
    return {
        "selection_split": "validation only", "test_loaded": False, "rows": len(target),
        "utility": float(utility), "semantics": semantics,
        "shared_target": {
            "prediction_mse": float(prediction_error.mean()),
            "controls_mse": {name: float(value.mean()) for name, value in control_errors.items()},
            "improvement_ci95": shared_gates,
            "cosine_true": float(cosine_true.mean()), "cosine_shuffled": float(cosine_false.mean()),
            "cosine_margin": float(cosine_margin.mean()),
            "cosine_margin_ci95": bootstrap_mean_ci(cosine_margin, samples=bootstrap, seed=seed + 40),
            "retrieval": retrieval,
        },
        "physics": {
            "teacher_side_h_only": True, "prediction_mse": float(physics_error.mean()),
            "dynamic_mse": float(physics_error[dynamic].mean()),
            "controls_mse": {name: float(value.mean()) for name, value in physics_controls.items()},
            "strongest_control": strongest_physics,
            "improvement_ci95": bootstrap_mean_ci(physics_gain, samples=bootstrap, seed=seed + 41),
            "dynamic_improvement_ci95": bootstrap_mean_ci(physics_gain[dynamic], samples=bootstrap, seed=seed + 42),
        },
        "action_temporal": {"exact_ar_transform": True, "correct_dynamic_mse": float(prediction_error[dynamic].mean()), "variants": temporal},
        "invalid_h_misuse": {"baseline": baseline, "gate": misuse_gate},
        "geometry": geometry, "noncollapse": noncollapse,
        "gates": {**gates, "all": all(gates.values())},
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    artifact_root = ROOT / config["runtime"]["artifact_root"]
    experiment_root = ROOT / config["runtime"]["experiment_root"]
    if not (artifact_root / "availability_baseline.json").is_file():
        raise RuntimeError("run C4.0 availability baseline before fallback training")
    identities_before = identity_snapshot(config)
    if not identities_before["pass"]:
        raise RuntimeError("STRUCTURAL_FAIL: frozen identity mismatch")
    device, lock_handle, gpu = resolve_device(args.device, allowed_physical=("0", "1", "2", "3"))
    try:
        set_seed(int(config["seed"]))
        train = load_split(config, "train")
        validation = load_split(config, "validation")
        parent = load_parent_config(config)
        shared_space, _, shared_before = load_frozen_shared_space(parent, device)
        s2 = load_s2_model(ROOT / config["runtime"]["s2_checkpoint"], device)
        decoder = s2.decoder.eval().requires_grad_(False)
        oracle = oracle_semantics(train, validation)
        trials = list(config["training"]["trials"])
        trials = trials[:min(len(trials), int(args.max_trials or len(trials)), 6)]
        atomic_json(artifact_root / "trial_manifest.json", {
            "schema": "tactile3d-unit.vac-c4-trials.v1", "planned": trials,
            "maximum_allowed": 6, "test_loaded": False, "selection_split": "validation only",
            "gpu": {**gpu, "preferred_physical": 1, "fallback": gpu.get("actual_physical") != 1},
        })
        spec = config["training"]
        results = []
        if args.append_existing:
            existing = artifact_root / "fallback_training_summary.json"
            if not existing.is_file():
                raise RuntimeError("--append-existing requires fallback_training_summary.json")
            results = json.loads(existing.read_text())["trials"]
        indexed_trials = list(enumerate(trials))
        if args.only_trial:
            indexed_trials = [row for row in indexed_trials if row[1]["id"] == args.only_trial]
            if not indexed_trials:
                raise RuntimeError(f"unknown trial {args.only_trial}")
        for trial_index, trial in indexed_trials:
            set_seed(int(config["seed"]) + trial_index * 100)
            trial_root = experiment_root / f"trial_{trial_index:02d}_{trial['id']}"
            model = ContactFallbackPredictor(
                trial["source"], blocks=int(config["architecture"]["blocks"]),
                heads=int(config["architecture"]["heads"]),
                mlp_width=int(config["architecture"]["mlp_width"]),
            ).to(device)
            if trial.get("initialize_from"):
                parent_index, parent_trial = next(
                    row for row in enumerate(trials) if row[1]["id"] == trial["initialize_from"]
                )
                from gr00t.tactile_unit.c4_availability_conditioning import load_fallback_checkpoint
                initial, _ = load_fallback_checkpoint(
                    experiment_root / f"trial_{parent_index:02d}_{parent_trial['id']}" / "best.pt",
                    device,
                )
                if initial.source != model.source:
                    raise RuntimeError("STRUCTURAL_FAIL: optional trial source changed")
                model.load_state_dict(initial.state_dict(), strict=True)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=float(trial.get("learning_rate", spec["learning_rate"])),
                weight_decay=float(spec["weight_decay"]),
            )
            weights = C4FallbackLossWeights(**trial.get("loss_weights", spec["loss_weights"]))
            if args.finalize_only:
                saved = json.loads((trial_root / "history.json").read_text())
                history = saved["history"]
                best = saved["best"]
            else:
                history = []
                best = None
            stale = 0
            epochs = int(args.epochs or trial.get("epochs", spec["epochs"]))
            started = time.monotonic()
            for epoch in (() if args.finalize_only else range(1, epochs + 1)):
                model.train()
                order = np.random.default_rng(int(config["seed"]) + trial_index * 1000 + epoch).permutation(len(train["u_c"]))
                totals: dict[str, float] = {}
                batches = 0
                for start in range(0, len(order), int(spec["batch_size"])):
                    indices = order[start:start + int(spec["batch_size"])]
                    if len(indices) < 2:
                        continue
                    def tensor(name):
                        return torch.from_numpy(np.array(train[name][indices], copy=True)).to(device)
                    u_a, u_c, dynamic = tensor("u_a"), tensor("u_c"), tensor("dynamic")
                    u_v = tensor("u_v") if model.source == "VA" else None
                    enhanced = bool(trial["physics"] or trial["covariance"])
                    optimizer.zero_grad(set_to_none=True)
                    loss, terms = fallback_prediction_loss(
                        model, shared_space, decoder, u_a=u_a, u_c=u_c,
                        dynamic=dynamic, u_v=u_v, teacher_h_current=tensor("h_current") if enhanced else None,
                        invalid_u_a=(tensor("u_a_reversed"), tensor("u_a_shuffled")) if enhanced else (),
                        enhanced=enhanced, dynamic_weight=float(spec["dynamic_weight"]),
                        order_margin=float(spec["order_margin"]), variance_floor=float(spec["variance_floor"]),
                        weights=weights,
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite C4 fallback loss")
                    loss.backward()
                    if any(parameter.grad is not None for parameter in shared_space.parameters()):
                        raise RuntimeError("STRUCTURAL_FAIL: fallback gradient reached shared space")
                    if any(parameter.grad is not None for parameter in decoder.parameters()):
                        raise RuntimeError("STRUCTURAL_FAIL: fallback gradient reached D_c parameters")
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(spec["gradient_clip"]))
                    optimizer.step()
                    for name, value in terms.items():
                        totals[name] = totals.get(name, 0.0) + float(value)
                    batches += 1
                quick = _quick_metrics(model, train, validation, oracle, device, int(spec["batch_size"]), shared_space, decoder)
                score = quick["contact"] * 0.35 + quick["force"] * 0.30 + quick["future_change"] * 0.10 + quick["shared_gain"] * 0.20 + float(quick["noncollapse"]) * 0.05
                history.append({"epoch": epoch, "train": {name: value / max(batches, 1) for name, value in totals.items()}, "validation": quick, "selection_score": float(score)})
                if best is None or score > best["selection_score"]:
                    checkpoint = trial_root / "best.pt"
                    digest = save_fallback_checkpoint(checkpoint, model, {
                        "trial": trial["id"], "source": trial["source"], "epoch": epoch,
                        "selection_score": float(score), "selection_split": "validation only",
                        "test_loaded": False, "teacher_h_predictor_input": False,
                        "private_residual": False,
                    })
                    best = {"epoch": epoch, "selection_score": float(score), "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": digest}
                    stale = 0
                else:
                    stale += 1
                atomic_json(trial_root / "history.json", {"trial": trial, "history": history, "best": best, "test_loaded": False})
                if stale >= int(spec["patience"]):
                    break
            from gr00t.tactile_unit.c4_availability_conditioning import load_fallback_checkpoint
            selected_model, metadata = load_fallback_checkpoint(ROOT / best["checkpoint"], device)
            metrics = complete_validation(
                selected_model.eval().requires_grad_(False), train, validation, oracle,
                config, device, int(spec["batch_size"]), shared_space, decoder,
                int(config["seed"]) + 5000 + trial_index * 100,
            )
            results.append({
                "trial_index": trial_index, "trial": trial, "parameters": selected_model.parameter_summary(),
                "epochs_ran": len(history), "seconds": time.monotonic() - started,
                "best": {**best, "validation": metrics},
            })
        unique = {row["trial"]["id"]: row for row in results}
        results = [unique[name] for name in sorted(unique, key=lambda value: int(value[1:]))]
        passing = [row for row in results if row["best"]["validation"]["gates"]["all"]]
        pool = passing if passing else results
        best_a = max((row for row in pool if row["trial"]["source"] == "A"), key=lambda row: row["best"]["validation"]["utility"], default=None)
        best_va = max((row for row in pool if row["trial"]["source"] == "VA"), key=lambda row: row["best"]["validation"]["utility"], default=None)
        best_overall = max(pool, key=lambda row: row["best"]["validation"]["utility"])
        tolerance = float(config["validation"]["simplicity_tolerance"])
        if best_a is not None and best_a["best"]["validation"]["gates"]["all"] and best_a["best"]["validation"]["utility"] >= best_overall["best"]["validation"]["utility"] - tolerance:
            selected, rationale = best_a, "A passes all fallback gates and is within 0.01 of best validation utility"
        else:
            selected, rationale = (best_va or best_overall), "A does not satisfy the frozen simplicity rule; select best validation-only VA fallback"
        summary = {
            "schema": "tactile3d-unit.vac-c4-fallback-training.v1", "trials": results,
            "total_trials": len(results), "maximum_trials": 6, "test_loaded": False,
            "selection_split": "validation only", "identity_before": identities_before,
            "shared_state_before": shared_before,
        }
        atomic_json(artifact_root / "fallback_training_summary.json", summary)
        selection = {
            "schema": "tactile3d-unit.vac-c4-fallback-selection.v1",
            "candidate": selected["trial"]["id"], "source": selected["trial"]["source"],
            "trial": selected["trial"], "checkpoint": selected["best"]["checkpoint"],
            "checkpoint_sha256": selected["best"]["checkpoint_sha256"],
            "validation_utility": selected["best"]["validation"]["utility"],
            "validation": selected["best"]["validation"],
            "all_validation_gates": selected["best"]["validation"]["gates"]["all"],
            "rationale": rationale, "simplicity_tolerance": tolerance,
            "selected_via": "VALIDATION ONLY", "selection_split": "validation only",
            "test_loaded": False, "identity": identity_snapshot(config),
        }
        path = artifact_root / "fallback_selection.json"
        atomic_json(path, selection)
        digest = sha256_file(path)
        (artifact_root / "fallback_selection.sha256").write_text(digest + "  fallback_selection.json\n")
        if not identity_snapshot(config)["pass"] or state_dict_digest(shared_space) != shared_before:
            raise RuntimeError("STRUCTURAL_FAIL: frozen identity changed during fallback training")
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    main()
