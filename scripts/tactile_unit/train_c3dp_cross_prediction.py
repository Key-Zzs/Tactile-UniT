#!/usr/bin/env python3
"""Train at most the preregistered bounded C3-DP predictor candidates."""

from __future__ import annotations

import argparse
import json
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
from gr00t.tactile_unit.c3dp_shared_private import (  # noqa: E402
    ORDERED_DIRECTIONS,
    SHORT_DIRECTION,
    C3DPLossWeights,
    SharedCrossModalPredictor,
    cross_modal_prediction_loss,
    load_predictor_checkpoint,
    output_geometry_gate,
    save_predictor_checkpoint,
    sha256_file,
)
from gr00t.tactile_unit.continuous_vac_shared_space import (  # noqa: E402
    bootstrap_mean_ci,
    different_episode_permutation,
    numpy_flatten_normalize,
    retrieval_metrics,
)
from scripts.tactile_unit.c3dp_runtime import (  # noqa: E402
    DEFAULT_CONFIG,
    atomic_json,
    ensure_cache_identities,
    load_config,
    load_derived_split,
    load_frozen_shared_space,
    shared_digest,
)
from scripts.tactile_unit.vac_runtime_common import (  # noqa: E402
    resolve_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--epochs", type=int)
    return parser.parse_args()


def predictor_for(config: Mapping[str, Any], candidate: str) -> SharedCrossModalPredictor:
    value = config["candidates"][candidate]
    return SharedCrossModalPredictor(
        candidate,
        hidden_dim=int(value["hidden_dim"]),
        attention_layers=int(value["attention_layers"]),
        heads=int(value["heads"]),
    )


def predict_numpy(predictor, source, source_name, target_name, device, batch_size):
    result = np.empty((len(source), 8, 32), dtype=np.float32)
    predictor.eval()
    with torch.inference_mode():
        for start in range(0, len(source), batch_size):
            stop = min(start + batch_size, len(source))
            value = torch.from_numpy(np.array(source[start:stop], copy=True)).to(device)
            result[start:stop] = predictor(value, source_name, target_name).float().cpu().numpy()
    return result


def fit_contact_probe(train_x, train_y):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), RidgeClassifier(alpha=10.0, class_weight="balanced"))
    model.fit(np.asarray(train_x).reshape(len(train_x), -1), np.asarray(train_y, dtype=np.int64))
    return model


def probe_metrics(model, value, target, majority_class):
    prediction = model.predict(np.asarray(value).reshape(len(value), -1))
    majority = np.full(len(target), int(majority_class), dtype=np.int64)
    return {
        **classification_metrics(target, prediction),
        "majority": classification_metrics(target, majority),
    }


def validation_metrics(
    predictor,
    train,
    validation,
    majority_class,
    config,
    device,
    batch_size,
    seed,
):
    count = min(int(config["validation"]["metric_rows"]), len(validation["u_v"]))
    indices = np.linspace(0, len(validation["u_v"]) - 1, count, dtype=np.int64)
    episode = np.asarray(validation["episode_id"])[indices]
    different = different_episode_permutation(episode, seed + 1)
    shuffled_target = np.random.default_rng(seed + 2).permutation(count)
    names = {"vision": "u_v", "action": "u_a", "contact": "u_c"}
    direction_metrics = {}
    contact_predictions = {}
    utilities = []
    all_pass = True
    for offset, (source_name, target_name) in enumerate(ORDERED_DIRECTIONS):
        source = np.asarray(validation[names[source_name]][indices])
        target = np.asarray(validation[names[target_name]][indices])
        prediction = predict_numpy(predictor, source, source_name, target_name, device, batch_size)
        mean_target = (
            np.asarray(train[names[target_name]], dtype=np.float64).mean(0).astype(np.float32)
        )
        mean_prediction = np.broadcast_to(mean_target[None], target.shape)
        shuffled_prediction = predict_numpy(
            predictor, source[shuffled_target], source_name, target_name, device, batch_size
        )
        different_prediction = predict_numpy(
            predictor, source[different], source_name, target_name, device, batch_size
        )
        per_sample = np.square(prediction.astype(np.float64) - target).reshape(count, -1).mean(1)
        controls = {
            "mean": np.square(mean_prediction.astype(np.float64) - target)
            .reshape(count, -1)
            .mean(1),
            "shuffled_source": np.square(shuffled_prediction.astype(np.float64) - target)
            .reshape(count, -1)
            .mean(1),
            "different_episode": np.square(different_prediction.astype(np.float64) - target)
            .reshape(count, -1)
            .mean(1),
        }
        strongest_name = min(controls, key=lambda name: controls[name].mean())
        improvement = controls[strongest_name] - per_sample
        prediction_norm = numpy_flatten_normalize(prediction)
        target_norm = numpy_flatten_normalize(target)
        true_cosine = np.sum(prediction_norm * target_norm, axis=1)
        shuffled_cosine = np.sum(prediction_norm * target_norm[shuffled_target], axis=1)
        margin = true_cosine - shuffled_cosine
        retrieval = retrieval_metrics(
            prediction, target, chunk=int(config["validation"]["retrieval_chunk"])
        )
        r10_multiplier = retrieval["recall_at_10"] / retrieval["chance"]["recall_at_10"]
        gate = bool(
            improvement.mean() > 0
            and bootstrap_mean_ci(
                improvement,
                samples=int(config["validation"]["bootstrap_samples"]),
                seed=seed + 10 + offset,
            )[0]
            > 0
            and margin.mean() > 0
            and bootstrap_mean_ci(
                margin,
                samples=int(config["validation"]["bootstrap_samples"]),
                seed=seed + 20 + offset,
            )[0]
            > 0
            and r10_multiplier >= float(config["validation"]["retrieval_r10_chance_multiplier_min"])
        )
        utilities.append(float(improvement.mean() / max(controls[strongest_name].mean(), 1e-12)))
        all_pass = all_pass and gate
        key = SHORT_DIRECTION[(source_name, target_name)]
        direction_metrics[key] = {
            "prediction_mse": float(per_sample.mean()),
            "controls": {name: float(value.mean()) for name, value in controls.items()},
            "strongest_control": strongest_name,
            "improvement": float(improvement.mean()),
            "improvement_ci95": bootstrap_mean_ci(
                improvement,
                samples=int(config["validation"]["bootstrap_samples"]),
                seed=seed + 10 + offset,
            ),
            "cosine_true": float(true_cosine.mean()),
            "cosine_shuffled": float(shuffled_cosine.mean()),
            "cosine_margin": float(margin.mean()),
            "cosine_margin_ci95": bootstrap_mean_ci(
                margin,
                samples=int(config["validation"]["bootstrap_samples"]),
                seed=seed + 20 + offset,
            ),
            "retrieval": retrieval,
            "r10_chance_multiplier": float(r10_multiplier),
            "gate": gate,
        }
        if target_name == "contact":
            contact_predictions[key] = prediction
    target = np.asarray(validation["contact_transition"])[indices]
    oracle_model = fit_contact_probe(train["u_c"], train["contact_transition"])
    oracle_probe = probe_metrics(oracle_model, validation["u_c"][indices], target, majority_class)
    contact_semantics = {"oracle": oracle_probe}
    for key, prediction in contact_predictions.items():
        source_name = "vision" if key == "V->C" else "action"
        train_prediction = predict_numpy(
            predictor,
            np.asarray(train[names[source_name]]),
            source_name,
            "contact",
            device,
            batch_size,
        )
        predicted_model = fit_contact_probe(train_prediction, train["contact_transition"])
        predicted_probe = probe_metrics(predicted_model, prediction, target, majority_class)
        majority_f1 = float(oracle_probe["majority"]["macro_f1"])
        retention = (float(predicted_probe["macro_f1"]) - majority_f1) / max(
            float(oracle_probe["macro_f1"]) - majority_f1, 1e-12
        )
        contact_semantics[key] = {"probe": predicted_probe, "retention": float(retention)}
    geometry = {}
    geometry_pass = True
    for target_name in ("vision", "action", "contact"):
        source_name = next(source for source, target in ORDERED_DIRECTIONS if target == target_name)
        value = predict_numpy(
            predictor,
            np.asarray(validation[names[source_name]][indices]),
            source_name,
            target_name,
            device,
            batch_size,
        )
        diagnostics, passed = output_geometry_gate(value)
        geometry[target_name] = diagnostics
        geometry_pass = geometry_pass and passed
    semantic_utility = np.mean([contact_semantics[key]["retention"] for key in ("V->C", "A->C")])
    parameter_penalty = np.log10(max(predictor.parameter_summary()["total"], 1)) * 1e-3
    utility = float(np.mean(utilities) + 0.05 * semantic_utility - parameter_penalty)
    return {
        "selection_split": "validation only",
        "test_loaded": False,
        "rows": count,
        "directions": direction_metrics,
        "all_direction_gates": bool(all_pass),
        "contact_semantics": contact_semantics,
        "geometry": geometry,
        "noncollapse": bool(geometry_pass),
        "utility": utility,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runtime = config["runtime"]
    cache_root = ROOT / runtime["cache_root"]
    experiment_root = ROOT / runtime["experiment_root"]
    artifact_root = ROOT / runtime["artifact_root"]
    device, lock_handle, gpu = resolve_device(args.device, allowed_physical=("2", "3"))
    try:
        set_seed(int(config["seed"]))
        train, train_manifest = load_derived_split(cache_root, "train")
        validation, validation_manifest = load_derived_split(cache_root, "validation")
        ensure_cache_identities(config, train_manifest, validation_manifest)
        shared_space, shared_metadata, checkpoint_sha = load_frozen_shared_space(config, device)
        shared_before = shared_digest(shared_space)
        if train_manifest["shared_state_sha256"] != shared_before:
            raise RuntimeError("STRUCTURAL_FAIL: derived shared cache does not match C2-R")
        majority_class = int(
            np.bincount(
                np.asarray(train["contact_transition"], dtype=np.int64), minlength=4
            ).argmax()
        )
        trials = list(config["training"]["bounded_trials"])
        maximum_trials = args.max_trials or len(trials)
        trials = trials[: min(maximum_trials, len(trials), 12)]
        artifact_root.mkdir(parents=True, exist_ok=True)
        experiment_root.mkdir(parents=True, exist_ok=True)
        trial_manifest = {
            "schema": "tactile3d-unit.vac-c3dp-trials.v1",
            "selection_split": "validation only",
            "test_loaded": False,
            "maximum_allowed": 12,
            "planned": trials,
        }
        atomic_json(experiment_root / "trial_manifest.json", trial_manifest)
        results = []
        training = config["training"]
        epochs = int(args.epochs or training["epochs"])
        batch_size = int(training["batch_size"])
        names = {"vision": "u_v", "action": "u_a", "contact": "u_c"}
        for trial_id, trial in enumerate(trials):
            set_seed(int(config["seed"]) + trial_id * 1000)
            predictor = predictor_for(config, str(trial["candidate"])).to(device)
            optimizer = torch.optim.AdamW(
                predictor.parameters(),
                lr=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
            )
            weights = C3DPLossWeights(
                shared=float(training["loss_weights"]["shared"]),
                shared_native=float(trial["shared_native_weight"]),
                relational=float(
                    trial.get("relational_weight", training["loss_weights"]["relational"])
                ),
                variance=float(trial.get("variance_weight", training["loss_weights"]["variance"])),
                cosine=float(training["loss_weights"]["cosine"]),
            )
            trial_root = experiment_root / f"trial_{trial_id:02d}_{trial['candidate']}"
            history = []
            best = None
            stale = 0
            started = time.monotonic()
            for epoch in range(1, epochs + 1):
                predictor.train()
                order = np.random.default_rng(
                    int(config["seed"]) + trial_id * 1000 + epoch
                ).permutation(len(train["u_v"]))
                totals: dict[str, float] = {}
                batches = 0
                for start in range(0, len(order), batch_size):
                    indices = order[start : start + batch_size]
                    if len(indices) < 2:
                        continue
                    shared = {
                        modality: torch.from_numpy(np.array(train[key][indices], copy=True)).to(
                            device
                        )
                        for modality, key in names.items()
                    }
                    dynamic = torch.from_numpy(
                        np.array(train["dynamic"][indices], dtype=np.bool_, copy=True)
                    ).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss, breakdown = cross_modal_prediction_loss(
                        predictor,
                        shared_space,
                        shared,
                        dynamic,
                        dynamic_weight=float(trial["dynamic_weight"]),
                        weights=weights,
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite C3-DP training loss")
                    loss.backward()
                    if any(parameter.grad is not None for parameter in shared_space.parameters()):
                        raise RuntimeError("STRUCTURAL_FAIL: C3 gradient reached C2-R")
                    torch.nn.utils.clip_grad_norm_(
                        predictor.parameters(), float(training["gradient_clip"])
                    )
                    optimizer.step()
                    for name, value in breakdown.items():
                        totals[name] = totals.get(name, 0.0) + float(value)
                    batches += 1
                if shared_digest(shared_space) != shared_before:
                    raise RuntimeError("STRUCTURAL_FAIL: C2-R changed during C3-DP training")
                validation_result = validation_metrics(
                    predictor,
                    train,
                    validation,
                    majority_class,
                    config,
                    device,
                    batch_size,
                    int(config["seed"]) + trial_id * 100 + epoch,
                )
                row = {
                    "epoch": epoch,
                    "train": {name: value / max(batches, 1) for name, value in totals.items()},
                    "validation": validation_result,
                }
                history.append(row)
                score = float(validation_result["utility"])
                if best is None or score > float(best["utility"]) + 1e-12:
                    checkpoint = trial_root / "best.pt"
                    digest = save_predictor_checkpoint(
                        checkpoint,
                        predictor,
                        {
                            "trial_id": trial_id,
                            "trial": trial,
                            "epoch": epoch,
                            "selection_split": "validation only",
                            "test_loaded": False,
                            "validation": validation_result,
                        },
                    )
                    best = {
                        "epoch": epoch,
                        "utility": score,
                        "checkpoint": str(checkpoint.relative_to(ROOT)),
                        "checkpoint_sha256": digest,
                        "validation": validation_result,
                    }
                    stale = 0
                else:
                    stale += 1
                atomic_json(
                    trial_root / "history.json",
                    {
                        "trial_id": trial_id,
                        "trial": trial,
                        "history": history,
                        "best": best,
                        "test_loaded": False,
                    },
                )
                if stale >= int(training["patience"]):
                    break
            assert best is not None
            results.append(
                {
                    "trial_id": trial_id,
                    "trial": trial,
                    "parameter_summary": predictor.parameter_summary(),
                    "epochs": len(history),
                    "seconds": time.monotonic() - started,
                    "best": best,
                }
            )

        maximum = max(float(row["best"]["utility"]) for row in results)
        tolerance = float(config["validation"]["effective_tie_tolerance"])
        tied = [row for row in results if float(row["best"]["utility"]) >= maximum - tolerance]
        selected = min(tied, key=lambda row: row["parameter_summary"]["total"])
        selected_source = ROOT / selected["best"]["checkpoint"]
        selected_path = experiment_root / "selected.pt"
        shutil.copyfile(selected_source, selected_path)
        checkpoint_digest = sha256_file(selected_path)
        reloaded, metadata = load_predictor_checkpoint(selected_path, device)
        if metadata.get("test_loaded") is not False:
            raise RuntimeError("STRUCTURAL_FAIL: selected C3-DP checkpoint permits test")
        selection = {
            "schema": "tactile3d-unit.vac-c3dp-selection.v1",
            "candidate": selected["trial"]["candidate"],
            "hyperparameters": selected["trial"],
            "epoch": selected["best"]["epoch"],
            "validation_metrics": selected["best"]["validation"],
            "parameter_summary": reloaded.parameter_summary(),
            "checkpoint": str(selected_path.relative_to(ROOT)),
            "checkpoint_sha256": checkpoint_digest,
            "selection_rationale": "maximum validation-only utility; effective ties prefer fewer parameters",
            "selection_split": "validation only",
            "test_loaded": False,
            "frozen_c2r_checkpoint_sha256": checkpoint_sha,
            "frozen_shared_state_sha256": shared_before,
        }
        selection_path = artifact_root / "selection.json"
        atomic_json(selection_path, selection)
        selection_sha = sha256_file(selection_path)
        (artifact_root / "selection.sha256").write_text(selection_sha + "  selection.json\n")
        summary = {
            "schema": "tactile3d-unit.vac-c3dp-training.v1",
            "gpu": gpu,
            "selection_split": "validation only",
            "test_loaded": False,
            "trial_count": len(results),
            "trials": results,
            "selected": selection,
            "selection_artifact_sha256": selection_sha,
            "shared_space_metadata": shared_metadata,
            "shared_state_before": shared_before,
            "shared_state_after": shared_digest(shared_space),
            "shared_space_unchanged": shared_before == shared_digest(shared_space),
        }
        atomic_json(experiment_root / "training_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    main()
