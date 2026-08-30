#!/usr/bin/env python3
"""Run the single locked C3-MS-CC benchmark evaluation after selection freeze."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_teacher.evaluation import classification_metrics  # noqa: E402
from gr00t.tactile_unit.c3mscc_contact_context import load_checkpoint, sha256_file  # noqa: E402
from gr00t.tactile_unit.c3r0_conditional_sufficiency import (  # noqa: E402
    bootstrap_f1_difference, evaluate_prediction, semantic_ratio,
)
from gr00t.tactile_unit.continuous_vac_shared_space import (  # noqa: E402
    bootstrap_mean_ci, different_episode_permutation, geometry_diagnostics,
    linear_cka, numpy_flatten_normalize, retrieval_metrics, state_dict_digest,
)
from scripts.tactile_unit.c3mscc_runtime import (  # noqa: E402
    DEFAULT_CONFIG, atomic_json, identity_snapshot, load_aligned_split,
    load_config, load_frozen_shared_space, validate_selection_lock,
)
from scripts.tactile_unit.continuous_contact_bridge_common import load_s2_model  # noqa: E402
from scripts.tactile_unit.train_c3mscc_contact_prediction import (  # noqa: E402
    fit_probe, majority, physics_prediction, predict_numpy, row_mse,
)
from scripts.tactile_unit.vac_runtime_common import resolve_device, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--bootstrap-samples", type=int)
    return parser.parse_args()


def per_class_metrics(target: np.ndarray, prediction: np.ndarray, classes: int) -> dict[str, Any]:
    result = {}
    for label in range(classes):
        truth = np.asarray(target) == label
        guessed = np.asarray(prediction) == label
        tp = int(np.sum(truth & guessed))
        fp = int(np.sum(~truth & guessed))
        fn = int(np.sum(truth & ~guessed))
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
        result[str(label)] = {
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "support": int(truth.sum()),
        }
    return result


def semantic_probe(
    train_prediction: np.ndarray,
    test_prediction: np.ndarray,
    train: Mapping[str, np.ndarray],
    test: Mapping[str, np.ndarray],
    oracle: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    result = {}
    labels = {}
    for metric, classes in (("contact_transition", 4), ("force_trend_class", 3)):
        probe = fit_probe(train_prediction, train[metric])
        predicted = np.asarray(
            probe.predict(test_prediction.reshape(len(test_prediction), -1)), dtype=np.int64
        )
        labels[metric] = predicted
        value = evaluate_prediction(
            test[metric], predicted, majority(train[metric], len(predicted), classes), classes
        )
        value["semantic_ratio"] = semantic_ratio(
            value["macro_f1"], oracle[metric]["majority"]["macro_f1"],
            oracle[metric]["macro_f1"],
        )
        result[metric] = value
    return result, labels


def oracle_probe(train: Mapping[str, np.ndarray], test: Mapping[str, np.ndarray]):
    result = {}
    for metric, classes in (("contact_transition", 4), ("force_trend_class", 3)):
        model = fit_probe(train["u_c"], train[metric])
        prediction = model.predict(np.asarray(test["u_c"]).reshape(len(test["u_c"]), -1))
        result[metric] = evaluate_prediction(
            test[metric], prediction, majority(train[metric], len(prediction), classes), classes
        )
    return result


def wrong_time_indices(episode: np.ndarray, t: np.ndarray) -> np.ndarray:
    result = np.arange(len(episode), dtype=np.int64)
    for value in np.unique(episode):
        rows = np.flatnonzero(episode == value)
        if len(rows) > 1:
            ordered = rows[np.argsort(t[rows], kind="stable")]
            result[ordered] = np.roll(ordered, 1)
    return result


def cosine_and_retrieval(prediction, target, seed, chunk):
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(target))
    left = numpy_flatten_normalize(prediction)
    right = numpy_flatten_normalize(target)
    true = np.sum(left * right, axis=1)
    false = np.sum(left * right[shuffled], axis=1)
    margin = true - false
    return {
        "cosine_true": float(true.mean()), "cosine_shuffled": float(false.mean()),
        "cosine_margin": float(margin.mean()),
        "cosine_margin_ci95": bootstrap_mean_ci(margin, samples=5000, seed=seed + 1),
        "retrieval": retrieval_metrics(prediction, target, chunk=chunk),
    }


def model_evaluation(
    model, train, test, oracle, shared_space, decoder, config, device, batch_size, seed
):
    bootstrap = int(config["evaluation"]["bootstrap_samples"])
    train_prediction = predict_numpy(model, train, device, batch_size)
    prediction = predict_numpy(model, test, device, batch_size)
    semantics, labels = semantic_probe(train_prediction, prediction, train, test, oracle)
    target = np.asarray(test["u_c"])
    episode = np.asarray(test["episode_id"])
    t = np.asarray(test["t"])
    different = different_episode_permutation(episode, seed + 1)
    shuffled = np.random.default_rng(seed + 2).permutation(len(target))
    wrong_time = wrong_time_indices(episode, t)
    mean_target = np.asarray(train["u_c"], dtype=np.float64).mean(0).astype(np.float32)
    mean_control = np.broadcast_to(mean_target, target.shape)
    prediction_error = row_mse(prediction, target)
    source_controls = {
        "mean_target": mean_control,
        "shuffled_sources": predict_numpy(
            model, test, device, batch_size, u_a=np.asarray(test["u_a"])[shuffled],
            h_current=np.asarray(test["h_current"])[shuffled],
            u_v=np.asarray(test["u_v"])[shuffled] if model.source == "VAH" else None,
        ),
        "different_episode_sources": predict_numpy(
            model, test, device, batch_size, u_a=np.asarray(test["u_a"])[different],
            h_current=np.asarray(test["h_current"])[different],
            u_v=np.asarray(test["u_v"])[different] if model.source == "VAH" else None,
        ),
        "wrong_time_sources": predict_numpy(
            model, test, device, batch_size, u_a=np.asarray(test["u_a"])[wrong_time],
            h_current=np.asarray(test["h_current"])[wrong_time],
            u_v=np.asarray(test["u_v"])[wrong_time] if model.source == "VAH" else None,
        ),
    }
    control_errors = {name: row_mse(value, target) for name, value in source_controls.items()}
    strongest = min(control_errors, key=lambda name: control_errors[name].mean())
    improvement = control_errors[strongest] - prediction_error

    h_mean = np.asarray(train["h_current"], dtype=np.float64).mean(0).astype(np.float32)
    h_inputs = {
        "zero": np.zeros_like(test["h_current"]),
        "mean": np.broadcast_to(h_mean, np.asarray(test["h_current"]).shape),
        "different_episode": np.asarray(test["h_current"])[different],
        "wrong_time": np.asarray(test["h_current"])[wrong_time],
        "time_shuffled": np.asarray(test["h_current"])[shuffled],
    }
    h_predictions = {
        name: predict_numpy(model, test, device, batch_size, h_current=value)
        for name, value in h_inputs.items()
    }
    h_errors = {name: row_mse(value, target) for name, value in h_predictions.items()}
    strongest_h = min(h_errors, key=lambda name: h_errors[name].mean())
    h_gain = h_errors[strongest_h] - prediction_error
    contact_probe_model = fit_probe(train_prediction, train["contact_transition"])
    h_semantics = {}
    for name, value in h_predictions.items():
        predicted_labels = contact_probe_model.predict(value.reshape(len(value), -1))
        h_semantics[name] = evaluate_prediction(
            test["contact_transition"], predicted_labels,
            majority(train["contact_transition"], len(value), 4), 4,
        )

    action_inputs = {
        "reversed_surrogate": np.asarray(test["u_a"])[:, ::-1].copy(),
        "shuffled": np.asarray(test["u_a"])[shuffled],
        "different_episode": np.asarray(test["u_a"])[different],
    }
    action_predictions = {
        name: predict_numpy(model, test, device, batch_size, u_a=value)
        for name, value in action_inputs.items()
    }
    action_errors = {name: row_mse(value, target) for name, value in action_predictions.items()}
    dynamic = np.asarray(test["dynamic"], dtype=bool)
    action_metrics = {}
    for name, value in action_predictions.items():
        semantic_value, _ = semantic_probe(train_prediction, value, train, test, oracle)
        gain = action_errors[name][dynamic] - prediction_error[dynamic]
        action_metrics[name] = {
            "dynamic_mse": float(action_errors[name][dynamic].mean()),
            "improvement_over_correct": float(gain.mean()),
            "improvement_ci95": bootstrap_mean_ci(gain, samples=bootstrap, seed=seed + 20 + len(action_metrics)),
            "contact_f1": float(semantic_value["contact_transition"]["macro_f1"]),
            "force_f1": float(semantic_value["force_trend_class"]["macro_f1"]),
            "future_change_f1": float(semantic_value["contact_transition"]["future_change"]["macro_f1"]),
        }

    oracle_future = physics_prediction(
        shared_space, decoder, target, test["h_current"], device, batch_size
    )
    predicted_future = physics_prediction(
        shared_space, decoder, prediction, test["h_current"], device, batch_size
    )
    native_future = np.asarray(test["h_future"])
    physics_controls = {
        "mean_target": physics_prediction(shared_space, decoder, mean_control, test["h_current"], device, batch_size),
        "shuffled_source": physics_prediction(shared_space, decoder, source_controls["shuffled_sources"], test["h_current"], device, batch_size),
        "invalid_h": physics_prediction(shared_space, decoder, h_predictions[strongest_h], test["h_current"], device, batch_size),
    }
    physics_error = row_mse(predicted_future, oracle_future)
    physics_control_errors = {name: row_mse(value, oracle_future) for name, value in physics_controls.items()}
    strongest_physics = min(physics_control_errors, key=lambda name: physics_control_errors[name].mean())
    physics_gain = physics_control_errors[strongest_physics] - physics_error
    geometry = geometry_diagnostics(prediction)
    geometry["cka_with_oracle"] = linear_cka(prediction, target)
    noncollapse = bool(
        geometry["per_dimension_variance"]["near_zero_fraction"] < 0.5
        and geometry["query_diversity"]["collapsed_pair_fraction"] < 0.5
    )
    retrieval = cosine_and_retrieval(
        prediction, target, seed + 30, int(config["evaluation"]["retrieval_chunk"])
    )
    retrieval_value = retrieval["retrieval"]
    latent_gate = bool(
        improvement.mean() > 0
        and bootstrap_mean_ci(improvement, samples=bootstrap, seed=seed + 31)[0] > 0
        and retrieval["cosine_margin_ci95"][0] > 0
        and retrieval_value["recall_at_10"] >= 1.5 * retrieval_value["chance"]["recall_at_10"]
    )
    physics_gate = bool(
        bootstrap_mean_ci(physics_gain, samples=bootstrap, seed=seed + 32)[0] > 0
        and bootstrap_mean_ci(physics_gain[dynamic], samples=bootstrap, seed=seed + 33)[0] > 0
    )
    h_gate = bool(
        bootstrap_mean_ci(h_gain, samples=bootstrap, seed=seed + 34)[0] > 0
        and semantics["contact_transition"]["macro_f1"]
        > max(value["macro_f1"] for value in h_semantics.values())
    )
    return {
        "source": model.source, "prediction": prediction,
        "semantics": semantics, "semantic_labels": labels,
        "shared_target": {
            "prediction_mse": float(prediction_error.mean()),
            "dynamic_mse": float(prediction_error[dynamic].mean()),
            "controls_mse": {name: float(value.mean()) for name, value in control_errors.items()},
            "strongest_control": strongest, "improvement": float(improvement.mean()),
            "improvement_ci95": bootstrap_mean_ci(improvement, samples=bootstrap, seed=seed + 31),
            **retrieval, "gate": latent_gate,
        },
        "h_context": {
            "correct_mse": float(prediction_error.mean()),
            "controls_mse": {name: float(value.mean()) for name, value in h_errors.items()},
            "strongest_control": strongest_h, "improvement": float(h_gain.mean()),
            "improvement_ci95": bootstrap_mean_ci(h_gain, samples=bootstrap, seed=seed + 34),
            "correct_contact_f1": float(semantics["contact_transition"]["macro_f1"]),
            "invalid_semantics": h_semantics, "gate": h_gate,
        },
        "action_temporal": {
            "method": "shared-token reversal surrogate; exact frozen A-R raw-action reversal unavailable",
            "exact_ar_transform": False,
            "correct_dynamic_mse": float(prediction_error[dynamic].mean()),
            "variants": action_metrics, "gate": False,
        },
        "physics": {
            "oracle_shared_self_mse": 0.0,
            "prediction_mse": float(physics_error.mean()),
            "prediction_dynamic_mse": float(physics_error[dynamic].mean()),
            "controls_mse": {name: float(value.mean()) for name, value in physics_control_errors.items()},
            "strongest_control": strongest_physics,
            "improvement": float(physics_gain.mean()),
            "improvement_ci95": bootstrap_mean_ci(physics_gain, samples=bootstrap, seed=seed + 32),
            "dynamic_improvement_ci95": bootstrap_mean_ci(physics_gain[dynamic], samples=bootstrap, seed=seed + 33),
            "native_future_mse": float(row_mse(predicted_future, native_future).mean()),
            "native_future_dynamic_mse": float(row_mse(predicted_future, native_future)[dynamic].mean()),
            "gate": physics_gate,
        },
        "geometry": geometry, "noncollapse": noncollapse,
    }


def strip_arrays(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return None
    if isinstance(value, dict):
        return {key: strip_arrays(item) for key, item in value.items() if key not in {"prediction", "semantic_labels"}}
    if isinstance(value, list):
        return [strip_arrays(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.bootstrap_samples is not None:
        config["evaluation"]["bootstrap_samples"] = int(args.bootstrap_samples)
    artifact_root = ROOT / config["runtime"]["artifact_root"]
    selection = validate_selection_lock(config)
    identities_before = identity_snapshot(config)
    if not identities_before["pass"]:
        raise RuntimeError("STRUCTURAL_FAIL: frozen identities changed before locked test")
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
        # The locked benchmark is loaded only after the selection hash and checkpoint pass.
        train = load_aligned_split(config, "train")
        test = load_aligned_split(config, "test")
        shared_space, _, shared_before = load_frozen_shared_space(config, device)
        s2 = load_s2_model(ROOT / config["runtime"]["s2_checkpoint"], device)
        decoder = s2.decoder.eval().requires_grad_(False)
        oracle = oracle_probe(train, test)
        training = json.loads((artifact_root / "training_summary.json").read_text())
        trial_by_source = {}
        for source in ("AH", "VAH"):
            candidates = [row for row in training["trials"] if row["trial"]["source"] == source]
            trial_by_source[source] = max(candidates, key=lambda row: float(row["best"]["utility"]))
        evaluated = {}
        for offset, source in enumerate(("AH", "VAH")):
            trial = trial_by_source[source]
            if sha256_file(ROOT / trial["best"]["checkpoint"]) != trial["best"]["checkpoint_sha256"]:
                raise RuntimeError("STRUCTURAL_FAIL: frozen ablation checkpoint changed")
            model, metadata = load_checkpoint(ROOT / trial["best"]["checkpoint"], device)
            model.eval().requires_grad_(False)
            if metadata.get("test_loaded") is not False:
                raise RuntimeError("STRUCTURAL_FAIL: ablation checkpoint saw test")
            evaluated[source] = model_evaluation(
                model, train, test, oracle, shared_space, decoder, config, device,
                args.batch_size, int(config["seed"]) + offset * 1000,
            )
        clean = {name: strip_arrays(value) for name, value in evaluated.items()}
        ah_labels = evaluated["AH"]["semantic_labels"]
        vah_labels = evaluated["VAH"]["semantic_labels"]
        bootstrap = int(config["evaluation"]["bootstrap_samples"])
        increments = {}
        for offset, metric in enumerate(("contact_transition", "force_trend_class")):
            increments[metric] = {
                "f1_gain": float(
                    clean["VAH"]["semantics"][metric]["macro_f1"]
                    - clean["AH"]["semantics"][metric]["macro_f1"]
                ),
                "bootstrap_ci95": bootstrap_f1_difference(
                    test[metric], vah_labels[metric], ah_labels[metric],
                    samples=bootstrap, seed=int(config["seed"]) + 5000 + offset,
                ),
            }
        contact_ci = increments["contact_transition"]["bootstrap_ci95"]
        force_ci = increments["force_trend_class"]["bootstrap_ci95"]
        if contact_ci[0] > 0 and force_ci[0] > 0:
            vision_classification = "VISION_MATERIALLY_IMPROVES_CONTACT_PREDICTION"
        elif contact_ci[0] > 0 or force_ci[0] > 0:
            vision_classification = "VISION_SMALL_BUT_POSITIVE_GAIN"
        elif (
            clean["AH"]["semantics"]["contact_transition"]["semantic_ratio"] >= 0.75
            and clean["AH"]["semantics"]["force_trend_class"]["semantic_ratio"] >= 0.75
            and clean["AH"]["shared_target"]["gate"]
            and clean["AH"]["physics"]["gate"]
            and clean["AH"]["h_context"]["gate"]
        ):
            vision_classification = "A_PLUS_H_SUFFICIENT_VISION_OPTIONAL"
        else:
            vision_classification = "VISION_NO_MEASURABLE_GAIN"
        selected = clean[selection["source"]]
        semantic_gate = bool(
            selected["semantics"]["contact_transition"]["semantic_ratio"] >= 0.75
            and selected["semantics"]["force_trend_class"]["semantic_ratio"] >= 0.75
        )
        if not semantic_gate:
            decision = "C3MSCC_SEMANTIC_INSUFFICIENT"
        elif not selected["h_context"]["gate"]:
            decision = "C3MSCC_CAUSAL_CONTEXT_BYPASS"
        elif not selected["action_temporal"]["gate"]:
            decision = "C3MSCC_ACTION_TEMPORAL_FAIL"
        elif not selected["physics"]["gate"]:
            decision = "C3MSCC_PHYSICS_INSUFFICIENT"
        elif not selected["noncollapse"]:
            decision = "STRUCTURAL_FAIL"
        else:
            warning = selected["geometry"]["effective_rank"] < 0.5 * float(config["evaluation"]["oracle_contact_effective_rank"])
            decision = "C3MSCC_READY_WITH_RANK_WARNING" if warning else (
                "C3MSCC_READY_AH_MINIMAL" if selection["source"] == "AH" else "C3MSCC_READY_VAH"
            )
        evaluation = {
            "schema": "tactile3d-unit.vac-c3mscc-locked-evaluation.v1",
            "evaluation_type": "LOCKED BENCHMARK RE-EVALUATION AFTER C3-R0 DIAGNOSIS",
            "first_look_untouched": False, "rows": len(test["u_c"]),
            "selection_sha256": sha256_file(artifact_root / "selection.json"),
            "selection": selection, "oracle": oracle,
            "sources": clean, "vision_incremental": increments,
            "vision_classification": vision_classification,
            "selected_source": selection["source"], "semantic_gate": semantic_gate,
            "decision": decision, "gpu": gpu,
            "identity_before": identities_before, "identity_after": identity_snapshot(config),
            "shared_state_before": shared_before,
            "shared_state_after": state_dict_digest(shared_space),
            "test_loaded": True,
        }
        atomic_json(artifact_root / "locked_test_evaluation.json", evaluation)
        atomic_json(artifact_root / "source_ablation.json", {
            "sources": clean, "vision_incremental": increments,
            "vision_classification": vision_classification, "test_loaded": True,
        })
        atomic_json(artifact_root / "context_ablation.json", {
            name: value["h_context"] for name, value in clean.items()
        })
        atomic_json(artifact_root / "temporal_ablation.json", {
            name: value["action_temporal"] for name, value in clean.items()
        })
        final = {
            "decision": decision,
            "reasons": [
                f"semantic gate={'PASS' if semantic_gate else 'FAIL'}",
                f"H-context gate={'PASS' if selected['h_context']['gate'] else 'FAIL'}",
                "exact frozen A-R reversed Action audit unavailable; temporal gate fails closed",
                f"shared physics gate={'PASS' if selected['physics']['gate'] else 'FAIL'}",
            ],
            "c4_readiness": "NOT READY" if not decision.startswith("C3MSCC_READY") else "READY",
            "m3": "NOT ESTABLISHED", "c4": "NOT STARTED", "c5": "NOT STARTED",
            "c6": "NOT STARTED", "test_loaded": True,
        }
        atomic_json(artifact_root / "final_decision.json", final)
        print(artifact_root / "locked_test_evaluation.json")
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    main()
