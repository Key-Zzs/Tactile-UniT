#!/usr/bin/env python3
"""Run the six bounded validation-only C5 causal fallback trials and freeze selection."""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.c3r0_conditional_sufficiency import bootstrap_f1_difference  # noqa: E402
from gr00t.tactile_unit.c5_causal_visual import (  # noqa: E402
    C5LossWeights, CausalVisualEncoder, CausalVisionSubstituter,
    DirectCausalContactPredictor, ModularCausalContactPredictor, VisualSupport,
    causal_fallback_loss, load_causal_checkpoint, save_causal_checkpoint,
)
from gr00t.tactile_unit.continuous_vac_shared_space import (  # noqa: E402
    bootstrap_mean_ci, different_episode_permutation, geometry_diagnostics,
    linear_cka, numpy_flatten_normalize, retrieval_metrics, state_dict_digest,
)
from scripts.tactile_unit.c3mscc_runtime import load_frozen_shared_space  # noqa: E402
from scripts.tactile_unit.c5_runtime import (  # noqa: E402
    DEFAULT_CONFIG, atomic_json, identity_snapshot, load_c4_fallbacks,
    load_config, load_split, predict_causal, visual_batch,
)
from scripts.tactile_unit.continuous_contact_bridge_common import load_s2_model  # noqa: E402
from scripts.tactile_unit.train_c3mscc_contact_prediction import (  # noqa: E402
    fit_probe, oracle_semantics, physics_prediction, row_mse, semantic_evaluation,
)
from scripts.tactile_unit.vac_runtime_common import resolve_device, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--only-trial")
    parser.add_argument("--append-existing", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def trainable_parameters(visual: torch.nn.Module, predictor: torch.nn.Module) -> list[torch.nn.Parameter]:
    values = [parameter for parameter in visual.parameters() if parameter.requires_grad]
    values.extend(parameter for parameter in predictor.parameters() if parameter.requires_grad)
    return values


def make_models(trial: Mapping[str, Any], config: Mapping[str, Any], offline_va, device: torch.device):
    support = VisualSupport(trial["support"])
    visual = CausalVisualEncoder(
        support, layers=int(config["visual"]["temporal_layers"]),
        heads=int(config["visual"]["heads"]), mlp_width=64,
    ).to(device)
    if trial["family"] == "direct":
        predictor = DirectCausalContactPredictor(visual_head=bool(trial["visual_distillation"])).to(device)
    elif trial["family"] == "modular":
        predictor = ModularCausalContactPredictor(CausalVisionSubstituter(), offline_va).to(device)
    else:
        raise ValueError("unknown bounded C5 family")
    return support, visual, predictor


def labels_for_metrics(train_prediction, prediction, train, validation):
    return {
        metric: np.asarray(fit_probe(train_prediction, train[metric]).predict(prediction.reshape(len(prediction), -1)), dtype=np.int64)
        for metric in ("contact_transition", "force_trend_class")
    }


def mean_visual_features(split, support, batch_size=4096):
    total = None
    count = 0
    for start in range(0, len(split["u_c"]), batch_size):
        stop = min(start + batch_size, len(split["u_c"]))
        value = visual_batch(split, support, slice(start, stop)).astype(np.float64)
        current = value.sum(0)
        total = current if total is None else total + current
        count += len(value)
    return (total / count).astype(np.float32)


def past_time_indices(episode: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Select the immediately preceding available row, never a future row."""
    result = np.arange(len(episode), dtype=np.int64)
    for value in np.unique(episode):
        rows = np.flatnonzero(episode == value)
        ordered = rows[np.argsort(anchor[rows], kind="stable")]
        if len(ordered) > 1:
            result[ordered[1:]] = ordered[:-1]
    if np.any(anchor[result] > anchor):
        raise RuntimeError("CAUSAL_LEAKAGE_FAIL: wrong-time visual control selected a future row")
    return result


def quick_validation(visual, predictor, support, train, validation, oracle, device, batch_size):
    train_prediction = predict_causal(visual, predictor, train, support, device, batch_size)
    prediction = predict_causal(visual, predictor, validation, support, device, batch_size)
    semantics, _ = semantic_evaluation(train_prediction, prediction, train, validation, oracle)
    error = row_mse(prediction, validation["u_c"])
    mean = np.asarray(train["u_c"], dtype=np.float64).mean(0).astype(np.float32)
    mean_error = row_mse(np.broadcast_to(mean, prediction.shape), validation["u_c"])
    geometry = geometry_diagnostics(prediction)
    noncollapse = geometry["per_dimension_variance"]["near_zero_fraction"] < 0.5 and geometry["query_diversity"]["collapsed_pair_fraction"] < 0.5
    return {
        "contact": float(semantics["contact_transition"]["semantic_ratio"]),
        "force": float(semantics["force_trend_class"]["semantic_ratio"]),
        "future_change": float(semantics["contact_transition"]["future_change"]["macro_f1"]),
        "shared_gain": float((mean_error.mean() - error.mean()) / max(mean_error.mean(), 1e-12)),
        "rank": float(geometry["effective_rank"]), "noncollapse": bool(noncollapse),
    }


def complete_validation(
    visual, predictor, support, train, validation, oracle, offline_va_prediction,
    a_prediction, a_train_prediction, config, shared_space, decoder, device, batch_size, seed,
) -> dict[str, Any]:
    bootstrap = int(config["validation"]["bootstrap_samples"])
    train_prediction = predict_causal(visual, predictor, train, support, device, batch_size)
    prediction = predict_causal(visual, predictor, validation, support, device, batch_size)
    semantics, contact_labels = semantic_evaluation(train_prediction, prediction, train, validation, oracle)
    labels = labels_for_metrics(train_prediction, prediction, train, validation)
    a_semantics, _ = semantic_evaluation(a_train_prediction, a_prediction, train, validation, oracle)
    a_labels = labels_for_metrics(a_train_prediction, a_prediction, train, validation)
    target = np.asarray(validation["u_c"])
    error = row_mse(prediction, target)
    a_error = row_mse(a_prediction, target)
    offline_error = row_mse(offline_va_prediction, target)
    mean_target = np.asarray(train["u_c"], dtype=np.float64).mean(0).astype(np.float32)
    mean_prediction = np.broadcast_to(mean_target, target.shape)
    episode, anchor = np.asarray(validation["episode_id"]), np.asarray(validation["t"])
    shuffled = np.random.default_rng(seed + 1).permutation(len(target))
    different = different_episode_permutation(episode, seed + 2)
    wrong = past_time_indices(episode, anchor)
    base_visual = visual_batch(validation, support, slice(None))
    mean_visual = np.broadcast_to(mean_visual_features(train, support), base_visual.shape)
    visual_variants = {
        "zero": np.zeros_like(base_visual), "mean": mean_visual,
        "shuffled": base_visual[shuffled], "different_episode": base_visual[different],
        "wrong_time_past": base_visual[wrong],
    }
    if support is VisualSupport.CAUSAL_HISTORY_8:
        visual_variants["reversed_history"] = base_visual[:, ::-1].copy()
        history_shuffle = np.random.default_rng(seed + 3).random((len(base_visual), 8)).argsort(1)
        visual_variants["shuffled_history"] = np.take_along_axis(base_visual, history_shuffle[:, :, None, None], axis=1)
    invalid_predictions = {
        name: predict_causal(visual, predictor, validation, support, device, batch_size, visual_override=value)
        for name, value in visual_variants.items()
    }
    invalid_errors = {name: row_mse(value, target) for name, value in invalid_predictions.items()}
    mean_error = row_mse(mean_prediction, target)
    shared_controls = {"mean_target": mean_error, "a_only": a_error, **invalid_errors}
    shared_cis = {name: bootstrap_mean_ci(control - error, samples=bootstrap, seed=seed + 20 + index) for index, (name, control) in enumerate(shared_controls.items())}
    visual_control_name = min(invalid_errors, key=lambda name: invalid_errors[name].mean())
    visual_use_ci = shared_cis[visual_control_name]

    exact_predictions = {
        name: predict_causal(visual, predictor, validation, support, device, batch_size, u_a=np.asarray(validation[f"u_a_{name}"]))
        for name in ("reversed", "shuffled", "different")
    }
    dynamic = np.asarray(validation["dynamic"], dtype=bool)
    exact_errors = {name: row_mse(value, target) for name, value in exact_predictions.items()}
    temporal = {}
    for index, name in enumerate(("reversed", "shuffled", "different")):
        difference = exact_errors[name][dynamic] - error[dynamic]
        temporal[name] = {"dynamic_mse": float(exact_errors[name][dynamic].mean()), "difference": float(difference.mean()), "difference_ci95": bootstrap_mean_ci(difference, samples=bootstrap, seed=seed + 50 + index)}

    left, right = numpy_flatten_normalize(prediction), numpy_flatten_normalize(target)
    cosine_true = np.sum(left * right, axis=1)
    cosine_shuffled = np.sum(left * right[shuffled], axis=1)
    cosine_margin = cosine_true - cosine_shuffled
    retrieval = retrieval_metrics(prediction, target, chunk=256)
    geometry = geometry_diagnostics(prediction)
    geometry["cka_with_oracle"] = linear_cka(prediction, target)
    noncollapse = geometry["per_dimension_variance"]["near_zero_fraction"] < 0.5 and geometry["query_diversity"]["collapsed_pair_fraction"] < 0.5

    oracle_future = physics_prediction(shared_space, decoder, target, validation["h_current"], device, batch_size)
    physics_prediction_rows = physics_prediction(shared_space, decoder, prediction, validation["h_current"], device, batch_size)
    physics_error = row_mse(physics_prediction_rows, oracle_future)
    physics_controls = {
        "a_only": row_mse(physics_prediction(shared_space, decoder, a_prediction, validation["h_current"], device, batch_size), oracle_future),
        "offline_oracle_va": row_mse(physics_prediction(shared_space, decoder, offline_va_prediction, validation["h_current"], device, batch_size), oracle_future),
        "mean_target": row_mse(physics_prediction(shared_space, decoder, mean_prediction, validation["h_current"], device, batch_size), oracle_future),
    }
    for name, value in invalid_predictions.items():
        physics_controls[name] = row_mse(physics_prediction(shared_space, decoder, value, validation["h_current"], device, batch_size), oracle_future)
    nonoracle_physics = {name: value for name, value in physics_controls.items() if name != "offline_oracle_va"}
    strongest_physics = min(nonoracle_physics, key=lambda name: nonoracle_physics[name][dynamic].mean())
    physics_gain = nonoracle_physics[strongest_physics] - physics_error

    contact_ci = bootstrap_f1_difference(validation["contact_transition"], labels["contact_transition"], a_labels["contact_transition"], samples=bootstrap, seed=seed + 80)
    force_ci = bootstrap_f1_difference(validation["force_trend_class"], labels["force_trend_class"], a_labels["force_trend_class"], samples=bootstrap, seed=seed + 81)
    mse_a_ci = bootstrap_mean_ci(a_error - error, samples=bootstrap, seed=seed + 82)
    visual_gain_gate = bool(mse_a_ci[0] > 0 and (contact_ci[0] > 0 or force_ci[0] > 0 or semantics["contact_transition"]["future_change"]["macro_f1"] > a_semantics["contact_transition"]["future_change"]["macro_f1"]))
    action_gate = bool(temporal["reversed"]["difference_ci95"][0] > 0 and temporal["shuffled"]["difference_ci95"][0] > 0 and temporal["different"]["difference"] > 0)
    shared_gate = bool(mean_error.mean() > error.mean() and shared_cis["mean_target"][0] > 0 and visual_use_ci[0] > 0 and bootstrap_mean_ci(cosine_margin, samples=bootstrap, seed=seed + 83)[0] > 0 and retrieval["recall_at_10"] >= 1.5 * retrieval["chance"]["recall_at_10"])
    physics_gate = bool(bootstrap_mean_ci(physics_gain[dynamic], samples=bootstrap, seed=seed + 84)[0] > 0)
    gates = {
        "contact": semantics["contact_transition"]["semantic_ratio"] >= float(config["evaluation"]["contact_retention_min"]),
        "force": semantics["force_trend_class"]["semantic_ratio"] >= float(config["evaluation"]["force_retention_min"]),
        "future_change": semantics["contact_transition"]["future_change"]["macro_f1"] > a_semantics["contact_transition"]["future_change"]["macro_f1"],
        "visual_gain_over_a_only": visual_gain_gate, "shared_prediction": shared_gate,
        "physics": physics_gate, "action_temporal": action_gate, "visual_context": visual_use_ci[0] > 0,
        "noncollapse": bool(noncollapse),
    }
    weights = config["validation"]["utility"]
    shared_gain = float((a_error.mean() - error.mean()) / max(a_error.mean(), 1e-12))
    physics_normalized = float(physics_gain[dynamic].mean() / max(nonoracle_physics[strongest_physics][dynamic].mean(), 1e-12))
    temporal_normalized = float(np.mean([temporal[name]["difference"] for name in ("reversed", "shuffled")]) / max(error[dynamic].mean(), 1e-12))
    visual_normalized = float((invalid_errors[visual_control_name].mean() - error.mean()) / max(invalid_errors[visual_control_name].mean(), 1e-12))
    utility = (weights["contact"] * semantics["contact_transition"]["semantic_ratio"] + weights["force"] * semantics["force_trend_class"]["semantic_ratio"] + weights["future_change"] * semantics["contact_transition"]["future_change"]["macro_f1"] + weights["shared"] * shared_gain + weights["physics"] * physics_normalized + weights["temporal"] * temporal_normalized + weights["visual"] * visual_normalized + weights["noncollapse"] * float(noncollapse))
    return {
        "selection_split": "validation only", "test_loaded": False, "rows": len(target), "utility": float(utility),
        "support": support.value, "semantics": semantics, "a_only_semantics": a_semantics,
        "visual_contribution": {"a_only_mse": float(a_error.mean()), "prediction_mse": float(error.mean()), "mse_gain_ci95": mse_a_ci, "contact_f1_gain_ci95": contact_ci, "force_f1_gain_ci95": force_ci, "gate": visual_gain_gate},
        "shared_target": {"prediction_mse": float(error.mean()), "a_only_mse": float(a_error.mean()), "offline_oracle_mse": float(offline_error.mean()), "controls_mse": {name: float(value.mean()) for name, value in shared_controls.items()}, "improvement_ci95": shared_cis, "cosine_true": float(cosine_true.mean()), "cosine_shuffled": float(cosine_shuffled.mean()), "cosine_margin": float(cosine_margin.mean()), "cosine_margin_ci95": bootstrap_mean_ci(cosine_margin, samples=bootstrap, seed=seed + 83), "retrieval": retrieval},
        "visual_context": {"correct_mse": float(error.mean()), "variants_mse": {name: float(value.mean()) for name, value in invalid_errors.items()}, "strongest_invalid": visual_control_name, "improvement_ci95": visual_use_ci, "gate": visual_use_ci[0] > 0},
        "physics": {"teacher_side_h_only": True, "prediction_mse": float(physics_error.mean()), "dynamic_mse": float(physics_error[dynamic].mean()), "controls_mse": {name: float(value.mean()) for name, value in physics_controls.items()}, "strongest_nonoracle_dynamic_control": strongest_physics, "dynamic_improvement_ci95": bootstrap_mean_ci(physics_gain[dynamic], samples=bootstrap, seed=seed + 84), "gate": physics_gate},
        "action_temporal": {"exact_raw_action_transform": True, "correct_dynamic_mse": float(error[dynamic].mean()), "variants": temporal, "gate": action_gate},
        "geometry": geometry, "noncollapse": bool(noncollapse), "gates": {**gates, "all": all(gates.values())},
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    artifact_root, experiment_root = ROOT / config["runtime"]["artifact_root"], ROOT / config["runtime"]["experiment_root"]
    if not json.loads((artifact_root / "causal_visual_cache_manifest.json").read_text())["splits"].get("train", {}).get("complete"):
        raise RuntimeError("complete train/validation C5 visual cache required")
    identities_before = identity_snapshot(config)
    if not identities_before["pass"]:
        raise RuntimeError("STRUCTURAL_FAIL: frozen identity mismatch")
    device, lock_handle, gpu = resolve_device(args.device, allowed_physical=("0", "1", "2", "3"))
    try:
        set_seed(int(config["seed"]))
        train, validation = load_split(config, "train"), load_split(config, "validation")
        parent = json.loads((ROOT / "configs/tactile_unit/c3mscc_contact_context_prediction.json").read_text())
        shared_space, _, shared_before = load_frozen_shared_space(parent, device)
        decoder = load_s2_model(ROOT / config["runtime"]["s2_checkpoint"], device).decoder.eval().requires_grad_(False)
        offline_va, a_only, _, _ = load_c4_fallbacks(config, device)
        oracle = oracle_semantics(train, validation)
        a_train_prediction = np.load(ROOT / ".local/cache/tactile_unit/vac_c4/train/prediction_FALLBACK_A.npy", mmap_mode="r")
        a_prediction = np.load(ROOT / ".local/cache/tactile_unit/vac_c4/validation/prediction_FALLBACK_A.npy", mmap_mode="r")
        offline_va_prediction = np.load(ROOT / ".local/cache/tactile_unit/vac_c4/validation/prediction_FALLBACK_VA.npy", mmap_mode="r")
        trials = list(config["training"]["trials"])[:min(6, int(args.max_trials or 6))]
        atomic_json(artifact_root / "trial_manifest.json", {"schema": "tactile3d-unit.vac-c5-trials.v1", "planned": trials, "maximum_allowed": 6, "selection_split": "validation only", "test_loaded": False, "gpu": {**gpu, "preferred_physical": 1, "fallback": gpu.get("actual_physical") != 1}})
        results = []
        if args.append_existing:
            results = json.loads((artifact_root / "training_summary.json").read_text())["trials"]
        indexed = list(enumerate(trials))
        if args.only_trial:
            indexed = [row for row in indexed if row[1]["id"] == args.only_trial]
        for trial_index, trial in indexed:
            set_seed(int(config["seed"]) + trial_index * 100)
            support, visual, predictor = make_models(trial, config, offline_va, device)
            parameters = trainable_parameters(visual, predictor)
            optimizer = torch.optim.AdamW(parameters, lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
            weights = C5LossWeights(**config["training"]["loss_weights"])
            trial_root = experiment_root / f"trial_{trial_index:02d}_{trial['id']}"
            history, best, stale = [], None, 0
            epochs = int(args.epochs or config["training"]["epochs"])
            started = time.monotonic()
            for epoch in (() if args.finalize_only else range(1, epochs + 1)):
                visual.train(); predictor.train()
                order = np.random.default_rng(int(config["seed"]) + trial_index * 1000 + epoch).permutation(len(train["u_c"]))
                totals, batches = {}, 0
                for start in range(0, len(order), int(config["training"]["batch_size"])):
                    rows = order[start:start + int(config["training"]["batch_size"])]
                    if len(rows) < 2:
                        continue
                    tensor = lambda name: torch.from_numpy(np.array(train[name][rows], copy=True)).to(device)
                    frozen_features = torch.from_numpy(visual_batch(train, support, rows)).to(device).detach()
                    c_v = visual(frozen_features)
                    enhanced = bool(trial["physics_covariance"])
                    optimizer.zero_grad(set_to_none=True)
                    loss, terms = causal_fallback_loss(
                        predictor, shared_space, decoder, c_v=c_v, u_a=tensor("u_a"), u_c=tensor("u_c"),
                        dynamic=tensor("dynamic"), teacher_h_current=tensor("h_current").detach() if enhanced else None,
                        teacher_u_v=tensor("u_v").detach() if trial["visual_distillation"] else None,
                        invalid_u_a=(tensor("u_a_reversed"), tensor("u_a_shuffled")) if enhanced else (),
                        enhanced=enhanced, dynamic_weight=float(config["training"]["dynamic_weight"]),
                        order_margin=float(config["training"]["order_margin"]), variance_floor=float(config["training"]["variance_floor"]), weights=weights,
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite C5 loss")
                    loss.backward()
                    if any(parameter.grad is not None for parameter in shared_space.parameters()) or any(parameter.grad is not None for parameter in decoder.parameters()) or any(parameter.grad is not None for parameter in offline_va.parameters()):
                        raise RuntimeError("STRUCTURAL_FAIL: C5 gradient reached a frozen accepted component")
                    torch.nn.utils.clip_grad_norm_(parameters, float(config["training"]["gradient_clip"]))
                    optimizer.step()
                    for name, value in terms.items(): totals[name] = totals.get(name, 0.0) + float(value)
                    batches += 1
                quick = quick_validation(visual, predictor, support, train, validation, oracle, device, int(config["training"]["batch_size"]))
                score = 0.32 * quick["contact"] + 0.26 * quick["force"] + 0.12 * quick["future_change"] + 0.22 * quick["shared_gain"] + 0.08 * float(quick["noncollapse"])
                history.append({"epoch": epoch, "train": {name: value / max(batches, 1) for name, value in totals.items()}, "validation": quick, "selection_score": float(score)})
                if best is None or score > best["selection_score"]:
                    checkpoint = trial_root / "best.pt"
                    digest = save_causal_checkpoint(checkpoint, visual, predictor, {"trial": trial["id"], "family": trial["family"], "support": support.value, "epoch": epoch, "selection_split": "validation only", "test_loaded": False, "true_u_v_input": False, "teacher_h_predictor_input": False, "private_residual": False})
                    best = {"epoch": epoch, "selection_score": float(score), "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": digest}
                    stale = 0
                else:
                    stale += 1
                atomic_json(trial_root / "history.json", {"trial": trial, "history": history, "best": best})
                if stale >= int(config["training"]["patience"]): break
            if args.finalize_only:
                saved = json.loads((trial_root / "history.json").read_text()); history, best = saved["history"], saved["best"]
            selected_visual, selected_predictor, metadata = load_causal_checkpoint(ROOT / best["checkpoint"], device, frozen_f_va=offline_va)
            metrics = complete_validation(selected_visual, selected_predictor, support, train, validation, oracle, offline_va_prediction, a_prediction, a_train_prediction, config, shared_space, decoder, device, int(config["training"]["batch_size"]), int(config["seed"]) + trial_index * 1000)
            results.append({"index": trial_index, "trial": trial, "parameters": {"trainable": sum(parameter.numel() for parameter in trainable_parameters(selected_visual, selected_predictor)), "frozen_offline_va": sum(parameter.numel() for parameter in offline_va.parameters()) if trial["family"] == "modular" else 0}, "epochs_ran": len(history), "seconds": time.monotonic() - started, "best": {**best, "validation": metrics}})
            atomic_json(artifact_root / "training_summary.partial.json", {"trials": results, "test_loaded": False})
        unique = {row["trial"]["id"]: row for row in results}
        results = [unique[f"T{index}"] for index in range(len(trials))]
        valid = [row for row in results if row["best"]["validation"]["gates"]["all"]]
        pool = valid or results
        best_overall = max(pool, key=lambda row: row["best"]["validation"]["utility"])
        tolerance = float(config["validation"]["simplicity_tolerance"])
        current_direct = [row for row in valid if row["trial"]["family"] == "direct" and row["trial"]["support"] == "CURRENT_FRAME"]
        direct = [row for row in valid if row["trial"]["family"] == "direct"]
        if current_direct and max(current_direct, key=lambda row: row["best"]["validation"]["utility"])["best"]["validation"]["utility"] >= best_overall["best"]["validation"]["utility"] - tolerance:
            selected = max(current_direct, key=lambda row: row["best"]["validation"]["utility"]); rationale = "current-frame direct passes all gates and is within 0.01 of best validation utility"
        elif direct and max(direct, key=lambda row: row["best"]["validation"]["utility"])["best"]["validation"]["utility"] >= best_overall["best"]["validation"]["utility"] - float(config["validation"]["direct_tolerance"]):
            selected = max(direct, key=lambda row: row["best"]["validation"]["utility"]); rationale = "direct candidate passes all gates and is within 0.01 of best valid candidate"
        else:
            selected = best_overall; rationale = "validation-only best valid candidate; frozen simplicity preferences did not apply"
        support_best = {
            name: max((row for row in results if row["trial"]["support"] == name), key=lambda row: row["best"]["validation"]["utility"])
            for name in ("CURRENT_FRAME", "CAUSAL_HISTORY_8")
        }
        support_errors = {}
        for name, row in support_best.items():
            compare_visual, compare_predictor, _ = load_causal_checkpoint(
                ROOT / row["best"]["checkpoint"], device, frozen_f_va=offline_va,
            )
            compare_prediction = predict_causal(
                compare_visual, compare_predictor, validation, VisualSupport(name), device,
                int(config["training"]["batch_size"]),
            )
            support_errors[name] = row_mse(compare_prediction, validation["u_c"])
        current_minus_history = support_errors["CURRENT_FRAME"] - support_errors["CAUSAL_HISTORY_8"]
        family_best = {
            family: max((row for row in results if row["trial"]["family"] == family), key=lambda row: row["best"]["validation"]["utility"])
            for family in ("direct", "modular")
        }
        comparisons = {
            "current_vs_history": {
                "current_candidate": support_best["CURRENT_FRAME"]["trial"]["id"],
                "history_candidate": support_best["CAUSAL_HISTORY_8"]["trial"]["id"],
                "current_utility": support_best["CURRENT_FRAME"]["best"]["validation"]["utility"],
                "history_utility": support_best["CAUSAL_HISTORY_8"]["best"]["validation"]["utility"],
                "current_minus_history_shared_mse": float(current_minus_history.mean()),
                "current_minus_history_shared_mse_ci95": bootstrap_mean_ci(
                    current_minus_history, samples=int(config["validation"]["bootstrap_samples"]),
                    seed=int(config["seed"]) + 9900,
                ),
            },
            "direct_vs_modular": {
                "direct_candidate": family_best["direct"]["trial"]["id"],
                "modular_candidate": family_best["modular"]["trial"]["id"],
                "direct_utility": family_best["direct"]["best"]["validation"]["utility"],
                "modular_utility": family_best["modular"]["best"]["validation"]["utility"],
            },
        }
        summary = {"schema": "tactile3d-unit.vac-c5-training.v1", "trials": results, "total_trials": len(results), "maximum_trials": 6, "selection_split": "validation only", "test_loaded": False, "identity_before": identities_before, "shared_state_before": shared_before}
        atomic_json(artifact_root / "training_summary.json", summary)
        selection = {"schema": "tactile3d-unit.vac-c5-causal-selection.v1", "candidate": selected["trial"]["id"], "visual_support": selected["trial"]["support"], "family": selected["trial"]["family"], "architecture": "bounded causal visual encoder + " + selected["trial"]["family"] + " Contact fallback", "epoch": selected["best"]["epoch"], "loss_weights": config["training"]["loss_weights"], "checkpoint": selected["best"]["checkpoint"], "checkpoint_sha256": selected["best"]["checkpoint_sha256"], "validation_utility": selected["best"]["validation"]["utility"], "validation_metrics": selected["best"]["validation"], "comparisons": comparisons, "all_validation_gates": selected["best"]["validation"]["gates"]["all"], "rationale": rationale, "selected_via": "VALIDATION ONLY", "selection_split": "validation only", "test_loaded": False, "identity": identity_snapshot(config)}
        path = artifact_root / "causal_visual_selection.json"; atomic_json(path, selection)
        digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest(); (artifact_root / "causal_visual_selection.sha256").write_text(digest + "  causal_visual_selection.json\n")
        if not identity_snapshot(config)["pass"] or state_dict_digest(shared_space) != shared_before:
            raise RuntimeError("STRUCTURAL_FAIL: frozen identity changed during C5 training")
        print(json.dumps({"selected": selection["candidate"], "support": selection["visual_support"], "family": selection["family"], "all_gates": selection["all_validation_gates"], "sha256": digest}, indent=2))
    finally:
        if lock_handle is not None: lock_handle.close()


if __name__ == "__main__":
    main()
