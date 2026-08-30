#!/usr/bin/env python3
"""Audit exact Action evidence and perform validation-only C3-MS-CC-R closure."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.c3mscc_contact_context import (  # noqa: E402
    C3MSCCLossWeights, contact_prediction_loss, load_checkpoint, save_checkpoint,
    sha256_file,
)
from gr00t.tactile_unit.c3msccr_exact_action_closure import (  # noqa: E402
    VARIANTS, per_query_distance, row_cosine, row_mse,
)
from gr00t.tactile_unit.continuous_vac_shared_space import (  # noqa: E402
    bootstrap_mean_ci, effective_rank, geometry_diagnostics, linear_cka,
    state_dict_digest,
)
from scripts.tactile_unit.c3mscc_runtime import (  # noqa: E402
    atomic_json, identity_snapshot, load_aligned_split, load_config as load_parent_config,
    load_frozen_shared_space,
)
from scripts.tactile_unit.continuous_contact_bridge_common import load_s2_model  # noqa: E402
from scripts.tactile_unit.evaluate_c3mscc_contact_prediction import (  # noqa: E402
    model_evaluation, oracle_probe, strip_arrays,
)
from scripts.tactile_unit.train_c3mscc_contact_prediction import (  # noqa: E402
    oracle_semantics, physics_prediction, predict_numpy, semantic_evaluation,
    validation_metrics,
)
from scripts.tactile_unit.vac_runtime_common import resolve_device, set_seed  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/tactile_unit/c3msccr_exact_action_closure.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--phase", choices=(
            "integrity", "source-audit", "validation", "remediation", "freeze",
            "locked", "all-pretest",
        ),
        default="all-pretest",
    )
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def exact_split(config: Mapping[str, Any], split: str) -> dict[str, np.ndarray]:
    root = ROOT / config["runtime"]["exact_cache_root"] / split
    result = {}
    for space in ("z_a", "u_a"):
        for variant in VARIANTS:
            path = root / f"{space}_{variant}.npy"
            if not path.is_file():
                raise RuntimeError(f"C3MSCCR_EXACT_ACTION_EVIDENCE_UNAVAILABLE: {path.name}")
            result[f"{space}_{variant}"] = np.load(path, mmap_mode="r", allow_pickle=False)
    for name in ("pair_id", "source_index", "shuffle_order", "different_source_index"):
        result[name] = np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
    for variant in VARIANTS:
        path = root / f"decoder_mse_{variant}.npy"
        if path.is_file():
            result[f"decoder_mse_{variant}"] = np.load(path, mmap_mode="r", allow_pickle=False)
    return result


def latent_comparison(correct: np.ndarray, variant: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    left = np.asarray(correct)[mask]
    right = np.asarray(variant)[mask]
    return {
        "rows": len(left),
        "cosine": float(row_cosine(left, right).mean()),
        "mse": float(row_mse(left, right).mean()),
        "per_query_distance": per_query_distance(left, right),
        "variant_effective_rank": effective_rank(right),
    }


def temporal_integrity(config: Mapping[str, Any]) -> dict[str, Any]:
    split = "validation"
    exact = exact_split(config, split)
    dynamic = np.asarray(
        np.load(
            ROOT / config["runtime"]["c1_cache_root"] / split / "dynamic.npy",
            mmap_mode="r", allow_pickle=False,
        ),
        dtype=bool,
    )
    masks = {"all": np.ones(len(dynamic), dtype=bool), "dynamic": dynamic}
    latent = {}
    for space in ("z_a", "u_a"):
        latent[space] = {}
        for subset, mask in masks.items():
            latent[space][subset] = {
                variant: latent_comparison(
                    exact[f"{space}_correct"], exact[f"{space}_{variant}"], mask
                )
                for variant in ("reversed", "shuffled", "different")
            }
    decoder = {}
    correct = np.asarray(exact["decoder_mse_correct"], dtype=np.float64)
    bootstrap = 5000
    for subset, mask in masks.items():
        decoder[subset] = {"correct": float(correct[mask].mean())}
        for offset, variant in enumerate(("reversed", "shuffled", "different")):
            error = np.asarray(exact[f"decoder_mse_{variant}"], dtype=np.float64)
            difference = error[mask] - correct[mask]
            decoder[subset][variant] = {
                "mse": float(error[mask].mean()),
                "ratio_to_correct": float(error[mask].mean() / max(correct[mask].mean(), 1e-12)),
                "difference": float(difference.mean()),
                "difference_ci95": bootstrap_mean_ci(
                    difference, samples=bootstrap, seed=int(config["seed"]) + offset + (100 if subset == "dynamic" else 0)
                ),
            }
    dynamic_decoder = decoder["dynamic"]
    passed = bool(
        latent["z_a"]["dynamic"]["reversed"]["mse"] > 0
        and latent["z_a"]["dynamic"]["shuffled"]["mse"] > 0
        and dynamic_decoder["reversed"]["difference_ci95"][0] > 0
        and dynamic_decoder["shuffled"]["difference_ci95"][0] > 0
        and dynamic_decoder["different"]["difference_ci95"][0] > 0
    )
    result = {
        "schema": "tactile3d-unit.vac-c3msccr-ar-temporal-integrity.v1",
        "split": "validation",
        "raw_perturbation": True,
        "exact_full_ar_pipeline": True,
        "rows": len(dynamic),
        "dynamic_rows": int(dynamic.sum()),
        "latent": latent,
        "decoder": decoder,
        "a_r_temporal_property_reproduced": passed,
        "decision": "PASS" if passed else "C3MSCCR_AR_TEMPORAL_PROVENANCE_FAIL",
    }
    atomic_json(ROOT / config["runtime"]["artifact_root"] / "ar_temporal_integrity.json", result)
    if not passed:
        raise RuntimeError("C3MSCCR_AR_TEMPORAL_PROVENANCE_FAIL")
    return result


def frozen_reducer(trials: list[Mapping[str, Any]], tolerance: float) -> tuple[Mapping[str, Any], str]:
    best_utility = max(float(row["best"]["utility"]) for row in trials)
    best_ah = max(
        (row for row in trials if row["trial"]["source"] == "AH"),
        key=lambda row: float(row["best"]["utility"]),
    )
    if (
        bool(best_ah["best"]["validation"]["gates"]["all"])
        and float(best_ah["best"]["utility"]) >= best_utility - float(tolerance)
    ):
        return best_ah, "A+H passes all validation gates and is within 0.01 of best utility"
    selected = max(
        (row for row in trials if row["trial"]["source"] == "VAH"),
        key=lambda row: float(row["best"]["utility"]),
    )
    return selected, "best validation-only V+A+H trial; A+H all-gates simplicity condition not met"


def source_selection_audit(config: Mapping[str, Any], parent: Mapping[str, Any]) -> dict[str, Any]:
    artifact_root = ROOT / parent["runtime"]["artifact_root"]
    training = json.loads((artifact_root / "training_summary.json").read_text())
    selection = json.loads((artifact_root / "selection.json").read_text())
    trials = training["trials"]
    selected, rationale = frozen_reducer(trials, parent["validation"]["simplicity_tolerance"])
    t0 = next(row for row in trials if row["trial"]["id"] == "T0")
    failed = sorted(
        name for name, value in t0["best"]["validation"]["gates"].items()
        if name != "all" and not value
    )
    reducer_source = (ROOT / "scripts/tactile_unit/freeze_c3mscc_selection.py").read_text()
    implementation_valid = bool(
        training.get("test_loaded") is False
        and selection.get("test_loaded") is False
        and selected["trial"]["id"] == selection["trial"] == "T1"
        and float(parent["validation"]["simplicity_tolerance"]) == 0.01
        and 'best_ah["best"]["validation"]["gates"]["all"]' in reducer_source
        and 'row["trial"]["source"] == "VAH"' in reducer_source
    )
    result = {
        "schema": "tactile3d-unit.vac-c3msccr-source-selection-audit.v1",
        "selection_split": "validation only",
        "test_loaded": False,
        "frozen_rule": "choose best-utility A+H iff it passes all validation gates and is within 0.01 of global best; otherwise choose best-utility V+A+H",
        "simplicity_tolerance": 0.01,
        "utilities": {
            row["trial"]["id"]: float(row["best"]["utility"]) for row in trials
        },
        "t0_original_failed_gates": failed,
        "selected_trial": selected["trial"]["id"],
        "selected_source": selected["trial"]["source"],
        "why_t1_selected": rationale,
        "reducer_implementation": "VALID" if implementation_valid else "INVALID",
        "selection_bug": not implementation_valid,
    }
    atomic_json(ROOT / config["runtime"]["artifact_root"] / "source_selection_audit.json", result)
    if not implementation_valid:
        raise RuntimeError("C3MSCCR_SELECTION_IMPLEMENTATION_INVALID")
    return result


def exact_action_metrics(
    model,
    train: Mapping[str, np.ndarray],
    validation: Mapping[str, np.ndarray],
    exact: Mapping[str, np.ndarray],
    oracle: Mapping[str, Any],
    shared_space,
    decoder,
    parent: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
    seed: int,
    bootstrap_samples: int | None = None,
) -> dict[str, Any]:
    train_prediction = predict_numpy(model, train, device, batch_size)
    predictions = {
        variant: predict_numpy(
            model, validation, device, batch_size, u_a=np.asarray(exact[f"u_a_{variant}"])
        )
        for variant in VARIANTS
    }
    target = np.asarray(validation["u_c"])
    dynamic = np.asarray(validation["dynamic"], dtype=bool)
    errors = {name: row_mse(value, target) for name, value in predictions.items()}
    oracle_future = physics_prediction(
        shared_space, decoder, target, validation["h_current"], device, batch_size
    )
    bootstrap = int(
        parent["validation"]["bootstrap_samples"]
        if bootstrap_samples is None else bootstrap_samples
    )
    variants = {}
    for offset, variant in enumerate(("reversed", "shuffled", "different")):
        semantic, _ = semantic_evaluation(
            train_prediction, predictions[variant], train, validation, oracle
        )
        difference = errors[variant][dynamic] - errors["correct"][dynamic]
        future = physics_prediction(
            shared_space, decoder, predictions[variant], validation["h_current"],
            device, batch_size,
        )
        variants[variant] = {
            "all_mse": float(errors[variant].mean()),
            "dynamic_mse": float(errors[variant][dynamic].mean()),
            "dynamic_difference_over_correct": float(difference.mean()),
            "dynamic_difference_ci95": bootstrap_mean_ci(
                difference, samples=bootstrap, seed=seed + offset
            ),
            "contact_f1": float(semantic["contact_transition"]["macro_f1"]),
            "force_f1": float(semantic["force_trend_class"]["macro_f1"]),
            "future_change_f1": float(semantic["contact_transition"]["future_change"]["macro_f1"]),
            "shared_physics_mse": float(row_mse(future, oracle_future).mean()),
        }
    minimum_invalid = np.minimum(errors["reversed"], errors["shuffled"])
    normalized_rows = (
        minimum_invalid[dynamic] - errors["correct"][dynamic]
    ) / max(float(minimum_invalid[dynamic].mean()), 1e-12)
    gate = bool(
        variants["reversed"]["dynamic_difference_ci95"][0] > 0
        and variants["shuffled"]["dynamic_difference_ci95"][0] > 0
        and variants["different"]["dynamic_difference_ci95"][0] > 0
    )
    return {
        "method": "raw [16,58] perturbation before full accepted A-R transition pipeline and frozen P_a",
        "exact_ar_transform": True,
        "correct_all_mse": float(errors["correct"].mean()),
        "correct_dynamic_mse": float(errors["correct"][dynamic].mean()),
        "variants": variants,
        "normalized_improvement": float(normalized_rows.mean()),
        "normalized_improvement_ci95": bootstrap_mean_ci(
            normalized_rows, samples=bootstrap, seed=seed + 20
        ),
        "gate": gate,
    }


def exact_validation(
    config: Mapping[str, Any], parent: Mapping[str, Any], device: torch.device, batch_size: int
) -> dict[str, Any]:
    identities = identity_snapshot(parent)
    if not identities["pass"]:
        raise RuntimeError("STRUCTURAL_FAIL: frozen identity mismatch")
    train = load_aligned_split(parent, "train")
    validation = load_aligned_split(parent, "validation")
    train_exact = exact_split(config, "train")
    validation_exact = exact_split(config, "validation")
    train = dict(train, u_a=train_exact["u_a_correct"])
    validation = dict(validation, u_a=validation_exact["u_a_correct"])
    shared_space, _, shared_digest = load_frozen_shared_space(parent, device)
    s2 = load_s2_model(ROOT / parent["runtime"]["s2_checkpoint"], device)
    decoder = s2.decoder.eval().requires_grad_(False)
    oracle = oracle_semantics(train, validation)
    summary = json.loads(
        (ROOT / parent["runtime"]["artifact_root"] / "training_summary.json").read_text()
    )
    evaluated = []
    for trial_index, row in enumerate(summary["trials"]):
        path = ROOT / row["best"]["checkpoint"]
        if sha256_file(path) != row["best"]["checkpoint_sha256"]:
            raise RuntimeError("STRUCTURAL_FAIL: frozen trial checkpoint changed")
        model, metadata = load_checkpoint(path, device)
        model.eval().requires_grad_(False)
        if metadata.get("test_loaded") is not False:
            raise RuntimeError("STRUCTURAL_FAIL: frozen trial saw test")
        seed = int(config["seed"]) + trial_index * 100
        metrics = validation_metrics(
            model, train, validation, oracle, shared_space, decoder,
            parent, device, batch_size, seed,
        )
        old_normalized = float(metrics["action_temporal"]["normalized_improvement"])
        exact_metrics = exact_action_metrics(
            model, train, validation, validation_exact, oracle, shared_space, decoder,
            parent, device, batch_size, seed + 50,
        )
        action_weight = float(parent["validation"]["utility"]["action_temporal_sensitivity"])
        metrics["utility"] = float(
            metrics["utility"]
            - action_weight * old_normalized
            + action_weight * float(exact_metrics["normalized_improvement"])
        )
        metrics["action_temporal"] = exact_metrics
        metrics["gates"]["action_surrogate"] = bool(exact_metrics["gate"])
        metrics["gates"]["action_exact_ar"] = bool(exact_metrics["gate"])
        metrics["gates"]["all"] = all(
            value for name, value in metrics["gates"].items() if name != "all"
        )
        evaluated.append({
            "trial_index": row["trial_index"], "trial": row["trial"],
            "checkpoint": row["best"]["checkpoint"],
            "checkpoint_sha256": row["best"]["checkpoint_sha256"],
            "original_utility": float(row["best"]["utility"]),
            "exact_action_utility": float(metrics["utility"]),
            "validation": metrics,
        })
    reducer_rows = [
        {"trial": row["trial"], "best": {"utility": row["exact_action_utility"], "validation": row["validation"]}}
        for row in evaluated
    ]
    selected, rationale = frozen_reducer(
        reducer_rows, parent["validation"]["simplicity_tolerance"]
    )
    t0 = next(row for row in evaluated if row["trial"]["id"] == "T0")
    failed = sorted(
        name for name, value in t0["validation"]["gates"].items()
        if name != "all" and not value
    )
    result = {
        "schema": "tactile3d-unit.vac-c3msccr-frozen-candidate-validation.v1",
        "selection_split": "validation only",
        "test_loaded": False,
        "exact_action_evidence": True,
        "trials": evaluated,
        "frozen_reducer_result": {
            "trial": selected["trial"]["id"], "source": selected["trial"]["source"],
            "rationale": rationale,
        },
        "t0_failed_gates_after_exact_evidence": failed,
        "t0_all_validation_hard_gates": bool(t0["validation"]["gates"]["all"]),
        "remediation_required": not bool(t0["validation"]["gates"]["all"]),
        "identity": identities,
        "shared_state_sha256": shared_digest,
    }
    atomic_json(
        ROOT / config["runtime"]["artifact_root"] / "frozen_candidate_validation.json",
        result,
    )
    return result


def patch_validation_with_exact_action(
    metrics: dict[str, Any],
    exact_metrics: Mapping[str, Any],
    parent: Mapping[str, Any],
) -> dict[str, Any]:
    old_normalized = float(metrics["action_temporal"]["normalized_improvement"])
    action_weight = float(parent["validation"]["utility"]["action_temporal_sensitivity"])
    metrics["utility"] = float(
        metrics["utility"]
        - action_weight * old_normalized
        + action_weight * float(exact_metrics["normalized_improvement"])
    )
    metrics["action_temporal"] = dict(exact_metrics)
    metrics["gates"]["action_surrogate"] = bool(exact_metrics["gate"])
    metrics["gates"]["action_exact_ar"] = bool(exact_metrics["gate"])
    metrics["gates"]["all"] = all(
        value for name, value in metrics["gates"].items() if name != "all"
    )
    return metrics


def remediation(
    config: Mapping[str, Any], parent: Mapping[str, Any], device: torch.device, batch_size: int
) -> dict[str, Any]:
    artifact_root = ROOT / config["runtime"]["artifact_root"]
    frozen_path = artifact_root / "frozen_candidate_validation.json"
    if not frozen_path.is_file():
        raise RuntimeError("run frozen exact validation before remediation")
    frozen = json.loads(frozen_path.read_text())
    failed = list(frozen["t0_failed_gates_after_exact_evidence"])
    if not frozen.get("remediation_required") or failed != ["physics"]:
        raise RuntimeError("remediation is allowed only for the established T0 physics gap")
    if int(config["remediation"]["maximum_trials"]) > 2:
        raise RuntimeError("STRUCTURAL_FAIL: remediation trial bound changed")
    identities_before = identity_snapshot(parent)
    if not identities_before["pass"]:
        raise RuntimeError("STRUCTURAL_FAIL: frozen identity mismatch before remediation")
    train = load_aligned_split(parent, "train")
    validation = load_aligned_split(parent, "validation")
    train_exact = exact_split(config, "train")
    validation_exact = exact_split(config, "validation")
    train = dict(train, u_a=train_exact["u_a_correct"])
    validation = dict(validation, u_a=validation_exact["u_a_correct"])
    shared_space, _, shared_before = load_frozen_shared_space(parent, device)
    s2 = load_s2_model(ROOT / parent["runtime"]["s2_checkpoint"], device)
    decoder = s2.decoder.eval().requires_grad_(False)
    oracle = oracle_semantics(train, validation)
    t0_path = ROOT / config["runtime"]["c3mscc_root"] / "trial_00_T0/best.pt"
    if sha256_file(t0_path) != config["accepted"]["t0_checkpoint_sha256"]:
        raise RuntimeError("C3MSCCR_MINIMAL_SOURCE_ARTIFACT_UNAVAILABLE")
    model, metadata = load_checkpoint(t0_path, device)
    if model.source != "AH" or metadata.get("test_loaded") is not False:
        raise RuntimeError("C3MSCCR_MINIMAL_SOURCE_ARTIFACT_UNAVAILABLE")
    initial_state = state_dict_digest(model)
    initial_architecture = model.parameter_summary()
    model.train().requires_grad_(True)
    spec = config["remediation"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    weights = C3MSCCLossWeights(**spec["loss_weights"])
    epochs = int(spec["epochs"])
    train_batch = int(spec["batch_size"])
    history = []
    best = None
    trial_root = ROOT / config["runtime"]["experiment_root"] / "trial_00_R1"
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.default_rng(int(config["seed"]) + epoch).permutation(len(train["u_c"]))
        totals: dict[str, float] = {}
        batches = 0
        for start in range(0, len(order), train_batch):
            indices = order[start:start + train_batch]
            if len(indices) < 2:
                continue
            u_a = torch.from_numpy(np.array(train_exact["u_a_correct"][indices], copy=True)).to(device)
            u_rev = torch.from_numpy(np.array(train_exact["u_a_reversed"][indices], copy=True)).to(device)
            u_shuf = torch.from_numpy(np.array(train_exact["u_a_shuffled"][indices], copy=True)).to(device)
            u_c = torch.from_numpy(np.array(train["u_c"][indices], copy=True)).to(device)
            h = torch.from_numpy(np.array(train["h_current"][indices], copy=True)).to(device)
            dynamic = torch.from_numpy(np.array(train["dynamic"][indices], copy=True)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = contact_prediction_loss(
                model, shared_space, decoder,
                u_a=u_a, h_current=h, u_c=u_c, dynamic=dynamic, u_v=None,
                invalid_u_a=(u_rev, u_shuf), enhanced=True,
                dynamic_weight=float(spec["dynamic_weight"]),
                order_margin=float(spec["order_margin"]),
                variance_floor=float(spec["variance_floor"]), weights=weights,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite remediation loss")
            loss.backward()
            if any(parameter.grad is not None for parameter in shared_space.parameters()):
                raise RuntimeError("STRUCTURAL_FAIL: remediation gradient reached shared space")
            if any(parameter.grad is not None for parameter in decoder.parameters()):
                raise RuntimeError("STRUCTURAL_FAIL: remediation gradient reached D_c")
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(spec["gradient_clip"]))
            optimizer.step()
            for name, value in terms.items():
                totals[name] = totals.get(name, 0.0) + float(value)
            batches += 1
        if state_dict_digest(shared_space) != shared_before:
            raise RuntimeError("STRUCTURAL_FAIL: shared space changed during remediation")
        seed = int(config["seed"]) + 1000 + epoch
        metrics = validation_metrics(
            model, train, validation, oracle, shared_space, decoder,
            parent, device, batch_size, seed,
        )
        exact_metrics = exact_action_metrics(
            model, train, validation, validation_exact, oracle, shared_space,
            decoder, parent, device, batch_size, seed + 50,
        )
        metrics = patch_validation_with_exact_action(metrics, exact_metrics, parent)
        row = {
            "epoch": epoch,
            "train": {name: value / max(batches, 1) for name, value in totals.items()},
            "validation": metrics,
        }
        history.append(row)
        eligible = bool(metrics["gates"]["all"])
        if eligible and (best is None or float(metrics["utility"]) > float(best["utility"])):
            checkpoint = trial_root / "best.pt"
            digest = save_checkpoint(
                checkpoint, model,
                {
                    "mode": "BOUNDED_AH_REMEDIATION", "trial": "R1",
                    "initialized_from": str(t0_path.relative_to(ROOT)),
                    "initialized_from_sha256": config["accepted"]["t0_checkpoint_sha256"],
                    "objective": "exact temporal ranking plus existing shared physics",
                    "epoch": epoch, "validation": metrics,
                    "selection_split": "validation only", "test_loaded": False,
                    "vision_used": False, "architecture_changed": False,
                },
            )
            best = {
                "epoch": epoch, "utility": float(metrics["utility"]),
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": digest, "validation": metrics,
            }
        atomic_json(trial_root / "history.json", {
            "trial": "R1", "source": "AH", "history": history, "best": best,
            "test_loaded": False, "vision_used": False, "architecture_changed": False,
        })
    identities_after = identity_snapshot(parent)
    result = {
        "schema": "tactile3d-unit.vac-c3msccr-remediation-trials.v1",
        "selection_split": "validation only", "test_loaded": False,
        "trigger": {"frozen_t0_failed_gates": failed},
        "total_new_trials": 1, "maximum_trials": int(spec["maximum_trials"]),
        "trials": [{
            "id": "R1", "source": "AH", "vision_used": False,
            "architecture_changed": False,
            "initialized_from": "frozen T0",
            "initial_state_sha256": initial_state,
            "parameter_summary": initial_architecture,
            "objective": "exact temporal ranking plus existing shared physics",
            "epochs": epochs, "best": best,
        }],
        "selected": best,
        "all_validation_hard_gates": bool(best and best["validation"]["gates"]["all"]),
        "identity_before": identities_before, "identity_after": identities_after,
        "seconds": time.monotonic() - started,
    }
    atomic_json(artifact_root / "remediation_trials.json", result)
    if not identities_after["pass"]:
        raise RuntimeError("STRUCTURAL_FAIL: frozen identity changed during remediation")
    return result


def freeze_closure_selection(
    config: Mapping[str, Any], parent: Mapping[str, Any]
) -> dict[str, Any]:
    artifact_root = ROOT / config["runtime"]["artifact_root"]
    remediation_path = artifact_root / "remediation_trials.json"
    if not remediation_path.is_file():
        raise RuntimeError("remediation result is required before closure freeze")
    remediation_value = json.loads(remediation_path.read_text())
    best = remediation_value.get("selected")
    if (
        remediation_value.get("test_loaded") is not False
        or remediation_value.get("total_new_trials") != 1
        or not remediation_value.get("all_validation_hard_gates")
        or best is None
    ):
        raise RuntimeError("NO_ELIGIBLE_MODEL")
    checkpoint = ROOT / best["checkpoint"]
    if sha256_file(checkpoint) != best["checkpoint_sha256"]:
        raise RuntimeError("STRUCTURAL_FAIL: remediation checkpoint changed")
    model, metadata = load_checkpoint(checkpoint)
    if (
        model.source != "AH"
        or metadata.get("test_loaded") is not False
        or metadata.get("vision_used") is not False
        or metadata.get("architecture_changed") is not False
    ):
        raise RuntimeError("STRUCTURAL_FAIL: invalid remediation checkpoint metadata")
    identities = identity_snapshot(parent)
    if not identities["pass"]:
        raise RuntimeError("STRUCTURAL_FAIL: frozen identity mismatch before closure freeze")
    selection = {
        "schema": "tactile3d-unit.vac-c3msccr-closure-selection.v1",
        "mode": "BOUNDED_AH_REMEDIATION",
        "source": "AH",
        "trial": "R1",
        "checkpoint": best["checkpoint"],
        "SHA256": best["checkpoint_sha256"],
        "validation_utility": float(best["utility"]),
        "validation_hard_gates": best["validation"]["gates"],
        "exact_Action_temporal_gate": best["validation"]["action_temporal"],
        "selection_rationale": "frozen T0 passed exact Action evidence but retained one real validation physics gap; one bounded A+H R1 closed that gap",
        "selected_via": "VALIDATION ONLY",
        "selection_split": "validation only",
        "test_loaded": False,
        "identity": identities,
    }
    path = artifact_root / "closure_selection.json"
    atomic_json(path, selection)
    digest = sha256_file(path)
    (artifact_root / "closure_selection.sha256").write_text(
        digest + "  closure_selection.json\n"
    )
    return {**selection, "selection_sha256": digest}


def validate_closure_selection(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    root = ROOT / config["runtime"]["artifact_root"]
    path = root / "closure_selection.json"
    digest_path = root / "closure_selection.sha256"
    if not path.is_file() or not digest_path.is_file():
        raise RuntimeError("closure selection is not frozen")
    digest = sha256_file(path)
    if digest != digest_path.read_text().split()[0]:
        raise RuntimeError("STRUCTURAL_FAIL: closure selection hash mismatch")
    selection = json.loads(path.read_text())
    if (
        selection.get("test_loaded") is not False
        or selection.get("selected_via") != "VALIDATION ONLY"
        or selection.get("source") != "AH"
    ):
        raise RuntimeError("STRUCTURAL_FAIL: invalid pretest closure selection")
    if sha256_file(ROOT / selection["checkpoint"]) != selection["SHA256"]:
        raise RuntimeError("STRUCTURAL_FAIL: selected closure checkpoint changed")
    return selection, digest


def locked_evaluation(
    config: Mapping[str, Any], parent: Mapping[str, Any], device: torch.device, batch_size: int
) -> dict[str, Any]:
    selection, selection_digest = validate_closure_selection(config)
    identities_before = identity_snapshot(parent)
    if not identities_before["pass"]:
        raise RuntimeError("STRUCTURAL_FAIL: frozen identity mismatch before locked evaluation")
    # Locked data is loaded only after the unhashed selection and checkpoint checks above.
    train = load_aligned_split(parent, "train")
    test = load_aligned_split(parent, "test")
    train_exact = exact_split(config, "train")
    test_exact = exact_split(config, "test")
    train = dict(train, u_a=train_exact["u_a_correct"])
    test = dict(test, u_a=test_exact["u_a_correct"])
    if len(test["u_c"]) != 17504:
        raise RuntimeError("STRUCTURAL_FAIL: locked benchmark row count changed")
    shared_space, _, shared_before = load_frozen_shared_space(parent, device)
    s2 = load_s2_model(ROOT / parent["runtime"]["s2_checkpoint"], device)
    decoder = s2.decoder.eval().requires_grad_(False)
    model, metadata = load_checkpoint(ROOT / selection["checkpoint"], device)
    model.eval().requires_grad_(False)
    if model.source != "AH" or metadata.get("test_loaded") is not False:
        raise RuntimeError("STRUCTURAL_FAIL: selected model violates locked protocol")
    oracle = oracle_probe(train, test)
    seed = int(config["seed"]) + 9000
    evaluated = model_evaluation(
        model, train, test, oracle, shared_space, decoder, parent,
        device, batch_size, seed,
    )
    exact_metrics = exact_action_metrics(
        model, train, test, test_exact, oracle, shared_space, decoder,
        parent, device, batch_size, seed + 100,
        bootstrap_samples=int(parent["evaluation"]["bootstrap_samples"]),
    )
    evaluated["action_temporal"] = exact_metrics
    first = predict_numpy(model, test, device, batch_size)
    second = predict_numpy(model, test, device, batch_size)
    repeat_exact = bool(np.array_equal(first, second))
    if not repeat_exact:
        raise RuntimeError("STRUCTURAL_FAIL: repeated locked predictor inference changed")
    clean = strip_arrays(evaluated)
    semantic_gate = bool(
        clean["semantics"]["contact_transition"]["semantic_ratio"] >= 0.75
        and clean["semantics"]["force_trend_class"]["semantic_ratio"] >= 0.75
    )
    h_gate = bool(clean["h_context"]["gate"])
    action_gate = bool(clean["action_temporal"]["gate"])
    shared_gate = bool(clean["shared_target"]["gate"])
    physics_gate = bool(clean["physics"]["gate"])
    noncollapse = bool(clean["noncollapse"])
    contact = clean["semantics"]["contact_transition"]
    boundaries_valid = bool(
        np.isfinite(contact["future_change"]["macro_f1"])
        and all(
            np.isfinite(contact[name][metric])
            for name in ("free_to_contact", "contact_to_free")
            for metric in ("precision", "recall", "f1")
        )
    )
    rank_warning = bool(
        clean["geometry"]["effective_rank"]
        < 0.5 * float(parent["evaluation"]["oracle_contact_effective_rank"])
    )
    identities_after = identity_snapshot(parent)
    integrity = bool(
        identities_after["pass"]
        and state_dict_digest(shared_space) == shared_before
    )
    if not semantic_gate:
        decision = "C3MSCCR_SEMANTIC_FAIL"
    elif not action_gate:
        decision = "C3MSCCR_ACTION_TEMPORAL_FAIL"
    elif not physics_gate:
        decision = "C3MSCCR_PHYSICS_FAIL"
    elif not (h_gate and shared_gate and noncollapse and boundaries_valid and integrity):
        decision = "STRUCTURAL_FAIL"
    else:
        decision = (
            "C3MSCCR_READY_AH_MINIMAL_WITH_RANK_WARNING"
            if rank_warning else "C3MSCCR_READY_AH_MINIMAL"
        )
    result = {
        "schema": "tactile3d-unit.vac-c3msccr-locked-closure.v1",
        "evaluation": "LOCKED POST-HOC CLOSURE RE-EVALUATION",
        "first_look_untouched": False,
        "rows": len(test["u_c"]),
        "selection_frozen_before_test": True,
        "selection_sha256": selection_digest,
        "selection": selection,
        "source": "AH",
        "oracle": oracle,
        "metrics": clean,
        "hard_gates": {
            "contact_and_force_semantics": semantic_gate,
            "h_context": h_gate,
            "exact_action_temporal": action_gate,
            "shared_latent_and_retrieval": shared_gate,
            "all_and_dynamic_physics": physics_gate,
            "future_change_and_boundaries": boundaries_valid,
            "noncollapse": noncollapse,
            "frozen_integrity": integrity,
        },
        "rank_warning": rank_warning,
        "repeated_evaluation_exact": repeat_exact,
        "identity_before": identities_before,
        "identity_after": identities_after,
        "shared_state_before": shared_before,
        "shared_state_after": state_dict_digest(shared_space),
        "decision": decision,
        "test_loaded": True,
    }
    artifact_root = ROOT / config["runtime"]["artifact_root"]
    atomic_json(artifact_root / "locked_closure_evaluation.json", result)
    ready = decision.startswith("C3MSCCR_READY")
    atomic_json(artifact_root / "final_decision.json", {
        "decision": decision,
        "reasons": [
            f"semantic gate={'PASS' if semantic_gate else 'FAIL'}",
            f"exact Action temporal gate={'PASS' if action_gate else 'FAIL'}",
            f"all/dynamic shared physics gate={'PASS' if physics_gate else 'FAIL'}",
            f"rank warning={'YES' if rank_warning else 'NO'}",
        ],
        "canonical_source": "A+H" if ready else "NONE",
        "classification": "A_PLUS_H_CANONICAL_MINIMAL_SOURCE" if ready else "NO_ACCEPTED_SOURCE",
        "vision": "OPTIONAL_SHORT_HORIZON_CONTEXT" if ready else "ablation only",
        "c4_readiness": "READY_WITH_RANK_WARNING" if ready and rank_warning else ("READY" if ready else "NOT READY"),
        "c4": "NOT STARTED", "c5": "NOT STARTED", "c6_m3": "NOT STARTED",
        "m3": "NOT ESTABLISHED", "test_loaded": True,
    })
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    parent = load_parent_config(ROOT / config["runtime"]["parent_config"])
    set_seed(int(config["seed"]))
    device, lock_handle, gpu = resolve_device(
        args.device, allowed_physical=tuple(str(value) for value in config["gpu"]["allowed_physical"])
    )
    try:
        phases = (
            ("integrity", "source-audit", "validation")
            if args.phase == "all-pretest" else (args.phase,)
        )
        if "integrity" in phases:
            temporal_integrity(config)
        if "source-audit" in phases:
            source_selection_audit(config, parent)
        if "validation" in phases:
            exact_validation(config, parent, device, args.batch_size)
        if "remediation" in phases:
            remediation(config, parent, device, args.batch_size)
        if "freeze" in phases:
            freeze_closure_selection(config, parent)
        if "locked" in phases:
            locked_evaluation(config, parent, device, args.batch_size)
        print(json.dumps({"status": "PASS", "phases": phases, "gpu": gpu}, sort_keys=True))
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    main()
