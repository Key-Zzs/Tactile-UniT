#!/usr/bin/env python3
"""Run the preregistered six-or-fewer Contact-only C2-R trials."""

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

from gr00t.tactile_unit.c2r_contact_preservation import (  # noqa: E402
    C2RLossWeights,
    c2r_contact_loss,
    canonical_contact_probe,
    configure_contact_only_trainability,
    frozen_state_digest,
    retention,
    sha256_file,
    verify_accepted_c2_checkpoint,
)
from gr00t.tactile_unit.compatibility import parameter_digest  # noqa: E402
from gr00t.tactile_unit.continuous_vac_shared_space import (  # noqa: E402
    geometry_diagnostics,
    load_checkpoint,
    pairwise_alignment_metrics,
    save_checkpoint,
)
from gr00t.tactile_unit.vac_latent_dataset import load_split  # noqa: E402
from scripts.tactile_unit.continuous_contact_bridge_common import load_s2_model  # noqa: E402
from scripts.tactile_unit.vac_runtime_common import resolve_device, set_seed  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/tactile_unit/c2r_contact_preservation_remediation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--s2-checkpoint", type=Path, default=ROOT / ".local/experiments/contact_dynamics/s2_models/proposed_best.pt")
    parser.add_argument("--max-trials", type=int)
    return parser.parse_args()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def trial_grid(config: Mapping[str, Any]) -> list[dict[str, float]]:
    training = config["training"]
    ratio = float(training["lambda_delta_ratio"])
    return [
        {
            "lambda_future": float(future),
            "lambda_delta": float(future) * ratio,
            "boundary_weight": float(boundary),
        }
        for future in training["lambda_future"]
        for boundary in training["boundary_weight"]
    ]


def encode_modality(model, split, modality: str, device: torch.device, batch_size: int) -> np.ndarray:
    source = {"vision": "z_v", "action": "z_a", "contact": "z_c"}[modality]
    result = np.empty((len(split), 8, 32), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(split), batch_size):
            stop = min(start + batch_size, len(split))
            value = torch.from_numpy(np.array(split.arrays[source][start:stop], copy=True)).to(device)
            result[start:stop] = model.encode(modality, value).float().cpu().numpy()
    return result


def encode_contact_and_recovery(model, split, device: torch.device, batch_size: int):
    shared = np.empty((len(split), 8, 32), dtype=np.float32)
    recovered = np.empty_like(shared)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(split), batch_size):
            stop = min(start + batch_size, len(split))
            native = torch.from_numpy(np.array(split.arrays["z_c"][start:stop], copy=True)).to(device)
            value = model.encode("contact", native)
            shared[start:stop] = value.float().cpu().numpy()
            recovered[start:stop] = model.recover("contact", value).float().cpu().numpy()
    return shared, recovered


def physics_metrics(decoder, split, recovered: np.ndarray, device: torch.device, batch_size: int):
    squared = np.empty(len(split), dtype=np.float64)
    decoder.eval()
    with torch.inference_mode():
        for start in range(0, len(split), batch_size):
            stop = min(start + batch_size, len(split))
            z_value = torch.from_numpy(recovered[start:stop]).to(device)
            current = torch.from_numpy(np.array(split.arrays["h_current"][start:stop], copy=True)).to(device)
            future = torch.from_numpy(np.array(split.arrays["h_future"][start:stop], copy=True)).to(device)
            predicted = decoder(z_value, current)
            squared[start:stop] = (
                torch.square(predicted - future).mean(dim=1).double().cpu().numpy()
            )
    dynamic = np.asarray(split.arrays["dynamic"], dtype=bool)
    return {
        "future_mse": float(squared.mean()),
        "dynamic_mse": float(squared[dynamic].mean()),
    }


def frozen_output_identity(model, split, expected, device, batch_size) -> dict[str, bool]:
    result = {}
    for modality in ("vision", "action"):
        actual = encode_modality(model, split, modality, device, batch_size)
        result[modality] = bool(np.array_equal(actual, expected[modality]))
    return result


def validate(
    model,
    decoder,
    train,
    validation,
    frozen_validation,
    native_transition_probe,
    baseline_physics,
    baseline_contact_rank,
    config,
    device,
    batch_size,
    seed,
):
    shared_train = encode_modality(model, train, "contact", device, batch_size)
    shared_validation, recovered_validation = encode_contact_and_recovery(
        model, validation, device, batch_size
    )
    shared_probe = canonical_contact_probe(
        shared_train,
        shared_validation,
        train.arrays["contact_transition"],
        validation.arrays["contact_transition"],
        4,
    )
    transition_retention = retention(shared_probe, native_transition_probe)
    physics = physics_metrics(decoder, validation, recovered_validation, device, batch_size)
    sample_count = min(int(config["validation"]["alignment_samples"]), len(validation))
    indices = np.linspace(0, len(validation) - 1, sample_count, dtype=np.int64)
    episode = np.asarray(validation.arrays["episode_id"])[indices]
    alignment = {}
    for offset, (name, frozen_name) in enumerate((("V-C", "vision"), ("A-C", "action"))):
        alignment[name] = pairwise_alignment_metrics(
            frozen_validation[frozen_name][indices],
            shared_validation[indices],
            episode,
            bootstrap_samples=int(config["validation"]["bootstrap_samples"]),
            seed=seed + offset,
            retrieval_chunk=int(config["validation"]["retrieval_chunk"]),
        )
    geometry = geometry_diagnostics(shared_validation)
    collapse = bool(
        geometry["per_dimension_variance"]["near_zero_fraction"] >= 0.5
        or geometry["query_diversity"]["collapsed_pair_fraction"] >= 0.5
    )
    multiplier_min = float(config["validation"]["retrieval_r10_chance_multiplier_min"])
    alignment_gates = {}
    for name, value in alignment.items():
        multipliers = [
            value["retrieval"][direction]["recall_at_10"]
            / value["retrieval"][direction]["chance"]["recall_at_10"]
            for direction in ("forward", "reverse")
        ]
        alignment_gates[name] = {
            "margin": value["paired_minus_shuffled_margin"] > 0
            and value["margin_bootstrap_ci95"][0] > 0,
            "retrieval": min(multipliers) >= multiplier_min,
            "r10_chance_multiplier": multipliers,
        }
        alignment_gates[name]["pass"] = bool(
            alignment_gates[name]["margin"] and alignment_gates[name]["retrieval"]
        )
    material = float(config["validation"]["physics_material_regression_fraction"])
    physics_gate = all(
        physics[name] <= baseline_physics[name] * (1.0 + material)
        for name in ("future_mse", "dynamic_mse")
    )
    physics_gain = float(np.mean([
        (baseline_physics[name] - physics[name]) / max(baseline_physics[name], 1e-12)
        for name in ("future_mse", "dynamic_mse")
    ]))
    rank_retention = geometry["effective_rank"] / max(baseline_contact_rank, 1e-12)
    utility_weights = config["validation"]["utility"]
    utility = (
        float(utility_weights["contact_retention"]) * transition_retention
        + float(utility_weights["physics_gain"]) * physics_gain
        + float(utility_weights["minimum_alignment_margin"])
        * min(value["paired_minus_shuffled_margin"] for value in alignment.values())
        + float(utility_weights["rank_retention"]) * rank_retention
    )
    selectable = bool(
        all(value["pass"] for value in alignment_gates.values())
        and physics_gate
        and not collapse
    )
    return {
        "selection_split": "validation only",
        "test_loaded": False,
        "contact_transition": {
            "native": native_transition_probe,
            "shared": shared_probe,
            "retention": transition_retention,
        },
        "physics": physics,
        "physics_baseline": baseline_physics,
        "physics_gain": physics_gain,
        "alignment": alignment,
        "alignment_gates": alignment_gates,
        "geometry": geometry,
        "rank_retention": rank_retention,
        "collapse": collapse,
        "physics_gate": physics_gate,
        "selectable": selectable,
        "utility": utility,
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    runtime = config["runtime"]
    cache_root = ROOT / runtime["cache_root"]
    baseline_checkpoint = ROOT / runtime["accepted_c2_checkpoint"]
    experiment_root = ROOT / runtime["experiment_root"]
    artifact_root = ROOT / runtime["artifact_root"]
    audit = json.loads((artifact_root / "metric_audit.json").read_text())
    if audit.get("decision") != "C2R0_METRIC_AUDIT_PASS":
        raise RuntimeError("C2-R0 metric audit did not pass")
    baseline_sha = verify_accepted_c2_checkpoint(baseline_checkpoint)
    grid = trial_grid(config)
    if len(grid) != int(config["training"]["bounded_trials"]) or len(grid) > 6:
        raise RuntimeError("C2-R bounded trial preregistration changed")
    if args.max_trials is not None:
        if not 1 <= args.max_trials <= len(grid):
            raise ValueError("--max-trials must remain within the preregistered grid")
        grid = grid[: args.max_trials]
    atomic_json(experiment_root / "trial_manifest.json", {
        "schema": "tactile3d-unit.vac-c2r-trials.v1",
        "initialization_sha256": baseline_sha,
        "trials": grid,
        "max_epochs": int(config["training"]["max_epochs"]),
        "patience": int(config["training"]["patience"]),
        "selection_split": "validation only",
        "test_loaded": False,
    })
    device, lock_handle, gpu = resolve_device(
        args.device, allowed_physical=("1", "2", "3")
    )
    try:
        set_seed(int(config["seed"]))
        train = load_split(cache_root, "train", verify_hashes=True)
        validation = load_split(cache_root, "validation", verify_hashes=True)
        baseline, baseline_metadata = load_checkpoint(baseline_checkpoint, device)
        if baseline.candidate != config["accepted_c2"]["candidate"]:
            raise RuntimeError("accepted C2 architecture mismatch")
        baseline.eval().requires_grad_(False).to(device)
        s2 = load_s2_model(args.s2_checkpoint, device)
        s2.eval().requires_grad_(False)
        accepted_evaluation = json.loads((ROOT / runtime["accepted_c2_evaluation"]).read_text())
        expected_decoder = accepted_evaluation["native_identity_before"]["s2_decoder"]
        if parameter_digest(s2.decoder) != expected_decoder:
            raise RuntimeError("frozen D_c identity mismatch")
        batch_size = int(config["training"]["batch_size"])
        frozen_validation = {
            modality: encode_modality(baseline, validation, modality, device, batch_size)
            for modality in ("vision", "action")
        }
        native_transition_probe = canonical_contact_probe(
            train.arrays["z_c"], validation.arrays["z_c"],
            train.arrays["contact_transition"], validation.arrays["contact_transition"], 4,
        )
        baseline_shared, baseline_recovered = encode_contact_and_recovery(
            baseline, validation, device, batch_size
        )
        baseline_physics = physics_metrics(
            s2.decoder, validation, baseline_recovered, device, batch_size
        )
        baseline_contact_rank = geometry_diagnostics(baseline_shared)["effective_rank"]
        results = []
        accepted = config["accepted_c2"]
        training = config["training"]
        for trial_id, trial in enumerate(grid):
            set_seed(int(config["seed"]))
            model, _ = load_checkpoint(baseline_checkpoint, device)
            model.to(device)
            boundary = configure_contact_only_trainability(model)
            frozen_digest_before = frozen_state_digest(model)
            decoder_digest_before = parameter_digest(s2.decoder)
            optimizer = torch.optim.AdamW(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
            )
            weights = C2RLossWeights(
                alignment=float(accepted["loss_weights"]["alignment"]),
                native_z=float(accepted["loss_weights"]["native_z"]),
                future=float(trial["lambda_future"]),
                delta=float(trial["lambda_delta"]),
                relational=float(accepted["loss_weights"]["relational"]),
                variance=float(accepted["loss_weights"]["variance"]),
            )
            generator = np.random.default_rng(int(config["seed"]))
            history = []
            best = None
            epochs_without_improvement = 0
            trial_root = experiment_root / f"trial_{trial_id:02d}"
            started = time.monotonic()
            for epoch in range(1, int(training["max_epochs"]) + 1):
                model.train()
                order = generator.permutation(len(train))
                totals = {name: 0.0 for name in (
                    "total", "alignment_contact", "native_z", "future", "delta",
                    "relational_contact", "relational_pairwise",
                    "relational_neighborhood", "relational_ordering", "variance_contact",
                )}
                batches = 0
                for start in range(0, len(order), batch_size):
                    indices = order[start:start + batch_size]
                    if len(indices) < 2:
                        continue
                    native = {
                        "vision": torch.from_numpy(np.asarray(train.arrays["z_v"][indices])).to(device),
                        "action": torch.from_numpy(np.asarray(train.arrays["z_a"][indices])).to(device),
                        "contact": torch.from_numpy(np.asarray(train.arrays["z_c"][indices])).to(device),
                    }
                    episode = torch.from_numpy(np.asarray(train.arrays["episode_id"][indices], dtype=np.int64)).to(device)
                    dynamic = torch.from_numpy(np.asarray(train.arrays["dynamic"][indices], dtype=np.bool_)).to(device)
                    transition = torch.from_numpy(np.asarray(train.arrays["contact_transition"][indices], dtype=np.int64)).to(device)
                    current = torch.from_numpy(np.asarray(train.arrays["h_current"][indices])).to(device)
                    future = torch.from_numpy(np.asarray(train.arrays["h_future"][indices])).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss, breakdown = c2r_contact_loss(
                        model, s2.decoder, native, episode, dynamic, transition,
                        current, future,
                        temperature=float(accepted["temperature"]),
                        dynamic_weight=float(accepted["dynamic_weight"]),
                        boundary_weight=float(trial["boundary_weight"]),
                        weights=weights,
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite C2-R loss")
                    loss.backward()
                    if any(parameter.grad is not None for parameter in s2.decoder.parameters()):
                        raise RuntimeError("C2R_GRADIENT_ISOLATION_FAIL")
                    unexpected = [
                        name for name, parameter in model.named_parameters()
                        if parameter.grad is not None and not name.startswith(("projectors.contact.", "recovery.contact."))
                    ]
                    if unexpected:
                        raise RuntimeError("C2R_GRADIENT_ISOLATION_FAIL")
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in model.parameters() if parameter.requires_grad],
                        float(training["gradient_clip"]),
                    )
                    optimizer.step()
                    for name, value in breakdown.items():
                        totals[name] += float(value)
                    batches += 1
                if frozen_state_digest(model) != frozen_digest_before:
                    raise RuntimeError("frozen C2 state changed")
                if parameter_digest(s2.decoder) != decoder_digest_before:
                    raise RuntimeError("frozen D_c changed")
                identity = frozen_output_identity(
                    model, validation, frozen_validation, device, batch_size
                )
                if not all(identity.values()):
                    raise RuntimeError("frozen Vision/Action shared output changed")
                validation_result = validate(
                    model, s2.decoder, train, validation, frozen_validation,
                    native_transition_probe, baseline_physics, baseline_contact_rank,
                    config, device, batch_size, int(config["seed"]) + epoch * 31,
                )
                validation_result["frozen_output_identity"] = identity
                row = {
                    "epoch": epoch,
                    "train": {name: value / max(batches, 1) for name, value in totals.items()},
                    "validation": validation_result,
                }
                history.append(row)
                score = float(validation_result["utility"])
                current_selectable = bool(validation_result["selectable"])
                improved = (
                    best is None
                    or (current_selectable and not bool(best["validation"]["selectable"]))
                    or (
                        current_selectable == bool(best["validation"]["selectable"])
                        and score > float(best["utility"]) + 1e-12
                    )
                )
                if improved:
                    checkpoint_path = trial_root / "best.pt"
                    digest = save_checkpoint(
                        checkpoint_path,
                        model,
                        {
                            "trial_id": trial_id,
                            "trial": trial,
                            "epoch": epoch,
                            "selection_split": "validation only",
                            "test_loaded": False,
                            "validation": validation_result,
                            "accepted_c2_initialization_sha256": baseline_sha,
                        },
                    )
                    best = {
                        "epoch": epoch,
                        "utility": score,
                        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
                        "sha256": digest,
                        "validation": validation_result,
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                atomic_json(trial_root / "history.json", {
                    "trial_id": trial_id, "trial": trial, "history": history, "best": best,
                })
                if epochs_without_improvement >= int(training["patience"]):
                    break
            assert best is not None
            results.append({
                "trial_id": trial_id,
                "trial": trial,
                "epochs": len(history),
                "seconds": time.monotonic() - started,
                "parameter_boundary": boundary,
                "frozen_state_digest": frozen_digest_before,
                "decoder_digest": decoder_digest_before,
                "best": best,
            })

        selectable = [row for row in results if row["best"]["validation"]["selectable"]]
        if not selectable:
            raise RuntimeError("no validation-safe C2-R candidate")
        maximum = max(float(row["best"]["utility"]) for row in selectable)
        tolerance = float(config["validation"]["effective_tie_tolerance"])
        tied = [row for row in selectable if float(row["best"]["utility"]) >= maximum - tolerance]
        selected = min(
            tied,
            key=lambda row: (
                float(row["trial"]["lambda_future"]),
                float(row["trial"]["boundary_weight"]),
                -float(row["best"]["utility"]),
            ),
        )
        selected_source = ROOT / selected["best"]["checkpoint"]
        selected_path = experiment_root / "selected.pt"
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(selected_source, selected_path)
        selected_sha = sha256_file(selected_path)
        selection = {
            "schema": "tactile3d-unit.vac-c2r-selection.v1",
            "trial_id": selected["trial_id"],
            "config": selected["trial"],
            "validation_metrics": selected["best"]["validation"],
            "checkpoint": str(selected_path.relative_to(ROOT)),
            "checkpoint_sha256": selected_sha,
            "selection_rationale": "maximum frozen validation utility; effective ties prefer smallest lambda_future then boundary weight 1",
            "selection_split": "validation only",
            "test_loaded": False,
            "accepted_c2_initialization_sha256": baseline_sha,
            "architecture_changed": False,
        }
        selection_path = artifact_root / "selection.json"
        atomic_json(selection_path, selection)
        selection_sha = sha256_file(selection_path)
        (artifact_root / "selection.sha256").write_text(selection_sha + "  selection.json\n")
        summary = {
            "schema": "tactile3d-unit.vac-c2r-training.v1",
            "gpu": gpu,
            "selection_split": "validation only",
            "test_loaded": False,
            "accepted_c2_checkpoint_sha256": baseline_sha,
            "accepted_c2_metadata": baseline_metadata,
            "baseline_validation_physics": baseline_physics,
            "trial_count": len(results),
            "trials": results,
            "selected": selection,
            "selection_artifact_sha256": selection_sha,
        }
        atomic_json(experiment_root / "training_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    main()
