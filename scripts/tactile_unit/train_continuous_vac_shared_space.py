#!/usr/bin/env python3
"""Train the bounded C2 independent continuous shared-space candidates."""

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
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.continuous_vac_shared_space import (  # noqa: E402
    ContinuousVACSharedSpace,
    MODALITIES,
    VACLossWeights,
    continuous_vac_loss,
    effective_rank,
    geometry_diagnostics,
    pairwise_alignment_metrics,
    save_checkpoint,
)
from gr00t.tactile_unit.vac_latent_dataset import load_split  # noqa: E402
from scripts.tactile_unit.vac_runtime_common import resolve_device, set_seed  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/tactile_unit/c2_continuous_shared_space.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--validation-samples", type=int, default=2048)
    return parser.parse_args()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def encode_arrays(
    model: ContinuousVACSharedSpace,
    split: Any,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    shared = {name: np.empty((len(indices), 8, 32), dtype=np.float32) for name in MODALITIES}
    recovered = {name: np.empty_like(shared[name]) for name in MODALITIES}
    source_names = {"vision": "z_v", "action": "z_a", "contact": "z_c"}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            stop = min(start + batch_size, len(indices))
            source_index = indices[start:stop]
            for modality in MODALITIES:
                native = torch.from_numpy(np.asarray(split.arrays[source_names[modality]][source_index])).to(device)
                encoded = model.encode(modality, native)
                shared[modality][start:stop] = encoded.float().cpu().numpy()
                recovered[modality][start:stop] = model.recover(modality, encoded).float().cpu().numpy()
    return shared, recovered


def recovery_metrics(native: np.ndarray, recovered: np.ndarray) -> dict[str, float]:
    source = np.asarray(native, dtype=np.float64).reshape(len(native), -1)
    prediction = np.asarray(recovered, dtype=np.float64).reshape(len(recovered), -1)
    mse = float(np.mean(np.square(prediction - source)))
    cosine = float(np.mean(np.sum(source * prediction, axis=1) / np.maximum(
        np.linalg.norm(source, axis=1) * np.linalg.norm(prediction, axis=1), 1e-12
    )))
    denominator = float(np.sum(np.square(source - source.mean(axis=0, keepdims=True))))
    r2 = 1.0 - float(np.sum(np.square(prediction - source))) / max(denominator, 1e-12)
    return {"mse": mse, "cosine": cosine, "r2": r2}


def validate_candidate(
    model: ContinuousVACSharedSpace,
    split: Any,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    shared, recovered = encode_arrays(model, split, indices, device, batch_size)
    episode = np.asarray(split.arrays["episode_id"][indices])
    native_names = {"vision": "z_v", "action": "z_a", "contact": "z_c"}
    alignment = {}
    for offset, (name, left, right) in enumerate((
        ("V-A", "vision", "action"), ("V-C", "vision", "contact"), ("A-C", "action", "contact")
    )):
        alignment[name] = pairwise_alignment_metrics(
            shared[left], shared[right], episode,
            bootstrap_samples=200, seed=seed + offset, retrieval_chunk=512,
        )
    recovery = {
        modality: recovery_metrics(
            np.asarray(split.arrays[native_names[modality]][indices]), recovered[modality]
        ) for modality in MODALITIES
    }
    geometry = {modality: geometry_diagnostics(shared[modality]) for modality in MODALITIES}
    native_rank = {
        modality: effective_rank(np.asarray(split.arrays[native_names[modality]][indices]))
        for modality in MODALITIES
    }
    margins = [alignment[name]["paired_minus_shuffled_margin"] for name in alignment]
    r10_multipliers = []
    for value in alignment.values():
        for direction in ("forward", "reverse"):
            retrieval = value["retrieval"][direction]
            r10_multipliers.append(retrieval["recall_at_10"] / retrieval["chance"]["recall_at_10"])
    recovery_cosine = float(np.mean([value["cosine"] for value in recovery.values()]))
    rank_retention = float(min(
        min(geometry[name]["effective_rank"] / max(native_rank[name], 1e-12), 1.0)
        for name in MODALITIES
    ))
    collapse = any(
        geometry[name]["per_dimension_variance"]["near_zero_fraction"] > 0.5
        or geometry[name]["query_diversity"]["collapsed_pair_fraction"] > 0.5
        for name in MODALITIES
    )
    score = (
        float(np.mean(margins))
        + 0.05 * float(np.mean(np.log1p(np.maximum(r10_multipliers, 0.0))))
        + 0.25 * recovery_cosine
        + 0.10 * rank_retention
        - (1.0 if collapse else 0.0)
    )
    return {
        "selection_split": "validation",
        "samples": len(indices),
        "alignment": alignment,
        "recovery": recovery,
        "geometry": geometry,
        "native_effective_rank": native_rank,
        "preservation_proxies": {
            "contact": {
                "recovery_cosine": recovery["contact"]["cosine"],
                "effective_rank_retention": geometry["contact"]["effective_rank"] / max(native_rank["contact"], 1e-12),
            },
            "action_temporal": {
                "recovery_cosine": recovery["action"]["cosine"],
                "relational_geometry_retained": geometry["action"]["effective_rank"] / max(native_rank["action"], 1e-12),
                "role": "validation proxy; frozen decoder temporal audit is test-only",
            },
        },
        "summary": {
            "mean_margin": float(np.mean(margins)),
            "mean_r10_chance_multiplier": float(np.mean(r10_multipliers)),
            "mean_recovery_cosine": recovery_cosine,
            "minimum_rank_retention": rank_retention,
            "collapse": collapse,
            "comprehensive_score": score,
        },
    }


def trial_grid() -> list[dict[str, Any]]:
    return [
        {"candidate": "C1-linear", "temperature": 0.10, "native_weight": 5.0, "dynamic_weight": 2.0},
        {"candidate": "C1-mlp", "temperature": 0.07, "native_weight": 5.0, "dynamic_weight": 2.0},
        {"candidate": "C1-mlp", "temperature": 0.10, "native_weight": 10.0, "dynamic_weight": 2.0},
        {"candidate": "C1-mlp", "temperature": 0.20, "native_weight": 1.0, "dynamic_weight": 1.0},
        {"candidate": "C2-slot", "temperature": 0.07, "native_weight": 10.0, "dynamic_weight": 2.0},
        {"candidate": "C2-slot", "temperature": 0.10, "native_weight": 5.0, "dynamic_weight": 2.0},
    ]


def train_trial(
    trial_id: int,
    trial: Mapping[str, Any],
    train: Any,
    validation: Any,
    validation_indices: np.ndarray,
    config: Mapping[str, Any],
    experiment_root: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    seed = int(config["seed"]) + trial_id * 1009
    set_seed(seed)
    model = ContinuousVACSharedSpace(str(trial["candidate"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    weights = VACLossWeights(
        alignment=float(config["training"]["loss_weights"]["alignment"]),
        native=float(trial["native_weight"]),
        relational=float(config["training"]["loss_weights"]["relational"]),
        variance=float(config["training"]["loss_weights"]["variance"]),
    )
    source_names = {"vision": "z_v", "action": "z_a", "contact": "z_c"}
    generator = np.random.default_rng(seed)
    history = []
    best: dict[str, Any] | None = None
    trial_path = experiment_root / f"trial_{trial_id:02d}_{trial['candidate']}"
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        order = generator.permutation(len(train))
        totals: dict[str, float] = {name: 0.0 for name in ("total", "alignment", "native", "relational", "variance")}
        batches = 0
        model.train()
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            if len(indices) < 2:
                continue
            native = {
                modality: torch.from_numpy(np.asarray(train.arrays[source_names[modality]][indices])).to(device)
                for modality in MODALITIES
            }
            episode = torch.from_numpy(np.asarray(train.arrays["episode_id"][indices], dtype=np.int64)).to(device)
            dynamic = torch.from_numpy(np.asarray(train.arrays["dynamic"][indices], dtype=np.bool_)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, breakdown = continuous_vac_loss(
                model, native, episode, dynamic,
                temperature=float(trial["temperature"]),
                dynamic_weight=float(trial["dynamic_weight"]),
                weights=weights,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite C2 training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for name, value in breakdown.items():
                totals[name] += float(value)
            batches += 1
        validation_result = validate_candidate(
            model, validation, validation_indices, device, batch_size,
            seed=seed + epoch * 17,
        )
        row = {
            "epoch": epoch,
            "train": {name: value / max(batches, 1) for name, value in totals.items()},
            "validation": validation_result,
        }
        history.append(row)
        score = validation_result["summary"]["comprehensive_score"]
        if best is None or score > best["score"]:
            checkpoint_path = trial_path / "best.pt"
            digest = save_checkpoint(
                checkpoint_path, model,
                {"trial_id": trial_id, "trial": dict(trial), "epoch": epoch,
                 "selection_split": "validation", "validation": validation_result},
            )
            best = {"score": score, "epoch": epoch, "checkpoint": str(checkpoint_path.relative_to(ROOT)), "sha256": digest, "validation": validation_result}
        atomic_json(trial_path / "history.json", {"trial": dict(trial), "history": history, "best": best})
    assert best is not None
    return {
        "trial_id": trial_id,
        "trial": dict(trial),
        "parameter_summary": model.parameter_summary(),
        "epochs": epochs,
        "seconds": time.monotonic() - started,
        "best": best,
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    cache_root = args.cache_root or ROOT / config["runtime"]["cache_root"]
    experiment_root = args.experiment_root or ROOT / config["runtime"]["experiment_root"]
    epochs = args.epochs or int(config["training"]["epochs"])
    batch_size = args.batch_size or int(config["training"]["batch_size"])
    device, lock_handle, gpu = resolve_device(args.device)
    try:
        train = load_split(cache_root, "train", verify_hashes=True)
        validation = load_split(cache_root, "validation", verify_hashes=True)
        if (cache_root / "test").is_dir():
            # Deliberately do not load test; selection is validation-only.
            pass
        validation_indices = np.linspace(
            0, len(validation) - 1, min(args.validation_samples, len(validation)), dtype=np.int64
        )
        c0 = ContinuousVACSharedSpace("C0").to(device)
        c0_validation = validate_candidate(
            c0, validation, validation_indices, device, batch_size, seed=int(config["seed"])
        )
        trials = trial_grid()
        if len(trials) != int(config["training"]["bounded_trials"]):
            raise RuntimeError("bounded trial preregistration changed")
        results = [
            train_trial(index, trial, train, validation, validation_indices, config,
                        experiment_root, device, epochs, batch_size)
            for index, trial in enumerate(trials)
        ]
        c0_record = {
            "trial_id": -1,
            "trial": {"candidate": "C0", "temperature": None, "native_weight": None, "dynamic_weight": None},
            "parameter_summary": c0.parameter_summary(),
            "best": {"score": c0_validation["summary"]["comprehensive_score"], "epoch": 0, "validation": c0_validation},
        }
        candidate_pool = [c0_record, *results]
        maximum = max(value["best"]["score"] for value in candidate_pool)
        eligible = [value for value in candidate_pool if value["best"]["score"] >= maximum - 0.01]
        selected = min(
            eligible,
            key=lambda value: (value["parameter_summary"]["total"], -value["best"]["score"]),
        )
        if selected["trial"]["candidate"] == "C0":
            selected_model = ContinuousVACSharedSpace("C0")
            selected_metadata = {"trial_id": -1, "trial": selected["trial"], "epoch": 0,
                                 "selection_split": "validation", "validation": c0_validation}
        else:
            checkpoint_path = ROOT / selected["best"]["checkpoint"]
            model_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            selected_model = ContinuousVACSharedSpace(model_payload["candidate"])
            selected_model.load_state_dict(model_payload["state_dict"], strict=True)
            selected_metadata = dict(model_payload["metadata"])
        selected_path = experiment_root / "selected.pt"
        selected_sha = save_checkpoint(
            selected_path, selected_model,
            {**selected_metadata, "selection_rule": "smallest within 0.01 of maximum validation comprehensive score"},
        )
        summary = {
            "schema": "tactile3d-unit.vac-c2-training.v1",
            "selection_split": "validation only",
            "test_loaded": False,
            "gpu": gpu,
            "training": {"epochs": epochs, "batch_size": batch_size, "trials": len(trials)},
            "C0_native": c0_validation,
            "trials": results,
            "selected": {
                "trial_id": selected["trial_id"],
                "candidate": selected["trial"]["candidate"],
                "checkpoint": str(selected_path.relative_to(ROOT)),
                "sha256": selected_sha,
                "validation": selected["best"]["validation"],
                "selection_rule": "smallest parameter count within 0.01 of maximum comprehensive score",
            },
        }
        atomic_json(experiment_root / "training_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    main()
