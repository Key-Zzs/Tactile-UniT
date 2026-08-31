#!/usr/bin/env python3
"""Train and validate the preregistered S4.2 simulated Contact-state teacher."""

from __future__ import annotations

import json
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_2_dataset import sha256_file  # noqa: E402
from gr00t.simulation.s4_2_normalization import TactileNormalization  # noqa: E402
from gr00t.simulation.sim_contact_models import (  # noqa: E402
    build_sim_contact_teacher,
    load_teacher_checkpoint,
    parameter_count,
    save_teacher_checkpoint,
)
from gr00t.tactile_unit.continuous_vac_shared_space import (  # noqa: E402
    bootstrap_mean_ci,
    effective_rank,
)

PAIR_ROOT = ROOT / ".local/cache/simulation/s4_2/pairs"
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_2"
EXPERIMENT_ROOT = ROOT / ".local/experiments/simulation/s4_2/contact_teacher"
LATENT_ROOT = ROOT / ".local/cache/simulation/s4_2/contact_teacher"
CONTRACT = ROOT / "configs/simulation/s4_2_representation_contract.json"
CANDIDATES = ROOT / "configs/simulation/s4_2_model_candidate_registry.json"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def normalized_arrays(
    raw: dict[str, np.ndarray], normalization: TactileNormalization
) -> tuple[np.ndarray, np.ndarray]:
    history = normalization.transform(raw["current_history"]).astype(np.float32)
    future = normalization.transform(raw["teacher_future"]).astype(np.float32)
    return history, future


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    history: np.ndarray,
    future: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray | float]:
    model.eval()
    predictions = np.empty_like(future)
    latents = np.empty((len(history), 256), dtype=np.float32)
    reconstruction_sum = 0.0
    reconstruction_elements = 0
    for start in range(0, len(history), batch_size):
        stop = min(start + batch_size, len(history))
        value = torch.from_numpy(history[start:stop]).to(device)
        with autocast(device):
            output = model(value)
        predictions[start:stop] = output["future"].float().cpu().numpy()
        latents[start:stop] = output["latent"].float().cpu().numpy()
        if "reconstruction" in output:
            reconstruction_sum += (
                output["reconstruction"].float() - value.float()
            ).square().sum().item()
            reconstruction_elements += value.numel()
    error = np.mean(np.square(predictions - future), axis=(1, 2))
    return {
        "prediction": predictions,
        "latent": latents,
        "error": error,
        "future_mse": float(error.mean()),
        "history_reconstruction_mse": (
            reconstruction_sum / reconstruction_elements if reconstruction_elements else None
        ),
    }


def train_one(
    candidate: str,
    family: str,
    normalization: TactileNormalization,
    reconstruction_weight: float,
    train_history: np.ndarray,
    train_future: np.ndarray,
    validation_history: np.ndarray,
    validation_future: np.ndarray,
    contract: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    seed = int(contract["seed"])
    seed_everything(seed)
    model = build_sim_contact_teacher(
        family, reconstruction_weight=reconstruction_weight
    ).to(device)
    if parameter_count(model) > int(contract["contact_teacher"]["parameter_max"]):
        raise RuntimeError(f"{candidate} exceeds teacher parameter budget")
    training = contract["training"]
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_history), torch.from_numpy(train_future)),
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(training["epochs"]), eta_min=float(training["learning_rate"]) * 0.05
    )
    output_dir = EXPERIMENT_ROOT / "trials" / candidate
    checkpoint_path = output_dir / "best.pt"
    best = float("inf")
    best_epoch = 0
    patience = 0
    history_rows = []
    started = time.monotonic()
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        totals = []
        for batch_history, batch_future in loader:
            batch_history = batch_history.to(device, non_blocking=True)
            batch_future = batch_future.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device):
                output = model(batch_history)
                future_loss = F.mse_loss(output["future"], batch_future)
                reconstruction = (
                    F.mse_loss(output["reconstruction"], batch_history)
                    if "reconstruction" in output
                    else future_loss.new_zeros(())
                )
                loss = future_loss + reconstruction_weight * reconstruction
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
            optimizer.step()
            totals.append((float(loss.detach()), float(future_loss.detach()), float(reconstruction.detach())))
        scheduler.step()
        validation = predict(
            model,
            validation_history,
            validation_future,
            device,
            int(training["batch_size"]),
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean([value[0] for value in totals])),
            "train_future_mse": float(np.mean([value[1] for value in totals])),
            "train_reconstruction_mse": float(np.mean([value[2] for value in totals])),
            "validation_future_mse": validation["future_mse"],
        }
        history_rows.append(row)
        print(json.dumps({"candidate": candidate, **row}), flush=True)
        score = float(validation["future_mse"])
        if score < best - 1e-8:
            best = score
            best_epoch = epoch
            patience = 0
            save_teacher_checkpoint(
                checkpoint_path,
                model,
                candidate=candidate,
                normalization=normalization.to_json(),
                metadata={
                    "reconstruction_weight": reconstruction_weight,
                    "best_epoch": epoch,
                    "validation_future_mse": score,
                    "test_loaded": False,
                    "selection_uses_test": False,
                },
            )
        else:
            patience += 1
            if patience >= int(training["patience"]):
                break
    reloaded, checkpoint = load_teacher_checkpoint(checkpoint_path, device)
    reloaded = reloaded.to(device).eval()
    validation = predict(
        reloaded,
        validation_history,
        validation_future,
        device,
        int(training["batch_size"]),
    )
    return {
        "candidate": candidate,
        "family": family,
        "normalization": normalization.candidate,
        "reconstruction_weight": reconstruction_weight,
        "parameters": parameter_count(reloaded),
        "best_epoch": best_epoch,
        "training_seconds": time.monotonic() - started,
        "validation_future_mse": validation["future_mse"],
        "validation_reconstruction_mse": validation["history_reconstruction_mse"],
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "history": history_rows,
        "test_loaded": False,
        "_model": reloaded,
        "_validation": validation,
        "_checkpoint": checkpoint,
    }


def labels(raw: dict[str, np.ndarray], trend_deadband: float | None = None):
    current = raw["current_history"][:, -1].reshape(-1, 5, 6)
    future = raw["future_history"][:, -1].reshape(-1, 5, 6)
    current_force = current[:, :, 1].sum(axis=1)
    future_force = future[:, :, 1].sum(axis=1)
    delta = future_force - current_force
    if trend_deadband is None:
        trend_deadband = float(np.quantile(np.abs(delta), 0.33))
    trend = np.ones(len(delta), dtype=np.int64)
    trend[delta < -trend_deadband] = 0
    trend[delta > trend_deadband] = 2
    return {
        "contact": (current[:, :, 0].sum(axis=1) > 0).astype(np.int64),
        "force": current_force.astype(np.float32),
        "trend": trend,
        "region": (current[:, :, 0] > 0).astype(np.int64),
    }, trend_deadband


def majority_f1(target: np.ndarray) -> float:
    majority = int(np.bincount(target).argmax())
    return float(f1_score(target, np.full_like(target, majority), average="macro"))


def probe_latent(train_latent, validation_latent, train_raw, validation_raw) -> dict[str, Any]:
    train_labels, deadband = labels(train_raw)
    validation_labels, _ = labels(validation_raw, deadband)
    contact_model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=4242)
    contact_model.fit(train_latent, train_labels["contact"])
    contact_prediction = contact_model.predict(validation_latent)
    trend_model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=4242)
    trend_model.fit(train_latent, train_labels["trend"])
    trend_prediction = trend_model.predict(validation_latent)
    force_model = Ridge(alpha=10.0)
    force_model.fit(train_latent, train_labels["force"])
    force_prediction = force_model.predict(validation_latent)
    region_rows = []
    for region in range(5):
        target = train_labels["region"][:, region]
        if len(np.unique(target)) < 2:
            region_rows.append(
                {"region": region, "usable": False, "reason": "single_class_train_label"}
            )
            continue
        model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=4242)
        model.fit(train_latent, target)
        prediction = model.predict(validation_latent)
        validation_target = validation_labels["region"][:, region]
        region_rows.append(
            {
                "region": region,
                "usable": True,
                "macro_f1": float(f1_score(validation_target, prediction, average="macro")),
                "balanced_accuracy": float(balanced_accuracy_score(validation_target, prediction)),
                "positive_train": int(target.sum()),
                "positive_validation": int(validation_target.sum()),
            }
        )
    return {
        "trend_deadband_newton": deadband,
        "contact": {
            "macro_f1": float(f1_score(validation_labels["contact"], contact_prediction, average="macro")),
            "balanced_accuracy": float(
                balanced_accuracy_score(validation_labels["contact"], contact_prediction)
            ),
            "majority_macro_f1": majority_f1(validation_labels["contact"]),
        },
        "force": {
            "r2": float(r2_score(validation_labels["force"], force_prediction)),
            "mae": float(mean_absolute_error(validation_labels["force"], force_prediction)),
        },
        "trend": {
            "macro_f1": float(f1_score(validation_labels["trend"], trend_prediction, average="macro")),
            "balanced_accuracy": float(
                balanced_accuracy_score(validation_labels["trend"], trend_prediction)
            ),
            "majority_macro_f1": majority_f1(validation_labels["trend"]),
        },
        "per_region_contact": region_rows,
    }


def collapse_metrics(latent: np.ndarray) -> dict[str, Any]:
    variance = latent.var(axis=0)
    return {
        "effective_rank": effective_rank(latent),
        "mean_variance": float(variance.mean()),
        "minimum_variance": float(variance.min()),
        "near_zero_variance_fraction": float(np.mean(variance < 1e-8)),
        "mean_pairwise_distance_sample": float(
            np.linalg.norm(latent[:512, None] - latent[None, :512], axis=-1).mean()
        ),
    }


def corrupt(history: np.ndarray, kind: str, strength: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    value = history.copy()
    magnitude = 0.05 if strength == "mild" else 0.15
    force_indices = np.asarray([1, 2, 7, 8, 13, 14, 19, 20, 25, 26])
    if kind == "gaussian_force_noise":
        value[..., force_indices] += rng.normal(0, magnitude, value[..., force_indices].shape)
    elif kind == "force_bias":
        value[..., force_indices] += magnitude
    elif kind == "frame_dropout":
        mask = rng.random(value.shape[:2]) < (0.10 if strength == "mild" else 0.30)
        value[mask] = 0
    elif kind == "timestamp_jitter_surrogate":
        probability = 0.10 if strength == "mild" else 0.30
        for row in range(len(value)):
            for index in range(1, value.shape[1] - 1):
                if rng.random() < probability:
                    value[row, index] = value[row, index + rng.choice([-1, 1])]
    else:
        raise ValueError(kind)
    return value


def main() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    registry = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    normalization_candidates = json.loads(
        (ARTIFACT_ROOT / "normalization_candidates.json").read_text(encoding="utf-8")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("S4.2 Contact teacher requires an allocated CUDA device")
    raw = {split: load_npz(PAIR_ROOT / f"{split}.npz") for split in ("train", "validation")}
    normalized: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    normalizers = {}
    for name, value in normalization_candidates["candidates"].items():
        normalizer = TactileNormalization.from_json(value)
        normalizers[name] = normalizer
        normalized[name] = {
            split: normalized_arrays(raw[split], normalizer)
            for split in ("train", "validation")
        }

    proposed = []
    for trial in registry["contact_teacher"]["trials"]:
        name = trial["normalization"]
        train_history, train_future = normalized[name]["train"]
        val_history, val_future = normalized[name]["validation"]
        proposed.append(
            train_one(
                trial["id"],
                "B3",
                normalizers[name],
                float(trial["reconstruction_weight"]),
                train_history,
                train_future,
                val_history,
                val_future,
                contract,
                device,
            )
        )
    proposed.sort(
        key=lambda row: (
            round(float(row["validation_future_mse"]), 6),
            registry["contact_teacher"]["trials"].index(
                next(value for value in registry["contact_teacher"]["trials"] if value["id"] == row["candidate"])
            ),
        )
    )
    selected = proposed[0]
    selected_normalization = normalizers[selected["normalization"]]
    train_history, train_future = normalized[selected["normalization"]]["train"]
    val_history, val_future = normalized[selected["normalization"]]["validation"]

    baselines = []
    for family in ("B0", "B1", "B2"):
        baselines.append(
            train_one(
                family,
                family,
                selected_normalization,
                0.0,
                train_history,
                train_future,
                val_history,
                val_future,
                contract,
                device,
            )
        )

    selected_model = selected["_model"]
    selected_validation = selected["_validation"]
    baseline_by_name = {row["candidate"]: row for row in baselines}
    strongest = min(baselines, key=lambda row: row["validation_future_mse"])
    proposed_error = np.asarray(selected_validation["error"])
    strongest_error = np.asarray(strongest["_validation"]["error"])
    overall_improvement = 1.0 - float(proposed_error.mean() / strongest_error.mean())
    overall_ci = bootstrap_mean_ci(
        strongest_error - proposed_error,
        samples=int(contract["bootstrap_samples"]),
        seed=int(contract["seed"]) + 10,
    )
    last_prediction = np.repeat(val_history[:, -1:, :], 13, axis=1)
    last_error = np.mean(np.square(last_prediction - val_future), axis=(1, 2))
    threshold_candidates = {}
    for quantile in (0.70, 0.75):
        threshold = float(np.quantile(raw["train"]["force_delta_abs"], quantile))
        dynamic = raw["validation"]["force_delta_abs"] > threshold
        b0_error = np.asarray(baseline_by_name["B0"]["_validation"]["error"])
        improvements = {
            "B0": 1.0 - float(proposed_error[dynamic].mean() / b0_error[dynamic].mean()),
            "last_frame": 1.0 - float(proposed_error[dynamic].mean() / last_error[dynamic].mean()),
        }
        threshold_candidates[f"q{int(quantile * 100)}"] = {
            "threshold_newton": threshold,
            "validation_dynamic": int(dynamic.sum()),
            "improvements": improvements,
            "selection_score": min(improvements.values()),
        }
    dynamic_name = max(
        threshold_candidates,
        key=lambda name: (threshold_candidates[name]["selection_score"], name == "q75"),
    )
    dynamic_threshold = threshold_candidates[dynamic_name]["threshold_newton"]
    dynamic = raw["validation"]["force_delta_abs"] > dynamic_threshold
    b0_error = np.asarray(baseline_by_name["B0"]["_validation"]["error"])
    dynamic_rows = {}
    for name, control_error in (("B0", b0_error), ("last_frame", last_error)):
        difference = control_error[dynamic] - proposed_error[dynamic]
        dynamic_rows[name] = {
            "control_mse": float(control_error[dynamic].mean()),
            "proposed_mse": float(proposed_error[dynamic].mean()),
            "relative_improvement": 1.0 - float(
                proposed_error[dynamic].mean() / control_error[dynamic].mean()
            ),
            "improvement_ci95": bootstrap_mean_ci(
                difference,
                samples=int(contract["bootstrap_samples"]),
                seed=int(contract["seed"]) + 20 + len(dynamic_rows),
            ),
        }

    controls = {
        "last_frame_repeated": np.repeat(val_history[:, -1:, :], 26, axis=1),
        "shuffled_history": val_history[:, np.random.default_rng(4242).permutation(26)],
        "reversed_history": val_history[:, ::-1].copy(),
    }
    temporal = {}
    for index, (name, value) in enumerate(controls.items()):
        result = predict(
            selected_model, value, val_future, device, int(contract["training"]["batch_size"])
        )
        control_error = np.asarray(result["error"])
        difference = control_error[dynamic] - proposed_error[dynamic]
        temporal[name] = {
            "dynamic_mse": float(control_error[dynamic].mean()),
            "full_dynamic_mse": float(proposed_error[dynamic].mean()),
            "relative_improvement": 1.0 - float(
                proposed_error[dynamic].mean() / control_error[dynamic].mean()
            ),
            "improvement_ci95": bootstrap_mean_ci(
                difference,
                samples=int(contract["bootstrap_samples"]),
                seed=int(contract["seed"]) + 30 + index,
            ),
        }

    train_prediction = predict(
        selected_model,
        train_history,
        train_future,
        device,
        int(contract["training"]["batch_size"]),
    )
    probes = probe_latent(
        np.asarray(train_prediction["latent"]),
        np.asarray(selected_validation["latent"]),
        raw["train"],
        raw["validation"],
    )
    collapse = collapse_metrics(np.asarray(selected_validation["latent"]))
    robustness = {"clean": float(proposed_error.mean())}
    for corruption in contract["contact_teacher"]["robustness"]:
        robustness[corruption] = {}
        for strength_index, strength in enumerate(("mild", "strong")):
            damaged = corrupt(val_history, corruption, strength, 5000 + strength_index)
            result = predict(
                selected_model,
                damaged,
                val_future,
                device,
                int(contract["training"]["batch_size"]),
            )
            robustness[corruption][strength] = float(result["future_mse"])

    checkpoint_path = EXPERIMENT_ROOT / "selected.pt"
    save_teacher_checkpoint(
        checkpoint_path,
        selected_model,
        candidate=selected["candidate"],
        normalization=selected_normalization.to_json(),
        metadata={
            "reconstruction_weight": selected["reconstruction_weight"],
            "selected_from_validation": True,
            "dynamic_threshold_candidate": dynamic_name,
            "dynamic_threshold_newton": dynamic_threshold,
            "test_loaded": False,
            "selection_uses_test": False,
        },
    )
    reloaded, _ = load_teacher_checkpoint(checkpoint_path, device)
    reloaded = reloaded.to(device).eval()
    reference = predict(selected_model, val_history[:128], val_future[:128], device, 128)
    cold = predict(reloaded, val_history[:128], val_future[:128], device, 128)
    deterministic_reload = bool(
        np.array_equal(reference["prediction"], cold["prediction"])
        and np.array_equal(reference["latent"], cold["latent"])
    )

    gates = {
        "finite": bool(np.isfinite(proposed_error).all() and np.isfinite(selected_validation["latent"]).all()),
        "no_collapse": bool(
            collapse["effective_rank"] >= contract["contact_teacher"]["gates"]["effective_rank_min"]
            and collapse["near_zero_variance_fraction"]
            <= contract["contact_teacher"]["gates"]["near_zero_variance_fraction_max"]
        ),
        "overall_baseline_improvement": bool(
            overall_improvement
            >= contract["contact_teacher"]["gates"]["overall_improvement_over_strongest_baseline"]
            and overall_ci[0] > 0
        ),
        "dynamic_improvement": bool(
            all(
                row["relative_improvement"]
                >= contract["contact_teacher"]["gates"]["dynamic_improvement_over_current_and_last_frame"]
                and row["improvement_ci95"][0] > 0
                for row in dynamic_rows.values()
            )
        ),
        "temporal_value": bool(
            all(row["improvement_ci95"][0] > 0 for row in temporal.values())
        ),
        "deterministic_reload": deterministic_reload,
        "contact_probe": bool(
            probes["contact"]["macro_f1"]
            >= probes["contact"]["majority_macro_f1"]
            + contract["contact_teacher"]["gates"]["contact_macro_f1_over_majority"]
        ),
        "force_probe": bool(
            probes["force"]["r2"] >= contract["contact_teacher"]["gates"]["force_r2"]
        ),
        "trend_probe": bool(
            probes["trend"]["macro_f1"]
            >= probes["trend"]["majority_macro_f1"]
            + contract["contact_teacher"]["gates"]["trend_macro_f1_over_majority"]
        ),
    }
    status = "S4_2_SIM_CONTACT_STATE_READY" if all(gates.values()) else "S4_2_CONTACT_STATE_FAIL"
    cleanups = ("_model", "_validation", "_checkpoint")
    public_proposed = [{key: value for key, value in row.items() if key not in cleanups} for row in proposed]
    public_baselines = [{key: value for key, value in row.items() if key not in cleanups} for row in baselines]
    selection = {
        "schema": "tactile3d-unit.s4-2-contact-teacher-selection.v1",
        "selected": selected["candidate"],
        "normalization": selected_normalization.candidate,
        "dynamic_threshold_candidate": dynamic_name,
        "dynamic_threshold_candidates": threshold_candidates,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "proposed_trials": public_proposed,
        "baselines": public_baselines,
        "test_loaded": False,
        "selection_uses_test": False,
        "training_uses_test": False,
    }
    evaluation = {
        "schema": "tactile3d-unit.s4-2-contact-teacher-validation.v1",
        "status": status,
        "input_shape": [26, 30],
        "future_target_shape": [13, 30],
        "future_target_relative_steps": [1, 13],
        "selected": selected["candidate"],
        "parameters": selected["parameters"],
        "validation_future_mse": float(proposed_error.mean()),
        "strongest_baseline": strongest["candidate"],
        "strongest_baseline_mse": float(strongest_error.mean()),
        "overall_relative_improvement": overall_improvement,
        "overall_improvement_ci95": overall_ci,
        "dynamic": dynamic_rows,
        "temporal_controls": temporal,
        "probes": probes,
        "collapse": collapse,
        "robustness": robustness,
        "deterministic_reload": deterministic_reload,
        "gates": gates,
        "warnings": [
            "right_thumb per-region probe unusable due single-class train label"
        ],
        "test_loaded": False,
    }
    atomic_json(ARTIFACT_ROOT / "teacher_selection.json", selection)
    atomic_json(ARTIFACT_ROOT / "teacher_evaluation.json", evaluation)
    atomic_json(
        ARTIFACT_ROOT / "normalization.json",
        {
            "schema": "tactile3d-unit.s4-2-selected-normalization.v1",
            **selected_normalization.to_json(),
            "selection_split": "validation",
            "test_loaded": False,
        },
    )
    if status == "S4_2_SIM_CONTACT_STATE_READY":
        LATENT_ROOT.mkdir(parents=True, exist_ok=True)
        latent_manifest = {"schema": "tactile3d-unit.s4-2-contact-state-cache.v1", "splits": {}}
        for split, history, future in (
            ("train", train_history, normalized[selected["normalization"]]["train"][1]),
            ("validation", val_history, normalized[selected["normalization"]]["validation"][1]),
        ):
            current_state = predict(selected_model, history, future, device, 1024)["latent"]
            future_history = selected_normalization.transform(raw[split]["future_history"]).astype(np.float32)
            future_state = predict(
                selected_model,
                future_history,
                np.zeros((len(future_history), 13, 30), dtype=np.float32),
                device,
                1024,
            )["latent"]
            destination = LATENT_ROOT / f"{split}.npz"
            np.savez_compressed(
                destination,
                pair_id=raw[split]["pair_id"],
                episode_id=raw[split]["episode_id"],
                task=raw[split]["task"],
                source_trajectory_id=raw[split]["source_trajectory_id"],
                h_current=current_state,
                h_future=future_state,
                contact_transition=raw[split]["contact_transition"],
                current_total_force=raw[split]["current_total_force"],
                future_total_force=raw[split]["future_total_force"],
                force_delta_abs=raw[split]["force_delta_abs"],
            )
            latent_manifest["splits"][split] = {
                "path": str(destination.relative_to(ROOT)),
                "sha256": sha256_file(destination),
                "pairs": len(current_state),
                "shape": [len(current_state), 256],
            }
        latent_manifest["test_cached"] = False
        atomic_json(ARTIFACT_ROOT / "contact_state_cache_manifest.json", latent_manifest)
    print(json.dumps({"selection": selection, "evaluation": evaluation}, indent=2))
    if status != "S4_2_SIM_CONTACT_STATE_READY":
        raise SystemExit(status)


if __name__ == "__main__":
    main()
