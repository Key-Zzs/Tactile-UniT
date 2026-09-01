#!/usr/bin/env python3
"""Run the three preregistered bounded S4.2-DR Dynamics trials."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.contact_dynamics.models import (  # noqa: E402
    ContactDynamicsEncoder,
    ContactDynamicsModel,
    LatentTransitionDecoder,
)
from scripts.simulation.s4_2dr_common import (  # noqa: E402
    ARTIFACT_ROOT,
    CACHE_ROOT,
    DR_EXPERIMENT_ROOT,
    ROOT,
    atomic_json,
    canonical_hash,
    ensure_local_roots,
    load_json,
    load_npz,
    per_sample_mse,
    seed_everything,
    sha256_file,
)

CONFIG_PATH = ROOT / "configs/simulation/s4_2dr_contact_dynamics_remediation.json"


def build_model() -> ContactDynamicsModel:
    return ContactDynamicsModel(ContactDynamicsEncoder(), LatentTransitionDecoder())


@torch.inference_mode()
def evaluate(
    model: ContactDynamicsModel,
    arrays: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> dict[str, float | int]:
    model.eval()
    predictions = []
    for start in range(0, len(arrays["h_current"]), batch_size):
        current = torch.from_numpy(arrays["h_current"][start:start + batch_size]).to(device)
        future = torch.from_numpy(arrays["h_future"][start:start + batch_size]).to(device)
        predictions.append(model(current, future)["future"].cpu().numpy())
    prediction = np.concatenate(predictions)
    error = per_sample_mse(prediction, arrays["h_future"])
    dynamic = arrays["dynamic"].astype(bool)
    return {
        "overall_future_mse": float(error.mean()),
        "dynamic_future_mse": float(error[dynamic].mean()),
        "dynamic_windows": int(dynamic.sum()),
    }


def different_episode_indices(episode_id: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = rng.permutation(len(episode_id))
    bad = episode_id[result] == episode_id
    for _ in range(20):
        if not bad.any():
            return result
        result[bad] = result[bad][rng.permutation(int(bad.sum()))]
        bad = episode_id[result] == episode_id
    # Deterministic fallback searches forward for a different episode.
    for index in np.flatnonzero(bad):
        candidate = (int(result[index]) + 1) % len(result)
        while episode_id[candidate] == episode_id[index]:
            candidate = (candidate + 1) % len(result)
        result[index] = candidate
    return result


def contrastive_loss(
    model: ContactDynamicsModel,
    current: torch.Tensor,
    future: torch.Tensor,
    negative_future: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    positive = model.encoder(current, future).flatten(1)
    negative = model.encoder(current, negative_future).flatten(1)
    reversed_code = model.encoder(future, current).flatten(1)
    negative_distance = 1.0 - F.cosine_similarity(positive, negative, dim=1)
    reversed_distance = 1.0 - F.cosine_similarity(positive, reversed_code, dim=1)
    return 0.5 * (
        F.relu(margin - negative_distance).mean()
        + F.relu(margin - reversed_distance).mean()
    )


def train_trial(
    trial: str,
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    device: torch.device,
    config: dict[str, Any],
    median_dynamic_delta_norm: float,
) -> dict[str, Any]:
    seed_everything(4242)
    model = build_model().to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters > config["remediation"]["parameter_max"]:
        raise RuntimeError("Dynamics parameter budget exceeded")
    training = config["remediation"]["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training["epochs_max"], eta_min=training["learning_rate"] * 0.05
    )
    batch_size = int(training["batch_size"])
    rng = np.random.default_rng(4242)
    negative_indices = different_episode_indices(train["episode_id"], 4242)
    output = DR_EXPERIMENT_ROOT / trial
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best.pt"
    best_dynamic = float("inf")
    best_epoch = 0
    stale = 0
    optimizer_steps = 0
    history = []
    transition_scale = 1.0
    scale_audit: dict[str, Any] | None = None
    start_time = time.monotonic()
    for epoch in range(1, int(training["epochs_max"]) + 1):
        model.train()
        order = rng.permutation(len(train["h_current"]))
        losses = []
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            current = torch.from_numpy(train["h_current"][indices]).to(device)
            future = torch.from_numpy(train["h_future"][indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            output_value = model(current, future)
            residual_per_sample = (output_value["future"] - current - (future - current)).square().mean(dim=1)
            absolute_future_loss = F.mse_loss(output_value["future"], future)
            if trial == "R2":
                delta_norm = torch.linalg.vector_norm(future - current, dim=1)
                weight = 1.0 + torch.clamp(delta_norm / median_dynamic_delta_norm, max=3.0)
                prediction_loss = (weight * residual_per_sample).mean() + 0.25 * absolute_future_loss
            else:
                prediction_loss = residual_per_sample.mean() + 0.25 * absolute_future_loss
            if trial == "R3":
                negative_future = torch.from_numpy(train["h_future"][negative_indices[indices]]).to(device)
                transition = contrastive_loss(model, current, future, negative_future, margin=0.2)
                if scale_audit is None:
                    raw_ratio = float(transition.detach() / prediction_loss.detach().clamp_min(1e-12))
                    rescaled = raw_ratio > 10.0 or raw_ratio < 0.01
                    if rescaled and float(transition.detach()) > 0.0:
                        transition_scale = float(prediction_loss.detach() / transition.detach())
                    scale_audit = {
                        "prediction_loss": float(prediction_loss.detach()),
                        "contrastive_loss": float(transition.detach()),
                        "raw_ratio": raw_ratio,
                        "rescaled": rescaled,
                        "deterministic_train_only_scale": transition_scale,
                    }
                loss = prediction_loss + 0.01 * transition_scale * transition
            else:
                loss = prediction_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer_steps += 1
            losses.append(float(loss.detach()))
        scheduler.step()
        metrics = evaluate(model, validation, device, batch_size)
        row = {"epoch": epoch, "train_objective": float(np.mean(losses)), "learning_rate": scheduler.get_last_lr()[0], "validation": metrics}
        history.append(row)
        print(json.dumps({"trial": trial, "epoch": epoch, "dynamic_mse": metrics["dynamic_future_mse"], "overall_mse": metrics["overall_future_mse"]}), flush=True)
        if metrics["dynamic_future_mse"] < best_dynamic:
            best_dynamic = float(metrics["dynamic_future_mse"])
            best_epoch = epoch
            stale = 0
            torch.save({
                "schema": "tactile3d-unit.s4-2dr-contact-dynamics-checkpoint.v1",
                "model": "C3", "trial": trial, "seed": 4242, "epoch": epoch,
                "state_dict": model.state_dict(), "validation": metrics,
                "median_train_dynamic_delta_norm": median_dynamic_delta_norm,
                "transition_scale_audit": scale_audit,
                "test_loaded": False,
            }, checkpoint_path)
        else:
            stale += 1
        if stale >= int(training["patience"]):
            break
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reloaded = build_model()
    reloaded.load_state_dict(checkpoint["state_dict"], strict=True)
    reloaded = reloaded.to(device).eval()
    final_metrics = evaluate(reloaded, validation, device, batch_size)
    original_max_steps = math.ceil(len(train["h_current"]) / batch_size) * int(training["epochs_max"])
    allowed_steps = math.floor(original_max_steps * float(training["optimizer_steps_max_relative_to_original"]))
    if optimizer_steps > allowed_steps:
        raise RuntimeError("optimizer-step budget exceeded")
    return {
        "trial": trial,
        "objective": next(row["objective"] for row in config["remediation"]["trials"] if row["id"] == trial),
        "parameters": parameters,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "optimizer_steps": optimizer_steps,
        "optimizer_steps_allowed": allowed_steps,
        "validation": final_metrics,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "transition_scale_audit": scale_audit,
        "training_seconds": time.monotonic() - start_time,
        "history": history,
        "test_loaded": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_local_roots()
    followup = load_json(ARTIFACT_ROOT / "existing_c3_followup.json")
    freeze = load_json(ARTIFACT_ROOT / "protocol_freeze.json")
    if followup["classification"] != "DYNAMICS_REMEDIATION_REQUIRED":
        raise RuntimeError("DR6 is not authorized")
    if freeze["test_loaded"] or freeze["new_training_started"]:
        raise RuntimeError("invalid protocol freeze")
    if sha256_file(CACHE_ROOT / "train.npz") != freeze["dataset"]["train_sha256"]:
        raise RuntimeError("frozen TRAIN cache changed")
    if sha256_file(CACHE_ROOT / "validation.npz") != freeze["dataset"]["validation_sha256"]:
        raise RuntimeError("frozen validation cache changed")
    config = load_json(CONFIG_PATH)
    train = load_npz(CACHE_ROOT / "train.npz")
    validation = load_npz(CACHE_ROOT / "validation.npz")
    dynamic = train["dynamic"].astype(bool)
    delta_norm = np.linalg.norm(train["h_future"] - train["h_current"], axis=1)
    median_dynamic_delta_norm = float(np.median(delta_norm[dynamic]))
    trials = [
        train_trial(name, train, validation, torch.device(args.device), config, median_dynamic_delta_norm)
        for name in ("R1", "R2", "R3")
    ]
    summary = {
        "schema": "tactile3d-unit.s4-2dr-remediation-trials.v1",
        "status": "TRAINING_COMPLETE_VALIDATION_GATES_PENDING",
        "maximum_trials": 3,
        "total_trials": len(trials),
        "median_train_dynamic_delta_norm": median_dynamic_delta_norm,
        "frozen_contact_state_checkpoint_sha256": freeze["contact_state_checkpoint_sha256"],
        "frozen_C2_checkpoint_sha256": freeze["C2_checkpoint_sha256"],
        "trials": trials,
        "selection_uses_test": False,
        "test_loaded": False,
    }
    summary["training_result_hash"] = canonical_hash(summary)
    atomic_json(ARTIFACT_ROOT / "remediation_trials.json", summary)
    print(json.dumps({"status": summary["status"], "trials": [{"trial": row["trial"], **row["validation"]} for row in trials], "test_loaded": False}, indent=2))


if __name__ == "__main__":
    main()
