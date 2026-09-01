#!/usr/bin/env python3
"""Train frozen S4.2-3 Contact-Dynamics baselines and proposed candidates."""

from __future__ import annotations

import argparse
import json
import random
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
    CurrentOnlyPredictor,
    DeltaMLPEncoder,
    LatentTransitionDecoder,
)
from gr00t.simulation.s4_2_dataset import sha256_file  # noqa: E402

CACHE_ROOT = ROOT / ".local/cache/simulation/s4_2r/contact_dynamics"
EXPERIMENT_ROOT = ROOT / ".local/experiments/simulation/s4_2r/contact_dynamics"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_2r"
CONTRACT = ROOT / "configs/simulation/s4_2_representation_contract.json"
REGISTRY = ROOT / "configs/simulation/s4_2_model_candidate_registry.json"
ACCEPTANCE = ROOT / "configs/simulation/s4_2r_contact_state_acceptance.json"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(name: str) -> torch.nn.Module:
    if name == "C1":
        return CurrentOnlyPredictor()
    if name == "C2":
        return ContactDynamicsModel(DeltaMLPEncoder(), LatentTransitionDecoder())
    if name == "C3":
        return ContactDynamicsModel(ContactDynamicsEncoder(), LatentTransitionDecoder())
    raise ValueError(name)


def forward_model(
    model: torch.nn.Module,
    name: str,
    current: torch.Tensor,
    future: torch.Tensor,
) -> torch.Tensor:
    if name == "C1":
        return model(current)
    return model(current, future)["future"]


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    name: str,
    arrays: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    squared = []
    for start in range(0, len(arrays["h_current"]), batch_size):
        stop = min(start + batch_size, len(arrays["h_current"]))
        current = torch.from_numpy(arrays["h_current"][start:stop]).to(device)
        future = torch.from_numpy(arrays["h_future"][start:stop]).to(device)
        prediction = forward_model(model, name, current, future)
        squared.append((prediction - future).square().mean(dim=1).cpu().numpy())
    error = np.concatenate(squared)
    dynamic = arrays["dynamic"].astype(bool)
    return {
        "overall_future_mse": float(error.mean()),
        "dynamic_future_mse": float(error[dynamic].mean()),
        "dynamic_windows": int(dynamic.sum()),
    }


def train_one(
    name: str,
    candidate: str,
    lambda_delta: float,
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    device: torch.device,
    contract: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    seed = int(contract["seed"])
    seed_everything(seed)
    model = build_model(name).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters > int(contract["contact_dynamics"]["parameter_max"]):
        raise RuntimeError(f"{candidate} exceeds the frozen parameter budget")
    training = contract["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(training["epochs"])
    patience = int(training["patience"])
    batch_size = int(training["batch_size"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(training["learning_rate"]) * 0.05
    )
    best_dynamic = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    start_time = time.monotonic()
    rng = np.random.default_rng(seed)
    checkpoint_path = output / "best.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train["h_current"]))
        losses = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            current = torch.from_numpy(train["h_current"][indices]).to(device)
            future = torch.from_numpy(train["h_future"][indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = forward_model(model, name, current, future)
            future_loss = F.mse_loss(prediction, future)
            delta_loss = F.mse_loss(prediction - current, future - current)
            loss = future_loss + lambda_delta * delta_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        validation_metrics = evaluate(model, name, validation, device, batch_size)
        row = {
            "epoch": epoch,
            "train_objective": float(np.mean(losses)),
            "learning_rate": scheduler.get_last_lr()[0],
            "validation": validation_metrics,
        }
        history.append(row)
        print(
            json.dumps(
                {
                    "candidate": candidate,
                    "epoch": epoch,
                    "train": row["train_objective"],
                    "validation_dynamic": validation_metrics["dynamic_future_mse"],
                }
            ),
            flush=True,
        )
        if validation_metrics["dynamic_future_mse"] < best_dynamic:
            best_dynamic = float(validation_metrics["dynamic_future_mse"])
            best_epoch = epoch
            stale = 0
            output.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema": "tactile3d-unit.s4-2r-contact-dynamics-checkpoint.v1",
                    "model": name,
                    "candidate": candidate,
                    "lambda_delta": lambda_delta,
                    "seed": seed,
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "validation": validation_metrics,
                    "test_loaded": False,
                },
                checkpoint_path,
            )
        else:
            stale += 1
        if stale >= patience:
            break
    best = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reloaded = build_model(name)
    reloaded.load_state_dict(best["state_dict"], strict=True)
    reloaded = reloaded.to(device).eval()
    final_validation = evaluate(reloaded, name, validation, device, batch_size)
    return {
        "model": name,
        "candidate": candidate,
        "lambda_delta": lambda_delta,
        "seed": seed,
        "parameters": parameters,
        "best_epoch": best_epoch,
        "validation": final_validation,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_seconds": time.monotonic() - start_time,
        "history": history,
        "test_loaded": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    acceptance = json.loads(ACCEPTANCE.read_text(encoding="utf-8"))
    if not acceptance["s4_2_3_allowed"] or acceptance["test_loaded"]:
        raise RuntimeError("Contact-State acceptance does not authorize training")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    train = load_npz(args.cache_root / "train.npz")
    validation = load_npz(args.cache_root / "validation.npz")
    device = torch.device(args.device)
    trials = []
    trials.append(
        train_one(
            "C1", "C1-current-only", 0.0, train, validation, device, contract, args.experiment_root / "C1"
        )
    )
    trials.append(
        train_one(
            "C2", "C2-delta-mlp", 0.0, train, validation, device, contract, args.experiment_root / "C2"
        )
    )
    for trial in registry["contact_dynamics"]["trials"]:
        trials.append(
            train_one(
                "C3",
                trial["id"],
                float(trial["delta_weight"]),
                train,
                validation,
                device,
                contract,
                args.experiment_root / trial["id"],
            )
        )
    proposed = [row for row in trials if row["model"] == "C3"]
    selected = min(
        proposed,
        key=lambda row: (
            row["validation"]["dynamic_future_mse"],
            row["validation"]["overall_future_mse"],
            row["candidate"],
        ),
    )
    selected_path = ROOT / selected["checkpoint"]
    args.experiment_root.mkdir(parents=True, exist_ok=True)
    promoted = args.experiment_root / "selected.pt"
    shutil.copyfile(selected_path, promoted)
    if sha256_file(promoted) != selected["checkpoint_sha256"]:
        raise RuntimeError("promoted dynamics checkpoint identity mismatch")
    current = validation["h_current"]
    target = validation["h_future"]
    dynamic = validation["dynamic"].astype(bool)
    persistence_error = np.mean(np.square(current - target), axis=1)
    selection = {
        "schema": "tactile3d-unit.s4-2r-contact-dynamics-selection.v1",
        "selected": selected["candidate"],
        "selected_checkpoint": str(promoted.relative_to(ROOT)),
        "selected_checkpoint_sha256": sha256_file(promoted),
        "selection_split": "validation",
        "C0_persistence": {
            "overall_future_mse": float(persistence_error.mean()),
            "dynamic_future_mse": float(persistence_error[dynamic].mean()),
        },
        "trials": trials,
        "parameter_budget": contract["contact_dynamics"]["parameter_max"],
        "test_loaded": False,
        "selection_uses_test": False,
    }
    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "contact_dynamics_selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected": selected["candidate"],
                "checkpoint_sha256": sha256_file(promoted),
                "validation": selected["validation"],
                "test_loaded": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
