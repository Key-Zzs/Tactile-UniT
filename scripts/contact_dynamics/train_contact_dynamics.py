#!/usr/bin/env python3
"""Validation-select and train S2 contact-transition baselines and model."""

from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from gr00t.contact_dynamics.models import (
    ContactDynamicsEncoder,
    ContactDynamicsModel,
    CurrentOnlyPredictor,
    DeltaMLPEncoder,
    LatentTransitionDecoder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".local/cache/contact_dynamics/s2_transition_pairs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".local/experiments/contact_dynamics/s2_models"),
    )
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pilot-epochs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LatentPairDataset(Dataset):
    def __init__(self, cache_dir: Path, split: str) -> None:
        split_dir = cache_dir / split
        self.current = np.load(split_dir / "current.npy", mmap_mode="r")
        self.future = np.load(split_dir / "future.npy", mmap_mode="r")
        self.dynamic = np.load(split_dir / "dynamic.npy", mmap_mode="r")
        if len({len(self.current), len(self.future), len(self.dynamic)}) != 1:
            raise ValueError("latent pair arrays do not align")

    def __len__(self) -> int:
        return len(self.current)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "current": torch.from_numpy(np.array(self.current[index], copy=True)),
            "future": torch.from_numpy(np.array(self.future[index], copy=True)),
            "dynamic": torch.tensor(bool(self.dynamic[index])),
        }


def make_loader(
    args: argparse.Namespace,
    split: str,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        LatentPairDataset(args.cache_dir, split),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        drop_last=shuffle,
        generator=generator,
    )


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def build_model(name: str) -> torch.nn.Module:
    if name == "C1":
        return CurrentOnlyPredictor()
    if name == "C2":
        return ContactDynamicsModel(DeltaMLPEncoder(), LatentTransitionDecoder())
    if name == "proposed":
        return ContactDynamicsModel(ContactDynamicsEncoder(), LatentTransitionDecoder())
    raise ValueError(f"unknown model: {name}")


def predict(
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
    loader: DataLoader,
    device: torch.device,
) -> dict[str, dict[str, float | int]]:
    model.eval()
    totals = {
        "all": {"squared": 0.0, "elements": 0},
        "dynamic": {"squared": 0.0, "elements": 0, "windows": 0},
    }
    for batch in loader:
        current = batch["current"].to(device, non_blocking=True)
        future = batch["future"].to(device, non_blocking=True)
        dynamic = batch["dynamic"].to(device, non_blocking=True).bool()
        with autocast_context(device):
            prediction = predict(model, name, current, future)
        error = prediction.float() - future.float()
        totals["all"]["squared"] += float(error.square().sum())
        totals["all"]["elements"] += error.numel()
        if dynamic.any():
            dynamic_error = error[dynamic]
            totals["dynamic"]["squared"] += float(dynamic_error.square().sum())
            totals["dynamic"]["elements"] += dynamic_error.numel()
            totals["dynamic"]["windows"] += int(dynamic.sum())
    return {
        subset: {
            "future_mse": values["squared"] / values["elements"],
            "delta_mse": values["squared"] / values["elements"],
            "windows": len(loader.dataset)
            if subset == "all"
            else int(values["windows"]),
        }
        for subset, values in totals.items()
    }


def train_one(
    args: argparse.Namespace,
    *,
    name: str,
    lambda_delta: float,
    epochs: int,
    seed: int,
    checkpoint_path: Path | None,
) -> dict:
    seed_everything(seed)
    device = torch.device(args.device)
    train_loader = make_loader(args, "train", shuffle=True, seed=seed)
    val_loader = make_loader(args, "val", shuffle=False, seed=seed)
    model = build_model(name).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=args.learning_rate * 0.05
    )
    best_dynamic = float("inf")
    best_epoch = 0
    history = []
    run_start = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        for batch in train_loader:
            current = batch["current"].to(device, non_blocking=True)
            future = batch["future"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                prediction = predict(model, name, current, future)
                future_loss = F.mse_loss(prediction, future)
                delta_loss = F.mse_loss(prediction - current, future - current)
                loss = future_loss + lambda_delta * delta_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        scheduler.step()
        validation = evaluate(model, name, val_loader, device)
        row = {
            "epoch": epoch,
            "learning_rate": scheduler.get_last_lr()[0],
            "train_objective": total / batches,
            "validation": validation,
        }
        history.append(row)
        print(
            json.dumps(
                {
                    "model": name,
                    "lambda_delta": lambda_delta,
                    "epoch": epoch,
                    "train": row["train_objective"],
                    "val_dynamic_mse": validation["dynamic"]["future_mse"],
                }
            ),
            flush=True,
        )
        dynamic_mse = float(validation["dynamic"]["future_mse"])
        if dynamic_mse < best_dynamic:
            best_dynamic = dynamic_mse
            best_epoch = epoch
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "schema": "tactile3d-unit.s2-contact-dynamics-checkpoint.v1",
                        "model": name,
                        "lambda_delta": lambda_delta,
                        "epoch": epoch,
                        "seed": seed,
                        "state_dict": model.state_dict(),
                        "validation": validation,
                    },
                    checkpoint_path,
                )
    return {
        "model": name,
        "lambda_delta": lambda_delta,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_dynamic_mse": best_dynamic,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": time.monotonic() - run_start,
        "history": history,
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_start = time.monotonic()
    pilot_results = []
    for lambda_delta in (0.0, 0.5, 1.0):
        pilot_results.append(
            train_one(
                args,
                name="proposed",
                lambda_delta=lambda_delta,
                epochs=args.pilot_epochs,
                seed=args.seed,
                checkpoint_path=None,
            )
        )
    selected = min(pilot_results, key=lambda value: value["best_val_dynamic_mse"])
    selected_lambda = float(selected["lambda_delta"])
    print(json.dumps({"selected_lambda_delta": selected_lambda}), flush=True)
    final_results = {}
    for name in ("C1", "C2", "proposed"):
        final_results[name] = train_one(
            args,
            name=name,
            lambda_delta=selected_lambda,
            epochs=args.epochs,
            seed=args.seed,
            checkpoint_path=args.output_dir / f"{name.lower()}_best.pt",
        )
    summary = {
        "schema": "tactile3d-unit.s2-contact-dynamics-training.v1",
        "selection_partition": "validation",
        "test_used_for_selection": False,
        "pilot": pilot_results,
        "selected_lambda_delta": selected_lambda,
        "models": final_results,
        "device": str(device),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "total_seconds": time.monotonic() - total_start,
        "status": "PASS",
    }
    write_json(args.output_dir / "s2_training_summary.json", summary)
    print(
        json.dumps(
            {
                "status": "PASS",
                "selected_lambda_delta": selected_lambda,
                "best_val_dynamic_mse": {
                    name: value["best_val_dynamic_mse"]
                    for name, value in final_results.items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
