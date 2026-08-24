#!/usr/bin/env python3
"""Train the S1.3 continuous predictive contact-state teacher."""

from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gr00t.tactile_teacher.cache import WrenchWindowDataset
from gr00t.tactile_teacher.models import PredictiveContactTeacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".local/cache/tactile_teacher/s1_wrench_windows"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".local/experiments/tactile_teacher/s1_teacher"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(args: argparse.Namespace, split: str, shuffle: bool) -> DataLoader:
    return DataLoader(
        WrenchWindowDataset(args.cache_dir, split),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        drop_last=shuffle,
    )


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.inference_mode()
def evaluate(
    model: PredictiveContactTeacher, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    future_squared = 0.0
    future_absolute = 0.0
    history_squared = 0.0
    target_sum = 0.0
    target_squared_sum = 0.0
    future_elements = 0
    history_elements = 0
    for batch in loader:
        history = batch["history"].to(device, non_blocking=True)
        target = batch["future"].to(device, non_blocking=True)
        with autocast_context(device):
            output = model(history)
        future_error = output["future"].float() - target.float()
        history_error = output["reconstruction"].float() - history.float()
        future_squared += future_error.square().sum().item()
        future_absolute += future_error.abs().sum().item()
        history_squared += history_error.square().sum().item()
        target_sum += target.sum().item()
        target_squared_sum += target.square().sum().item()
        future_elements += target.numel()
        history_elements += history.numel()
    target_mean = target_sum / future_elements
    target_variance = target_squared_sum - future_elements * target_mean**2
    return {
        "future_mse": future_squared / future_elements,
        "future_mae": future_absolute / future_elements,
        "future_r2": 1.0 - future_squared / max(target_variance, 1e-12),
        "history_reconstruction_mse": history_squared / history_elements,
        "windows": len(loader.dataset),
    }


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    train_loader = make_loader(args, "train", True)
    val_loader = make_loader(args, "val", False)
    test_loader = make_loader(args, "test", False)
    model = PredictiveContactTeacher(
        latent_dim=args.latent_dim, channels=args.channels
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.05
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best.pt"
    best_val = float("inf")
    best_epoch = 0
    log = []
    start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        future_loss_total = 0.0
        reconstruction_total = 0.0
        batches = 0
        for batch in train_loader:
            history = batch["history"].to(device, non_blocking=True)
            future = batch["future"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                output = model(history)
                future_loss = F.mse_loss(output["future"], future)
                reconstruction = F.mse_loss(output["reconstruction"], history)
                loss = future_loss + args.reconstruction_weight * reconstruction
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach())
            future_loss_total += float(future_loss.detach())
            reconstruction_total += float(reconstruction.detach())
            batches += 1
        scheduler.step()
        validation = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "learning_rate": scheduler.get_last_lr()[0],
            "train_loss": total_loss / batches,
            "train_future_mse": future_loss_total / batches,
            "train_reconstruction_mse": reconstruction_total / batches,
            "val": validation,
        }
        log.append(row)
        print(json.dumps(row), flush=True)
        if validation["future_mse"] < best_val:
            best_val = validation["future_mse"]
            best_epoch = epoch
            torch.save(
                {
                    "schema": "tactile3d-unit.s1-contact-teacher-checkpoint.v1",
                    "latent_dim": args.latent_dim,
                    "channels": args.channels,
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_metrics": validation,
                },
                checkpoint_path,
            )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    result = {
        "schema": "tactile3d-unit.s1.3-predictive-contact-teacher.v1",
        "status": "PASS",
        "architecture": "residual dilated TCN + learned query pooling + MLP projection",
        "input": "normalized 16x60 wrench history with deterministic first differences",
        "latent_dim": args.latent_dim,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "objectives": {
            "future_wrench_mse_weight": 1.0,
            "history_reconstruction_mse_weight": args.reconstruction_weight,
        },
        "seed": args.seed,
        "device": str(device),
        "dataset_revision": manifest["dataset_revision"],
        "split_manifest_sha256": manifest["split_manifest_sha256"],
        "best_epoch": best_epoch,
        "training_seconds": time.monotonic() - start,
        "val": evaluate(model, val_loader, device),
        "test": evaluate(model, test_loader, device),
        "checkpoint": str(checkpoint_path),
        "history": log,
    }
    (args.output_dir / "s1_3_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
