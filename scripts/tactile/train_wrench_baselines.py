#!/usr/bin/env python3
"""Train S1.2 future-predictive wrench-history baselines on the real cache."""

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
from torch.utils.data import DataLoader

from gr00t.tactile_teacher.cache import WrenchWindowDataset
from gr00t.tactile_teacher.models import TactileVQBaseline, build_baseline


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
        default=Path(".local/experiments/tactile_teacher/s1_baselines"),
    )
    parser.add_argument("--models", nargs="+", default=["B0", "B1", "B2", "B3"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--vq-future-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def make_loader(
    cache_dir: Path,
    split: str,
    batch_size: int,
    workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = WrenchWindowDataset(cache_dir, split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=shuffle,
    )


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def vq_reconstruction_loss(
    output: dict[str, torch.Tensor], history: torch.Tensor
) -> torch.Tensor:
    per_sample = (output["reconstruction"] - history).square().mean(dim=(1, 2))
    finger_force = history.view(len(history), 16, 10, 6)[..., :3]
    magnitude = finger_force.square().sum(dim=-1).sqrt().amax(dim=(1, 2))
    weight = 1.0 + 2.0 * torch.sigmoid(magnitude / 4.0 - 1.0)
    return (per_sample * weight).sum() / weight.sum().clamp_min(1e-8)


def training_loss(
    model: torch.nn.Module,
    history: torch.Tensor,
    future: torch.Tensor,
    vq_future_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = model(history)
    future_loss = F.mse_loss(output["future"], future)
    components = {"future_mse": float(future_loss.detach())}
    if isinstance(model, TactileVQBaseline):
        reconstruction = vq_reconstruction_loss(output, history)
        loss = reconstruction + output["commitment"] + vq_future_weight * future_loss
        components.update(
            reconstruction_mse=float(reconstruction.detach()),
            commitment=float(output["commitment"].detach()),
        )
        return loss, components
    return future_loss, components


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    absolute_error = 0.0
    squared_error = 0.0
    target_sum = 0.0
    target_squared_sum = 0.0
    elements = 0
    recon_squared_error = 0.0
    recon_elements = 0
    code_counts = None
    for batch in loader:
        history = batch["history"].to(device, non_blocking=True)
        target = batch["future"].to(device, non_blocking=True)
        with autocast_context(device):
            output = model(history)
        prediction = output["future"].float()
        target = target.float()
        error = prediction - target
        absolute_error += error.abs().sum().item()
        squared_error += error.square().sum().item()
        target_sum += target.sum().item()
        target_squared_sum += target.square().sum().item()
        elements += target.numel()
        if "reconstruction" in output:
            recon_error = output["reconstruction"].float() - history.float()
            recon_squared_error += recon_error.square().sum().item()
            recon_elements += recon_error.numel()
            counts = torch.bincount(
                output["indices"].reshape(-1).cpu(),
                minlength=model.config.codebook_size,
            )
            code_counts = counts if code_counts is None else code_counts + counts
    mean = target_sum / elements
    total_variance = target_squared_sum - elements * mean * mean
    result = {
        "future_mse": squared_error / elements,
        "future_mae": absolute_error / elements,
        "future_r2": 1.0 - squared_error / max(total_variance, 1e-12),
        "windows": len(loader.dataset),
    }
    if recon_elements:
        probabilities = code_counts.double() / code_counts.sum()
        positive = probabilities > 0
        entropy = -(probabilities[positive] * probabilities[positive].log()).sum().item()
        active = int(positive.sum().item())
        result.update(
            reconstruction_mse=recon_squared_error / recon_elements,
            active_codes=active,
            code_entropy=entropy,
            code_perplexity=float(np.exp(entropy)),
            dead_code_ratio=1.0 - active / len(code_counts),
        )
    return result


def train_one(args: argparse.Namespace, name: str, device: torch.device) -> dict[str, Any]:
    seed_everything(args.seed)
    train_loader = make_loader(
        args.cache_dir, "train", args.batch_size, args.num_workers, True
    )
    val_loader = make_loader(args.cache_dir, "val", args.batch_size, args.num_workers, False)
    test_loader = make_loader(args.cache_dir, "test", args.batch_size, args.num_workers, False)
    model = build_baseline(name, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.05
    )
    model_dir = args.output_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "best.pt"
    best_val = float("inf")
    history_log = []
    start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        batches = 0
        for batch in train_loader:
            history = batch["history"].to(device, non_blocking=True)
            future = batch["future"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                loss, components = training_loss(
                    model, history, future, args.vq_future_weight
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach())
            for key, value in components.items():
                totals[key] = totals.get(key, 0.0) + value
            batches += 1
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "learning_rate": scheduler.get_last_lr()[0],
            "train": {key: value / batches for key, value in totals.items()},
            "val": val_metrics,
        }
        history_log.append(row)
        print(json.dumps({"model": name, **row}), flush=True)
        if val_metrics["future_mse"] < best_val:
            best_val = val_metrics["future_mse"]
            torch.save(
                {
                    "schema": "tactile3d-unit.s1-baseline-checkpoint.v1",
                    "model_name": name,
                    "latent_dim": args.latent_dim,
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                },
                best_path,
            )
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    elapsed = time.monotonic() - start
    result = {
        "model": name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "latent_dim": args.latent_dim,
        "best_epoch": checkpoint["epoch"],
        "training_seconds": elapsed,
        "val": evaluate(model, val_loader, device),
        "test": evaluate(model, test_loader, device),
        "checkpoint": str(best_path),
        "history": history_log,
    }
    write_json(model_dir / "result.json", result)
    return result


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in args.models:
        results[name] = train_one(args, name, device)
    manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    summary = {
        "schema": "tactile3d-unit.s1.2-baselines.v1",
        "status": "PASS",
        "seed": args.seed,
        "device": str(device),
        "dataset_revision": manifest["dataset_revision"],
        "split_manifest_sha256": manifest["split_manifest_sha256"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "vq_future_weight": args.vq_future_weight,
        "results": results,
    }
    write_json(args.output_dir / "s1_2_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
