#!/usr/bin/env python3
"""Train one canonical task/variant/seed S4.3 ACT job."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_3_act import CausalACTPolicy, FrozenS42PolicyStack  # noqa: E402
from gr00t.simulation.s4_3_training import (  # noqa: E402
    TASKS,
    TRAINING_SEEDS,
    VARIANTS,
    PolicyCacheDataset,
    atomic_json,
    infinite_batches,
    load_normalization,
    set_deterministic,
    sha256_file,
)

CACHE_ROOT = ROOT / ".local/cache/simulation/s4_3_restart"
EXPERIMENT_ROOT = ROOT / ".local/experiments/simulation/s4_3_restart"
LOG_ROOT = ROOT / ".local/logs/simulation/s4_3_restart/training"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--seed", required=True, type=int, choices=TRAINING_SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--dev-interval", type=int, default=1_000)
    parser.add_argument("--minimum-steps", type=int, default=5_000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_kwargs(batch: dict[str, torch.Tensor], variant: str) -> dict[str, torch.Tensor]:
    kwargs: dict[str, torch.Tensor] = {}
    if variant == "P1":
        kwargs["tactile_history"] = batch["tactile_history"]
    if variant in {"P2", "P3"}:
        kwargs["contact_state"] = batch["contact_state"]
    return kwargs


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


@torch.inference_mode()
def evaluate_dev(
    model: CausalACTPolicy,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    stack: FrozenS42PolicyStack | None,
) -> dict[str, float]:
    model.eval()
    absolute = 0.0
    squared = 0.0
    contact_squared = 0.0
    elements = 0
    contact_elements = 0
    finite = 0
    predictions = 0
    for raw_batch in loader:
        batch = to_device(raw_batch, device)
        output = model(batch["vision"], batch["proprio"], **model_kwargs(batch, model.variant))
        target = model.normalize_action(batch["action"]).clamp(-1.0, 1.0)
        delta = output["normalized_action"] - target
        absolute += float(delta.abs().sum())
        squared += float(delta.square().sum())
        elements += delta.numel()
        finite += int(torch.isfinite(output["physical_action"]).all(dim=(1, 2)).sum())
        predictions += len(delta)
        if model.variant == "P3":
            assert stack is not None
            prediction = stack.predict_shared_contact(
                batch["proprio"], output["physical_action"], batch["contact_state"]
            )
            contact_delta = prediction - batch["contact_target"]
            contact_squared += float(contact_delta.square().sum())
            contact_elements += contact_delta.numel()
    model.train()
    return {
        "normalized_full_chunk_action_l1": absolute / elements,
        "full_chunk_mse": squared / elements,
        "p3_contact_auxiliary_mse": (
            contact_squared / contact_elements if contact_elements else float("nan")
        ),
        "prediction_finite_rate": finite / predictions,
    }


def save_checkpoint(
    path: Path,
    model: CausalACTPolicy,
    args: argparse.Namespace,
    step: int,
    metrics: dict[str, float],
) -> None:
    payload = {
        "schema": "tactile3d-unit.s4-3-act-checkpoint.v1",
        "task": args.task,
        "variant": args.variant,
        "training_seed": args.seed,
        "training_step": step,
        "selection_metric": "POLICY_DEV normalized full-chunk Action L1",
        "dev_metrics": metrics,
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "trainable_parameter_count": model.trainable_parameter_count,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if (
        args.max_steps != 20_000
        or args.dev_interval != 1_000
        or args.minimum_steps != 5_000
        or args.patience != 5
        or args.batch_size != 128
    ):
        raise RuntimeError("canonical S4.3 training hyperparameters may not be changed")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("canonical CuBLAS deterministic environment is missing")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical_gpu = os.environ.get("S4_3_PHYSICAL_GPU")
    if not visible or physical_gpu != visible or "," in visible:
        raise RuntimeError("one physical GPU must be isolated and recorded")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("canonical S4.3 training requires exactly one visible CUDA GPU")

    started_at = now()
    generator = set_deterministic(args.seed)
    normalization = load_normalization(CACHE_ROOT, args.task)
    train_dataset = PolicyCacheDataset(CACHE_ROOT, args.task, "train", args.variant)
    dev_dataset = PolicyCacheDataset(CACHE_ROOT, args.task, "dev", args.variant)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
        generator=generator,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    batches = infinite_batches(train_loader)
    model = CausalACTPolicy(args.variant, normalization).to(device).train()
    stack = FrozenS42PolicyStack().to(device) if args.variant == "P3" else None
    if stack is not None and any(parameter.requires_grad for parameter in stack.parameters()):
        raise RuntimeError("P3 S4.2 auxiliary parameters are not frozen")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    run_root = EXPERIMENT_ROOT / args.task / args.variant / f"seed_{args.seed}"
    log_root = LOG_ROOT / args.task / args.variant / f"seed_{args.seed}"
    checkpoint_path = run_root / "best.pt"
    trace_path = log_root / "trace.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    best_step = 0
    stale_evaluations = 0
    last_train: dict[str, float] = {}
    stopped_early = False
    with trace_path.open("w", encoding="utf-8") as trace:
        for step in range(1, args.max_steps + 1):
            batch = to_device(next(batches), device)
            output = model(
                batch["vision"],
                batch["proprio"],
                **model_kwargs(batch, args.variant),
                target_action=batch["action"],
                sample_posterior=True,
            )
            losses = model.act_loss(output, batch["action"])
            total = losses["act_total"]
            contact_loss = torch.zeros((), device=device)
            if args.variant == "P3":
                assert stack is not None
                prediction = stack.predict_shared_contact(
                    batch["proprio"], output["physical_action"], batch["contact_state"]
                )
                contact_loss = F.mse_loss(prediction, batch["contact_target"])
                total = total + 0.1 * contact_loss
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            gradient_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            last_train = {
                "step": step,
                "act_total": float(losses["act_total"].detach()),
                "action_l1": float(losses["action_l1"].detach()),
                "kl": float(losses["kl"].detach()),
                "contact_mse": float(contact_loss.detach()),
                "combined_loss": float(total.detach()),
                "gradient_norm_before_clip": float(gradient_norm),
            }
            if step % args.dev_interval != 0:
                continue
            metrics = evaluate_dev(model, dev_loader, device, stack)
            if metrics["prediction_finite_rate"] != 1.0:
                raise RuntimeError("non-finite POLICY_DEV inference")
            improved = metrics["normalized_full_chunk_action_l1"] < best
            if improved:
                best = metrics["normalized_full_chunk_action_l1"]
                best_step = step
                stale_evaluations = 0
                save_checkpoint(checkpoint_path, model, args, step, metrics)
            else:
                stale_evaluations += 1
            event: dict[str, Any] = {
                **last_train,
                "dev": metrics,
                "best_dev_l1": best,
                "best_step": best_step,
                "improved": improved,
                "stale_evaluations": stale_evaluations,
            }
            trace.write(json.dumps(event, sort_keys=True) + "\n")
            trace.flush()
            print(json.dumps(event, sort_keys=True), flush=True)
            if step >= args.minimum_steps and stale_evaluations >= args.patience:
                stopped_early = True
                break

    steps = int(last_train["step"])
    if not checkpoint_path.is_file() or not best_step:
        raise RuntimeError("training did not produce a selected checkpoint")
    summary = {
        "schema": "tactile3d-unit.s4-3-act-training-job.v1",
        "stage": "R9",
        "task": args.task,
        "variant": args.variant,
        "training_seed": args.seed,
        "physical_gpu": int(physical_gpu),
        "logical_device": args.device,
        "start_time": started_at,
        "end_time": now(),
        "exit_code": 0,
        "training_steps": steps,
        "maximum_steps": args.max_steps,
        "best_dev_step": best_step,
        "best_dev_normalized_full_chunk_action_l1": best,
        "stopped_early": stopped_early,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "trace": str(trace_path.relative_to(ROOT)),
        "trace_sha256": sha256_file(trace_path),
        "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": 1e-4},
        "batch_size": args.batch_size,
        "grad_clip": 1.0,
        "deterministic": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "status": "PASS",
    }
    atomic_json(run_root / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
