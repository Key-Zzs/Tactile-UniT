#!/usr/bin/env python3
"""Train and validation-select the private Contact RQ information ceiling."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from gr00t.tactile_unit.s3_2_r import build_contact_rq
from s3_2_r_common import (
    DEFAULT_EXPERIMENTS,
    DEFAULT_SPEC,
    checkpoint_sha256,
    frozen_guard,
    load_runtime,
    set_seed,
    verify_frozen,
    verify_gpu,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXPERIMENTS / "r0")
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--codes", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.003, 0.01])
    return parser.parse_args()


@torch.inference_mode()
def validation_metrics(
    rq: torch.nn.Module,
    codes: np.ndarray,
    arrays: dict[str, np.ndarray],
    decoder: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    squared = 0.0
    squared_dynamic = 0.0
    count = 0
    dynamic_count = 0
    distortion = 0.0
    rq.eval()
    for start in range(0, len(codes), batch_size):
        stop = min(start + batch_size, len(codes))
        z = torch.from_numpy(np.array(codes[start:stop], copy=True)).to(device)
        current = torch.from_numpy(np.array(arrays["current"][start:stop], copy=True)).to(device)
        future = torch.from_numpy(np.array(arrays["future"][start:stop], copy=True)).to(device)
        dynamic = torch.from_numpy(np.array(arrays["dynamic"][start:stop], copy=True)).to(device).bool()
        q, _, _ = rq(z)
        prediction = decoder(q, current)
        per_sample = (prediction - future).square().mean(dim=1)
        squared += float(per_sample.sum().item())
        count += len(z)
        if dynamic.any():
            squared_dynamic += float(per_sample[dynamic].sum().item())
            dynamic_count += int(dynamic.sum().item())
        distortion += float((q - z).square().sum().item())
    return {
        "future_mse": squared / count,
        "dynamic_future_mse": squared_dynamic / dynamic_count,
        "quantization_mse": distortion / (count * 8 * 32),
    }


def train_candidate(
    *,
    runtime: dict[str, Any],
    stages: int,
    codes_per_stage: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    set_seed(seed)
    rq = build_contact_rq(stages=stages, codes=codes_per_stage).to(device)
    optimizer = torch.optim.AdamW(rq.parameters(), lr=learning_rate, weight_decay=0.0)
    train_codes = runtime["codes"]["train"]
    train_arrays = runtime["arrays"]["train"]
    val_codes = runtime["codes"]["validation"]
    val_arrays = runtime["arrays"]["validation"]
    decoder = runtime["s2"].decoder.eval().requires_grad_(False)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    history = []
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        rq.train()
        order = torch.randperm(len(train_codes), generator=generator).numpy()
        sums = {"loss": 0.0, "vq": 0.0, "future": 0.0, "delta": 0.0, "examples": 0}
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            z = torch.from_numpy(np.array(train_codes[selected], copy=True)).to(device)
            current = torch.from_numpy(np.array(train_arrays["current"][selected], copy=True)).to(device)
            future = torch.from_numpy(np.array(train_arrays["future"][selected], copy=True)).to(device)
            optimizer.zero_grad(set_to_none=True)
            q, _, vq_loss = rq(z)
            prediction = decoder(q, current)
            future_loss = F.mse_loss(prediction, future)
            delta_loss = F.mse_loss(prediction - current, future - current)
            # The repository RQ's straight-through route intentionally sends decoder
            # gradients toward an encoder input. With frozen native z_c and no adaptor,
            # private codebooks legitimately learn through their native VQ objective;
            # reconstruction/delta remain objective monitors and selection signals.
            loss = vq_loss + future_loss + delta_loss
            loss.backward()
            optimizer.step()
            n = len(z)
            sums["loss"] += float(loss.item()) * n
            sums["vq"] += float(vq_loss.item()) * n
            sums["future"] += float(future_loss.item()) * n
            sums["delta"] += float(delta_loss.item()) * n
            sums["examples"] += n
        validation = validation_metrics(rq, val_codes, val_arrays, decoder, device, batch_size)
        score = validation["dynamic_future_mse"]
        history.append(
            {
                "epoch": epoch,
                "train": {key: value / sums["examples"] for key, value in sums.items() if key != "examples"},
                "validation": validation,
                "selection_score": score,
                "internal_steps": [int(layer.internal_step.item()) for layer in rq.layers],
            }
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(rq.state_dict())
    if best_state is None:
        raise RuntimeError("R0 training did not produce a checkpoint")
    return best_state, {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_validation_dynamic_future_mse": best_score,
        "history": history,
        "runtime_seconds": time.monotonic() - started,
    }


def main() -> int:
    args = parse_args()
    device = verify_gpu()
    runtime = load_runtime(spec_path=args.spec, device=device)
    guard = frozen_guard(runtime)
    seed = int(runtime["spec"]["seed"])
    candidates = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, learning_rate in enumerate(args.learning_rates):
        state, summary = train_candidate(
            runtime=runtime,
            stages=args.stages,
            codes_per_stage=args.codes,
            learning_rate=learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            seed=seed + index,
        )
        checkpoint = args.output_dir / f"rq_s{args.stages}_c{args.codes}_lr{learning_rate:g}.pt"
        torch.save(
            {
                "schema": "tactile3d-unit.s3-2-r-private-contact-rq.v1",
                "stages": args.stages,
                "codes_per_stage": args.codes,
                "embedding_dim": 32,
                "state_dict": state,
                "training": summary,
                "selection_partition": "validation",
                "test_used_for_selection": False,
                "frozen_identity": runtime["identity"],
            },
            checkpoint,
        )
        candidates.append({**summary, "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha256(checkpoint)})
    selected = min(candidates, key=lambda row: row["best_validation_dynamic_future_mse"])
    integrity = verify_frozen(guard, runtime)
    output = {
        "schema": "tactile3d-unit.s3-2-r-r0-training.v1",
        "status": "PASS",
        "architecture": {"queries": 8, "embedding_dim": 32, "stages": args.stages, "codes_per_stage": args.codes},
        "parameter_count": args.stages * args.codes * 32,
        "training_behavior": {
            "implementation": "repository ResidualVectorQuantizer",
            "codebook_update": "gradient VQ objective with repository dead-code restart",
            "decoder_reconstruction_role": "monitored in objective and used for validation selection; repository STE sends its gradient to the frozen native input",
        },
        "candidates": candidates,
        "selected": selected,
        "test_used_for_selection": False,
        "frozen_identity": runtime["identity"],
        "frozen_integrity": integrity,
        "environment": {
            "torch": torch.__version__,
            "physical_gpu": 3,
            "logical_device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
    }
    write_json(args.output_dir / "training_summary.json", output)
    print(json.dumps({"status": "PASS", "selected": selected["checkpoint"], "validation_dynamic_future_mse": selected["best_validation_dynamic_future_mse"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
