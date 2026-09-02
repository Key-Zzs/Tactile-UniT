#!/usr/bin/env python3
"""Cold-reload and evaluate all 36 selected ACT checkpoints on POLICY_DEV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_3_act import CausalACTPolicy, FrozenS42PolicyStack  # noqa: E402
from gr00t.simulation.s4_3_training import (  # noqa: E402
    PolicyCacheDataset,
    atomic_json,
    load_normalization,
    read_json,
    set_deterministic,
    sha256_file,
)
from scripts.simulation.train_s4_3_act import model_kwargs, to_device  # noqa: E402

CACHE_ROOT = ROOT / ".local/cache/simulation/s4_3_restart"
TRAINING_JOBS = ROOT / ".local/artifacts/simulation/s4_3_restart/training_jobs.json"
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_restart"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def cold_load(row: dict[str, Any], device: torch.device) -> CausalACTPolicy:
    checkpoint_path = ROOT / row["checkpoint"]
    if sha256_file(checkpoint_path) != row["checkpoint_sha256"]:
        raise RuntimeError("selected ACT checkpoint SHA256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    identity = (checkpoint["task"], checkpoint["variant"], checkpoint["training_seed"])
    expected = (row["task"], row["variant"], row["training_seed"])
    if identity != expected or checkpoint["training_step"] != row["best_dev_step"]:
        raise RuntimeError("selected ACT checkpoint metadata mismatch")
    model = CausalACTPolicy(row["variant"], load_normalization(CACHE_ROOT, row["task"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if model.trainable_parameter_count != checkpoint["trainable_parameter_count"]:
        raise RuntimeError("cold-loaded ACT parameter count mismatch")
    return model.to(device).eval()


@torch.inference_mode()
def evaluate(
    row: dict[str, Any],
    device: torch.device,
    batch_size: int,
    stack: FrozenS42PolicyStack,
) -> dict[str, Any]:
    model = cold_load(row, device)
    dataset = PolicyCacheDataset(CACHE_ROOT, row["task"], "dev", row["variant"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    sums = {
        "normalized_full_chunk_action_l1": 0.0,
        "full_chunk_mse": 0.0,
        "first_step_action_mae": 0.0,
        "chunk_endpoint_action_mae": 0.0,
        "tcp_xyz_mae": 0.0,
        "rotvec_mae": 0.0,
        "hand_joint_mae": 0.0,
        "action_velocity_mae": 0.0,
        "action_second_difference_mae": 0.0,
        "p3_contact_auxiliary_mse": 0.0,
    }
    counts = {name: 0 for name in sums}
    finite_rows = 0
    rows = 0
    shape_pass = True
    bound_pass = True
    auxiliary_shape_pass = True
    auxiliary_finite_pass = True
    repeat_max_absolute_difference = 0.0
    first_batch = True
    for raw_batch in loader:
        batch = to_device(raw_batch, device)
        kwargs = model_kwargs(batch, row["variant"])
        output = model(batch["vision"], batch["proprio"], **kwargs)
        prediction = output["physical_action"]
        target = batch["action"]
        normalized_target = model.normalize_action(target).clamp(-1.0, 1.0)
        normalized_delta = output["normalized_action"] - normalized_target
        physical_delta = prediction - target
        if first_batch:
            repeat = model(batch["vision"], batch["proprio"], **kwargs)["physical_action"]
            repeat_max_absolute_difference = float((prediction - repeat).abs().max())
            first_batch = False
        shape_pass = shape_pass and tuple(prediction.shape[1:]) == (27, 22)
        finite_rows += int(torch.isfinite(prediction).all(dim=(1, 2)).sum())
        rows += len(prediction)
        bound_pass = bound_pass and bool(
            torch.all(prediction >= model.action_min[None, None] - 1e-6)
            and torch.all(prediction <= model.action_max[None, None] + 1e-6)
        )

        def add(name: str, value: torch.Tensor) -> None:
            sums[name] += float(value.abs().sum())
            counts[name] += value.numel()

        add("normalized_full_chunk_action_l1", normalized_delta)
        sums["full_chunk_mse"] += float(physical_delta.square().sum())
        counts["full_chunk_mse"] += physical_delta.numel()
        add("first_step_action_mae", physical_delta[:, 0])
        add("chunk_endpoint_action_mae", physical_delta[:, -1])
        add("tcp_xyz_mae", physical_delta[:, :, :3])
        add("rotvec_mae", physical_delta[:, :, 3:6])
        add("hand_joint_mae", physical_delta[:, :, 6:22])
        prediction_velocity = prediction[:, 1:] - prediction[:, :-1]
        target_velocity = target[:, 1:] - target[:, :-1]
        add("action_velocity_mae", prediction_velocity - target_velocity)
        prediction_acceleration = prediction_velocity[:, 1:] - prediction_velocity[:, :-1]
        target_acceleration = target_velocity[:, 1:] - target_velocity[:, :-1]
        add(
            "action_second_difference_mae",
            prediction_acceleration - target_acceleration,
        )
        if row["variant"] == "P3":
            contact_prediction = stack.predict_shared_contact(
                batch["proprio"], prediction, batch["contact_state"]
            )
            auxiliary_shape_pass = auxiliary_shape_pass and tuple(contact_prediction.shape[1:]) == (
                8,
                32,
            )
            auxiliary_finite_pass = auxiliary_finite_pass and bool(
                torch.isfinite(contact_prediction).all()
            )
            contact_delta = contact_prediction - batch["contact_target"]
            sums["p3_contact_auxiliary_mse"] += float(contact_delta.square().sum())
            counts["p3_contact_auxiliary_mse"] += contact_delta.numel()

    metrics = {name: (sums[name] / counts[name] if counts[name] else None) for name in sums}
    metrics["prediction_finite_rate"] = finite_rows / rows
    sanity = {
        "selected_checkpoint_loaded_cold": True,
        "prediction_shape_27x22": shape_pass,
        "all_predictions_finite": finite_rows == rows,
        "p3_contact_auxiliary_shape_8x32": auxiliary_shape_pass,
        "p3_contact_auxiliary_finite": auxiliary_finite_pass,
        "action_within_train_bounds": bound_pass,
        "deterministic_repeat_max_absolute_difference": repeat_max_absolute_difference,
        "deterministic_within_tolerance": repeat_max_absolute_difference <= 1e-7,
        "status": "PASS",
    }
    if not all(
        (
            sanity["prediction_shape_27x22"],
            sanity["all_predictions_finite"],
            sanity["p3_contact_auxiliary_shape_8x32"],
            sanity["p3_contact_auxiliary_finite"],
            sanity["action_within_train_bounds"],
            sanity["deterministic_within_tolerance"],
        )
    ):
        sanity["status"] = "FAIL"
        raise RuntimeError(f"offline hard sanity failed: {row['task']}/{row['variant']}")
    return {**row, "policy_dev_rows": rows, "metrics": metrics, "hard_sanity": sanity}


def main() -> None:
    args = parse_args()
    set_deterministic(4243)
    device = torch.device(args.device)
    jobs = read_json(TRAINING_JOBS)
    if jobs.get("status") != "PASS" or jobs.get("canonical_checkpoints") != 36:
        raise RuntimeError("R9 complete training gate has not passed")
    stack = FrozenS42PolicyStack().to(device)
    results = []
    for row in sorted(
        jobs["jobs"], key=lambda value: (value["task"], value["variant"], value["training_seed"])
    ):
        result = evaluate(row, device, args.batch_size, stack)
        results.append(result)
        print(
            json.dumps(
                {
                    "task": row["task"],
                    "variant": row["variant"],
                    "seed": row["training_seed"],
                    "dev_l1": result["metrics"]["normalized_full_chunk_action_l1"],
                    "status": "PASS",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    artifact = {
        "schema": "tactile3d-unit.s4-3-policy-dev-evaluation.v1",
        "stage": "R10",
        "checkpoint_selection_changed": False,
        "jobs": results,
        "hard_sanity": {
            "checkpoints_loaded": len(results),
            "all_predictions_finite": all(
                row["hard_sanity"]["all_predictions_finite"] for row in results
            ),
            "all_shapes_27x22": all(
                row["hard_sanity"]["prediction_shape_27x22"] for row in results
            ),
            "all_action_bounds_pass": all(
                row["hard_sanity"]["action_within_train_bounds"] for row in results
            ),
            "all_p3_contact_auxiliary_shapes_8x32": all(
                row["hard_sanity"]["p3_contact_auxiliary_shape_8x32"] for row in results
            ),
            "all_p3_contact_auxiliary_predictions_finite": all(
                row["hard_sanity"]["p3_contact_auxiliary_finite"] for row in results
            ),
            "all_deterministic": all(
                row["hard_sanity"]["deterministic_within_tolerance"] for row in results
            ),
            "status": "PASS",
        },
        "status": "PASS",
    }
    atomic_json(ARTIFACT_ROOT / "offline_dev_evaluation.json", artifact)
    print(json.dumps({"checkpoint_evaluations": len(results), "status": "PASS"}, sort_keys=True))


if __name__ == "__main__":
    main()
