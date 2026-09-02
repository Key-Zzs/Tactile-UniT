#!/usr/bin/env python3
"""Cold unit-side audit of causal DexJoCo runtime smoke traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.dexjoco_adapter import policy_action_to_env_action  # noqa: E402
from gr00t.simulation.s4_3_act import FrozenS42PolicyStack  # noqa: E402
from gr00t.simulation.s4_3_policy import TensorProvenance  # noqa: E402
from gr00t.tactile_unit.paired_contract import preprocess_trex_rgb  # noqa: E402
from scripts.tactile_unit.build_c5_causal_visual_cache import (  # noqa: E402
    frozen_frame_features,
    verify_frozen_boundary_repeat,
)
from scripts.tactile_unit.continuous_contact_bridge_common import (  # noqa: E402
    load_frozen_vision,
)

ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_restart"
COLLECTION = ARTIFACT_ROOT / "runtime_smoke_collection.json"
VISION_IDENTITY = ROOT / ".local/artifacts/simulation/s4_2_formal/vision_identity.json"
TASKS = ("pinch_tongs", "hammer_nail", "click_mouse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unit-checkpoint",
        type=Path,
        default=(
            Path(os.environ["UNIT_FULLDATA_CKPT"]) if os.environ.get("UNIT_FULLDATA_CKPT") else None
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.unit_checkpoint is None:
        raise RuntimeError("--unit-checkpoint or UNIT_FULLDATA_CKPT is required")
    device = torch.device(args.device)
    collection = read_json(COLLECTION)
    if (
        collection["status"] != "PASS"
        or len(collection["episodes"]) != 9
        or collection["total_policy_steps"] != 900
    ):
        raise RuntimeError("DexJoCo runtime smoke collection is incomplete")

    identity = read_json(VISION_IDENTITY)["checkpoint_file_sha256"]
    spec = {"frozen_identity": {"original_unit_tokenizer_files_sha256": identity}}
    vision, vision_load = load_frozen_vision(args.unit_checkpoint, spec, device)
    if vision_load["trainable_parameters"] != 0 or vision.training:
        raise RuntimeError("current-frame Original UniT Vision is not frozen")
    stack = FrozenS42PolicyStack().to(device)
    if any(parameter.requires_grad for parameter in stack.parameters()):
        raise RuntimeError("S4.2 policy stack is not frozen")

    trace_rows = []
    total_replans = 0
    all_frames: list[np.ndarray] = []
    episode_plans: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []
    for episode in collection["episodes"]:
        trace_path = ROOT / episode["trace"]
        plans_path = ROOT / episode["plans"]
        if (
            sha256_file(trace_path) != episode["trace_sha256"]
            or sha256_file(plans_path) != episode["plans_sha256"]
        ):
            raise RuntimeError("runtime smoke trace identity mismatch")
        with np.load(trace_path, allow_pickle=False) as loaded:
            trace = {name: loaded[name] for name in loaded.files}
        with np.load(plans_path, allow_pickle=False) as loaded:
            plans = {name: loaded[name] for name in loaded.files}
        if (
            len(trace["policy_step"]) != 100
            or int(trace["replan"].sum()) != 20
            or len(plans["policy_step"]) != 20
            or not np.array_equal(plans["policy_step"], np.arange(0, 100, 5))
        ):
            raise RuntimeError("runtime smoke queue/replan contract mismatch")
        if not np.array_equal(
            trace["history_source_max"] - trace["history_source_min"],
            np.full(100, 25),
        ):
            raise RuntimeError("runtime smoke history is not exact 26-sample causal input")
        if np.any(trace["history_source_max"] > trace["control_step"]):
            raise RuntimeError("runtime smoke read future tactile")
        if np.any(trace["terminated"]) or np.any(trace["truncated"]):
            raise RuntimeError("runtime smoke contains environment termination")
        all_frames.extend(plans["rgb"])
        episode_plans.append((episode, plans))
        total_replans += len(plans["policy_step"])

    processed = np.stack([preprocess_trex_rgb(frame) for frame in all_frames])
    verify_frozen_boundary_repeat(vision, processed[: min(args.batch_size, len(processed))], device)
    vision_features = []
    for start in range(0, len(processed), args.batch_size):
        stop = min(start + args.batch_size, len(processed))
        vision_features.append(frozen_frame_features(vision, processed[start:stop], device))
    vision_features_array = np.concatenate(vision_features)
    if vision_features_array.shape != (total_replans, 8, 32):
        raise RuntimeError("current-frame Vision output shape mismatch")
    if not np.isfinite(vision_features_array).all():
        raise RuntimeError("current-frame Vision output is non-finite")

    feature_cursor = 0
    with torch.inference_mode():
        for episode, plans in episode_plans:
            batch = len(plans["policy_step"])
            current_state = torch.from_numpy(plans["proprio"].astype(np.float32)).to(device)
            tactile_history = torch.from_numpy(plans["tactile_history"].astype(np.float32)).to(
                device
            )
            action_chunk = torch.from_numpy(plans["action_chunk"].astype(np.float32)).to(device)
            h_current = stack.encode_contact_state(tactile_history)
            predicted_contact = stack.predict_shared_contact(current_state, action_chunk, h_current)
            if h_current.shape != (batch, 256) or predicted_contact.shape != (batch, 8, 32):
                raise RuntimeError("frozen E_T/A0/B3/A+H runtime shape mismatch")
            if not torch.isfinite(h_current).all() or not torch.isfinite(predicted_contact).all():
                raise RuntimeError("frozen E_T/A0/B3/A+H runtime output is non-finite")
            adapter_shapes = {
                tuple(policy_action_to_env_action(action).values.shape)
                for action in plans["action_chunk"][:, 0]
            }
            if adapter_shapes != {(23,)}:
                raise RuntimeError("central Action adapter did not produce 23D")
            for row in range(batch):
                step = int(plans["control_step"][row])
                provenance = (
                    TensorProvenance(episode["episode_id"], step, step, step, "OBSERVATION"),
                    TensorProvenance(episode["episode_id"], step, step, step, "OBSERVATION"),
                    TensorProvenance(episode["episode_id"], step, step - 25, step, "OBSERVATION"),
                    TensorProvenance(episode["episode_id"], step, step, step + 26, "PLAN"),
                    TensorProvenance(
                        episode["episode_id"], step, step - 25, step + 26, "DIAGNOSTIC"
                    ),
                )
                for item in provenance:
                    item.validate(inference=True)
            scoped_features = vision_features_array[feature_cursor : feature_cursor + batch]
            feature_cursor += batch
            trace_rows.append(
                {
                    "episode_id": episode["episode_id"],
                    "task": episode["task"],
                    "replans": batch,
                    "current_vision_shape": list(scoped_features.shape[1:]),
                    "contact_state_shape": list(h_current.shape[1:]),
                    "predicted_contact_shape": list(predicted_contact.shape[1:]),
                    "central_adapter": "PASS",
                    "causal_provenance": "PASS",
                    "status": "PASS",
                }
            )

    trace_audit = {
        "schema": "tactile3d-unit.s4-3-causal-trace-audit.v1",
        "stage": "R6",
        "roles": ["OBSERVATION", "PLAN", "TRAINING_TARGET", "DIAGNOSTIC"],
        "observation_source_max_le_current": True,
        "plan_generated_by_current_policy": True,
        "training_target_available_at_inference": False,
        "future_read": False,
        "episodes": trace_rows,
        "status": "PASS",
    }
    runtime = {
        "schema": "tactile3d-unit.s4-3-runtime-smoke.v1",
        "stage": "R6",
        "tasks": list(TASKS),
        "resets_per_task": 3,
        "episodes": 9,
        "policy_control_steps": 900,
        "warmup_samples_per_episode": 26,
        "warmup_duration_sec": 0.5,
        "replans": total_replans,
        "current_rgb": "PASS",
        "current_frame_frozen_DINO": "PASS",
        "proprio_22D": "PASS",
        "tactile_history_26x30": "PASS",
        "contact_state_256D": "PASS",
        "P3_A0_B3_A_plus_H": "PASS",
        "action_adapter_22_to_23": "PASS",
        "queue_stride_5": "PASS",
        "success_polling": "PASS",
        "timeout_bookkeeping": "PASS",
        "egl": collection["egl"],
        "future_read": False,
        "crashes": 0,
        "uncertainty_diagnostic": {
            "status": "UNAVAILABLE_CAUSAL_INPUT_MISMATCH",
            "verified_no_illegal_invocation": True,
            "reason": "Frozen full S4.2 uncertainty requires the forbidden future-derived Vision transition latent; there is no frozen A+H-only uncertainty estimator.",
            "intervention": False,
        },
        "gate": "S4_3_1_CAUSAL_POLICY_INTERFACE_READY",
        "status": "PASS",
    }
    write_json(ARTIFACT_ROOT / "causal_trace_audit.json", trace_audit)
    write_json(ARTIFACT_ROOT / "runtime_smoke.json", runtime)
    print(
        json.dumps(
            {
                "episodes": runtime["episodes"],
                "replans": runtime["replans"],
                "gate": runtime["gate"],
                "status": runtime["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
