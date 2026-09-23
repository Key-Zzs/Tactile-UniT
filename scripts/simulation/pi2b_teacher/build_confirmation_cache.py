#!/usr/bin/env python3
"""Build frozen native V/A/C features for the independent confirmation cohort."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from gr00t.contact_dynamics.models import (  # noqa: E402
    ContactDynamicsEncoder,
    ContactDynamicsModel,
    LatentTransitionDecoder,
)
from gr00t.simulation.s4_2_dataset import build_pair_arrays  # noqa: E402
from gr00t.simulation.s4_2_formal import FormalActionEncoder  # noqa: E402
from gr00t.simulation.s4_2_normalization import TactileNormalization  # noqa: E402
from gr00t.simulation.sim_contact_models import load_teacher_checkpoint  # noqa: E402
from scripts.simulation.build_s4_2ds_paired_pilot import encode_batch as encode_vision_batch  # noqa: E402
from scripts.simulation.build_s4_2r_contact_dynamics_cache import encode as encode_contact_state  # noqa: E402
from scripts.simulation.pi2b_teacher.common import atomic_json, load_json, sha256_file  # noqa: E402
from scripts.simulation.run_s4_2_4_formal_representations import infer as infer_action  # noqa: E402
from scripts.simulation.train_s4_2ds_action_pilot import features as action_features  # noqa: E402
from scripts.simulation.train_s4_2ds_action_pilot import policy_state  # noqa: E402
from scripts.tactile_unit.continuous_contact_bridge_common import load_frozen_vision  # noqa: E402


def load_action(path: Path, device: torch.device) -> tuple[FormalActionEncoder, dict[str, np.ndarray]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = FormalActionEncoder(str(payload["candidate"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    stats = {name: np.asarray(value, dtype=np.float32) for name, value in payload["stats"].items()}
    return model.eval().requires_grad_(False).to(device), stats


def load_dynamics(path: Path, device: torch.device) -> ContactDynamicsModel:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = ContactDynamicsModel(ContactDynamicsEncoder(), LatentTransitionDecoder())
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval().requires_grad_(False).to(device)


@torch.inference_mode()
def encode_dynamics(
    model: ContactDynamicsModel, current: np.ndarray, future: np.ndarray, device: torch.device
) -> np.ndarray:
    result = np.empty((len(current), 8, 32), dtype=np.float32)
    for start in range(0, len(current), 2048):
        stop = min(start + 2048, len(current))
        output = model(
            torch.from_numpy(current[start:stop]).to(device),
            torch.from_numpy(future[start:stop]).to(device),
        )
        result[start:stop] = output["code"].float().cpu().numpy()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--unit-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vision-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    runtime = load_json(args.runtime_manifest)
    write_root = Path(runtime["write_root"])
    pretest = load_json(write_root / "artifacts/pretest_freeze.json")
    if pretest["status"] != "PASS" or pretest["fresh_performance_loaded"]:
        raise RuntimeError("valid pretest freeze is required")
    for relative, expected in pretest["implementation_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"evaluation snapshot changed: {relative}")
    history = Path(runtime["worktree_root"])
    checkpoints = {
        "contact_state": history / ".local/experiments/simulation/s4_2r/contact_state/accepted.pt",
        "contact_C3": history / ".local/experiments/simulation/s4_2r/contact_dynamics/selected.pt",
        "action_A0": history / ".local/experiments/simulation/s4_2_formal/action/selected.pt",
    }
    for name, path in checkpoints.items():
        if sha256_file(path) != pretest["native_checkpoint_sha256"][name]:
            raise RuntimeError(f"frozen native checkpoint changed: {name}")
    device = torch.device(args.device)
    if device.type == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("cache extraction requires exactly one supervisor-assigned GPU")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    dataset_root = write_root / "datasets/fresh_confirmation"
    pairs = build_pair_arrays(
        "test",
        dataset_root,
        purpose="locked_test",
        pretest_freeze=write_root / "artifacts/pretest_freeze.json",
    )
    if len(pairs["pair_id"]) != 4860 or not np.all(pairs["future_step"] - pairs["anchor_step"] == 27):
        raise RuntimeError("fresh pair contract mismatch")
    status_path = write_root / "status/confirmation_cache.json"
    start_time = time.monotonic()
    teacher, payload = load_teacher_checkpoint(checkpoints["contact_state"], "cpu")
    teacher = teacher.eval().requires_grad_(False).to(device)
    normalization = TactileNormalization.from_json(payload["normalization"])
    h_current = encode_contact_state(teacher, pairs["current_history"], normalization, device, 1024)
    h_future = encode_contact_state(teacher, pairs["future_history"], normalization, device, 1024)
    dynamics = load_dynamics(checkpoints["contact_C3"], device)
    z_c = encode_dynamics(dynamics, h_current, h_future, device)
    z_c_reversed = encode_dynamics(dynamics, h_future, h_current, device)
    action_model, stats = load_action(checkpoints["action_A0"], device)
    state = policy_state(pairs["current_state"])
    action = np.asarray(pairs["action_chunk"], dtype=np.float32)
    feature, normalized_state, _ = action_features(state, action, stats)
    z_a, _ = infer_action(action_model, feature, normalized_state, device, 1024)
    reversed_feature, reversed_state, _ = action_features(state, action[:, ::-1].copy(), stats)
    z_a_reversed, _ = infer_action(action_model, reversed_feature, reversed_state, device, 1024)
    vision_spec = {"frozen_identity": {"original_unit_tokenizer_files_sha256": pretest["vision_checkpoint_file_sha256"]}}
    vision, identity = load_frozen_vision(args.unit_checkpoint, vision_spec, device)
    if identity["trainable_parameters"] != 0 or vision.training:
        raise RuntimeError("Original UniT Vision is not frozen")
    z_v = np.empty((len(pairs["pair_id"]), 8, 32), dtype=np.float32)
    z_v_current = np.empty_like(z_v)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for start in range(0, len(z_v), args.vision_batch_size):
            stop = min(start + args.vision_batch_size, len(z_v))
            z_v[start:stop] = encode_vision_batch(
                vision,
                pairs["current_frame"][start:stop],
                pairs["future_frame"][start:stop],
                executor,
            )
            z_v_current[start:stop] = encode_vision_batch(
                vision,
                pairs["current_frame"][start:stop],
                pairs["current_frame"][start:stop],
                executor,
            )
            if stop % 256 < args.vision_batch_size or stop == len(z_v):
                elapsed = time.monotonic() - start_time
                atomic_json(status_path, {
                    "state": "RUNNING",
                    "pid": __import__("os").getpid(),
                    "vision_pairs": stop,
                    "total": len(z_v),
                    "seconds_per_pair": elapsed / stop,
                    "eta_seconds": (len(z_v) - stop) * elapsed / stop,
                })
                print(json.dumps({"vision_pairs": stop, "total": len(z_v)}), flush=True)
    tactile_current = pairs["current_history"][:, -1].reshape(-1, 5, 6)
    tactile_future = pairs["future_history"][:, -1].reshape(-1, 5, 6)
    current_region = tactile_current[:, :, 0] > 0
    future_region = tactile_future[:, :, 0] > 0
    current_contact = current_region.any(axis=1)
    future_contact = future_region.any(axis=1)
    contact_transition = current_contact.astype(np.int8) * 2 + future_contact.astype(np.int8)
    current_force = tactile_current[:, :, 1].sum(axis=1).astype(np.float32)
    future_force = tactile_future[:, :, 1].sum(axis=1).astype(np.float32)
    force_delta = future_force - current_force
    cache_manifest = load_json(history / ".local/refs/base/artifacts/simulation/s4_2r/contact_dynamics_cache_manifest.json")
    dynamic = np.abs(force_delta) > float(cache_manifest["dynamic_q70_threshold_newton"])
    force_trend = np.ones(len(force_delta), dtype=np.int8)
    deadband = float(cache_manifest["force_trend_deadband_newton"])
    force_trend[force_delta < -deadband] = 0
    force_trend[force_delta > deadband] = 2
    destination = write_root / "cache/fresh_confirmation_native.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        pair_id=pairs["pair_id"],
        episode_id=pairs["episode_id"],
        task=pairs["task"],
        source_trajectory_id=pairs["source_trajectory_id"],
        anchor_step=pairs["anchor_step"],
        future_step=pairs["future_step"],
        current_state=pairs["current_state"],
        action_chunk=pairs["action_chunk"],
        z_v=z_v,
        z_v_current=z_v_current,
        z_a=z_a,
        z_a_reversed=z_a_reversed,
        z_c=z_c,
        z_c_reversed=z_c_reversed,
        h_current=h_current,
        h_future=h_future,
        contact_transition=contact_transition,
        force_trend=force_trend,
        force_delta=force_delta.astype(np.float32),
        region_transition=(current_region != future_region).astype(np.int8),
        dynamic=dynamic,
        contact_boundary=np.isin(contact_transition, [1, 2]),
        current_total_force=current_force,
        future_total_force=future_force,
    )
    manifest = {
        "schema": "tactile3d-unit.s4-3-pi2b-teacher-confirmation-native-cache.v1",
        "status": "COMPLETE_NOT_EVALUATED",
        "path": str(destination),
        "sha256": sha256_file(destination),
        "rows": len(pairs["pair_id"]),
        "groups": len(set(zip(pairs["task"].tolist(), pairs["source_trajectory_id"].tolist()))),
        "fields": ["z_v", "z_v_current", "z_a", "z_a_reversed", "z_c", "z_c_reversed"],
        "native_checkpoint_sha256": pretest["native_checkpoint_sha256"],
        "vision_checkpoint_file_sha256": identity["original_unit_tokenizer_files_sha256"],
        "teacher_performance_loaded": False,
    }
    atomic_json(write_root / "artifacts/confirmation_cache_manifest.json", manifest)
    atomic_json(status_path, {"state": "COMPLETE", "pid": __import__("os").getpid(), "rows": len(pairs["pair_id"]), "cache": str(destination), "cache_sha256": manifest["sha256"]})
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
