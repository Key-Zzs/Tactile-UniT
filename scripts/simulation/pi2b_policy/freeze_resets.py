#!/usr/bin/env python3
"""Reserve and freeze the smallest fresh four-block reset cohort for Track A."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.pi2b_policy.contract import Workspace
from gr00t.simulation.pi2b_policy.coordination import (
    coordination_paths,
    gpu_is_idle,
    gpu_snapshot,
    open_lock,
)
from gr00t.simulation.pi2b_policy.integrity import sha256_file
from gr00t.simulation.pi2b_policy.resets import (
    ordered_sequence_sha256,
    sha256_arrays,
    validate_reset_manifest,
)


ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2b_policy"
MANIFEST = ARTIFACTS / "reset_manifest.json"
AUDIT = ARTIFACTS / "reset_exposure_audit.json"
BLOCKED_SEEDS = set(range(16)) | {700042}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def reserved_seeds(workspace: Workspace) -> set[int]:
    directory = workspace.common_git_dir / "pi2b_coordination/exposure_reservations"
    result: set[int] = set()
    for path in sorted(directory.rglob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for key in ("seed_blocks", "evaluator_seed_blocks"):
            values = payload.get(key, [])
            if isinstance(values, list):
                result.update(int(value) for value in values if isinstance(value, int))
        for block in payload.get("blocks", []) if isinstance(payload.get("blocks"), list) else []:
            if isinstance(block, dict) and isinstance(block.get("seed"), int):
                result.add(block["seed"])
    return result


def historical_reset_identities(workspace: Workspace) -> set[str]:
    identities: set[str] = set()
    root = workspace.history_artifacts / "simulation"
    for path in root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        candidates = []
        for key in ("episode_results", "reset_specs", "ordered_resets"):
            if isinstance(payload.get(key), list):
                candidates.extend(payload[key])
        if isinstance(payload.get("ordered_reset_identities"), list):
            identities.update(
                value for value in payload["ordered_reset_identities"] if isinstance(value, str)
            )
        for row in candidates:
            if isinstance(row, dict) and isinstance(row.get("reset_identity"), str):
                identities.add(row["reset_identity"])
    return identities


def generate_block(workspace: Workspace, seed: int) -> list[dict[str, Any]]:
    import mujoco
    import yaml

    dexjoco_root = workspace.main_root / "third_party/dexjoco/dexjoco"
    sys.path.insert(0, str(dexjoco_root))
    from dexjoco_openpi_client import eval_dexjoco_openpi as official
    from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv

    config_path = workspace.main_root / "third_party/dexjoco/configs/rand_obj/pinch_tongs.yaml"
    config = yaml.safe_load(config_path.read_text())
    official._set_seed(seed)
    environment = DexJoCoOpenPIEnv(
        env_name=config["env_name"],
        camera_mapping=config["camera_mapping"],
        seed=seed,
        rand_full=False,
        randomize_dynamics=False,
        dual_arm=False,
        prompt=config["prompt"],
        render_mode="rgb_array",
        pad_state_dim46=False,
    )
    rows = []
    try:
        environment.start()
        raw = environment.env.unwrapped
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        for block_index in range(50):
            environment.reset()
            mj_state = np.empty(mujoco.mj_stateSize(raw.model, state_spec), dtype=np.float64)
            mujoco.mj_getState(raw.model, raw.data, mj_state, state_spec)
            processed = np.asarray(environment.obs["state"], dtype=np.float64)
            rows.append(
                {
                    "seed": seed,
                    "block_index": block_index,
                    "reset_identity": sha256_arrays(mj_state, processed),
                    "mjstate_sha256": sha256_arrays(mj_state),
                    "processed_state_sha256": sha256_arrays(processed),
                    "mjstate_shape": list(mj_state.shape),
                    "processed_state_shape": list(processed.shape),
                }
            )
    finally:
        environment.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2, 3), required=True)
    args = parser.parse_args()
    workspace = Workspace.load(ROOT)
    if MANIFEST.exists() or AUDIT.exists():
        raise SystemExit("refusing to overwrite the Track A reset freeze")
    paths = coordination_paths(workspace)
    exposure_path = workspace.common_git_dir / "pi2b_coordination/exposure.lock"
    exposure = open_lock(exposure_path, shared=False)
    reservation = None
    try:
        occupied = BLOCKED_SEEDS | reserved_seeds(workspace)
        start = 0
        while any(value in occupied for value in range(start, start + 4)):
            start += 1
        blocks = list(range(start, start + 4))
        if blocks != [16, 17, 18, 19]:
            # A different valid block is allowed only when a real shared reservation caused it.
            if not (set(blocks) - BLOCKED_SEEDS).isdisjoint(reserved_seeds(workspace)):
                raise RuntimeError("fresh-block selection collided with an existing reservation")
        reservation = (
            workspace.common_git_dir
            / "pi2b_coordination/exposure_reservations/policy/s4_3_pi2b_policy_final.json"
        )
        if reservation.exists():
            raise SystemExit("Track A exposure reservation already exists")
        atomic_json(
            reservation,
            {
                "schema": "tactile3d-unit.pi2b-exposure-reservation.v1",
                "track": "policy",
                "cohort": "S4.3_PI2B_POLICY_FINAL",
                "status": "RESERVED_METADATA_PENDING",
                "created_at_utc": now(),
                "seed_blocks": blocks,
                "resets_per_block": 50,
                "performance_seen": False,
            },
        )
    finally:
        exposure.close()

    first = gpu_snapshot()
    time.sleep(2)
    second = gpu_snapshot()
    if not gpu_is_idle(args.gpu, first, second):
        raise SystemExit("selected reset-freeze GPU is not idle in two snapshots")
    barrier = open_lock(paths["runtime_barrier"], shared=True)
    scheduler = open_lock(paths["scheduler"], shared=False)
    gpu_lock = None
    try:
        gpu_lock = open_lock(
            workspace.common_git_dir / f"tactile3d_unit_gpu{args.gpu}.lock", shared=False
        )
        third = gpu_snapshot()
        if not gpu_is_idle(args.gpu, third):
            raise SystemExit("reset-freeze GPU became busy after lock acquisition")
    finally:
        scheduler.close()
    try:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
        rows = []
        for seed in blocks:
            for row in generate_block(workspace, seed):
                rows.append({**row, "global_index": len(rows)})
        identities = [row["reset_identity"] for row in rows]
        historical = historical_reset_identities(workspace)
        config = workspace.main_root / "third_party/dexjoco/configs/rand_obj/pinch_tongs.yaml"
        manifest = {
            "schema": "tactile3d-unit.s4-3-pi2b-policy-reset-manifest.v1",
            "status": "PASS",
            "cohort": "S4.3_PI2B_POLICY_FINAL",
            "created_at_utc": now(),
            "task": "pinch_tongs",
            "regime": "rand_obj",
            "config_sha256": sha256_file(config),
            "blocks": [
                {"seed": seed, "episodes": 50, "fresh_environment_process_contract": True}
                for seed in blocks
            ],
            "episodes": 200,
            "ordered_reset_identities": identities,
            "ordered_reset_sequence_sha256": ordered_sequence_sha256(identities),
            "all_reset_identities_unique": len(set(identities)) == 200,
            "identity_definition": "sha256(dtype+shape+bytes of mjSTATE_INTEGRATION float64, then official processed state float64)",
            "reset_specs": rows,
            "policy_performance_seen": False,
            "scientific_policy_inference_performed": False,
            "shared_across_all_15_checkpoints": True,
            "gpu_snapshots": [first, second, third, gpu_snapshot()],
        }
        overlap = set(identities) & historical
        if overlap:
            manifest["status"] = "FAIL"
        validate_reset_manifest(manifest)
        audit = {
            "schema": "tactile3d-unit.s4-3-pi2b-policy-reset-exposure-audit.v1",
            "created_at_utc": now(),
            "status": "PASS" if not overlap else "FAIL",
            "known_seed_labels_blocked": sorted(BLOCKED_SEEDS),
            "shared_reserved_seed_labels_before_selection": sorted(occupied - BLOCKED_SEEDS),
            "selected_seed_blocks": blocks,
            "historical_unique_reset_identities": len(historical),
            "overlap_count": len(overlap),
            "performance_seen": False,
            "teacher_results_read": False,
        }
        atomic_json(MANIFEST, manifest)
        atomic_json(AUDIT, audit)
        exposure = open_lock(exposure_path, shared=False)
        try:
            atomic_json(
                reservation,
                {
                    **json.loads(reservation.read_text()),
                    "status": "FINAL_FROZEN" if audit["status"] == "PASS" else "CONFLICT",
                    "updated_at_utc": now(),
                    "reset_manifest_sha256": sha256_file(MANIFEST),
                    "ordered_reset_sequence_sha256": manifest["ordered_reset_sequence_sha256"],
                    "performance_seen": False,
                },
            )
        finally:
            exposure.close()
        print(json.dumps({"status": audit["status"], "seed_blocks": blocks, "resets": len(rows)}, sort_keys=True))
    finally:
        if gpu_lock is not None:
            gpu_lock.close()
        barrier.close()


if __name__ == "__main__":
    main()
