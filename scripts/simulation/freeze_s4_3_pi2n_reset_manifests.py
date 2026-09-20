#!/usr/bin/env python3
"""Freeze exact PI2N DEV/FINAL reset identities without policy inference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_PROTOCOL = ROOT / "configs/simulation/s4_3_pi2n_evaluation_protocol.json"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2n"
SIMULATION_ARTIFACTS = ROOT / ".local/artifacts/simulation"
RUNTIME_MANIFEST = ARTIFACTS / "runtime_reset_manifest.json"
DEV_MANIFEST = ARTIFACTS / "development_manifest.json"
FINAL_MANIFEST = ARTIFACTS / "final_reset_manifest.json"
EXPOSURE_LEDGER = ARTIFACTS / "reset_exposure_ledger.json"
CONFIG = ROOT / "third_party/dexjoco/configs/rand_obj/pinch_tongs.yaml"
DEV_BLOCKS = ((9, 10), (10, 10), (11, 10))
FINAL_BLOCKS = ((12, 50), (13, 50), (14, 50), (15, 50))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def sequence_sha256(identities: list[str]) -> str:
    return hashlib.sha256("\n".join(identities).encode()).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def gpu_snapshot() -> dict[str, Any]:
    inventory = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    applications = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    return {"inventory": inventory, "compute_applications": applications}


def gpu_is_idle(index: int, snapshot: dict[str, Any]) -> bool:
    row = next(
        (line for line in snapshot["inventory"] if int(line.split(",", 1)[0].strip()) == index),
        None,
    )
    if row is None:
        return False
    fields = [field.strip() for field in row.split(",")]
    uuid = fields[1]
    return int(fields[3]) <= 64 and not any(
        line.split(",", 1)[0].strip() == uuid for line in snapshot["compute_applications"]
    )


def acquire_gpu_lock(index: int):
    common = subprocess.check_output(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=ROOT,
        text=True,
    ).strip()
    handle = open(Path(common) / f"tactile3d_unit_gpu{index}.lock", "a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"GPU {index} advisory lock is held")
    return handle


def generate_block(seed: int, episodes: int) -> list[dict[str, Any]]:
    import mujoco
    import yaml

    sys.path.insert(0, str(ROOT / "third_party/dexjoco/dexjoco"))
    from dexjoco_openpi_client import eval_dexjoco_openpi as official
    from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv

    config = yaml.safe_load(CONFIG.read_text())
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
        for block_index in range(episodes):
            environment.reset()
            mj_state = np.empty(
                mujoco.mj_stateSize(raw.model, state_spec), dtype=np.float64
            )
            mujoco.mj_getState(raw.model, raw.data, mj_state, state_spec)
            processed_state = np.asarray(environment.obs["state"], dtype=np.float64)
            rows.append(
                {
                    "seed": seed,
                    "block_index": block_index,
                    "reset_identity": sha256_array(mj_state, processed_state),
                    "mjstate_sha256": sha256_array(mj_state),
                    "processed_state_sha256": sha256_array(processed_state),
                    "mjstate_shape": list(mj_state.shape),
                    "processed_state_shape": list(processed_state.shape),
                }
            )
    finally:
        environment.close()
    return rows


def build_manifest(
    name: str, blocks: tuple[tuple[int, int], ...], rows_by_seed: dict[int, list[dict[str, Any]]]
) -> dict[str, Any]:
    rows = []
    for seed, episodes in blocks:
        block = rows_by_seed[seed]
        if len(block) != episodes:
            raise RuntimeError(f"seed {seed} generated {len(block)} resets, expected {episodes}")
        for row in block:
            rows.append({**row, "global_index": len(rows)})
    identities = [row["reset_identity"] for row in rows]
    unique = len(set(identities)) == len(identities)
    return {
        "schema": "tactile3d-unit.s4-3-pi2n-reset-manifest.v1",
        "status": "PASS" if unique else "FAIL",
        "cohort": name,
        "created_at": now(),
        "scientific_policy_inference_performed": False,
        "policy_performance_seen": False,
        "task": "pinch_tongs",
        "regime": "rand_obj",
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": sha256_file(CONFIG),
        "official_environment_wrapper": "DexJoCoOpenPIEnv",
        "rand_full": False,
        "randomize_dynamics": False,
        "render_mode": "rgb_array",
        "blocks": [
            {"seed": seed, "episodes": episodes, "fresh_environment_process_contract": True}
            for seed, episodes in blocks
        ],
        "episodes": len(rows),
        "ordered_reset_identities": identities,
        "ordered_reset_sequence_sha256": sequence_sha256(identities),
        "all_reset_identities_unique": unique,
        "identity_definition": "sha256(dtype+shape+bytes of mjSTATE_INTEGRATION float64, then official processed state float64)",
        "reset_specs": rows,
    }


def historical_exposure_sources() -> tuple[list[dict[str, Any]], set[str]]:
    sources = []
    union: set[str] = set()
    for path in sorted(SIMULATION_ARTIFACTS.rglob("*.json")):
        if path in {DEV_MANIFEST, FINAL_MANIFEST, EXPOSURE_LEDGER}:
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        episodes = payload.get("episode_results") if isinstance(payload, dict) else None
        if not isinstance(episodes, list):
            continue
        identities = [
            row.get("reset_identity")
            for row in episodes
            if isinstance(row, dict) and isinstance(row.get("reset_identity"), str)
        ]
        if not identities:
            continue
        identity_set = set(identities)
        union.update(identity_set)
        sources.append(
            {
                "path": str(path.relative_to(ROOT)),
                "artifact_sha256": sha256_file(path),
                "evaluator_seed": payload.get("evaluator_seed"),
                "model": payload.get("model"),
                "episodes_declared": payload.get("episodes"),
                "reset_identities_found": len(identities),
                "unique_reset_identities": len(identity_set),
                "ordered_sequence_sha256": sequence_sha256(identities),
            }
        )
    return sources, union


def freeze(gpu: int) -> None:
    protocol = json.loads(EVALUATION_PROTOCOL.read_text())
    expected = protocol["cohorts"]
    if protocol.get("status") != "FROZEN_BEFORE_NEW_POLICY_TRAINING":
        raise SystemExit("PI2N evaluation protocol is not frozen")
    if expected["PI2N_DEV"]["seed_blocks"] != [9, 10, 11]:
        raise SystemExit("PI2N DEV blocks differ from tracked constants")
    if expected["PI2N_FINAL"]["seed_blocks"] != [12, 13, 14, 15]:
        raise SystemExit("PI2N FINAL blocks differ from tracked constants")
    if any(path.exists() for path in (DEV_MANIFEST, FINAL_MANIFEST, EXPOSURE_LEDGER)):
        raise SystemExit("refusing to overwrite PI2N reset manifests or exposure ledger")
    runtime = json.loads(RUNTIME_MANIFEST.read_text())
    if runtime.get("status") != "PASS" or len(runtime.get("ordered_resets", [])) != 10:
        raise SystemExit("runtime reset manifest is not a complete PASS reference")

    first = gpu_snapshot()
    time.sleep(2)
    second = gpu_snapshot()
    if not (gpu_is_idle(gpu, first) and gpu_is_idle(gpu, second)):
        raise SystemExit("selected reset-freeze GPU is not genuinely idle in two snapshots")
    lock = acquire_gpu_lock(gpu)
    try:
        third = gpu_snapshot()
        if not gpu_is_idle(gpu, third):
            raise SystemExit("selected reset-freeze GPU became busy after locking")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"

        rows_by_seed = {
            seed: generate_block(seed, episodes)
            for seed, episodes in ((8, 10), *DEV_BLOCKS, *FINAL_BLOCKS)
        }
        runtime_generated = [row["reset_identity"] for row in rows_by_seed[8]]
        runtime_frozen = [row["reset_identity"] for row in runtime["ordered_resets"]]
        runtime_reproduced = runtime_generated == runtime_frozen
        dev = build_manifest("PI2N_DEV", DEV_BLOCKS, rows_by_seed)
        final = build_manifest("PI2N_FINAL", FINAL_BLOCKS, rows_by_seed)
        dev_ids = set(dev["ordered_reset_identities"])
        final_ids = set(final["ordered_reset_identities"])
        runtime_ids = set(runtime_frozen)
        sources, historical_ids = historical_exposure_sources()
        gates = {
            "runtime_seed8_generator_reproduces_frozen_identity_sequence": runtime_reproduced,
            "dev_exactly_30_unique_resets": dev["episodes"] == 30
            and dev["all_reset_identities_unique"],
            "final_exactly_200_unique_resets": final["episodes"] == 200
            and final["all_reset_identities_unique"],
            "dev_disjoint_runtime": not (dev_ids & runtime_ids),
            "final_disjoint_runtime": not (final_ids & runtime_ids),
            "dev_disjoint_final": not (dev_ids & final_ids),
            "dev_disjoint_all_historical_raw_rollouts": not (dev_ids & historical_ids),
            "final_disjoint_all_historical_raw_rollouts": not (final_ids & historical_ids),
            "no_policy_inference_or_performance_access": True,
        }
        status = "PASS" if all(gates.values()) else "FAIL"
        dev["status"] = status if dev["status"] == "PASS" else "FAIL"
        final["status"] = status if final["status"] == "PASS" else "FAIL"
        dev["evaluation_protocol_sha256"] = sha256_file(EVALUATION_PROTOCOL)
        final["evaluation_protocol_sha256"] = sha256_file(EVALUATION_PROTOCOL)
        dev["gpu_snapshots"] = [first, second, third]
        final["gpu_snapshots"] = [first, second, third, gpu_snapshot()]
        ledger = {
            "schema": "tactile3d-unit.s4-3-pi2n-reset-exposure-ledger.v1",
            "status": status,
            "created_at": now(),
            "historical_raw_sources": sources,
            "historical_source_count": len(sources),
            "historical_unique_reset_identity_count": len(historical_ids),
            "historical_union_sha256": sequence_sha256(sorted(historical_ids)),
            "runtime_diag_sequence_sha256": runtime["sequence_sha256"],
            "runtime_diag_reproduced_sequence_sha256": sequence_sha256(runtime_generated),
            "development_sequence_sha256": dev["ordered_reset_sequence_sha256"],
            "final_sequence_sha256": final["ordered_reset_sequence_sha256"],
            "overlap_counts": {
                "dev_runtime": len(dev_ids & runtime_ids),
                "final_runtime": len(final_ids & runtime_ids),
                "dev_final": len(dev_ids & final_ids),
                "dev_historical": len(dev_ids & historical_ids),
                "final_historical": len(final_ids & historical_ids),
            },
            "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
            "scientific_policy_inference_performed": False,
            "policy_performance_seen": False,
        }
        atomic_json(DEV_MANIFEST, dev)
        atomic_json(FINAL_MANIFEST, final)
        atomic_json(EXPOSURE_LEDGER, ledger)
        print(
            json.dumps(
                {
                    "status": status,
                    "dev_resets": dev["episodes"],
                    "final_resets": final["episodes"],
                    "historical_sources": len(sources),
                    "gates": ledger["gates"],
                },
                sort_keys=True,
            )
        )
        if status != "PASS":
            raise SystemExit("PI2N_RESET_MANIFEST_FREEZE_FAILED")
    finally:
        lock.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    freeze(args.gpu)


if __name__ == "__main__":
    main()
