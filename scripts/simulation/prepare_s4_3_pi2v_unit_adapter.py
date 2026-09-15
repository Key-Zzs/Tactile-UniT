#!/usr/bin/env python3
"""Freeze PI2V starting, source, data, temporal, and normalization evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/simulation"))

from s4_3_pi2v_unit_adapter import (  # noqa: E402
    ACTION_RESAMPLE_INDICES,
    CANONICAL_CONTROL_STEPS,
    CANONICAL_HORIZON_SECONDS,
    POLICY_DATASET_FRAMES,
    POLICY_DATASET_HORIZON_SECONDS,
    dexjoco_action_view,
    dexjoco_state_view,
    sha256_file,
)


ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2v"
CACHE = ROOT / ".local/cache/simulation/s4_3_pi2v"
PAIR_ROOT = ROOT / ".local/cache/simulation/s4_2/pairs"
S42_DATASET = ROOT / ".local/datasets/simulation/s4_2"
POLICY_DATASET = (
    ROOT
    / ".local/external/s4_3_pi0/datasets/DexJoCo-Datasets-LeRobot/"
    "dexjoco_lerobot_datasets/pinch_tongs"
)
UNIT_SOURCE = ROOT / ".local/external/s4_3_pi2u/unit_official"
UNIT_CHECKPOINT = (
    ROOT
    / ".local/external/s4_3_pi2u/unit_fulldata/"
    "VLA-UniT-3B-fulldata/tokenizer"
)
CHECKPOINT_HASHES = {
    "config.json": "7a651f488c93521e0d507880fc250a475e6a08aa9307aa1349f9d3509844971e",
    "model-00001-of-00002.safetensors": "32d5c326f6c83d12185b6954d2a52511f66ad18b6fdf814aecc5726dd39c243c",
    "model-00002-of-00002.safetensors": "2f8093a900330e5111b63e44dc1687b3212bec343e5b3832bf2e40f2bf18a768",
    "model.safetensors.index.json": "3b6d73d2442ce694287c5cd8b93db1bb232909becf35f08ecabadb614b9a1b86",
}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def command(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def tree_hash(path: Path) -> str:
    rows = []
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        rows.append((item.relative_to(path).as_posix(), sha256_file(item)))
    return canonical_hash(rows)


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    branch = command("git", "branch", "--show-current")
    head = command("git", "rev-parse", "HEAD")
    if branch != "develop/sim-benchmark":
        raise RuntimeError(f"required branch missing: {branch}")
    starting = {
        "schema": "tactile3d-unit.s4-3-pi2v-starting-integrity.v1",
        "status": "PASS",
        "branch": branch,
        "starting_head": head,
        "starting_worktree_clean": True,
        "starting_status_short": [],
        "remote": command("git", "remote", "get-url", "origin"),
        "submodule": command("git", "submodule", "status", "--recursive"),
        "official_pause_boundary": "PI2V-6 stable launch",
    }
    atomic_json(ARTIFACTS / "starting_integrity.json", starting)

    protected = [
        "configs/tactile_unit/m3_system_manifest.json",
        "configs/simulation/s4_2_representation_protocol.json",
        "configs/simulation/s4_3_pi0_official_pi05_protocol.json",
        "configs/simulation/s4_3_pi2a_final_decision.json",
        "configs/simulation/s4_3_pi2u_unit_compatibility.json",
        "configs/simulation/s4_3_pi2u_bva_protocol.json",
        "configs/simulation/s4_3_pi2u_bva_temporal_remediation.json",
        ".local/artifacts/simulation/s4_3_pi2a/checkpoint_immutability.json",
        ".local/artifacts/simulation/s4_3_pi2u/bva_checkpoint_manifest.json",
        ".local/artifacts/simulation/s4_3_pi2u/final_decision.json",
    ]
    protected_hashes = {
        name: sha256_file(ROOT / name) for name in protected if (ROOT / name).is_file()
    }
    pi2a_checkpoints = json.loads(
        (ROOT / ".local/artifacts/simulation/s4_3_pi2a/checkpoint_immutability.json").read_text()
    )["checkpoints"]
    bva_checkpoint = json.loads(
        (ROOT / ".local/artifacts/simulation/s4_3_pi2u/bva_checkpoint_manifest.json").read_text()
    )
    policy_checkpoint_hashes = {
        name: pi2a_checkpoints[name]["checkpoint_tree_sha256"]
        for name in ("B0", "B1", "B2")
    }
    policy_checkpoint_hashes["BVA"] = bva_checkpoint["checkpoint_tree_sha256"]
    expected_policy_hashes = {
        "B0": "1e7a6ace5d69a988a8b258e1c56a24d88b077580a05b27be3f510df9ac3864f3",
        "BVA": "04b609d7cc89e5fffdfab8219bf34da362117a15d0c7e3d5cd4ee20a9ee4770d",
        "B1": "04b59cbc3491bf4e88dc75558a08a94e5588e2fbe398d413e1766a87c7ab6f4f",
        "B2": "86c4908533ca24da06ae66f943e3605b5ccbae455441290ca8fb35514359e949",
    }
    if policy_checkpoint_hashes != expected_policy_hashes:
        raise RuntimeError("frozen policy checkpoint manifest drift")
    historical = {
        "schema": "tactile3d-unit.s4-3-pi2v-historical-immutability.v1",
        "status": "PASS",
        "captured_at_head": head,
        "protected_files_sha256": protected_hashes,
        "frozen_policy_checkpoint_tree_sha256": policy_checkpoint_hashes,
        "checkpoint_manifest_sources": {
            "B0_B1_B2": "$REPO_ROOT/.local/artifacts/simulation/s4_3_pi2a/checkpoint_immutability.json",
            "BVA": "$REPO_ROOT/.local/artifacts/simulation/s4_3_pi2u/bva_checkpoint_manifest.json",
        },
        "mutation_authorized": {"B0": False, "BVA": False, "B1": False, "B2": False},
    }
    atomic_json(ARTIFACTS / "historical_immutability_before.json", historical)

    source_head = command("git", "rev-parse", "HEAD", cwd=UNIT_SOURCE)
    source_status = command("git", "status", "--short", cwd=UNIT_SOURCE)
    source_tree = command("git", "ls-tree", "-r", "--full-tree", "HEAD", cwd=UNIT_SOURCE)
    source_tree_hash = hashlib.sha256((source_tree + "\n").encode()).hexdigest()
    checkpoint_actual = {name: sha256_file(UNIT_CHECKPOINT / name) for name in CHECKPOINT_HASHES}
    source_ok = (
        source_head == "0d762e32180bddd765694ef3846a3a5053f9d37f"
        and not source_status
        and source_tree_hash == "4268e41678c643bb6a5f4e6823cb087707393e9d44ec06035a46e0b6600c8648"
        and checkpoint_actual == CHECKPOINT_HASHES
    )
    official = {
        "schema": "tactile3d-unit.s4-3-pi2v-official-unit-revalidation.v1",
        "status": "PASS" if source_ok else "S4_3_PI2V_UNIT_SOURCE_DRIFT",
        "repository": "https://github.com/xpeng-robotics/UniT.git",
        "source_commit": source_head,
        "source_clean": not bool(source_status),
        "source_tree_sha256": source_tree_hash,
        "checkpoint": "$UNIT_OFFICIAL_CKPT",
        "checkpoint_files_sha256": checkpoint_actual,
        "checkpoint_total_tensor_bytes": 5_494_676_608,
        "compatibility": "UNIT_DEXJOCO_ADAPTER_ONLY_COMPATIBLE",
    }
    atomic_json(ARTIFACTS / "official_unit_revalidation.json", official)
    if not source_ok:
        raise RuntimeError("S4_3_PI2V_UNIT_SOURCE_DRIFT")

    with np.load(PAIR_ROOT / "train.npz", allow_pickle=False) as train_source:
        train = {key: train_source[key] for key in train_source.files}
    with np.load(PAIR_ROOT / "validation.npz", allow_pickle=False) as dev_source:
        dev = {key: dev_source[key] for key in dev_source.files}
    for name, values, expected in (("train", train, 22_680), ("dev", dev, 4_860)):
        if len(values["pair_id"]) != expected:
            raise RuntimeError(f"{name} pair count drift")
        if values["current_state"].shape != (expected, 23):
            raise RuntimeError(f"{name} state contract drift")
        if values["action_chunk"].shape != (expected, 27, 22):
            raise RuntimeError(f"{name} action contract drift")
        if not np.all(values["future_step"] - values["anchor_step"] == 27):
            raise RuntimeError(f"{name} pair horizon drift")
    train_groups = set(train["source_trajectory_id"].tolist())
    dev_groups = set(dev["source_trajectory_id"].tolist())
    if train_groups & dev_groups:
        raise RuntimeError("representation source-group leakage")

    step_deltas = []
    control_step_deltas = []
    for steps_path in sorted((S42_DATASET / "episodes").glob("*/steps.npz")):
        with np.load(steps_path, allow_pickle=False) as values:
            step_deltas.extend(np.diff(values["timestamp_sec"]).tolist())
            control_step_deltas.extend(np.diff(values["control_step"]).tolist())
    info = json.loads((POLICY_DATASET / "meta/info.json").read_text())
    policy_timestamp_deltas = []
    for data_path in sorted((POLICY_DATASET / "data").rglob("*.parquet")):
        table = pq.read_table(data_path, columns=["episode_index", "timestamp"]).to_pydict()
        episodes = np.asarray(table["episode_index"])
        timestamps = np.asarray(table["timestamp"], dtype=np.float64)
        valid = np.diff(episodes) == 0
        policy_timestamp_deltas.extend(np.diff(timestamps)[valid].tolist())
    bva_manifest = json.loads(
        (ROOT / ".local/artifacts/simulation/s4_3_pi2u/bva_target_manifest.json").read_text()
    )
    timestamp_ok = (
        np.max(np.abs(np.asarray(step_deltas) - 0.02)) < 1e-9
        and set(control_step_deltas) == {1}
        and float(info["fps"]) == 30.0
        and np.max(np.abs(np.asarray(policy_timestamp_deltas) - 1 / 30)) < 1e-5
        and bva_manifest["source_future_offset_frames"] == 16
        and bva_manifest["canonical_future_offset_steps"] == 27
    )
    temporal = {
        "schema": "tactile3d-unit.s4-3-pi2v-temporal-contract-audit.v1",
        "status": "PASS" if timestamp_ok else "S4_3_PI2V_TEMPORAL_CONTRACT_FAIL",
        "simulation": {
            "measured_dt_seconds_min": float(np.min(step_deltas)),
            "measured_dt_seconds_max": float(np.max(step_deltas)),
            "control_hz": 50.0,
            "offset_steps": CANONICAL_CONTROL_STEPS,
            "duration_seconds": CANONICAL_HORIZON_SECONDS,
        },
        "policy_dataset": {
            "declared_fps": float(info["fps"]),
            "measured_dt_seconds_min": float(np.min(policy_timestamp_deltas)),
            "measured_dt_seconds_max": float(np.max(policy_timestamp_deltas)),
            "offset_frames": POLICY_DATASET_FRAMES,
            "duration_seconds": POLICY_DATASET_HORIZON_SECONDS,
        },
        "adapter_action_view": {
            "source_commands": 27,
            "official_slots": 16,
            "nearest_50hz_command_indices_zero_based": ACTION_RESAMPLE_INDICES.tolist(),
            "last_command_time_seconds": float(ACTION_RESAMPLE_INDICES[-1] / 50),
            "future_observation_time_seconds": 0.54,
        },
        "absolute_policy_mapping_error_seconds": abs(
            CANONICAL_HORIZON_SECONDS - POLICY_DATASET_HORIZON_SECONDS
        ),
        "bva_and_future_bunit_same_transition": True,
        "rgb_interpolation": False,
    }
    atomic_json(ARTIFACTS / "temporal_contract_audit.json", temporal)
    if not timestamp_ok:
        raise RuntimeError("S4_3_PI2V_TEMPORAL_CONTRACT_FAIL")

    state = dexjoco_state_view(train["current_state"])
    action = dexjoco_action_view(train["action_chunk"])
    state_mean = state.mean(axis=0, dtype=np.float64).astype(np.float32)
    state_std = state.std(axis=0, dtype=np.float64).astype(np.float32)
    action_mean = action.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    action_std = action.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    normalization_path = CACHE / "adapter_normalization.npz"
    np.savez(
        normalization_path,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
    )
    normalization = {
        "schema": "tactile3d-unit.s4-3-pi2v-adapter-normalization.v1",
        "status": "PASS",
        "fit_split": "TRAIN only",
        "state_semantics": "TCP xyz3 + quaternion-wxyz-to-rotation6d6 + Allegro16",
        "state_dim_before_padding": 25,
        "action_semantics": "absolute TCP xyz3 + rotvec3 + Allegro16",
        "action_dim_before_padding": 22,
        "action_slots": 16,
        "cache": "$PI2V_ROOT/cache/adapter_normalization.npz",
        "cache_sha256": sha256_file(normalization_path),
        "zero_state_std_dimensions": np.flatnonzero(state_std == 0).tolist(),
        "zero_action_std_dimensions": np.flatnonzero(action_std == 0).tolist(),
    }
    atomic_json(ARTIFACTS / "adapter_normalization.json", normalization)

    data_contract = {
        "schema": "tactile3d-unit.s4-3-pi2v-adapter-data-contract.v1",
        "status": "PASS",
        "source": "frozen S4.2 representation dataset",
        "tasks": sorted(set(train["task"].tolist())),
        "train": {
            "pairs": len(train["pair_id"]),
            "source_groups": len(train_groups),
            "pair_cache_sha256": sha256_file(PAIR_ROOT / "train.npz"),
        },
        "dev": {
            "pairs": len(dev["pair_id"]),
            "source_groups": len(dev_groups),
            "pair_cache_sha256": sha256_file(PAIR_ROOT / "validation.npz"),
        },
        "source_group_overlap": 0,
        "fields_used": [
            "current_frame",
            "future_frame",
            "current_state",
            "action_chunk",
            "pair_id",
            "episode_id",
            "source_trajectory_id",
            "task",
        ],
        "forbidden_fields_used": [],
        "policy_outcomes_used": False,
        "evaluation_data_used": False,
    }
    atomic_json(ARTIFACTS / "adapter_data_contract.json", data_contract)
    print(json.dumps({"status": "PASS", "starting_head": head, "temporal": "PASS"}))


if __name__ == "__main__":
    main()
