#!/usr/bin/env python3
"""Audit fresh data and bind checkpoints, probes, evaluator, and source before access."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_2_dataset import build_pair_arrays, episode_references, load_episode  # noqa: E402
from scripts.simulation.audit_s4_2_dataset import rgb_tree_sha256  # noqa: E402
from scripts.simulation.pi2b_teacher.common import (  # noqa: E402
    atomic_json,
    canonical_digest,
    git_output,
    load_json,
    sha256_file,
)

IMPLEMENTATIONS = (
    "gr00t/simulation/pi2b_teacher/matched_teacher.py",
    "gr00t/simulation/pi2b_teacher/evaluation.py",
    "scripts/simulation/pi2b_teacher/build_confirmation_cache.py",
    "scripts/simulation/pi2b_teacher/evaluate.py",
    "scripts/simulation/pi2b_teacher/analyze.py",
    "scripts/simulation/pi2b_teacher/independent_audit.py",
    "scripts/simulation/pi2b_teacher/finalize.py",
    "configs/simulation/pi2b_teacher/evaluation.json",
)


def digest_strings(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8") + b"\0")
    return digest.hexdigest()


def copy_tracked_snapshot(destination: Path) -> dict[str, str]:
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    hashes = {}
    for relative in [value for value in files if value]:
        source = ROOT / relative
        if not source.is_file():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source_hash = sha256_file(source)
        if target.exists() and sha256_file(target) != source_hash:
            raise RuntimeError(f"immutable evaluation snapshot collision: {target}")
        shutil.copy2(source, target)
        hashes[relative] = source_hash
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    args = parser.parse_args()
    runtime = load_json(args.runtime_manifest)
    root = Path(runtime["write_root"])
    if git_output("status", "--porcelain"):
        raise RuntimeError("evaluation freeze requires a clean committed worktree")
    if (root / "artifacts/evaluation_result.json").exists():
        raise RuntimeError("fresh performance result already exists")
    generation = load_json(root / "artifacts/generated_dataset_manifest.json")
    generation_status = load_json(root / "status/confirmation_generation.json")
    if generation["status"] != "STRUCTURALLY_GENERATED_NOT_EVALUATED" or generation_status["state"] != "COMPLETE":
        raise RuntimeError("fresh confirmation generation is incomplete")
    dataset_root = root / "datasets/fresh_confirmation"
    reservation = load_json(root / "artifacts/exposure_reservation.json")
    expected = {
        (row["task"], row["source_trajectory_id"], row["episode_id"], int(row["seed"]))
        for row in reservation["identities"]
    }
    references = list(episode_references(dataset_root))
    failures = []
    identities = set()
    task_counts: Counter[str] = Counter()
    for reference in references:
        manifest = load_json(reference.directory / "metadata.json")
        metadata = manifest["metadata"]
        identity = (reference.task, reference.source_trajectory_id, reference.episode_id, int(metadata["seed"]))
        identities.add(identity)
        task_counts[reference.task] += 1
        if metadata.get("formal_test_name") != "S4_3_PI2B_TEACHER_CONFIRMATION_V1" or metadata.get("exposure_reservation_digest") != reservation["identity_digest"]:
            failures.append(f"{reference.episode_id}:identity_metadata")
        if sha256_file(reference.directory / "steps.npz") != manifest["checksums"]["steps_npz_sha256"]:
            failures.append(f"{reference.episode_id}:steps_checksum")
        if rgb_tree_sha256(reference.directory / "frames") != manifest["checksums"]["rgb_tree_sha256"]:
            failures.append(f"{reference.episode_id}:rgb_checksum")
        arrays = load_episode(reference)
        expected_shapes = {"policy_action": (160, 22), "env_action": (160, 23), "sim_tactile": (160, 30), "contact_count_by_region": (160, 5), "rgb_reference": (160,)}
        for name, shape in expected_shapes.items():
            if arrays[name].shape != shape:
                failures.append(f"{reference.episode_id}:{name}_shape")
        if any(not np.isfinite(arrays[name]).all() for name in ("timestamp_sec", "proprio", "policy_action", "env_action", "sim_tactile")):
            failures.append(f"{reference.episode_id}:nonfinite")
        if not np.array_equal(np.diff(arrays["control_step"]), np.ones(159)):
            failures.append(f"{reference.episode_id}:control_step")
        if not np.allclose(np.diff(arrays["timestamp_sec"]), 0.02, atol=1e-9):
            failures.append(f"{reference.episode_id}:timestamp")
    if identities != expected:
        failures.append("reserved_identity_set")
    pairs = build_pair_arrays("test", dataset_root, purpose="dataset_quality")
    history = Path(runtime["worktree_root"])
    prior_paths = (
        history / ".local/refs/base/cache/simulation/s4_2_formal/paired_train.npz",
        history / ".local/refs/base/cache/simulation/s4_2_formal/paired_validation.npz",
        history / ".local/refs/base/cache/simulation/s4_2_formal/paired_test.npz",
    )
    prior_pair_ids: set[str] = set()
    prior_groups: set[tuple[str, str]] = set()
    for path in prior_paths:
        with np.load(path, allow_pickle=False) as source:
            prior_pair_ids |= set(source["pair_id"].tolist())
            prior_groups |= set(zip(source["task"].tolist(), source["source_trajectory_id"].tolist()))
    fresh_groups = set(zip(pairs["task"].tolist(), pairs["source_trajectory_id"].tolist()))
    if set(pairs["pair_id"].tolist()) & prior_pair_ids:
        failures.append("pair_overlap")
    if fresh_groups & prior_groups:
        failures.append("source_group_overlap")
    if len(references) != 45 or len(pairs["pair_id"]) != 4860 or len(fresh_groups) != 9 or task_counts != Counter({task: 15 for task in ("pinch_tongs", "hammer_nail", "click_mouse")}):
        failures.append("preregistered_counts")
    if not np.all(pairs["future_step"] - pairs["anchor_step"] == 27):
        failures.append("time_plus_27")
    integrity = {
        "schema": "tactile3d-unit.s4-3-pi2b-teacher-dataset-integrity.v1",
        "status": "PASS" if not failures else "FAIL",
        "episodes": len(references),
        "pairs": len(pairs["pair_id"]),
        "source_groups": len(fresh_groups),
        "task_episode_counts": dict(task_counts),
        "pair_identity_sha256": digest_strings(pairs["pair_id"]),
        "reservation_identity_digest": reservation["identity_digest"],
        "prior_pair_overlap": len(set(pairs["pair_id"].tolist()) & prior_pair_ids),
        "prior_group_overlap": len(fresh_groups & prior_groups),
        "time_plus_27": bool(np.all(pairs["future_step"] - pairs["anchor_step"] == 27)),
        "teacher_performance_loaded": False,
        "failures": failures,
    }
    atomic_json(root / "artifacts/dataset_integrity.json", integrity)
    if failures:
        raise SystemExit(f"fresh dataset integrity failed: {failures}")
    statuses = {mode: load_json(root / "status" / f"{mode}.json") for mode in ("T_VA_match", "T_VAC_match")}
    teacher_hashes = {}
    for mode, status in statuses.items():
        if status["state"] != "COMPLETE" or status["final_step"] != 800 or not status["cold_reload_finite"]:
            raise RuntimeError(f"{mode} final checkpoint is invalid")
        if sha256_file(Path(status["checkpoint"])) != status["checkpoint_sha256"]:
            raise RuntimeError(f"{mode} checkpoint hash mismatch")
        teacher_hashes[mode] = status["checkpoint_sha256"]
    if statuses["T_VA_match"]["batch_schedule_sha256"] != statuses["T_VAC_match"]["batch_schedule_sha256"]:
        raise RuntimeError("teacher batch schedules differ")
    native_paths = {
        "contact_state": history / ".local/experiments/simulation/s4_2r/contact_state/accepted.pt",
        "contact_C3": history / ".local/experiments/simulation/s4_2r/contact_dynamics/selected.pt",
        "action_A0": history / ".local/experiments/simulation/s4_2_formal/action/selected.pt",
    }
    native_hashes = {name: sha256_file(path) for name, path in native_paths.items()}
    vision_hashes = load_json(history / ".local/refs/base/artifacts/simulation/s4_2_formal/vision_identity.json")["checkpoint_file_sha256"]
    implementations = {relative: sha256_file(ROOT / relative) for relative in IMPLEMENTATIONS}
    freeze_identity = canonical_digest({"head": git_output("rev-parse", "HEAD"), "implementations": implementations, "teachers": teacher_hashes, "native": native_hashes, "dataset": integrity["pair_identity_sha256"]})
    snapshot = root / "snapshots/evaluation" / freeze_identity / "source"
    snapshot_hashes = copy_tracked_snapshot(snapshot)
    evaluation_runtime = {
        **runtime,
        "schema": "tactile3d-unit.s4-3-pi2b-teacher-evaluation-runtime.v1",
        "training_snapshot_root": runtime["snapshot_root"],
        "snapshot_root": str(snapshot),
        "evaluation_freeze_identity": freeze_identity,
        "git_head": git_output("rev-parse", "HEAD"),
        "git_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)).hexdigest(),
    }
    atomic_json(root / "artifacts/evaluation_runtime_manifest.json", evaluation_runtime)
    atomic_json(root / "artifacts/evaluation_source_snapshot_manifest.json", {"root": str(snapshot), "freeze_identity": freeze_identity, "files": snapshot_hashes})
    probe_protocol = load_json(ROOT / "configs/simulation/pi2b_teacher/evaluation.json")
    atomic_json(root / "artifacts/probe_protocol.json", probe_protocol)
    pretest = {
        "schema": "tactile3d-unit.s4-3-pi2b-teacher-pretest-freeze.v1",
        "status": "PASS",
        "freeze_identity": freeze_identity,
        "dataset_integrity_sha256": sha256_file(root / "artifacts/dataset_integrity.json"),
        "fresh_pair_identity_sha256": integrity["pair_identity_sha256"],
        "teacher_checkpoint_sha256": teacher_hashes,
        "native_checkpoint_sha256": native_hashes,
        "vision_checkpoint_file_sha256": vision_hashes,
        "implementation_sha256": implementations,
        "probe_protocol_sha256": sha256_file(root / "artifacts/probe_protocol.json"),
        "training_complete": True,
        "selection_complete": True,
        "test_loaded": False,
        "fresh_performance_loaded": False,
        "locked_evaluation_runs_allowed": 1,
        "deterministic_repeat_runs_allowed": 1,
        "teacher_retraining_allowed": False,
        "candidate_addition_allowed": False,
        "track_a_performance_read": False,
    }
    atomic_json(root / "artifacts/pretest_freeze.json", pretest)
    print(json.dumps({"status": "PASS", "freeze_identity": freeze_identity, "episodes": len(references), "pairs": len(pairs["pair_id"]), "snapshot": str(snapshot)}, indent=2))


if __name__ == "__main__":
    main()
