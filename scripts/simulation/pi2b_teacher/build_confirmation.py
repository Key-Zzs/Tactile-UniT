#!/usr/bin/env python3
"""Generate the frozen 45-episode PI2B teacher confirmation cohort."""

from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.simulation.generate_s4_2_dataset import generate_episode  # noqa: E402
from scripts.simulation.pi2b_teacher.common import (  # noqa: E402
    CONFIG_ROOT,
    atomic_json,
    canonical_digest,
    load_json,
    sha256_file,
)


def main() -> None:
    if os.environ.get("DISPLAY"):
        raise SystemExit("DISPLAY must be unset for deterministic offscreen generation")
    runtime_path = Path(os.environ["PI2B_TEACHER_RUNTIME_MANIFEST"])
    runtime = load_json(runtime_path)
    if Path(runtime["snapshot_root"]).resolve() != ROOT.resolve():
        raise RuntimeError("confirmation generator is not running from frozen snapshot")
    protocol = load_json(CONFIG_ROOT / "protocol.json")
    confirmation = protocol["confirmation"]
    reservation = load_json(Path(runtime["write_root"]) / "artifacts/exposure_reservation.json")
    if reservation["identity_digest"] != canonical_digest(reservation["identities"]):
        raise RuntimeError("confirmation reservation digest mismatch")
    coordination = Path(runtime["coordination_dir"])
    policy_request = coordination / "requests/policy.json"
    if policy_request.exists():
        scheduler = load_json(policy_request)
        if scheduler.get("state") == "EVAL_PENDING" or scheduler.get("status") == "EVAL_PENDING":
            raise RuntimeError("formal policy evaluation is pending; no new heavy shared job may start")
    barrier_path = coordination / "runtime_barrier.lock"
    dataset_root = Path(runtime["write_root"]) / "datasets/fresh_confirmation"
    status_path = Path(runtime["write_root"]) / "status/confirmation_generation.json"
    log_path = Path(runtime["write_root"]) / "logs/confirmation_generation.jsonl"
    contract = load_json(ROOT / "configs/simulation/s4_2_dataset_contract.json")
    source_profiles = load_json(ROOT / "configs/simulation/s4_2_tf_locked_test_v2.json")["formal_test_v2"]["group_parameters"]
    profile_for_group = {23: source_profiles["20"], 24: source_profiles["21"], 25: source_profiles["22"]}
    expected = {
        (row["task"], row["source_trajectory_id"], row["episode_id"], int(row["seed"]))
        for row in reservation["identities"]
    }
    completed = []
    start = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with barrier_path.open("a+") as barrier:
        try:
            fcntl.flock(barrier, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("runtime barrier is held exclusively") from error
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            for task in confirmation["tasks"]:
                for group in confirmation["source_group_indices"]:
                    source_id = f"{task}-script-{group:02d}"
                    for perturbation in range(confirmation["episodes_per_group"]):
                        episode_id = f"{task}-pi2b-teacher-g{group:02d}-p{perturbation:02d}"
                        seed = int(confirmation["seed_bases"][task] + 10 * group + perturbation)
                        identity = (task, source_id, episode_id, seed)
                        if identity not in expected:
                            raise RuntimeError(f"unreserved confirmation identity: {identity}")
                        manifest = generate_episode(
                            dataset_root,
                            task,
                            group,
                            perturbation,
                            "test",
                            contract,
                            source_id=source_id,
                            episode_id=episode_id,
                            seed=seed,
                            parameters=dict(profile_for_group[group]),
                            metadata_updates={
                                "formal_test_name": "S4_3_PI2B_TEACHER_CONFIRMATION_V1",
                                "exposure_classification": "FINAL_FROZEN",
                                "exposure_reservation_digest": reservation["identity_digest"],
                                "parameter_profile_source_group": group - 3,
                                "teacher_performance_filtering": False,
                            },
                        )
                        metadata = manifest["metadata"]
                        actual = (
                            metadata["task"],
                            metadata["source_trajectory_id"],
                            manifest["episode_id"],
                            int(metadata["seed"]),
                        )
                        if actual != identity:
                            raise RuntimeError("generated confirmation identity mismatch")
                        completed.append(episode_id)
                        elapsed = time.monotonic() - start
                        record = {
                            "episode": episode_id,
                            "completed": len(completed),
                            "total": int(confirmation["episodes"]),
                            "seconds_per_episode": elapsed / len(completed),
                            "eta_seconds": (int(confirmation["episodes"]) - len(completed)) * elapsed / len(completed),
                        }
                        log.write(json.dumps(record, sort_keys=True) + "\n")
                        atomic_json(status_path, {"state": "RUNNING", "pid": os.getpid(), "dataset_root": str(dataset_root), "log": str(log_path), **record})
        fcntl.flock(barrier, fcntl.LOCK_UN)
    if len(completed) != int(confirmation["episodes"]) or len(set(completed)) != len(completed):
        raise RuntimeError("confirmation episode count mismatch")
    metadata_files = sorted(dataset_root.glob("episodes/*/metadata.json"))
    if len(metadata_files) != int(confirmation["episodes"]):
        raise RuntimeError("confirmation directory contains an unexpected episode count")
    manifest = {
        "schema": "tactile3d-unit.s4-3-pi2b-teacher-confirmation-generation.v1",
        "status": "STRUCTURALLY_GENERATED_NOT_EVALUATED",
        "dataset_root": str(dataset_root),
        "episodes": len(metadata_files),
        "groups": len({load_json(path)["metadata"]["source_trajectory_id"] for path in metadata_files}),
        "metadata_sha256": {path.parent.name: sha256_file(path) for path in metadata_files},
        "reservation_digest": reservation["identity_digest"],
        "teacher_performance_read": False,
        "generator_sha256": sha256_file(ROOT / "scripts/simulation/generate_s4_2_dataset.py"),
        "contract_sha256": sha256_file(ROOT / "configs/simulation/s4_2_dataset_contract.json"),
    }
    atomic_json(Path(runtime["write_root"]) / "artifacts/generated_dataset_manifest.json", manifest)
    atomic_json(status_path, {"state": "COMPLETE", "pid": os.getpid(), "completed": len(completed), "total": len(completed), "dataset_root": str(dataset_root), "log": str(log_path), "manifest": str(Path(runtime["write_root"]) / "artifacts/generated_dataset_manifest.json")})
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
