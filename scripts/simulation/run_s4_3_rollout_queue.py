#!/usr/bin/env python3
"""Schedule 36 checkpoint workers and aggregate all 1080 frozen rollouts."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_3_training import atomic_json, read_json  # noqa: E402
from scripts.simulation.run_s4_3_training_queue import gpu_state  # noqa: E402

TRAINING_JOBS = ROOT / ".local/artifacts/simulation/s4_3_restart/training_jobs.json"
PRE_ROLLOUT = ROOT / ".local/artifacts/simulation/s4_3_restart/pre_rollout_freeze.json"
JOB_ROOT = ROOT / ".local/artifacts/simulation/s4_3_restart/rollout_jobs"
LOG_ROOT = ROOT / ".local/logs/simulation/s4_3_restart/rollout_queue"
ARTIFACT = ROOT / ".local/artifacts/simulation/s4_3_restart/closed_loop_rollouts.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def job_path(row: dict[str, Any]) -> Path:
    return JOB_ROOT / f"{row['task']}_{row['variant']}_seed{row['training_seed']}.json"


def complete(row: dict[str, Any]) -> dict[str, Any] | None:
    path = job_path(row)
    if not path.is_file():
        return None
    value = read_json(path)
    identity = (value.get("task"), value.get("variant"), value.get("training_seed"))
    expected = (row["task"], row["variant"], row["training_seed"])
    if (
        value.get("status") != "PASS"
        or value.get("rollouts") != 30
        or identity != expected
        or value.get("checkpoint_sha256") != row["checkpoint_sha256"]
    ):
        return None
    return value


def main() -> None:
    if not os.environ.get("DEXJOCO_PYTHON") or not os.environ.get("UNIT_FULLDATA_CKPT"):
        raise RuntimeError("DEXJOCO_PYTHON and UNIT_FULLDATA_CKPT are required")
    training = read_json(TRAINING_JOBS)
    freeze = read_json(PRE_ROLLOUT)
    if training.get("status") != "PASS" or training.get("canonical_checkpoints") != 36:
        raise RuntimeError("complete R9 training gate is required")
    if (
        freeze.get("status") != "PASS"
        or freeze.get("rollout_performance_seen") is not False
        or freeze.get("checkpoint_selection_complete") is not True
    ):
        raise RuntimeError("valid pre-rollout freeze is required")
    common_git = Path(
        subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if not common_git.is_absolute():
        common_git = ROOT / common_git
    queue = sorted(
        training["jobs"],
        key=lambda row: (row["task"], row["variant"], row["training_seed"]),
    )
    completed = []
    for row in list(queue):
        prior = complete(row)
        if prior is not None:
            completed.append(prior)
            queue.remove(row)
    running: dict[int, dict[str, Any]] = {}
    attempts: dict[tuple[str, str, int], int] = {}
    started_at = now()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    def progress() -> None:
        atomic_json(
            ARTIFACT,
            {
                "schema": "tactile3d-unit.s4-3-closed-loop-rollouts.v1",
                "stage": "R12",
                "start_time": started_at,
                "end_time": None,
                "expected_jobs": 36,
                "expected_rollouts": 1080,
                "completed_jobs": len(completed),
                "completed_rollouts": sum(row["rollouts"] for row in completed),
                "running": [item["record"] for item in running.values()],
                "queued": [[row["task"], row["variant"], row["training_seed"]] for row in queue],
                "rollout_performance_seen": bool(completed),
                "status": "RUNNING",
            },
        )

    progress()
    while queue or running:
        for gpu, item in list(running.items()):
            return_code = item["process"].poll()
            if return_code is None:
                continue
            item["log"].close()
            item["lock"].close()
            row = item["row"]
            key = (row["task"], row["variant"], row["training_seed"])
            if return_code == 0:
                result = complete(row)
                if result is None:
                    raise RuntimeError("successful rollout process lacks complete job artifact")
                completed.append(
                    {
                        **result,
                        "physical_gpu": gpu,
                        "logical_device": "cuda:0",
                        "scheduler_start_time": item["record"]["start_time"],
                        "scheduler_end_time": now(),
                        "scheduler_exit_code": 0,
                    }
                )
            elif return_code == 75 and attempts[key] == 1:
                queue.append(row)
            else:
                progress()
                raise SystemExit(f"S4_3_2_ENVIRONMENT_FAIL: {key}, exit={return_code}")
            del running[gpu]
            progress()

        for gpu in range(4):
            if not queue or gpu in running or not gpu_state(gpu)["idle"]:
                continue
            lock = (common_git / f"tactile3d_unit_gpu{gpu}.lock").open("a+")
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            state = gpu_state(gpu)
            if not state["idle"]:
                lock.close()
                continue
            row = queue.pop(0)
            key = (row["task"], row["variant"], row["training_seed"])
            attempts[key] = attempts.get(key, 0) + 1
            log_path = LOG_ROOT / f"{row['task']}_{row['variant']}_seed{row['training_seed']}.log"
            log = log_path.open("w", encoding="utf-8")
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "S4_3_PHYSICAL_GPU": str(gpu),
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                }
            )
            environment.pop("DISPLAY", None)
            command = [
                sys.executable,
                str(ROOT / "scripts/simulation/run_s4_3_rollout_job.py"),
                "--task",
                row["task"],
                "--variant",
                row["variant"],
                "--seed",
                str(row["training_seed"]),
                "--checkpoint",
                str(ROOT / row["checkpoint"]),
                "--checkpoint-sha256",
                row["checkpoint_sha256"],
            ]
            record = {
                "task": row["task"],
                "variant": row["variant"],
                "training_seed": row["training_seed"],
                "physical_gpu": gpu,
                "logical_device": "cuda:0",
                "prelaunch_gpu_state": state,
                "start_time": now(),
                "attempt": attempts[key],
                "log": str(log_path.relative_to(ROOT)),
            }
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            running[gpu] = {
                "row": row,
                "process": process,
                "lock": lock,
                "log": log,
                "record": record,
            }
            print(json.dumps({**record, "status": "LAUNCHED"}, sort_keys=True), flush=True)
            progress()
        if queue or running:
            time.sleep(10)

    metadata = []
    for job in completed:
        for item in job["metadata"]:
            path = ROOT / item["path"]
            if not path.is_file():
                raise RuntimeError("frozen rollout metadata disappeared")
            metadata.append(read_json(path))
    if len(completed) != 36 or len(metadata) != 1080:
        raise RuntimeError("closed-loop matrix cardinality mismatch")
    reset_ids = {}
    for row in metadata:
        key = (row["task"], row["variant"], row["training_seed"])
        reset_ids[key] = reset_ids.get(key, set()) | {row["evaluation_reset_id"]}
    if len(reset_ids) != 36 or any(len(value) != 30 for value in reset_ids.values()):
        raise RuntimeError("closed-loop shared-reset coverage mismatch")
    final = {
        "schema": "tactile3d-unit.s4-3-closed-loop-rollouts.v1",
        "stage": "R12",
        "start_time": started_at,
        "end_time": now(),
        "jobs": completed,
        "rollouts": metadata,
        "completed_jobs": 36,
        "completed_rollouts": 1080,
        "rollout_performance_seen": True,
        "same_reset_ids_per_checkpoint": True,
        "status": "PASS",
    }
    atomic_json(ARTIFACT, final)
    print(json.dumps({"jobs": 36, "rollouts": 1080, "status": "PASS"}, sort_keys=True))


if __name__ == "__main__":
    main()
