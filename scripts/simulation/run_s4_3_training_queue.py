#!/usr/bin/env python3
"""Schedule the frozen 36-job S4.3 ACT matrix on genuinely idle GPUs."""

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

from gr00t.simulation.s4_3_training import (  # noqa: E402
    TASKS,
    TRAINING_SEEDS,
    VARIANTS,
    atomic_json,
    read_json,
)

EXPERIMENT_ROOT = ROOT / ".local/experiments/simulation/s4_3_restart"
LOG_ROOT = ROOT / ".local/logs/simulation/s4_3_restart/training_queue"
ARTIFACT = ROOT / ".local/artifacts/simulation/s4_3_restart/training_jobs.json"
PYTHON = Path(sys.executable)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gpu_state(gpu: int) -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    memory, utilization = [int(value.strip()) for value in query.split(",")]
    processes = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "memory_mib": memory,
        "utilization_percent": utilization,
        "compute_processes": 0 if not processes else len(processes.splitlines()),
        "idle": not processes and memory <= 512 and utilization <= 5,
    }


def summary_path(job: tuple[str, str, int]) -> Path:
    task, variant, seed = job
    return EXPERIMENT_ROOT / task / variant / f"seed_{seed}" / "summary.json"


def existing_complete(job: tuple[str, str, int]) -> dict[str, Any] | None:
    path = summary_path(job)
    if not path.is_file():
        return None
    value = read_json(path)
    expected = (value.get("task"), value.get("variant"), value.get("training_seed"))
    if value.get("status") != "PASS" or expected != job or value.get("exit_code") != 0:
        return None
    checkpoint = ROOT / value["checkpoint"]
    if not checkpoint.is_file():
        return None
    return value


def main() -> None:
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
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    queued = [
        (task, variant, seed) for task in TASKS for variant in VARIANTS for seed in TRAINING_SEEDS
    ]
    records: list[dict[str, Any]] = []
    for job in list(queued):
        prior = existing_complete(job)
        if prior is not None:
            records.append({**prior, "scheduler_disposition": "EXISTING_CANONICAL"})
            queued.remove(job)
    running: dict[int, dict[str, Any]] = {}
    attempts: dict[tuple[str, str, int], int] = {}
    launched_at = now()

    def save() -> None:
        atomic_json(
            ARTIFACT,
            {
                "schema": "tactile3d-unit.s4-3-training-jobs.v1",
                "stage": "R9",
                "queue_start_time": launched_at,
                "queue_end_time": None,
                "expected_jobs": 36,
                "completed_jobs": len(records),
                "queued_jobs": [list(job) for job in queued],
                "running_jobs": [value["record"] for value in running.values()],
                "jobs": records,
                "status": "RUNNING",
            },
        )

    save()
    while queued or running:
        for gpu, item in list(running.items()):
            return_code = item["process"].poll()
            if return_code is None:
                continue
            item["log"].close()
            item["lock"].close()
            job = item["job"]
            record = item["record"]
            record["end_time"] = now()
            record["exit_code"] = return_code
            if return_code == 0:
                summary = existing_complete(job)
                if summary is None:
                    raise RuntimeError(f"successful process lacks canonical summary: {job}")
                records.append({**summary, "scheduler_disposition": "COMPLETED"})
            elif return_code == 75 and attempts[job] == 1:
                queued.append(job)
                record["scheduler_disposition"] = "RETRY_INFRASTRUCTURE_ONCE"
                records.append(record)
            else:
                record["scheduler_disposition"] = "FAILED_NO_SCIENTIFIC_RETRY"
                records.append(record)
                save()
                raise SystemExit(f"S4_3_2_ACT_TRAINING_FAIL: {job}, exit={return_code}")
            del running[gpu]
            save()

        for gpu in range(4):
            if not queued or gpu in running:
                continue
            if not gpu_state(gpu)["idle"]:
                continue
            lock_path = common_git / f"tactile3d_unit_gpu{gpu}.lock"
            lock = lock_path.open("a+")
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            recheck = gpu_state(gpu)
            if not recheck["idle"]:
                lock.close()
                continue
            job = queued.pop(0)
            task, variant, seed = job
            attempts[job] = attempts.get(job, 0) + 1
            log_path = LOG_ROOT / f"{task}_{variant}_seed{seed}_attempt{attempts[job]}.log"
            log_handle = log_path.open("w", encoding="utf-8")
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
                str(PYTHON),
                str(ROOT / "scripts/simulation/train_s4_3_act.py"),
                "--task",
                task,
                "--variant",
                variant,
                "--seed",
                str(seed),
                "--device",
                "cuda:0",
            ]
            record = {
                "task": task,
                "variant": variant,
                "training_seed": seed,
                "physical_gpu": gpu,
                "logical_device": "cuda:0",
                "start_time": now(),
                "end_time": None,
                "exit_code": None,
                "attempt": attempts[job],
                "prelaunch_gpu_state": recheck,
                "log": str(log_path.relative_to(ROOT)),
            }
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            running[gpu] = {
                "job": job,
                "process": process,
                "lock": lock,
                "log": log_handle,
                "record": record,
            }
            print(json.dumps({**record, "status": "LAUNCHED"}, sort_keys=True), flush=True)
            save()
        if queued or running:
            time.sleep(10)

    successful = [row for row in records if row.get("status") == "PASS"]
    if len(successful) != 36:
        raise SystemExit("S4_3_2_ACT_TRAINING_FAIL: canonical checkpoint count is not 36")
    final = {
        "schema": "tactile3d-unit.s4-3-training-jobs.v1",
        "stage": "R9",
        "queue_start_time": launched_at,
        "queue_end_time": now(),
        "expected_jobs": 36,
        "completed_jobs": 36,
        "canonical_checkpoints": 36,
        "jobs": successful,
        "failed_attempts": [row for row in records if row.get("status") != "PASS"],
        "status": "PASS",
    }
    atomic_json(ARTIFACT, final)
    print(json.dumps({"canonical_checkpoints": 36, "status": "PASS"}, sort_keys=True))


if __name__ == "__main__":
    main()
