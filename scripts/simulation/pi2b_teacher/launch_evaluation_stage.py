#!/usr/bin/env python3
"""Locked persistent supervisor for confirmation cache or representation evaluation."""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.simulation.pi2b_teacher.common import atomic_json, load_json  # noqa: E402
from scripts.simulation.pi2b_teacher.launch import compute_processes, gpu_snapshot  # noqa: E402

STAGES = {
    "cache": ("build_confirmation_cache.py", "s4_3_pi2b_teacher_confirmation_cache"),
    "evaluate": ("evaluate.py", "s4_3_pi2b_teacher_representation_evaluation"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--unit-checkpoint", type=Path)
    args = parser.parse_args()
    runtime = load_json(args.runtime_manifest)
    if args.stage == "cache" and args.unit_checkpoint is None:
        raise RuntimeError("cache stage requires --unit-checkpoint")
    coordination = Path(runtime["coordination_dir"])
    policy_request = coordination / "requests/policy.json"
    if policy_request.exists():
        value = load_json(policy_request)
        if value.get("state") == "EVAL_PENDING" or value.get("status") == "EVAL_PENDING":
            raise RuntimeError("policy formal evaluation is pending")
    local = load_json(Path(runtime["worktree_root"]) / ".local/config/pi2b_workspace.json")
    barrier = (coordination / "runtime_barrier.lock").open("a+")
    scheduler = (coordination / "scheduler.lock").open("a+")
    gpu_lock = (Path(local["common_git_dir"]) / f"tactile3d_unit_gpu{args.gpu}.lock").open("a+")
    script_name, job_id = STAGES[args.stage]
    lease_path = coordination / "leases" / f"{job_id}.json"
    request_path = coordination / "requests/teacher.json"
    selected = gpu_snapshot()[args.gpu]
    if int(selected["index"]) != args.gpu or int(selected["memory_used_mib"]) > 512 or int(selected["utilization_percent"]) > 5:
        raise RuntimeError(f"GPU{args.gpu} is not idle: {selected}")
    if any(row["gpu_uuid"] == selected["uuid"] for row in compute_processes()):
        raise RuntimeError(f"GPU{args.gpu} has an unrelated compute process")
    try:
        fcntl.flock(barrier, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(scheduler, fcntl.LOCK_EX)
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = Path(runtime["snapshot_root"])
        command = [
            runtime["python"],
            str(snapshot / "scripts/simulation/pi2b_teacher" / script_name),
            "--runtime-manifest",
            str(args.runtime_manifest),
            "--device",
            "cuda:0",
        ]
        if args.stage == "cache":
            command.extend(["--unit-checkpoint", str(args.unit_checkpoint)])
        env = os.environ.copy()
        env.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(args.gpu), "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONPATH": str(snapshot)})
        child = subprocess.Popen(command, cwd=snapshot, env=env)
        lease = {
            "schema": "tactile3d-unit.pi2b-gpu-lease.v1",
            "job_id": job_id,
            "track": "teacher",
            "state": "RUNNING",
            "mode": args.stage.upper(),
            "gpu_index": args.gpu,
            "gpu_uuid": selected["uuid"],
            "supervisor_pid": os.getpid(),
            "worker_pid": child.pid,
            "supervisor_start_time_utc": datetime.now(timezone.utc).isoformat(),
            "snapshot": str(snapshot),
            "freeze_identity": runtime["evaluation_freeze_identity"],
            "runtime_barrier": "shared",
            "command": command,
            "preflight_gpu": selected,
        }
        atomic_json(lease_path, lease)
        jobs = load_json(request_path).get("jobs", []) if request_path.exists() else []
        jobs = [job for job in jobs if job.get("job_id") != job_id]
        jobs.append({key: lease[key] for key in ("job_id", "state", "mode", "gpu_index", "gpu_uuid", "supervisor_pid", "worker_pid", "supervisor_start_time_utc", "snapshot", "freeze_identity")})
        atomic_json(request_path, {"schema": "tactile3d-unit.pi2b-teacher-request.v1", "track": "teacher", "state": "RUNNING", "jobs": jobs})
        fcntl.flock(scheduler, fcntl.LOCK_UN)

        def forward(signum, _frame):
            if child.poll() is None:
                child.send_signal(signum)

        signal.signal(signal.SIGTERM, forward)
        signal.signal(signal.SIGINT, forward)
        exit_code = child.wait()
        fcntl.flock(scheduler, fcntl.LOCK_EX)
        lease.update({"state": "COMPLETE" if exit_code == 0 else "FAILED", "exit_code": exit_code, "finished_utc": datetime.now(timezone.utc).isoformat()})
        atomic_json(lease_path, lease)
        jobs = load_json(request_path).get("jobs", []) if request_path.exists() else []
        jobs = [{**job, "state": lease["state"], "exit_code": exit_code} if job.get("job_id") == job_id else job for job in jobs]
        overall = "FAILED" if any(job.get("state") == "FAILED" for job in jobs) else ("COMPLETE" if jobs and all(job.get("state") == "COMPLETE" for job in jobs) else "RUNNING")
        atomic_json(request_path, {"schema": "tactile3d-unit.pi2b-teacher-request.v1", "track": "teacher", "state": overall, "jobs": jobs})
        fcntl.flock(scheduler, fcntl.LOCK_UN)
        raise SystemExit(exit_code)
    finally:
        for handle in (gpu_lock, barrier):
            try:
                fcntl.flock(handle, fcntl.LOCK_UN)
            except OSError:
                pass
        gpu_lock.close()
        scheduler.close()
        barrier.close()


if __name__ == "__main__":
    main()
