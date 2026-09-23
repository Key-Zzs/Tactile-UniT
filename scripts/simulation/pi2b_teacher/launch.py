#!/usr/bin/env python3
"""Persistent supervisor for one formal matched-teacher GPU run."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.simulation.pi2b_teacher.common import atomic_json, load_json  # noqa: E402


def gpu_snapshot() -> list[dict[str, str]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in output.strip().splitlines():
        index, uuid, name, used, total, utilization = [value.strip() for value in line.split(",")]
        rows.append({"index": index, "uuid": uuid, "name": name, "memory_used_mib": used, "memory_total_mib": total, "utilization_percent": utilization})
    return rows


def compute_processes() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0 or not completed.stdout.strip():
        return []
    result = []
    for line in completed.stdout.strip().splitlines():
        uuid, pid, name, used = [value.strip() for value in line.split(",", 3)]
        result.append({"gpu_uuid": uuid, "pid": pid, "process_name": name, "used_gpu_memory_mib": used})
    return result


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("T_VA_match", "T_VAC_match"), required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    runtime = load_json(args.runtime_manifest)
    coordination = Path(runtime["coordination_dir"])
    policy_request = coordination / "requests/policy.json"
    if policy_request.exists():
        value = load_json(policy_request)
        if value.get("state") == "EVAL_PENDING" or value.get("status") == "EVAL_PENDING":
            raise RuntimeError("policy formal evaluation is pending")
    barrier_path = coordination / "runtime_barrier.lock"
    scheduler_path = coordination / "scheduler.lock"
    contract = load_json(Path(runtime["worktree_root"]) / ".local/config/pi2b_workspace.json")
    common_git = Path(contract["common_git_dir"])
    gpu_lock_path = common_git / f"tactile3d_unit_gpu{args.gpu}.lock"
    job_id = f"s4_3_pi2b_teacher_{args.mode.lower()}_seed42"
    lease_path = coordination / "leases" / f"{job_id}.json"
    request_path = coordination / "requests/teacher.json"
    before = gpu_snapshot()
    selected = before[args.gpu]
    if int(selected["index"]) != args.gpu:
        raise RuntimeError("nvidia-smi GPU ordering mismatch")
    if int(selected["memory_used_mib"]) > 512 or int(selected["utilization_percent"]) > 5:
        raise RuntimeError(f"GPU{args.gpu} is not idle enough for formal allocation: {selected}")
    if any(row["gpu_uuid"] == selected["uuid"] for row in compute_processes()):
        raise RuntimeError(f"GPU{args.gpu} has an unrelated compute process")
    barrier = barrier_path.open("a+")
    scheduler = scheduler_path.open("a+")
    gpu_lock = gpu_lock_path.open("a+")
    try:
        fcntl.flock(barrier, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(scheduler, fcntl.LOCK_EX)
        try:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"GPU{args.gpu} coordination lock is busy") from error
        running = []
        if request_path.exists():
            running = [job for job in load_json(request_path).get("jobs", []) if job.get("state") == "RUNNING" and pid_alive(int(job.get("supervisor_pid", -1)))]
        if len(running) >= 2:
            raise RuntimeError("teacher track already has two running GPU jobs")
        if any(job.get("job_id") == job_id for job in running):
            raise RuntimeError(f"job already running: {job_id}")
        after_lock = gpu_snapshot()[args.gpu]
        if int(after_lock["memory_used_mib"]) > 512 or any(row["gpu_uuid"] == after_lock["uuid"] for row in compute_processes()):
            raise RuntimeError(f"GPU{args.gpu} became busy after lock acquisition")
        snapshot = Path(runtime["snapshot_root"])
        python = Path(runtime["python"])
        command = [str(python), str(snapshot / "scripts/simulation/pi2b_teacher/train.py"), "--mode", args.mode, "--runtime-manifest", str(args.runtime_manifest)]
        if args.resume:
            command.append("--resume")
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        env["PYTHONPATH"] = str(snapshot)
        child = subprocess.Popen(command, cwd=snapshot, env=env)
        started = datetime.now(timezone.utc).isoformat()
        lease = {
            "schema": "tactile3d-unit.pi2b-gpu-lease.v1",
            "job_id": job_id,
            "track": "teacher",
            "state": "RUNNING",
            "mode": args.mode,
            "seed": 42,
            "gpu_index": args.gpu,
            "gpu_uuid": selected["uuid"],
            "supervisor_pid": os.getpid(),
            "worker_pid": child.pid,
            "supervisor_start_time_utc": started,
            "snapshot": str(snapshot),
            "freeze_identity": runtime["freeze_identity"],
            "runtime_barrier": "shared",
            "command": command,
            "preflight_gpu": selected,
        }
        atomic_json(lease_path, lease)
        running.append({key: lease[key] for key in ("job_id", "state", "mode", "seed", "gpu_index", "gpu_uuid", "supervisor_pid", "worker_pid", "supervisor_start_time_utc", "snapshot", "freeze_identity")})
        atomic_json(request_path, {"schema": "tactile3d-unit.pi2b-teacher-request.v1", "track": "teacher", "state": "RUNNING", "jobs": running})
        fcntl.flock(scheduler, fcntl.LOCK_UN)

        def forward(signum, _frame):
            if child.poll() is None:
                child.send_signal(signum)

        signal.signal(signal.SIGTERM, forward)
        signal.signal(signal.SIGINT, forward)
        exit_code = child.wait()
        fcntl.flock(scheduler, fcntl.LOCK_EX)
        lease["state"] = "COMPLETE" if exit_code == 0 else "FAILED"
        lease["exit_code"] = exit_code
        lease["finished_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(lease_path, lease)
        jobs = []
        if request_path.exists():
            for job in load_json(request_path).get("jobs", []):
                if job.get("job_id") == job_id:
                    job = {**job, "state": lease["state"], "exit_code": exit_code}
                jobs.append(job)
        overall = "FAILED" if any(job.get("state") == "FAILED" for job in jobs) else ("COMPLETE" if jobs and all(job.get("state") == "COMPLETE" for job in jobs) else "RUNNING")
        atomic_json(request_path, {"schema": "tactile3d-unit.pi2b-teacher-request.v1", "track": "teacher", "state": overall, "jobs": jobs})
        fcntl.flock(scheduler, fcntl.LOCK_UN)
        raise SystemExit(exit_code)
    finally:
        try:
            fcntl.flock(gpu_lock, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            fcntl.flock(barrier, fcntl.LOCK_UN)
        except OSError:
            pass
        gpu_lock.close()
        scheduler.close()
        barrier.close()


if __name__ == "__main__":
    main()
