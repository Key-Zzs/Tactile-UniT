#!/usr/bin/env python3
"""Persistent locked supervisor for the frozen confirmation generation job."""

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


def resolve_dexjoco_python(runtime: dict[str, object]) -> Path:
    """Resolve the simulator interpreter without embedding a host-specific path."""
    explicit = runtime.get("dexjoco_python")
    if explicit is not None:
        candidate = Path(str(explicit))
    else:
        unit_python = Path(str(runtime["python"]))
        if len(unit_python.parents) < 3:
            raise RuntimeError(f"cannot derive Conda env root from unit interpreter: {unit_python}")
        candidate = unit_python.parents[2] / "tactile-unit-dexjoco/bin/python"
    if not candidate.is_file():
        raise RuntimeError(f"DexJoCo interpreter is unavailable: {candidate}")
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    args = parser.parse_args()
    runtime = load_json(args.runtime_manifest)
    coordination = Path(runtime["coordination_dir"])
    policy_request = coordination / "requests/policy.json"
    if policy_request.exists():
        value = load_json(policy_request)
        if value.get("state") == "EVAL_PENDING" or value.get("status") == "EVAL_PENDING":
            raise RuntimeError("policy formal evaluation is pending")
    local = load_json(Path(runtime["worktree_root"]) / ".local/config/pi2b_workspace.json")
    gpu_lock_path = Path(local["common_git_dir"]) / f"tactile3d_unit_gpu{args.gpu}.lock"
    barrier_path = coordination / "runtime_barrier.lock"
    scheduler_path = coordination / "scheduler.lock"
    request_path = coordination / "requests/teacher.json"
    job_id = "s4_3_pi2b_teacher_confirmation_generation"
    lease_path = coordination / "leases" / f"{job_id}.json"
    selected = gpu_snapshot()[args.gpu]
    if int(selected["index"]) != args.gpu or int(selected["memory_used_mib"]) > 512 or int(selected["utilization_percent"]) > 5:
        raise RuntimeError(f"GPU{args.gpu} is not idle enough: {selected}")
    if any(row["gpu_uuid"] == selected["uuid"] for row in compute_processes()):
        raise RuntimeError(f"GPU{args.gpu} has an unrelated compute process")
    barrier = barrier_path.open("a+")
    scheduler = scheduler_path.open("a+")
    gpu_lock = gpu_lock_path.open("a+")
    try:
        fcntl.flock(barrier, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(scheduler, fcntl.LOCK_EX)
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        after_lock = gpu_snapshot()[args.gpu]
        if int(after_lock["memory_used_mib"]) > 512 or any(row["gpu_uuid"] == after_lock["uuid"] for row in compute_processes()):
            raise RuntimeError(f"GPU{args.gpu} became busy after lock acquisition")
        snapshot = Path(runtime["snapshot_root"])
        dex_python = resolve_dexjoco_python(runtime)
        command = [str(dex_python), str(snapshot / "scripts/simulation/pi2b_teacher/build_confirmation.py")]
        env = os.environ.copy()
        env.pop("DISPLAY", None)
        env.update(
            {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": str(args.gpu),
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
                "MUJOCO_EGL_DEVICE_ID": "0",
                "PYTHONPATH": str(snapshot),
                "PI2B_TEACHER_RUNTIME_MANIFEST": str(args.runtime_manifest),
            }
        )
        child = subprocess.Popen(command, cwd=snapshot, env=env)
        lease = {
            "schema": "tactile3d-unit.pi2b-gpu-lease.v1",
            "job_id": job_id,
            "track": "teacher",
            "state": "RUNNING",
            "mode": "CONFIRMATION_GENERATION",
            "gpu_index": args.gpu,
            "gpu_uuid": selected["uuid"],
            "supervisor_pid": os.getpid(),
            "worker_pid": child.pid,
            "supervisor_start_time_utc": datetime.now(timezone.utc).isoformat(),
            "snapshot": str(snapshot),
            "freeze_identity": runtime["freeze_identity"],
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
        lease["state"] = "COMPLETE" if exit_code == 0 else "FAILED"
        lease["exit_code"] = exit_code
        lease["finished_utc"] = datetime.now(timezone.utc).isoformat()
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
