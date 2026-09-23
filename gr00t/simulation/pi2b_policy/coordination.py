"""Prompt1 lock and GPU inventory helpers for Track A."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
import subprocess
from typing import Any, TextIO

from .contract import Workspace


def open_lock(path: Path, *, shared: bool, nonblocking: bool = True) -> TextIO:
    handle = path.open("a+")
    operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
    if nonblocking:
        operation |= fcntl.LOCK_NB
    try:
        fcntl.flock(handle, operation)
    except BlockingIOError:
        handle.close()
        raise
    return handle


def gpu_snapshot() -> dict[str, list[str]]:
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


def gpu_inventory(snapshot: dict[str, list[str]]) -> dict[int, dict[str, Any]]:
    busy_uuids = {
        line.split(",", 1)[0].strip()
        for line in snapshot["compute_applications"]
        if line.strip()
    }
    result = {}
    for line in snapshot["inventory"]:
        fields = [part.strip() for part in line.split(",")]
        index = int(fields[0])
        result[index] = {
            "index": index,
            "uuid": fields[1],
            "name": fields[2],
            "memory_used_mib": int(fields[3]),
            "memory_total_mib": int(fields[4]),
            "utilization_percent": int(fields[5]),
            "compute_application": fields[1] in busy_uuids,
        }
    return result


def gpu_is_idle(index: int, *snapshots: dict[str, list[str]]) -> bool:
    for snapshot in snapshots:
        row = gpu_inventory(snapshot).get(index)
        if row is None:
            return False
        if row["memory_used_mib"] > 64 or row["utilization_percent"] > 5 or row["compute_application"]:
            return False
    return True


def coordination_paths(workspace: Workspace) -> dict[str, Path]:
    root = workspace.common_git_dir / "pi2b_coordination"
    return {
        "root": root,
        "runtime_barrier": root / "runtime_barrier.lock",
        "scheduler": root / "scheduler.lock",
        "request": root / "requests/policy.json",
        "teacher_request": root / "requests/teacher.json",
        "leases": root / "leases",
    }


def teacher_has_pending_request(workspace: Workspace) -> bool:
    path = coordination_paths(workspace)["teacher_request"]
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return True
    states = {
        str(payload.get(key, "")).upper()
        for key in ("status", "state", "request_state")
    }
    terminal = {"", "DONE", "COMPLETE", "COMPLETED", "CANCELLED", "FAILED", "IDLE"}
    return any(value not in terminal for value in states)
