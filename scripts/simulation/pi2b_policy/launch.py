#!/usr/bin/env python3
"""Launch a static wave of frozen Track A runs in persistent tmux sessions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.pi2b_policy.contract import Workspace
from gr00t.simulation.pi2b_policy.coordination import (
    coordination_paths,
    gpu_inventory,
    gpu_is_idle,
    gpu_snapshot,
    open_lock,
    teacher_has_pending_request,
)
from gr00t.simulation.pi2b_policy.integrity import sha256_file


ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2b_policy"
FREEZE = ARTIFACTS / "training_protocol_freeze.json"
RUNS = ARTIFACTS / "training_runs.json"
LAUNCHES = ARTIFACTS / "launches"
STEP_PATTERN = re.compile(r"Step\s+(\d+):")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def session_name(run_id: str) -> str:
    return f"pi2bA_{run_id}"


def tmux_alive(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def status_path(workspace: Workspace, run_id: str) -> Path:
    return workspace.write_root / "status/training" / f"{run_id}.json"


def heartbeat_path(workspace: Workspace, run_id: str) -> Path:
    return workspace.write_root / "status/training" / f"{run_id}.heartbeat.json"


def log_path(workspace: Workspace, run_id: str) -> Path:
    return workspace.write_root / "logs/training" / f"{run_id}.log"


def load_frozen() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    freeze = json.loads(FREEZE.read_text())
    manifest = json.loads(RUNS.read_text())
    if freeze.get("status") != "FROZEN_BEFORE_TRAINING":
        raise SystemExit("training protocol is not frozen")
    if freeze.get("training_runs_sha256") != sha256_file(RUNS):
        raise SystemExit("training_runs.json changed after freeze")
    rows = manifest.get("rows", [])
    if len(rows) != 10 or manifest.get("authorized_runs") != 10:
        raise SystemExit("frozen run cardinality is not ten")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != freeze.get("git_head"):
        raise SystemExit("current HEAD differs from the frozen training code")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise SystemExit("tracked worktree must be clean before launch")
    return freeze, rows


def select_runs(workspace: Workspace, rows: list[dict[str, Any]], requested: list[str], count: int) -> list[dict[str, Any]]:
    known = {row["run_id"]: row for row in rows}
    if requested:
        unknown = [run_id for run_id in requested if run_id not in known]
        if unknown:
            raise SystemExit(f"unknown frozen run IDs: {unknown}")
        candidates = [known[run_id] for run_id in requested]
    else:
        candidates = rows
    selected = []
    for row in candidates:
        run_id = row["run_id"]
        paths = (
            status_path(workspace, run_id),
            heartbeat_path(workspace, run_id),
            log_path(workspace, run_id),
            Path(row["output_path"]),
            LAUNCHES / f"{run_id}.json",
        )
        if any(path.exists() for path in paths) or tmux_alive(session_name(run_id)):
            if requested:
                raise SystemExit(f"refusing to duplicate or overwrite {run_id}")
            continue
        selected.append(row)
        if len(selected) == count:
            break
    return selected


def launch(args: argparse.Namespace) -> None:
    workspace = Workspace.load(ROOT)
    _, rows = load_frozen()
    teacher_pending = teacher_has_pending_request(workspace)
    maximum = min(args.max_jobs, 2 if teacher_pending else 4)
    selected_rows = select_runs(workspace, rows, args.run, maximum)
    if not selected_rows:
        raise SystemExit("no unstarted frozen run is eligible for this wave")
    first = gpu_snapshot()
    time.sleep(2)
    second = gpu_snapshot()
    eligible = [index for index in range(4) if gpu_is_idle(index, first, second)]
    if len(eligible) < len(selected_rows):
        selected_rows = selected_rows[: len(eligible)]
    if not selected_rows:
        raise SystemExit("no genuinely idle GPU is available")
    selected_gpus = eligible[: len(selected_rows)]
    paths = coordination_paths(workspace)
    barrier = open_lock(paths["runtime_barrier"], shared=True)
    scheduler = open_lock(paths["scheduler"], shared=False)
    probes = []
    try:
        for gpu in selected_gpus:
            handle = open_lock(workspace.common_git_dir / f"tactile3d_unit_gpu{gpu}.lock", shared=False)
            probes.append(handle)
        third = gpu_snapshot()
        if not all(gpu_is_idle(gpu, third) for gpu in selected_gpus):
            raise SystemExit("a selected GPU became busy after advisory lock acquisition")
    finally:
        for handle in reversed(probes):
            handle.close()
        scheduler.close()
        barrier.close()
    inventory = gpu_inventory(third)
    assignments = [
        {
            "run_id": row["run_id"],
            "model_id": row["model_id"],
            "seed": row["seed"],
            "gpu_index": gpu,
            "gpu_uuid": inventory[gpu]["uuid"],
            "session": session_name(row["run_id"]),
            "log_path": str(log_path(workspace, row["run_id"])),
            "status_path": str(status_path(workspace, row["run_id"])),
            "final_checkpoint": row["final_checkpoint"],
        }
        for row, gpu in zip(selected_rows, selected_gpus, strict=True)
    ]
    request = {
        "schema": "tactile3d-unit.pi2b-coordination-policy-request.v1",
        "track": "policy",
        "status": "LAUNCHING",
        "kind": "TRAINING_WAVE",
        "created_at_utc": now(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "runtime_barrier": "shared",
        "jobs": assignments,
        "gpu_count": len(assignments),
        "teacher_pending_at_selection": teacher_pending,
        "automatic_next_wave": False,
        "automatic_formal_evaluation": False,
    }
    scheduler = open_lock(paths["scheduler"], shared=False)
    try:
        atomic_json(paths["request"], request)
    finally:
        scheduler.close()
    wave = ARTIFACTS / "launches" / f"wave_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    atomic_json(
        wave,
        {
            **request,
            "gpu_snapshots": [first, second, third],
            "openpi_python": str(args.python.resolve()),
        },
    )
    for row, assignment in zip(selected_rows, assignments, strict=True):
        atomic_json(
            LAUNCHES / f"{row['run_id']}.json",
            {
                **assignment,
                "schema": "tactile3d-unit.s4-3-pi2b-policy-training-launch.v1",
                "status": "LAUNCHING",
                "created_at_utc": now(),
                "config_sha256": row["config_sha256"],
                "training_runs_sha256": sha256_file(RUNS),
                "openpi_python": str(args.python.resolve()),
                "command": [
                    str(args.python.resolve()),
                    "scripts/simulation/pi2b_policy/train.py",
                    "--model-id",
                    row["model_id"],
                    "--seed",
                    str(row["seed"]),
                    "--fsdp-devices",
                    "1",
                ],
                "automatic_successor": False,
            },
        )
        command = shlex.join(
            [
                str(args.python.resolve()),
                str(Path(__file__).resolve()),
                "supervise",
                "--run-id",
                row["run_id"],
                "--gpu",
                str(assignment["gpu_index"]),
                "--uuid",
                assignment["gpu_uuid"],
                "--python",
                str(args.python.resolve()),
            ]
        )
        result = subprocess.run(
            ["tmux", "new-session", "-d", "-s", assignment["session"], command],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or f"failed to create {assignment['session']}")
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if all(status_path(workspace, item["run_id"]).exists() for item in assignments):
            break
        time.sleep(1)
    missing = [item["run_id"] for item in assignments if not status_path(workspace, item["run_id"]).exists()]
    if missing:
        raise RuntimeError(f"supervisors did not write status within 60 seconds: {missing}")
    print(json.dumps({"wave": str(wave), "assignments": assignments}, indent=2, sort_keys=True))


def latest_step(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open("rb") as source:
        source.seek(0, os.SEEK_END)
        size = source.tell()
        source.seek(max(0, size - 256 * 1024))
        text = source.read().decode("utf-8", errors="replace")
    matches = STEP_PATTERN.findall(text)
    return int(matches[-1]) if matches else None


def supervise(args: argparse.Namespace) -> None:
    workspace = Workspace.load(ROOT)
    freeze, rows = load_frozen()
    matches = [row for row in rows if row["run_id"] == args.run_id]
    if len(matches) != 1:
        raise SystemExit("run ID is not present exactly once in the frozen manifest")
    row = matches[0]
    launch_manifest = json.loads((LAUNCHES / f"{args.run_id}.json").read_text())
    if launch_manifest["config_sha256"] != row["config_sha256"]:
        raise SystemExit("launch identity differs from frozen run")
    status = status_path(workspace, args.run_id)
    heartbeat = heartbeat_path(workspace, args.run_id)
    log = log_path(workspace, args.run_id)
    if status.exists() or heartbeat.exists() or log.exists() or Path(row["output_path"]).exists():
        raise SystemExit("refusing to overwrite existing run state")
    status.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    paths = coordination_paths(workspace)
    barrier = open_lock(paths["runtime_barrier"], shared=True)
    scheduler = open_lock(paths["scheduler"], shared=False)
    gpu_lock = None
    lease = paths["leases"] / f"policy_{args.run_id}.json"
    try:
        gpu_lock = open_lock(
            workspace.common_git_dir / f"tactile3d_unit_gpu{args.gpu}.lock", shared=False
        )
        snapshot = gpu_snapshot()
        inventory = gpu_inventory(snapshot)
        actual = inventory.get(args.gpu)
        if actual is None or actual["uuid"] != args.uuid or not gpu_is_idle(args.gpu, snapshot):
            raise SystemExit("GPU identity or occupancy changed after lock acquisition")
        running = {
            "schema": "tactile3d-unit.s4-3-pi2b-policy-job-status.v1",
            "state": "RUNNING",
            "exit_code": None,
            "updated_at_utc": now(),
            "run_id": args.run_id,
            "model_id": row["model_id"],
            "seed": row["seed"],
            "supervisor_pid": os.getpid(),
            "supervisor_start_time": subprocess.check_output(
                ["ps", "-p", str(os.getpid()), "-o", "lstart="], text=True
            ).strip(),
            "session": session_name(args.run_id),
            "physical_gpu_index": args.gpu,
            "physical_gpu_uuid": args.uuid,
            "logical_cuda_device": 0,
            "fsdp_devices": 1,
            "global_batch_size": 32,
            "optimizer_steps": 30000,
            "current_logged_step": None,
            "log_path": str(log),
            "checkpoint_root": row["output_path"],
            "expected_final_checkpoint": row["final_checkpoint"],
            "git_head": freeze["git_head"],
            "config_sha256": row["config_sha256"],
            "runtime_barrier": "shared",
        }
        atomic_json(status, running)
        atomic_json(heartbeat, running)
        atomic_json(
            lease,
            {
                **running,
                "track": "policy",
                "job_id": args.run_id,
                "kind": "TRAINING",
                "gpus": [args.gpu],
                "pid": os.getpid(),
                "status": "ACTIVE",
            },
        )
    finally:
        scheduler.close()
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "WANDB_MODE": "offline",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(ROOT),
        }
    )
    command = [
        str(args.python.resolve()),
        str(ROOT / "scripts/simulation/pi2b_policy/train.py"),
        "--model-id",
        row["model_id"],
        "--seed",
        str(row["seed"]),
        "--fsdp-devices",
        "1",
    ]
    with log.open("x") as output:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            current = latest_step(log)
            update = {**running, "updated_at_utc": now(), "current_logged_step": current}
            atomic_json(status, update)
            atomic_json(heartbeat, update)
            time.sleep(60)
        exit_code = int(process.returncode)
    final = {
        **running,
        "state": "DONE" if exit_code == 0 else "FAILED",
        "exit_code": exit_code,
        "updated_at_utc": now(),
        "current_logged_step": latest_step(log),
        "message": "training process exited; completion audit required"
        if exit_code == 0
        else "training process exited nonzero; preserve attempt and log",
    }
    scheduler = open_lock(paths["scheduler"], shared=False)
    try:
        atomic_json(status, final)
        atomic_json(heartbeat, final)
        atomic_json(
            lease,
            {
                **json.loads(lease.read_text()),
                "status": "DONE" if exit_code == 0 else "FAILED",
                "state": final["state"],
                "exit_code": exit_code,
                "updated_at_utc": final["updated_at_utc"],
            },
        )
    finally:
        scheduler.close()
        if gpu_lock is not None:
            gpu_lock.close()
        barrier.close()
    raise SystemExit(exit_code)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--python", type=Path, required=True)
    launch_parser.add_argument("--max-jobs", type=int, choices=(1, 2, 3, 4), default=2)
    launch_parser.add_argument("--run", action="append", default=[])
    supervise_parser = sub.add_parser("supervise")
    supervise_parser.add_argument("--run-id", required=True)
    supervise_parser.add_argument("--gpu", type=int, choices=(0, 1, 2, 3), required=True)
    supervise_parser.add_argument("--uuid", required=True)
    supervise_parser.add_argument("--python", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "launch":
        launch(arguments)
    else:
        supervise(arguments)
