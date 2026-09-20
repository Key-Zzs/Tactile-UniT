#!/usr/bin/env python3
"""Read-only status summary for the active S4.3-PI2N batch.

This command does not load a model, hash a checkpoint tree, or mutate any
runtime/artifact file.  A finished trainer is deliberately reported as
EXITED_UNVERIFIED until a separate cold-load completion audit has passed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2n"
DEFAULT_RUN_ROOT = Path(
    "/mnt/ugreen_nas/storage/UniT_storage/experiments/simulation/s4_3_pi2n"
)
HEARTBEAT_MAX_AGE_SECONDS = 180.0


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    experiment: str
    session: str
    completion_manifest: str


MODEL_SPECS = (
    ModelSpec(
        "B_VA27",
        "s43_pi2n_b_va27_seed42",
        "s43_pi2n_b_va27_s42",
        "b_va27_checkpoint_manifest.json",
    ),
    ModelSpec(
        "B_VAC_V",
        "s43_pi2n_b_vac_v_seed42",
        "s43_pi2n_b_vac_v_s42",
        "b_vac_v_checkpoint_manifest.json",
    ),
)


def read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        return None, "missing"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return None, f"{type(error).__name__}: {error}"
    if not isinstance(value, dict):
        return None, "top-level JSON value is not an object"
    return value, None


def parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def heartbeat_age_seconds(payload: dict[str, Any] | None, now: datetime) -> float | None:
    timestamp = parse_utc(payload.get("updated_at_utc") if payload else None)
    if timestamp is None:
        return None
    return max(0.0, (now - timestamp).total_seconds())


def proc_identity(pid: Any) -> dict[str, Any]:
    if not isinstance(pid, int) or pid <= 0:
        return {"pid": pid, "live": False, "error": "invalid PID"}
    process = Path("/proc") / str(pid)
    try:
        stat = (process / "stat").read_text()
        close = stat.rfind(")")
        fields = stat[close + 2 :].split()
        start_ticks = int(fields[19])
        boot_time = next(
            int(line.split()[1])
            for line in Path("/proc/stat").read_text().splitlines()
            if line.startswith("btime ")
        )
        clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        started = datetime.fromtimestamp(
            boot_time + start_ticks / clock_ticks, tz=timezone.utc
        )
        command = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            errors="replace"
        ).strip()
        executable = os.readlink(process / "exe")
        return {
            "pid": pid,
            "live": True,
            "start_time_utc": started.isoformat(),
            "executable": executable,
            "cmdline": command,
        }
    except (FileNotFoundError, ProcessLookupError):
        return {"pid": pid, "live": False, "error": "process not present"}
    except (OSError, StopIteration, ValueError) as error:
        return {"pid": pid, "live": False, "error": f"{type(error).__name__}: {error}"}


def process_table() -> dict[int, tuple[int, str]]:
    rows: dict[int, tuple[int, str]] = {}
    for candidate in Path("/proc").iterdir():
        if not candidate.name.isdigit():
            continue
        try:
            stat = (candidate / "stat").read_text()
            close = stat.rfind(")")
            fields = stat[close + 2 :].split()
            parent = int(fields[1])
            command = (candidate / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            ).strip()
            rows[int(candidate.name)] = (parent, command)
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
    return rows


def descendants(parent: int, table: dict[int, tuple[int, str]]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    frontier = [parent]
    visited = {parent}
    while frontier:
        current = frontier.pop()
        for pid, (ppid, command) in table.items():
            if ppid != current or pid in visited:
                continue
            visited.add(pid)
            frontier.append(pid)
            found.append({"pid": pid, "ppid": ppid, "cmdline": command})
    return sorted(found, key=lambda row: row["pid"])


def tmux_session_exists(name: str) -> bool:
    try:
        return subprocess.run(
            ["tmux", "has-session", "-t", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
    except FileNotFoundError:
        return False


def tail_bytes(path: Path, maximum: int = 1024 * 1024) -> str:
    try:
        with path.open("rb") as source:
            source.seek(0, os.SEEK_END)
            size = source.tell()
            source.seek(max(0, size - maximum))
            return source.read().decode(errors="replace")
    except OSError:
        return ""


def training_progress(log_path: Path) -> dict[str, Any]:
    text = tail_bytes(log_path).replace("\r", "\n")
    progress = re.findall(
        r"Progress on:\s*([^\s]+/[^\s]+)\s+rate:([^\s]+)\s+remaining:([^\s]+)",
        text,
    )
    steps = [int(value) for value in re.findall(r"(?:^|\n)Step\s+(\d+):", text)]
    if not progress and not steps:
        return {"available": False}
    result: dict[str, Any] = {"available": True}
    if progress:
        display, rate, remaining = progress[-1]
        result.update(
            {"display": display, "rate": rate, "remaining": remaining}
        )
    if steps:
        result["last_diagnostic_step"] = steps[-1]
    return result


def completion_manifest_valid(
    payload: dict[str, Any] | None, spec: ModelSpec, final_checkpoint: Path
) -> bool:
    if not payload or payload.get("status") != "PASS":
        return False
    if payload.get("model_id") != spec.model_id:
        return False
    if payload.get("optimizer_steps") != 30_000:
        return False
    recorded = payload.get("checkpoint")
    if not isinstance(recorded, str) or not (
        recorded.endswith(f"/{spec.experiment}/29999") or recorded == str(final_checkpoint)
    ):
        return False
    return isinstance(payload.get("checkpoint_tree_sha256"), str) and len(
        payload["checkpoint_tree_sha256"]
    ) == 64


def training_freeze_valid(payload: dict[str, Any] | None, spec: ModelSpec) -> bool:
    return bool(
        payload
        and payload.get("status") == "FROZEN_BEFORE_TRAINING"
        and payload.get("model_id") == spec.model_id
        and payload.get("seed") == 42
        and payload.get("steps") == 30_000
        and payload.get("global_batch_size") == 32
        and payload.get("final_checkpoint_step") == 29_999
        and payload.get("checkpoint_selection") == "FINAL_ONLY"
        and payload.get("base_manifest_status") == "PASS"
        and isinstance(payload.get("target_sha256"), str)
        and len(payload["target_sha256"]) == 64
        and isinstance(payload.get("source_snapshot_sha256"), str)
        and len(payload["source_snapshot_sha256"]) == 64
    )


def small_file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def classify_run(
    *,
    job: dict[str, Any] | None,
    heartbeat: dict[str, Any] | None,
    heartbeat_age: float | None,
    supervisor: dict[str, Any],
    matching_trainers: int,
    runtime_identity_matches: bool,
    training_identity_frozen: bool,
    final_checkpoint_present: bool,
    completion_audit_passed: bool,
) -> tuple[str, list[str]]:
    gates = {
        "job_running": bool(job and job.get("state") == "RUNNING"),
        "heartbeat_running": bool(heartbeat and heartbeat.get("state") == "RUNNING"),
        "heartbeat_fresh": heartbeat_age is not None
        and heartbeat_age <= HEARTBEAT_MAX_AGE_SECONDS,
        "supervisor_live": bool(supervisor.get("live")),
        "one_matching_trainer": matching_trainers == 1,
        "runtime_identity_matches": runtime_identity_matches,
        "training_identity_frozen": training_identity_frozen,
    }
    if all(gates.values()):
        return "RUNNING", [name for name, value in gates.items() if not value]
    complete = bool(
        job
        and heartbeat
        and job.get("state") == "DONE"
        and heartbeat.get("state") == "DONE"
        and job.get("exit_code") == 0
        and heartbeat.get("exit_code") == 0
        and runtime_identity_matches
        and training_identity_frozen
        and final_checkpoint_present
        and completion_audit_passed
    )
    if complete:
        return "COMPLETE_VERIFIED", []
    return "EXITED_UNVERIFIED", [name for name, value in gates.items() if not value]


def mount_status(run_root: Path) -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            ["findmnt", "-rn", "-T", str(run_root), "-o", "TARGET,SOURCE,FSTYPE"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return {"status": "FAIL", "error": str(error)}
    rows = [line.split(maxsplit=2) for line in output.splitlines() if line.strip()]
    nfs = [row for row in rows if len(row) == 3 and row[2].startswith("nfs")]
    return {
        "status": "PASS" if nfs else "FAIL",
        "resolved_run_root": str(run_root.resolve()),
        "mount_rows": [
            {"target": row[0], "source": row[1], "filesystem": row[2]}
            for row in rows
            if len(row) == 3
        ],
    }


def summarize_model(
    spec: ModelSpec,
    run_root: Path,
    now: datetime,
    table: dict[int, tuple[int, str]],
) -> dict[str, Any]:
    state_dir = run_root / "runtime_state" / spec.model_id
    job_path = state_dir / "job_status.json"
    heartbeat_path = state_dir / "heartbeat.json"
    job, job_error = read_json(job_path)
    heartbeat, heartbeat_error = read_json(heartbeat_path)
    age = heartbeat_age_seconds(heartbeat, now)
    supervisor_pid = job.get("supervisor_pid") if job else None
    supervisor = proc_identity(supervisor_pid)
    children = descendants(supervisor_pid, table) if isinstance(supervisor_pid, int) else []
    expected_tokens = (
        "scripts/simulation/train_s4_3_pi2n.py",
        f"--model-id {spec.model_id}",
        f"--exp-name {spec.experiment}",
    )
    matching_rows = [
        row
        for row in children
        if all(token in row["cmdline"] for token in expected_tokens)
    ]
    matching = [{**row, **proc_identity(row["pid"])} for row in matching_rows]
    experiment = run_root / "runs" / spec.model_id / "pinch_tongs" / spec.experiment
    final_checkpoint = experiment / "29999"
    freeze_path = ARTIFACTS / f"{spec.model_id.lower()}_training_freeze.json"
    training_freeze, training_freeze_error = read_json(freeze_path)
    freeze_valid = training_freeze_valid(training_freeze, spec)
    completion_path = ARTIFACTS / spec.completion_manifest
    completion, completion_error = read_json(completion_path)
    completion_passed = completion_manifest_valid(
        completion, spec, final_checkpoint
    )
    session = tmux_session_exists(spec.session)
    expected_checkpoint = str(final_checkpoint)
    runtime_identity_matches = bool(
        job
        and heartbeat
        and job.get("model_id") == spec.model_id
        and heartbeat.get("model_id") == spec.model_id
        and job.get("supervisor_pid") == heartbeat.get("supervisor_pid")
        and job.get("physical_gpu_id") == heartbeat.get("physical_gpu_id")
        and job.get("physical_gpu_uuid") == heartbeat.get("physical_gpu_uuid")
        and job.get("checkpoint_directory") == str(experiment)
        and heartbeat.get("checkpoint_directory") == str(experiment)
        and job.get("expected_final_checkpoint") == expected_checkpoint
        and heartbeat.get("expected_final_checkpoint") == expected_checkpoint
    )
    state, failed_runtime_gates = classify_run(
        job=job,
        heartbeat=heartbeat,
        heartbeat_age=age,
        supervisor=supervisor,
        matching_trainers=len(matching),
        runtime_identity_matches=runtime_identity_matches,
        training_identity_frozen=freeze_valid,
        final_checkpoint_present=final_checkpoint.is_dir(),
        completion_audit_passed=completion_passed,
    )
    return {
        "model_id": spec.model_id,
        "state": state,
        "experiment": spec.experiment,
        "session": spec.session,
        "session_present": session,
        "job_status_path": str(job_path),
        "job_status": job,
        "job_status_read_error": job_error,
        "heartbeat_path": str(heartbeat_path),
        "heartbeat": heartbeat,
        "heartbeat_read_error": heartbeat_error,
        "heartbeat_age_seconds": age,
        "supervisor": supervisor,
        "matching_trainer_processes": matching,
        "runtime_identity_matches": runtime_identity_matches,
        "training_freeze": {
            "path": str(freeze_path),
            "sha256": small_file_sha256(freeze_path),
            "read_error": training_freeze_error,
            "valid": freeze_valid,
            "git_head": training_freeze.get("git_head") if training_freeze else None,
            "source_snapshot_sha256": training_freeze.get("source_snapshot_sha256")
            if training_freeze
            else None,
            "base_manifest_sha256": training_freeze.get("base_manifest_sha256")
            if training_freeze
            else None,
            "target_sha256": training_freeze.get("target_sha256")
            if training_freeze
            else None,
            "lambda_phys": training_freeze.get("lambda_phys")
            if training_freeze
            else None,
            "seed": training_freeze.get("seed") if training_freeze else None,
            "steps": training_freeze.get("steps") if training_freeze else None,
        },
        "failed_running_gates": failed_runtime_gates,
        "progress": training_progress(Path(job["log_path"]))
        if job and isinstance(job.get("log_path"), str)
        else {"available": False},
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_present": final_checkpoint.is_dir(),
        "completion_manifest": str(completion_path),
        "completion_manifest_read_error": completion_error,
        "completion_audit_passed": completion_passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    run_root = args.run_root.resolve()
    table = process_table()
    models = [summarize_model(spec, run_root, now, table) for spec in MODEL_SPECS]
    states = [row["state"] for row in models]
    if all(state == "COMPLETE_VERIFIED" for state in states):
        pause_state = "PI2N_TRAINING_COMPLETE_VERIFIED"
    elif any(state == "RUNNING" for state in states):
        pause_state = "PI2N_TRAINING_RUNNING"
    else:
        pause_state = "PI2N_TRAINING_EXITED_UNVERIFIED"
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2n-read-only-status.v1",
        "captured_at_utc": now.isoformat(),
        "read_only": True,
        "model_or_checkpoint_tree_loaded": False,
        "runtime_files_written": False,
        "pause_state": pause_state,
        "mount": mount_status(run_root),
        "models": models,
    }
    print(json.dumps(payload, sort_keys=True if args.compact else False, indent=None if args.compact else 2))


if __name__ == "__main__":
    main()
