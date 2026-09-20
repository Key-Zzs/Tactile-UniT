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
AUDIT_JOB_ROOT = ARTIFACTS / "candidate_completion_audit_jobs"
AUDIT_LAUNCHER = "launch_s4_3_pi2n_candidate_audit.py"
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


@dataclass(frozen=True)
class EvaluationSpec:
    phase: str
    session: str
    launcher: str
    pre_freeze: str
    launch: str
    job_status: str
    heartbeat: str
    required_outputs: tuple[tuple[str, str, int], ...]


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

EVALUATION_SPECS = (
    EvaluationSpec(
        "DEVELOPMENT",
        "s43_pi2n_dev",
        "launch_s4_3_pi2n_development.py",
        "pre_dev_freeze.json",
        "development_launch.json",
        "development_job_status.json",
        "development_heartbeat.json",
        (
            ("development_results.json", "total_canonical_outcomes", 90),
            ("vac_star_selection.json", "selected_vac_star", 0),
            ("development_gpu_execution.json", "maximum_heavy_workers", 0),
        ),
    ),
    EvaluationSpec(
        "FINAL",
        "s43_pi2n_final",
        "launch_s4_3_pi2n_final.py",
        "pre_final_freeze.json",
        "final_launch.json",
        "final_job_status.json",
        "final_heartbeat.json",
        (
            ("rollout_completeness.json", "total_canonical_outcomes", 1200),
            ("final_gpu_execution.json", "maximum_heavy_workers", 0),
        ),
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
    timestamp = parse_utc(
        (payload.get("updated_at_utc") or payload.get("updated_at"))
        if payload
        else None
    )
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


def completion_chain_status_valid(
    manifest: dict[str, Any] | None,
    completion: dict[str, Any] | None,
    completion_path: Path,
    spec: ModelSpec,
    final_checkpoint: Path,
) -> bool:
    if not completion_manifest_valid(manifest, spec, final_checkpoint):
        return False
    if not manifest or manifest.get("frozen") is not True or not completion:
        return False
    manifest_gates = manifest.get("gates")
    completion_gates = completion.get("gates")
    if not (
        isinstance(manifest_gates, dict)
        and manifest_gates
        and all(value == "PASS" for value in manifest_gates.values())
        and completion.get("status") == "PASS"
        and completion.get("model_id") == spec.model_id
        and completion.get("optimizer_steps") == 30_000
        and isinstance(completion_gates, dict)
        and completion_gates
        and all(value == "PASS" for value in completion_gates.values())
    ):
        return False
    expected_reference = "$REPO_ROOT/" + completion_path.relative_to(ROOT).as_posix()
    return bool(
        manifest.get("training_completion") == expected_reference
        and manifest.get("training_completion_sha256")
        == small_file_sha256(completion_path)
    )


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


def evaluation_output_valid(payload: dict[str, Any] | None, field: str, exact: int) -> bool:
    if not payload or payload.get("status") != "PASS":
        return False
    if field == "selected_vac_star":
        return payload.get(field) == "B_VAC_V"
    value = payload.get(field)
    if exact:
        return value == exact
    return isinstance(value, int) and 1 <= value <= 4


def summarize_evaluation(
    spec: EvaluationSpec, now: datetime
) -> dict[str, Any] | None:
    paths = {
        "pre_freeze": ARTIFACTS / spec.pre_freeze,
        "launch": ARTIFACTS / spec.launch,
        "job_status": ARTIFACTS / spec.job_status,
        "heartbeat": ARTIFACTS / spec.heartbeat,
    }
    if not any(path.exists() for path in paths.values()):
        return None
    payloads = {name: read_json(path) for name, path in paths.items()}
    pre_freeze, pre_freeze_error = payloads["pre_freeze"]
    launch, launch_error = payloads["launch"]
    job, job_error = payloads["job_status"]
    heartbeat, heartbeat_error = payloads["heartbeat"]
    age = heartbeat_age_seconds(heartbeat, now)
    supervisor_pid = job.get("supervisor_pid") if job else None
    supervisor = proc_identity(supervisor_pid)
    expected_command = bool(
        supervisor.get("live")
        and spec.launcher in str(supervisor.get("cmdline", ""))
        and " supervise " in f" {supervisor.get('cmdline', '')} "
    )
    runtime_identity_matches = bool(
        job
        and heartbeat
        and launch
        and launch.get("session") == spec.session
        and job.get("session") == spec.session
        and heartbeat.get("session") == spec.session
        and job.get("supervisor_pid") == heartbeat.get("supervisor_pid")
        and job.get("physical_gpu_ids") == heartbeat.get("physical_gpu_ids")
        and job.get("physical_gpu_uuids") == heartbeat.get("physical_gpu_uuids")
        and job.get("physical_gpu_ids") == launch.get("physical_gpu_ids")
        and job.get("physical_gpu_uuids") == launch.get("physical_gpu_uuids")
        and 1 <= len(job.get("physical_gpu_ids", [])) <= 4
        and len(job.get("physical_gpu_ids", []))
        == len(job.get("physical_gpu_uuids", []))
    )
    output_rows = []
    for filename, field, exact in spec.required_outputs:
        path = ARTIFACTS / filename
        payload, error = read_json(path)
        output_rows.append(
            {
                "path": str(path),
                "read_error": error,
                "valid": evaluation_output_valid(payload, field, exact),
            }
        )
    running_gates = {
        "pre_freeze_PASS": bool(pre_freeze and pre_freeze.get("status") == "PASS"),
        "launch_recorded": bool(launch and launch.get("status") == "LAUNCHING"),
        "job_running": bool(job and job.get("state") == "RUNNING"),
        "heartbeat_running": bool(heartbeat and heartbeat.get("state") == "RUNNING"),
        "heartbeat_fresh": age is not None and age <= HEARTBEAT_MAX_AGE_SECONDS,
        "supervisor_live": bool(supervisor.get("live")),
        "supervisor_command_matches": expected_command,
        "runtime_identity_matches": runtime_identity_matches,
    }
    if all(running_gates.values()):
        state = "RUNNING"
    elif bool(
        pre_freeze
        and pre_freeze.get("status") == "PASS"
        and job
        and heartbeat
        and job.get("state") == "DONE"
        and heartbeat.get("state") == "DONE"
        and job.get("exit_code") == 0
        and heartbeat.get("exit_code") == 0
        and runtime_identity_matches
        and all(row["valid"] for row in output_rows)
    ):
        state = "COMPLETE_VERIFIED"
    else:
        state = "EXITED_UNVERIFIED"
    return {
        "phase": spec.phase,
        "state": state,
        "session": spec.session,
        "session_present": tmux_session_exists(spec.session),
        "paths": {name: str(path) for name, path in paths.items()},
        "read_errors": {
            "pre_freeze": pre_freeze_error,
            "launch": launch_error,
            "job_status": job_error,
            "heartbeat": heartbeat_error,
        },
        "job_status": job,
        "heartbeat": heartbeat,
        "heartbeat_age_seconds": age,
        "supervisor": supervisor,
        "runtime_identity_matches": runtime_identity_matches,
        "failed_running_gates": [
            name for name, value in running_gates.items() if not value
        ],
        "required_outputs": output_rows,
        "performance_values_read": False,
    }


def summarize_candidate_audits(now: datetime) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        model_root = AUDIT_JOB_ROOT / spec.model_id.lower()
        if not model_root.is_dir():
            continue
        for directory in sorted(model_root.glob("attempt_[0-9][0-9][0-9]")):
            paths = {
                "launch": directory / "launch.json",
                "job_status": directory / "job_status.json",
                "heartbeat": directory / "heartbeat.json",
            }
            payloads = {name: read_json(path) for name, path in paths.items()}
            launch, launch_error = payloads["launch"]
            job, job_error = payloads["job_status"]
            heartbeat, heartbeat_error = payloads["heartbeat"]
            age = heartbeat_age_seconds(heartbeat, now)
            supervisor = proc_identity(job.get("supervisor_pid") if job else None)
            session = launch.get("session") if launch else None
            attempt_id = directory.name
            command = str(supervisor.get("cmdline", ""))
            command_matches = bool(
                supervisor.get("live")
                and AUDIT_LAUNCHER in command
                and " supervise " in f" {command} "
                and f"--model-id {spec.model_id}" in command
                and f"--attempt-id {attempt_id}" in command
            )
            runtime_identity_matches = bool(
                launch
                and job
                and heartbeat
                and all(
                    payload.get("model_id") == spec.model_id
                    and payload.get("attempt_id") == attempt_id
                    for payload in (launch, job, heartbeat)
                )
                and launch.get("session") == job.get("session")
                and job.get("session") == heartbeat.get("session")
                and job.get("supervisor_pid") == heartbeat.get("supervisor_pid")
            )
            session_present = bool(
                isinstance(session, str) and tmux_session_exists(session)
            )
            running_gates = {
                "launch_recorded": bool(
                    launch and launch.get("status") == "LAUNCHING"
                ),
                "job_running": bool(job and job.get("state") == "RUNNING"),
                "heartbeat_running": bool(
                    heartbeat and heartbeat.get("state") == "RUNNING"
                ),
                "heartbeat_fresh": age is not None
                and age <= HEARTBEAT_MAX_AGE_SECONDS,
                "supervisor_live": bool(supervisor.get("live")),
                "supervisor_command_matches": command_matches,
                "runtime_identity_matches": runtime_identity_matches,
                "session_present": session_present,
            }
            completion_path = (
                ARTIFACTS / f"{spec.model_id.lower()}_training_completion.json"
            )
            manifest_path = ARTIFACTS / spec.completion_manifest
            completion, completion_error = read_json(completion_path)
            manifest, manifest_error = read_json(manifest_path)
            final_checkpoint = (
                DEFAULT_RUN_ROOT
                / "runs"
                / spec.model_id
                / "pinch_tongs"
                / spec.experiment
                / "29999"
            )
            canonical_outputs_valid = completion_chain_status_valid(
                manifest,
                completion,
                completion_path,
                spec,
                final_checkpoint,
            )
            if all(running_gates.values()):
                state = "RUNNING"
            elif bool(
                job
                and heartbeat
                and job.get("state") == "DONE"
                and heartbeat.get("state") == "DONE"
                and job.get("exit_code") == 0
                and heartbeat.get("exit_code") == 0
                and runtime_identity_matches
                and canonical_outputs_valid
            ):
                state = "COMPLETE_VERIFIED"
            else:
                state = "EXITED_UNVERIFIED"
            rows.append(
                {
                    "model_id": spec.model_id,
                    "attempt_id": attempt_id,
                    "state": state,
                    "session": session,
                    "session_present": session_present,
                    "paths": {name: str(path) for name, path in paths.items()},
                    "read_errors": {
                        "launch": launch_error,
                        "job_status": job_error,
                        "heartbeat": heartbeat_error,
                        "training_completion": completion_error,
                        "checkpoint_manifest": manifest_error,
                    },
                    "job_status": job,
                    "heartbeat": heartbeat,
                    "heartbeat_age_seconds": age,
                    "supervisor": supervisor,
                    "runtime_identity_matches": runtime_identity_matches,
                    "failed_running_gates": [
                        name for name, value in running_gates.items() if not value
                    ],
                    "canonical_outputs_valid": canonical_outputs_valid,
                    "performance_values_read": False,
                }
            )
    return rows


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
    training_completion_path = (
        ARTIFACTS / f"{spec.model_id.lower()}_training_completion.json"
    )
    training_completion, training_completion_error = read_json(
        training_completion_path
    )
    completion_passed = completion_chain_status_valid(
        completion,
        training_completion,
        training_completion_path,
        spec,
        final_checkpoint,
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
        "training_completion": str(training_completion_path),
        "training_completion_read_error": training_completion_error,
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
    evaluations = [
        row
        for spec in EVALUATION_SPECS
        if (row := summarize_evaluation(spec, now)) is not None
    ]
    candidate_audits = summarize_candidate_audits(now)
    latest_candidate_audits = {
        model_id: next(
            row
            for row in reversed(candidate_audits)
            if row["model_id"] == model_id
        )
        for model_id in {row["model_id"] for row in candidate_audits}
    }
    states = [row["state"] for row in models]
    final = next(
        (row for row in evaluations if row["phase"] == "FINAL"), None
    )
    development = next(
        (row for row in evaluations if row["phase"] == "DEVELOPMENT"), None
    )
    if final and final["state"] == "RUNNING":
        pause_state = "PI2N_EVALUATION_RUNNING"
    elif final and final["state"] == "COMPLETE_VERIFIED":
        pause_state = "PI2N_FINAL_COMPLETE_VERIFIED"
    elif final:
        pause_state = "PI2N_FINAL_EXITED_UNVERIFIED"
    elif development and development["state"] == "RUNNING":
        pause_state = "PI2N_EVALUATION_RUNNING"
    elif development and development["state"] == "COMPLETE_VERIFIED":
        pause_state = "PI2N_DEVELOPMENT_COMPLETE_VERIFIED"
    elif development:
        pause_state = "PI2N_DEVELOPMENT_EXITED_UNVERIFIED"
    elif any(
        row["state"] == "RUNNING" for row in latest_candidate_audits.values()
    ):
        pause_state = "PI2N_CANDIDATE_AUDIT_RUNNING"
    elif latest_candidate_audits and any(
        row["state"] == "EXITED_UNVERIFIED"
        for row in latest_candidate_audits.values()
    ):
        pause_state = "PI2N_CANDIDATE_AUDIT_EXITED_UNVERIFIED"
    elif all(state == "COMPLETE_VERIFIED" for state in states):
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
        "candidate_audits": candidate_audits,
        "evaluations": evaluations,
    }
    print(json.dumps(payload, sort_keys=True if args.compact else False, indent=None if args.compact else 2))


if __name__ == "__main__":
    main()
