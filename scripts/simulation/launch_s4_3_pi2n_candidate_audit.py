#!/usr/bin/env python3
"""Persistently supervise one PI2N candidate cold-load completion audit."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2n"
JOB_ROOT = ARTIFACTS / "candidate_completion_audit_jobs"
RUN_ROOT = Path(
    "/mnt/ugreen_nas/storage/UniT_storage/experiments/simulation/s4_3_pi2n"
)
AUDITOR = ROOT / "scripts/simulation/audit_s4_3_pi2n_candidate_completion.py"
SESSION_PREFIX = "s43_pi2n_audit_"
ATTEMPT_PATTERN = re.compile(r"attempt_[0-9]{3}")


@dataclass(frozen=True)
class Candidate:
    model_id: str
    stem: str
    experiment: str


CANDIDATES = {
    "B_VA27": Candidate("B_VA27", "b_va27", "s43_pi2n_b_va27_seed42"),
    "B_VAC_V": Candidate("B_VAC_V", "b_vac_v", "s43_pi2n_b_vac_v_seed42"),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def checkpoint(spec: Candidate) -> Path:
    return RUN_ROOT / "runs" / spec.model_id / "pinch_tongs" / spec.experiment / "29999"


def runtime_path(spec: Candidate, name: str) -> Path:
    return RUN_ROOT / "runtime_state" / spec.model_id / name


def canonical_outputs(spec: Candidate) -> tuple[Path, Path]:
    return (
        ARTIFACTS / f"{spec.stem}_training_completion.json",
        ARTIFACTS / f"{spec.stem}_checkpoint_manifest.json",
    )


def trainer_alive(spec: Candidate) -> bool:
    result = subprocess.run(
        ["pgrep", "-af", "[t]rain_s4_3_pi2n.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    return any(
        f"--model-id {spec.model_id}" in row
        and f"--exp-name {spec.experiment}" in row
        for row in result.stdout.splitlines()
    )


def active_audit_sessions() -> list[str]:
    result = subprocess.run(
        ["tmux", "list-sessions", "-F", "#{session_name}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return sorted(
        row for row in result.stdout.splitlines() if row.startswith(SESSION_PREFIX)
    )


def tmux_alive(session: str) -> bool:
    return (
        subprocess.run(
            ["tmux", "has-session", "-t", session],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def reserve_attempt(spec: Candidate) -> Path:
    root = JOB_ROOT / spec.stem
    root.mkdir(parents=True, exist_ok=True)
    for index in range(1, 1_000):
        attempt = root / f"attempt_{index:03d}"
        try:
            attempt.mkdir()
        except FileExistsError:
            continue
        return attempt
    raise RuntimeError(f"too many completion audit jobs for {spec.model_id}")


def preflight(spec: Candidate) -> None:
    job_path = runtime_path(spec, "job_status.json")
    heartbeat_path = runtime_path(spec, "heartbeat.json")
    if not AUDITOR.is_file() or not checkpoint(spec).is_dir():
        raise SystemExit(f"{spec.model_id}_FINAL_CHECKPOINT_MISSING")
    if not job_path.is_file() or not heartbeat_path.is_file():
        raise SystemExit(f"{spec.model_id}_TRAINING_STATUS_MISSING")
    job = read_json(job_path)
    heartbeat = read_json(heartbeat_path)
    if not (
        job.get("model_id") == spec.model_id
        and heartbeat.get("model_id") == spec.model_id
        and job.get("state") == "DONE"
        and heartbeat.get("state") == "DONE"
        and job.get("exit_code") == 0
        and heartbeat.get("exit_code") == 0
        and job.get("expected_final_checkpoint") == str(checkpoint(spec))
        and heartbeat.get("expected_final_checkpoint") == str(checkpoint(spec))
    ):
        raise SystemExit(f"{spec.model_id}_TRAINING_NOT_DONE_EXIT_ZERO")
    if trainer_alive(spec):
        raise SystemExit(f"{spec.model_id}_TRAINING_PROCESS_STILL_LIVE")
    if any(path.exists() for path in canonical_outputs(spec)):
        raise SystemExit(f"refusing to overwrite completed {spec.model_id} audit")
    sessions = active_audit_sessions()
    if sessions:
        raise SystemExit("candidate completion audit already active: " + ",".join(sessions))
    if subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    ).strip():
        raise SystemExit("tracked worktree must be clean before candidate audit")


def attempt_dir(spec: Candidate, attempt_id: str) -> Path:
    if ATTEMPT_PATTERN.fullmatch(attempt_id) is None:
        raise SystemExit("invalid candidate audit attempt id")
    path = JOB_ROOT / spec.stem / attempt_id
    if not path.is_dir():
        raise SystemExit("candidate audit attempt directory missing")
    return path


def launch(spec: Candidate) -> None:
    preflight(spec)
    directory = reserve_attempt(spec)
    attempt_id = directory.name
    session = f"{SESSION_PREFIX}{spec.stem}_{attempt_id.removeprefix('attempt_')}"
    log = (
        ROOT
        / f".local/logs/simulation/s4_3_pi2n/{spec.model_id}/"
        f"completion_audit_{attempt_id}.log"
    )
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2n-candidate-audit-launch.v1",
        "status": "LAUNCHING",
        "launched_at": now(),
        "model_id": spec.model_id,
        "attempt_id": attempt_id,
        "session": session,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "supervisor_interpreter": sys.executable,
        "checkpoint": str(checkpoint(spec)),
        "log": str(log),
        "gpu_required": False,
        "policy_inference": False,
        "training_or_evaluation_launched": False,
        "automatic_pre_dev_or_followup": False,
        "completion_criteria": (
            "audit exit 0 plus canonical training completion and checkpoint "
            "manifest both present"
        ),
        "monitor": {
            "attach": f"tmux attach -t {session}",
            "capture": f"tmux capture-pane -pt {session} -S -80",
            "tail": f"tail -F {log}",
            "status": f"jq . {directory / 'job_status.json'}",
            "heartbeat": f"jq . {directory / 'heartbeat.json'}",
        },
    }
    atomic_json(directory / "launch.json", payload)
    command = shlex.join(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "supervise",
            "--model-id",
            spec.model_id,
            "--attempt-id",
            attempt_id,
        ]
    )
    result = subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        atomic_json(
            directory / "job_status.json",
            {
                **payload,
                "state": "FAILED",
                "exit_code": result.returncode,
                "updated_at": now(),
                "message": result.stderr.strip() or "tmux launch failed",
            },
        )
        raise RuntimeError(result.stderr.strip() or "failed to create audit tmux")
    time.sleep(3)
    status_path = directory / "job_status.json"
    if not tmux_alive(session) and not status_path.is_file():
        raise RuntimeError("candidate audit tmux exited before writing status")
    print(json.dumps(read_json(status_path), sort_keys=True))


def supervise(spec: Candidate, attempt_id: str) -> None:
    directory = attempt_dir(spec, attempt_id)
    launch_payload = read_json(directory / "launch.json")
    session = launch_payload["session"]
    log = Path(launch_payload["log"])
    running = {
        "schema": "tactile3d-unit.s4-3-pi2n-candidate-audit-job-status.v1",
        "state": "RUNNING",
        "exit_code": None,
        "updated_at": now(),
        "model_id": spec.model_id,
        "attempt_id": attempt_id,
        "session": session,
        "supervisor_pid": os.getpid(),
        "supervisor_start_time": subprocess.check_output(
            ["ps", "-p", str(os.getpid()), "-o", "lstart="], text=True
        ).strip(),
        "interpreter": sys.executable,
        "checkpoint": str(checkpoint(spec)),
        "log": str(log),
        "gpu_required": False,
        "message": "candidate checkpoint hash, Orbax restore and cold-load audit running",
    }
    atomic_json(directory / "job_status.json", running)
    atomic_json(directory / "heartbeat.json", running)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as output:
        process = subprocess.Popen(
            [sys.executable, str(AUDITOR), "--model-id", spec.model_id],
            cwd=ROOT,
            env=os.environ
            | {
                "CUDA_VISIBLE_DEVICES": "",
                "JAX_PLATFORMS": "cpu",
                "PYTHONPATH": str(ROOT),
                "PYTHONUNBUFFERED": "1",
            },
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            atomic_json(directory / "heartbeat.json", {**running, "updated_at": now()})
            time.sleep(60)
        audit_exit_code = int(process.returncode)
    canonical_complete = all(path.is_file() for path in canonical_outputs(spec))
    success = audit_exit_code == 0 and canonical_complete
    exit_code = 0 if success else audit_exit_code or 70
    final = {
        **running,
        "state": "DONE" if success else "FAILED",
        "exit_code": exit_code,
        "audit_exit_code": audit_exit_code,
        "canonical_completion_present": canonical_complete,
        "updated_at": now(),
        "message": "candidate completion audit passed; no follow-up launched"
        if success
        else "candidate completion audit failed; evidence preserved; no follow-up launched",
    }
    atomic_json(directory / "job_status.json", final)
    atomic_json(directory / "heartbeat.json", final)
    raise SystemExit(exit_code)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--model-id", choices=tuple(CANDIDATES), required=True)
    supervise_parser = subparsers.add_parser("supervise")
    supervise_parser.add_argument(
        "--model-id", choices=tuple(CANDIDATES), required=True
    )
    supervise_parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args()
    spec = CANDIDATES[args.model_id]
    if args.command == "launch":
        launch(spec)
    else:
        supervise(spec, args.attempt_id)


if __name__ == "__main__":
    main()
