#!/usr/bin/env python3
"""Persistently supervise the read-only-heavy PI2N closeout hash audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2n"
LAUNCH = ARTIFACTS / "closeout_launch.json"
STATUS = ARTIFACTS / "closeout_job_status.json"
HEARTBEAT = ARTIFACTS / "closeout_heartbeat.json"
LOG = ROOT / ".local/logs/simulation/s4_3_pi2n/closeout.log"
SESSION = "s43_pi2n_closeout"
PREREQUISITES = (
    ARTIFACTS / "paper_core_update_audit_final.json",
    ARTIFACTS / "regression_tests.json",
    ARTIFACTS / "plots_manifest.json",
    ARTIFACTS / "statistics_independent_audit.json",
    ARTIFACTS / "final_decision.json",
)
OUTPUTS = (
    ARTIFACTS / "protected_integrity_after.json",
    ARTIFACTS / "environment_integrity.json",
    ARTIFACTS / "retry_ledger.json",
    ARTIFACTS / "pi2b_recommendation_draft.json",
    ARTIFACTS / "final_completion_audit.json",
    ARTIFACTS / "HUMAN_ACCEPTANCE.md",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def tmux_alive() -> bool:
    return (
        subprocess.run(
            ["tmux", "has-session", "-t", SESSION],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def launch() -> None:
    if any(not path.is_file() for path in PREREQUISITES):
        raise SystemExit("PI2N closeout launch prerequisite missing")
    protected = (LAUNCH, STATUS, HEARTBEAT, LOG, *OUTPUTS)
    if tmux_alive() or any(path.exists() for path in protected):
        raise SystemExit("refusing to overwrite or duplicate PI2N closeout")
    if subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    ).strip():
        raise SystemExit("tracked worktree must be clean before PI2N closeout")
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2n-closeout-launch.v1",
        "status": "LAUNCHING",
        "launched_at": now(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "session": SESSION,
        "supervisor_interpreter": sys.executable,
        "log": str(LOG),
        "gpu_required": False,
        "model_or_policy_inference": False,
        "training_or_evaluation_launched": False,
        "automatic_PI2B": False,
    }
    atomic_json(LAUNCH, payload)
    command = shlex.join(
        [sys.executable, str(Path(__file__).resolve()), "supervise"]
    )
    result = subprocess.run(
        ["tmux", "new-session", "-d", "-s", SESSION, command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "failed to create PI2N closeout tmux")
    time.sleep(3)
    if not tmux_alive() and not STATUS.exists():
        raise RuntimeError("PI2N closeout tmux exited before writing status")
    print(json.dumps(json.loads(STATUS.read_text()), sort_keys=True))


def supervise() -> None:
    running = {
        "schema": "tactile3d-unit.s4-3-pi2n-closeout-job-status.v1",
        "state": "RUNNING",
        "exit_code": None,
        "updated_at": now(),
        "session": SESSION,
        "supervisor_pid": os.getpid(),
        "supervisor_start_time": subprocess.check_output(
            ["ps", "-p", str(os.getpid()), "-o", "lstart="], text=True
        ).strip(),
        "interpreter": sys.executable,
        "log": str(LOG),
        "gpu_required": False,
        "message": "PI2N protected identity, environment and closeout audit running",
    }
    atomic_json(STATUS, running)
    atomic_json(HEARTBEAT, running)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("x") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "scripts/simulation/finalize_s4_3_pi2n.py"),
            ],
            cwd=ROOT,
            env=os.environ | {"PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"},
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            atomic_json(HEARTBEAT, {**running, "updated_at": now()})
            time.sleep(60)
        exit_code = int(process.returncode)
    final = {
        **running,
        "state": "DONE" if exit_code == 0 else "FAILED",
        "exit_code": exit_code,
        "updated_at": now(),
        "message": "PI2N closeout complete; no further work launched"
        if exit_code == 0
        else "PI2N closeout exited nonzero; preserve evidence and inspect log",
    }
    atomic_json(STATUS, final)
    atomic_json(HEARTBEAT, final)
    raise SystemExit(exit_code)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("launch")
    sub.add_parser("supervise")
    args = parser.parse_args()
    if args.command == "launch":
        launch()
    else:
        supervise()


if __name__ == "__main__":
    main()
