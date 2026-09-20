#!/usr/bin/env python3
"""Launch and supervise the frozen PI2N DEV cohort in persistent tmux."""

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
sys.path.insert(0, str(ROOT))
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2n"
PRE_FREEZE = ARTIFACTS / "pre_dev_freeze.json"
LAUNCH = ARTIFACTS / "development_launch.json"
STATUS = ARTIFACTS / "development_job_status.json"
HEARTBEAT = ARTIFACTS / "development_heartbeat.json"
ORCHESTRATOR_LOG = (
    ROOT / ".local/logs/simulation/s4_3_pi2n/development_orchestrator.log"
)
SESSION = "s43_pi2n_dev"
MODELS = ("B_HVA", "B2", "B_VAC_V")
TOTAL_OUTCOMES = 90


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


def runtime_module():
    from scripts.simulation import run_s4_3_pi2u_eval as runtime

    return runtime


def launch() -> None:
    freeze = json.loads(PRE_FREEZE.read_text())
    if freeze.get("status") != "PASS":
        raise SystemExit("PI2N pre-DEV freeze is not PASS")
    protected = (
        LAUNCH,
        STATUS,
        HEARTBEAT,
        ORCHESTRATOR_LOG,
        ARTIFACTS / "development_raw",
        ARTIFACTS / "development_results.json",
        ARTIFACTS / "vac_star_selection.json",
        ARTIFACTS / "development_gpu_execution.json",
        ROOT / ".local/logs/simulation/s4_3_pi2n/development",
        ROOT / ".local/cache/simulation/s4_3_pi2n/development",
        ROOT / ".local/tmp/s43n_dev",
    )
    if tmux_alive() or any(path.exists() for path in protected):
        raise SystemExit("refusing to overwrite or duplicate PI2N DEV evaluation")
    runtime = runtime_module()
    snapshot1 = runtime.gpu_snapshot()
    time.sleep(2)
    snapshot2 = runtime.gpu_snapshot()
    eligible = [
        gpu
        for gpu in range(4)
        if runtime.gpu_is_idle(gpu, snapshot1)
        and runtime.gpu_is_idle(gpu, snapshot2)
    ]
    selected = eligible[:3]
    if not selected:
        raise SystemExit("no genuinely idle GPU available for PI2N DEV")
    probes = []
    try:
        probes = [runtime.acquire_gpu_lock(gpu) for gpu in selected]
        snapshot3 = runtime.gpu_snapshot()
        if not all(runtime.gpu_is_idle(gpu, snapshot3) for gpu in selected):
            raise SystemExit("PI2N DEV GPU became busy after advisory lock probe")
    finally:
        for handle in probes:
            handle.close()
    inventory = {
        int(parts[0].strip()): parts[1].strip()
        for row in snapshot3["inventory"]
        if len(parts := row.split(",")) >= 2
    }
    uuids = [inventory[gpu] for gpu in selected]
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2n-development-launch.v1",
        "status": "LAUNCHING",
        "launched_at": now(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "session": SESSION,
        "models": list(MODELS),
        "seed_blocks": [9, 10, 11],
        "episodes_per_model": 30,
        "total_canonical_outcomes": TOTAL_OUTCOMES,
        "physical_gpu_ids": selected,
        "physical_gpu_uuids": uuids,
        "maximum_heavy_workers": len(selected),
        "snapshots": [snapshot1, snapshot2, snapshot3],
        "pre_dev_freeze": str(PRE_FREEZE),
        "orchestrator_log": str(ORCHESTRATOR_LOG),
        "automatic_final_launch": False,
    }
    atomic_json(LAUNCH, payload)
    command = shlex.join(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "supervise",
            "--gpus",
            ",".join(str(gpu) for gpu in selected),
            "--uuids",
            ",".join(uuids),
        ]
    )
    result = subprocess.run(
        ["tmux", "new-session", "-d", "-s", SESSION, command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "failed to create PI2N DEV tmux")
    time.sleep(3)
    if not tmux_alive() and not STATUS.exists():
        raise RuntimeError("PI2N DEV tmux exited before writing status")
    print(json.dumps(json.loads(STATUS.read_text()), sort_keys=True))


def supervise(gpus: str, uuids: str) -> None:
    selected = [int(value) for value in gpus.split(",")]
    physical_uuids = uuids.split(",")
    running = {
        "schema": "tactile3d-unit.s4-3-pi2n-development-job-status.v1",
        "state": "RUNNING",
        "exit_code": None,
        "updated_at": now(),
        "session": SESSION,
        "supervisor_pid": os.getpid(),
        "supervisor_start_time": subprocess.check_output(
            ["ps", "-p", str(os.getpid()), "-o", "lstart="], text=True
        ).strip(),
        "interpreter": sys.executable,
        "physical_gpu_ids": selected,
        "physical_gpu_uuids": physical_uuids,
        "models": list(MODELS),
        "total_canonical_outcomes": TOTAL_OUTCOMES,
        "orchestrator_log": str(ORCHESTRATOR_LOG),
        "message": "frozen PI2N DEV 3-model x 3-seed-block evaluation running",
    }
    atomic_json(STATUS, running)
    atomic_json(HEARTBEAT, running)
    ORCHESTRATOR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ORCHESTRATOR_LOG.open("x") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "scripts/simulation/run_s4_3_pi2n_development.py"),
                "orchestrate",
                "--gpus",
                gpus,
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
        "message": "PI2N DEV completed; final evaluation remains manually blocked"
        if exit_code == 0
        else "PI2N DEV exited nonzero; preserve outputs and inspect logs",
    }
    atomic_json(STATUS, final)
    atomic_json(HEARTBEAT, final)
    raise SystemExit(exit_code)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("launch")
    supervise_parser = sub.add_parser("supervise")
    supervise_parser.add_argument("--gpus", required=True)
    supervise_parser.add_argument("--uuids", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "launch":
        launch()
    else:
        supervise(args.gpus, args.uuids)


if __name__ == "__main__":
    main()
