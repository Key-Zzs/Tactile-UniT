#!/usr/bin/env python3
"""Launch and supervise the frozen PI2M seed-7 evaluation in persistent tmux."""

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
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2m"
LAUNCH = ARTIFACTS / "evaluation_launch.json"
STATUS = ARTIFACTS / "evaluation_job_status.json"
PRE_FREEZE = ARTIFACTS / "pre_eval_freeze.json"
ORCHESTRATOR_LOG = (
    ROOT / ".local/logs/simulation/s4_3_pi2m/evaluation_seed7_orchestrator.log"
)
SESSION = "s43_pi2m_eval_s7"
FORMAL_LOGS = ROOT / ".local/logs/simulation/s4_3_pi2m/evaluation_seed7"
FORMAL_CACHE = ROOT / ".local/cache/simulation/s4_3_pi2m/evaluation/seed7"
FORMAL_TMP = ROOT / ".local/tmp/s43m7"
MODELS = ("B1", "B_HVA", "B2")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def configure():
    sys.path.insert(0, str(ROOT))
    from scripts.simulation import run_s4_3_pi2m_eval as formal

    return formal, formal.configure_frozen_runtime()


def tmux_alive() -> bool:
    return (
        subprocess.run(
            ["tmux", "has-session", "-t", SESSION],
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def launch() -> None:
    if json.loads(PRE_FREEZE.read_text()).get("status") != "PASS":
        raise SystemExit("PI2M pre-evaluation freeze is not PASS")
    protected = [
        LAUNCH,
        STATUS,
        ORCHESTRATOR_LOG,
        FORMAL_LOGS,
        FORMAL_CACHE,
        FORMAL_TMP,
        *[ARTIFACTS / f"{model.lower()}_raw_rollouts.json" for model in MODELS],
    ]
    if tmux_alive() or any(path.exists() for path in protected):
        raise SystemExit("refusing to overwrite or duplicate PI2M formal evaluation")
    formal, frozen = configure()
    snapshot1 = frozen.gpu_snapshot()
    time.sleep(2)
    snapshot2 = frozen.gpu_snapshot()
    eligible = [
        gpu
        for gpu in range(4)
        if frozen.gpu_is_idle(gpu, snapshot1) and frozen.gpu_is_idle(gpu, snapshot2)
    ]
    selected = eligible[:3]
    if not selected:
        atomic_json(
            STATUS,
            {
                "schema": "tactile3d-unit.s4-3-pi2m-evaluation-job-status.v1",
                "state": "WAITING_FOR_GPU",
                "updated_at": now(),
                "snapshots": [snapshot1, snapshot2],
                "message": "no genuinely idle GPU among physical GPU0-3",
            },
        )
        raise SystemExit(3)
    probes = []
    try:
        probes = [frozen.acquire_gpu_lock(gpu) for gpu in selected]
    finally:
        for handle in probes:
            handle.close()
    inventory = {
        int(parts[0].strip()): parts[1].strip()
        for row in snapshot2["inventory"]
        if len(parts := row.split(",")) >= 2
    }
    gpu_csv = ",".join(str(gpu) for gpu in selected)
    uuids = [inventory[gpu] for gpu in selected]
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2m-evaluation-launch.v1",
        "status": "LAUNCHING",
        "launched_at": now(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "session": SESSION,
        "physical_gpu_ids": selected,
        "physical_gpu_uuids": uuids,
        "maximum_heavy_workers": len(selected),
        "models": list(MODELS),
        "episodes_per_model": 200,
        "total_rollouts": 600,
        "evaluator_seed": 7,
        "snapshots": [snapshot1, snapshot2],
        "orchestrator_log": str(ORCHESTRATOR_LOG),
        "formal_logs": str(FORMAL_LOGS),
        "formal_cache": str(FORMAL_CACHE),
        "pre_eval_freeze": str(PRE_FREEZE),
    }
    atomic_json(LAUNCH, payload)
    command = shlex.join(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "supervise",
            "--gpus",
            gpu_csv,
            "--uuids",
            ",".join(uuids),
        ]
    )
    result = subprocess.run(
        ["tmux", "new-session", "-d", "-s", SESSION, command],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "failed to create PI2M evaluation tmux")
    time.sleep(3)
    if not tmux_alive() and not STATUS.exists():
        raise RuntimeError("PI2M evaluation tmux exited before writing job status")
    print(json.dumps(json.loads(STATUS.read_text()), sort_keys=True))


def supervise(gpus: str, uuids: str) -> None:
    selected = [int(value) for value in gpus.split(",")]
    physical_uuids = uuids.split(",")
    running = {
        "schema": "tactile3d-unit.s4-3-pi2m-evaluation-job-status.v1",
        "state": "RUNNING",
        "updated_at": now(),
        "session": SESSION,
        "supervisor_pid": os.getpid(),
        "physical_gpu_ids": selected,
        "physical_gpu_uuids": physical_uuids,
        "orchestrator_log": str(ORCHESTRATOR_LOG),
        "message": "frozen seed7 B1/B_HVA/B2 600-rollout evaluation starting",
    }
    atomic_json(STATUS, running)
    ORCHESTRATOR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ORCHESTRATOR_LOG.open("w") as output:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/simulation/run_s4_3_pi2m_eval.py"),
                "orchestrate",
                "--gpus",
                gpus,
            ],
            cwd=ROOT,
            env=os.environ | {"PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"},
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    final = {
        **running,
        "state": "DONE" if result.returncode == 0 else "FAILED",
        "exit_code": result.returncode,
        "updated_at": now(),
        "message": "formal evaluation completed"
        if result.returncode == 0
        else "formal evaluation exited nonzero; inspect preserved logs",
    }
    atomic_json(STATUS, final)
    raise SystemExit(result.returncode)


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
