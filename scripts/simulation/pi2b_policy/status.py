#!/usr/bin/env python3
"""Read-only status report for the ten frozen PI2B training runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.pi2b_policy.contract import Workspace
from scripts.simulation.pi2b_policy.launch import heartbeat_path, log_path, session_name, status_path


STEP_PATTERN = re.compile(r"Step\s+(\d+):")


def log_progress(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"logged_step": None, "seconds_per_step": None}
    text = path.read_text(errors="replace")[-1024 * 1024 :]
    steps = [int(value) for value in STEP_PATTERN.findall(text)]
    return {"logged_step": steps[-1] if steps else None, "logged_records_in_tail": len(steps)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    workspace = Workspace.load(ROOT)
    manifest = json.loads(
        (ROOT / ".local/artifacts/simulation/s4_3_pi2b_policy/training_runs.json").read_text()
    )
    rows = []
    for run in manifest["rows"]:
        run_id = run["run_id"]
        path = status_path(workspace, run_id)
        payload = json.loads(path.read_text()) if path.is_file() else {"state": "NOT_STARTED"}
        alive = subprocess.run(
            ["tmux", "has-session", "-t", session_name(run_id)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
        rows.append(
            {
                "run_id": run_id,
                "model_id": run["model_id"],
                "seed": run["seed"],
                "state": payload.get("state", "UNKNOWN"),
                "exit_code": payload.get("exit_code"),
                "supervisor_pid": payload.get("supervisor_pid"),
                "gpu": payload.get("physical_gpu_index"),
                "gpu_uuid": payload.get("physical_gpu_uuid"),
                "tmux_alive": alive,
                "heartbeat": str(heartbeat_path(workspace, run_id)),
                "log": str(log_path(workspace, run_id)),
                "expected_final_checkpoint": run["final_checkpoint"],
                **log_progress(log_path(workspace, run_id)),
            }
        )
    summary = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            state: sum(row["state"] == state for row in rows)
            for state in ("NOT_STARTED", "RUNNING", "DONE", "FAILED")
        },
        "runs": rows,
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    print(json.dumps(summary["counts"], sort_keys=True))
    for row in rows:
        print(
            f"{row['run_id']:18s} {row['state']:11s} gpu={row['gpu']} "
            f"step={row['logged_step']} tmux={row['tmux_alive']}"
        )


if __name__ == "__main__":
    main()
