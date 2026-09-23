#!/usr/bin/env python3
"""Report current matched-teacher jobs without reading Track A performance."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.simulation.pi2b_teacher.common import load_json, workspace  # noqa: E402


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def main() -> None:
    local = workspace()
    root = Path(local["write_root"])
    coordination = Path(local["coordination_contract_path"]).parent
    jobs = {}
    for mode in ("T_VA_match", "T_VAC_match"):
        path = root / "status" / f"{mode}.json"
        value = load_json(path) if path.exists() else {"state": "NOT_STARTED", "mode": mode}
        value["pid_alive"] = alive(value.get("pid"))
        if value.get("state") == "RUNNING" and value.get("seconds_per_step"):
            value["eta_seconds"] = max(0, int(value["steps"]) - int(value["step"])) * float(value["seconds_per_step"])
        jobs[mode] = value
    confirmation_path = root / "status/confirmation_generation.json"
    confirmation = load_json(confirmation_path) if confirmation_path.exists() else {"state": "NOT_STARTED"}
    confirmation["pid_alive"] = alive(confirmation.get("pid"))
    stages = {}
    for name in ("confirmation_cache", "evaluation", "analysis", "resume_state"):
        path = root / "status" / f"{name}.json"
        stages[name] = load_json(path) if path.exists() else {"state": "NOT_STARTED"}
        stages[name]["pid_alive"] = alive(stages[name].get("pid"))
    leases = []
    for path in sorted((coordination / "leases").glob("s4_3_pi2b_teacher_*.json")):
        value = load_json(path)
        value["supervisor_alive"] = alive(value.get("supervisor_pid"))
        value["worker_alive"] = alive(value.get("worker_pid"))
        leases.append(value)
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip().splitlines()
    result = {
        "schema": "tactile3d-unit.s4-3-pi2b-teacher-status.v1",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "jobs": jobs,
        "confirmation_generation": confirmation,
        "later_stages": stages,
        "leases": leases,
        "gpu": gpu,
        "track_a_performance_read": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
