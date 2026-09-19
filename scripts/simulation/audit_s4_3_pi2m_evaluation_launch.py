#!/usr/bin/env python3
"""Audit stable progress of the persistent frozen PI2M evaluation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2m"
LAUNCH = ARTIFACTS / "evaluation_launch.json"
STATUS = ARTIFACTS / "evaluation_job_status.json"
FREEZE = ARTIFACTS / "pre_eval_freeze.json"
OUTPUT = ARTIFACTS / "evaluation_job_manifest.json"
LOGS = ROOT / ".local/logs/simulation/s4_3_pi2m/evaluation_seed7"
CACHE = ROOT / ".local/cache/simulation/s4_3_pi2m/evaluation/seed7"
MODELS = ("B1", "B_HVA", "B2")
SESSION = "s43_pi2m_eval_s7"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def symbolic(path: Path) -> str:
    return "$REPO_ROOT/" + path.resolve().relative_to(ROOT).as_posix()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def process_commands() -> dict[int, str]:
    rows = {}
    for child in Path("/proc").iterdir():
        if not child.name.isdigit():
            continue
        try:
            command = (child / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeDecodeError):
            continue
        rows[int(child.name)] = command
    return rows


def main() -> None:
    launch = json.loads(LAUNCH.read_text())
    status = json.loads(STATUS.read_text())
    freeze = json.loads(FREEZE.read_text())
    if OUTPUT.exists():
        raise SystemExit("refusing to overwrite PI2M evaluation launch audit")
    started = datetime.fromisoformat(launch["launched_at"])
    now = datetime.now(timezone.utc)
    elapsed = max((now - started).total_seconds(), 1.0)
    counts = {
        model: len(
            list((CACHE / model.lower()).glob("episode_*_*"))
            if (CACHE / model.lower()).is_dir()
            else []
        )
        for model in MODELS
    }
    commands = process_commands()
    orchestrators = {
        pid: command
        for pid, command in commands.items()
        if "run_s4_3_pi2m_eval.py orchestrate" in command
    }
    servers = {
        model: {
            pid: command
            for pid, command in commands.items()
            if "serve_s4_3_pi2m_policy.py" in command and f"--model {model}" in command
        }
        for model in MODELS
    }
    first_wave = MODELS[: len(launch["physical_gpu_ids"])]
    active_models = tuple(model for model in MODELS if servers[model])
    source_hashes = freeze["sources_sha256"]
    source_integrity = all(
        sha256_file(ROOT / symbolic_path.removeprefix("$REPO_ROOT/")) == expected
        for symbolic_path, expected in source_hashes.items()
    )
    diagnostics = {}
    for model in MODELS:
        path = ROOT / ".local/tmp/s43m7" / f"{model.lower()}_inference.jsonl"
        rows = (
            [json.loads(line) for line in path.read_text().splitlines() if line]
            if path.is_file()
            else []
        )
        chunks = [row for row in rows if row.get("type") == "action_chunk"]
        diagnostics[model] = {
            "path": symbolic(path),
            "action_chunks": len(chunks),
            "all_finite_30x22": bool(chunks)
            and all(row.get("finite") and row.get("shape") == [30, 22] for row in chunks),
            "contact_state_sent": bool(chunks)
            and all(row.get("contact_state_sent") is True for row in chunks),
            "training_only_fields_absent": bool(chunks)
            and all(not row.get("training_only_fields_sent") for row in chunks),
        }
    gpu_rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    active_uuids = {row.split(",", 1)[0].strip() for row in gpu_rows if "," in row}
    common = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=ROOT,
            text=True,
        ).strip()
    )
    locks_held = all(
        subprocess.run(
            ["flock", "-n", str(common / f"tactile3d_unit_gpu{gpu}.lock"), "true"],
            capture_output=True,
        ).returncode
        != 0
        for gpu in launch["physical_gpu_ids"]
    )
    gates = {
        "pre_eval_freeze_PASS": freeze.get("status") == "PASS",
        "job_status_running": status.get("state") == "RUNNING",
        "tmux_session_alive": subprocess.run(
            ["tmux", "has-session", "-t", SESSION], capture_output=True
        ).returncode
        == 0,
        "orchestrator_alive": len(orchestrators) == 1,
        "active_policy_servers_match_first_wave": active_models == first_wave
        and all(len(servers[model]) == 1 for model in first_wave),
        "one_to_three_frozen_gpus": 1 <= len(launch["physical_gpu_ids"]) <= 3,
        "selected_gpu_uuids_active": all(
            uuid in active_uuids for uuid in launch["physical_gpu_uuids"]
        ),
        "gpu_locks_held": locks_held,
        "all_active_workers_advanced": all(counts[model] > 0 for model in first_wave),
        "all_active_workers_emitted_finite_actions": all(
            diagnostics[model]["all_finite_30x22"] for model in first_wave
        ),
        "matched_contact_state_delivery": all(
            diagnostics[model]["contact_state_sent"] for model in first_wave
        ),
        "training_targets_absent": all(
            diagnostics[model]["training_only_fields_absent"] for model in first_wave
        ),
        "frozen_scientific_sources_unchanged": source_integrity,
        "no_canonical_raw_completed_early": not any(
            (ARTIFACTS / f"{model.lower()}_raw_rollouts.json").exists()
            for model in MODELS
        ),
    }
    per_model_seconds = {
        model: elapsed / count if count else None for model, count in counts.items()
    }
    eta_candidates = [
        (200 - counts[model]) * per_model_seconds[model]
        for model in MODELS
        if per_model_seconds[model] is not None
    ]
    eta_seconds = max(eta_candidates) if len(eta_candidates) == len(MODELS) else None
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2m-evaluation-job-manifest.v1",
        "status": "RUNNING_STABLE" if all(gates.values()) else "FAIL",
        "audited_at": now.isoformat(),
        "session": SESSION,
        "supervisor_pid": status.get("supervisor_pid"),
        "orchestrator_pids": sorted(orchestrators),
        "policy_server_pids": {
            model: sorted(rows) for model, rows in servers.items()
        },
        "first_wave_models": list(first_wave),
        "active_models": list(active_models),
        "physical_gpu_ids": launch["physical_gpu_ids"],
        "physical_gpu_uuids": launch["physical_gpu_uuids"],
        "completed_or_active_episode_directories": counts,
        "diagnostics": diagnostics,
        "elapsed_seconds": elapsed,
        "observed_seconds_per_episode": per_model_seconds,
        "eta_hours": eta_seconds / 3600 if eta_seconds is not None else None,
        "expected_completion_utc": (now + timedelta(seconds=eta_seconds)).isoformat()
        if eta_seconds is not None
        else None,
        "orchestrator_log": launch["orchestrator_log"],
        "formal_logs": launch["formal_logs"],
        "formal_cache": launch["formal_cache"],
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "interim_success_analysis_performed": False,
    }
    atomic_json(OUTPUT, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "counts": counts,
                "eta_hours": payload["eta_hours"],
                "failed": [name for name, value in gates.items() if not value],
            },
            sort_keys=True,
        )
    )
    if payload["status"] != "RUNNING_STABLE":
        raise SystemExit("PI2M_EVALUATION_LAUNCH_AUDIT_FAIL")


if __name__ == "__main__":
    main()
