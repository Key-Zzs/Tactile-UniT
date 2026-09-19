#!/usr/bin/env python3
"""Audit the stable launch of the B2-only PI2M evaluation recovery."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2m"
LOGS = ROOT / ".local/logs/simulation/s4_3_pi2m/evaluation_seed7"
CACHE = ROOT / ".local/cache/simulation/s4_3_pi2m/evaluation/seed7/b2"
TMP = ROOT / ".local/tmp/s43m7"
LAUNCH = ARTIFACTS / "b2_recovery_launch.json"
STATUS = ARTIFACTS / "evaluation_job_status.json"
ARCHIVE = (
    ARTIFACTS
    / "evaluation_recovery/event_001/interruption_archive_manifest.json"
)
OUTPUT = ARTIFACTS / "b2_recovery_job_manifest.json"
RESUME = ARTIFACTS / "resume_state.json"
SESSION = "s43_pi2m_eval_s7_b2r1"
EXPECTED_RAW_SHA256 = {
    "B1": "8e6ac39c60d6a84e0df953f6c4076ebd55e0680b312b486a0814699e346680cf",
    "B_HVA": "078d8abb6fd933a9b4a42053212ec808098c6111de2e309e06599bc6bf69a18e",
}
EPISODE_PATTERN = re.compile(r"^episode_(\d+)_(success|failure|temp)$")


def now() -> datetime:
    return datetime.now(timezone.utc)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tmux_pane_pid() -> int | None:
    result = subprocess.run(
        ["tmux", "list-panes", "-t", SESSION, "-F", "#{pane_pid}"],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode or not result.stdout.strip().isdigit():
        return None
    return int(result.stdout.strip())


def process_table() -> list[str]:
    return subprocess.check_output(
        ["ps", "-eo", "pid,ppid,stat,etimes,args"], text=True
    ).splitlines()


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite {OUTPUT}")
    launch = read_json(LAUNCH)
    status = read_json(STATUS)
    archive = read_json(ARCHIVE)
    pre_freeze = read_json(ARTIFACTS / "pre_eval_freeze.json")
    controls = {
        model: read_json(ARTIFACTS / f"{model.lower()}_raw_rollouts.json")
        for model in ("B1", "B_HVA")
    }
    control_hashes = {
        model: sha256_file(ARTIFACTS / f"{model.lower()}_raw_rollouts.json")
        for model in controls
    }
    reset_sequences = {
        model: [row["reset_identity"] for row in payload["episode_results"]]
        for model, payload in controls.items()
    }
    entries: list[tuple[int, str]] = []
    for path in sorted(item for item in CACHE.iterdir() if item.is_dir()):
        match = EPISODE_PATTERN.fullmatch(path.name)
        if match is None:
            raise RuntimeError(f"unexpected recovery episode directory: {path.name}")
        entries.append((int(match.group(1)), match.group(2)))
    completed = sorted(index for index, result in entries if result != "temp")
    active = sorted(index for index, result in entries if result == "temp")

    diagnostics_path = TMP / "b2_inference.jsonl"
    diagnostics = [
        json.loads(line)
        for line in diagnostics_path.read_text().splitlines()
        if line.strip()
    ]
    chunks = [row for row in diagnostics if row.get("type") == "action_chunk"]
    errors = [row for row in diagnostics if row.get("type") == "server_client_error"]
    seen_indices = sorted({int(row["episode_index"]) for row in chunks})
    control_resets = reset_sequences["B1"]
    processes = process_table()
    pane_pid = tmux_pane_pid()
    selected_gpu = int(launch["physical_gpu_id"])

    sys.path.insert(0, str(ROOT))
    from scripts.simulation import run_s4_3_pi2m_eval as formal

    frozen = formal.configure_frozen_runtime()
    gpu_snapshot = frozen.gpu_snapshot()
    inventory = {
        int(fields[0].strip()): fields[1].strip()
        for row in gpu_snapshot["inventory"]
        if len(fields := row.split(",")) >= 2
    }
    selected_uuid = launch["physical_gpu_uuid"]
    selected_apps = [
        row
        for row in gpu_snapshot["compute_applications"]
        if row.split(",", 1)[0].strip() == selected_uuid
    ]
    lock_held = False
    try:
        handle = frozen.acquire_gpu_lock(selected_gpu)
    except RuntimeError:
        lock_held = True
    else:
        handle.close()

    source_hashes_exact = all(
        sha256_file(Path(path.replace("$REPO_ROOT", str(ROOT)))) == expected
        for path, expected in pre_freeze["sources_sha256"].items()
    )
    log_text = "\n".join(
        path.read_text(errors="replace")
        for path in sorted(LOGS.glob("b2_*.log"))
        if path.is_file()
    )
    required_markers = all(
        marker in log_text
        for marker in (
            "CONTACT_STATE_SERVICE_READY",
            "PI2M_POLICY_COLD_LOAD_PASS model=B2",
            "PI2M_POLICY_SERVER_READY model=B2",
        )
    )
    failure_markers_absent = not any(
        marker in log_text
        for marker in ("Traceback (most recent call last)", "CUDA out of memory")
    )
    current = now()
    launched = datetime.fromisoformat(launch["launched_at"])
    elapsed = (current - launched).total_seconds()
    seconds_per_episode = elapsed / max(len(completed), 1)
    remaining = 200 - len(completed)
    expected_completion = current + timedelta(seconds=remaining * seconds_per_episode)

    gates = {
        "recovery_status_running": status.get("state") == "RECOVERY_RUNNING",
        "B2_only": launch.get("models_relaunched") == ["B2"]
        and launch.get("models_not_relaunched") == ["B1", "B_HVA"],
        "archive_PASS": archive.get("status") == "PASS",
        "interrupted_boundary_31_complete_plus_one_active": archive.get("boundary", {}).get(
            "complete_episode_indices"
        )
        == list(range(31))
        and archive.get("boundary", {}).get("active_episode_index") == 31,
        "B1_B_HVA_raw_hashes_unchanged": control_hashes == EXPECTED_RAW_SHA256,
        "B1_B_HVA_reset_sequences_identical": reset_sequences["B1"]
        == reset_sequences["B_HVA"],
        "frozen_sources_unchanged": source_hashes_exact,
        "session_alive": pane_pid == status.get("supervisor_pid"),
        "supervisor_process_alive": any(
            str(status.get("supervisor_pid")) in row
            and "recover_s4_3_pi2m_b2_evaluation.py supervise" in row
            for row in processes
        ),
        "policy_contact_evaluator_processes_alive": all(
            any(fragment in row for row in processes)
            for fragment in (
                "serve_s4_3_pi2m_policy.py --model B2",
                "run_s4_3_pi2u_eval.py contact-service",
                "run_s4_3_pi2m_eval.py evaluate-model --model B2",
            )
        ),
        "selected_GPU_identity_exact": inventory.get(selected_gpu) == selected_uuid,
        "selected_GPU_has_only_recovery_policy_app": len(selected_apps) == 1
        and "openpi/bin/python" in selected_apps[0],
        "selected_GPU_lock_held": lock_held,
        "recovery_advanced": bool(completed) and len(active) <= 1,
        "episode_indices_contiguous": completed == list(range(len(completed)))
        and (not active or active == [len(completed)]),
        "diagnostics_advanced": bool(chunks),
        "all_seen_reset_identities_match_controls": all(
            row["reset_identity"] == control_resets[int(row["episode_index"])]
            for row in chunks
        ),
        "all_action_chunks_finite_30x22": all(
            row.get("finite") is True and row.get("shape") == [30, 22]
            for row in chunks
        ),
        "contact_state_sent_for_every_chunk": all(
            row.get("contact_state_sent") is True for row in chunks
        ),
        "training_targets_absent": all(
            row.get("training_only_fields_sent") == [] for row in chunks
        ),
        "no_server_client_errors": not errors,
        "readiness_markers_present": required_markers,
        "failure_markers_absent": failure_markers_absent,
        "B2_raw_not_prematurely_emitted": not (ARTIFACTS / "b2_raw_rollouts.json").exists(),
        "PI2B_not_started": True,
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2m-b2-recovery-job-manifest.v1",
        "status": "RUNNING_STABLE" if all(gates.values()) else "FAIL",
        "audited_at": current.isoformat(),
        "session": SESSION,
        "supervisor_pid": status.get("supervisor_pid"),
        "physical_gpu_id": selected_gpu,
        "physical_gpu_uuid": selected_uuid,
        "models_complete_and_preserved": ["B1", "B_HVA"],
        "active_model": "B2",
        "recovery_method": "B2_ONLY_CLEAN_REPLAY",
        "completed_episode_directories": len(completed),
        "active_episode_directories": len(active),
        "seen_episode_indices": seen_indices,
        "action_chunks": len(chunks),
        "elapsed_seconds": elapsed,
        "observed_seconds_per_completed_episode_including_cold_start": seconds_per_episode,
        "remaining_episode_slots": remaining,
        "eta_hours": remaining * seconds_per_episode / 3600,
        "expected_completion_utc": expected_completion.isoformat(),
        "cautious_eta_hours": [2.2, 3.2],
        "protected_raw_sha256": control_hashes,
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "interim_success_analysis_performed": False,
        "PI2B_started": False,
    }
    atomic_json(OUTPUT, payload)
    if payload["status"] != "RUNNING_STABLE":
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit("PI2M_B2_RECOVERY_STABLE_LAUNCH_AUDIT_FAILED")

    resume = read_json(RESUME)
    resume.update(
        {
            "updated_at": current.isoformat(),
            "stage": "M8_FROZEN_600_ROLLOUT_EVALUATION_B2_RECOVERY",
            "state": "PI2M_B2_RECOVERY_RUNNING_STABLE",
            "evaluation_started": True,
            "evaluation_completed": False,
            "interim_significance_analysis_performed": False,
            "PI2B_started": False,
            "evaluation_recovery": {
                "status": "RUNNING_STABLE",
                "method": "B2_ONLY_CLEAN_REPLAY",
                "B1_B_HVA_relaunched": False,
                "session": SESSION,
                "supervisor_pid": status.get("supervisor_pid"),
                "physical_gpu_id": selected_gpu,
                "physical_gpu_uuid": selected_uuid,
                "completed_episode_directories_at_audit": len(completed),
                "active_episode_directories_at_audit": len(active),
                "observed_seconds_per_episode": seconds_per_episode,
                "eta_hours_at_audit": remaining * seconds_per_episode / 3600,
                "expected_completion_utc": expected_completion.isoformat(),
                "manifest": str(OUTPUT),
                "status_file": str(STATUS),
                "client_log": str(LOGS / "b2_client.log"),
                "cache": str(CACHE),
            },
            "next_legal_operation": "After the user replies exactly '继续 S4.3-PI2M', inspect the existing B2 recovery without restarting it. If still running, report progress and pause. If DONE with 600 complete, audit completeness before M9 statistics.",
            "resume_phrase_for_running_job": "继续 S4.3-PI2M",
            "blocked_reason": None,
        }
    )
    atomic_json(RESUME, resume)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
