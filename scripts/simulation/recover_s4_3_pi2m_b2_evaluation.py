#!/usr/bin/env python3
"""Recover an interrupted PI2M evaluation by replaying only B2, end to end.

The frozen evaluator writes its scientific artifact only when all 200 episodes
close.  Consequently the interrupted B2 process cannot be spliced safely.  This
supervisor preserves the partial attempt byte-for-byte, verifies its exact
boundary, and then invokes the unchanged frozen B2 evaluator for one clean
seed-7 replay.  B1 and B_HVA are never launched or rewritten.
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2m"
LOGS = ROOT / ".local/logs/simulation/s4_3_pi2m/evaluation_seed7"
CACHE = ROOT / ".local/cache/simulation/s4_3_pi2m/evaluation/seed7"
TMP = ROOT / ".local/tmp/s43m7"
PRE_FREEZE = ARTIFACTS / "pre_eval_freeze.json"
STATUS = ARTIFACTS / "evaluation_job_status.json"
RECOVERY_ROOT = ARTIFACTS / "evaluation_recovery/event_001"
RECOVERY_LAUNCH = ARTIFACTS / "b2_recovery_launch.json"
RECOVERY_COMPLETION = ARTIFACTS / "b2_recovery_completion.json"
RECOVERY_LOG = (
    ROOT / ".local/logs/simulation/s4_3_pi2m/evaluation_seed7_b2_recovery.log"
)
SESSION = "s43_pi2m_eval_s7_b2r1"
EXPECTED_COMPLETE = list(range(31))
EXPECTED_ACTIVE = 31
EXPECTED_RAW_SHA256 = {
    "B1": "8e6ac39c60d6a84e0df953f6c4076ebd55e0680b312b486a0814699e346680cf",
    "B_HVA": "078d8abb6fd933a9b4a42053212ec808098c6111de2e309e06599bc6bf69a18e",
}
EPISODE_PATTERN = re.compile(r"^episode_(\d+)_(success|failure|temp)$")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def tree_manifest(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if path.is_file():
        rows.append(
            {"path": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
        )
    elif path.is_dir():
        for item in sorted(entry for entry in path.rglob("*") if entry.is_file()):
            rows.append(
                {
                    "path": item.relative_to(path).as_posix(),
                    "size": item.stat().st_size,
                    "sha256": sha256_file(item),
                }
            )
    else:
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "files": len(rows),
        "bytes": sum(row["size"] for row in rows),
        "tree_sha256": canonical_sha256(rows),
    }


def raw_path(model: str) -> Path:
    return ARTIFACTS / f"{model.lower()}_raw_rollouts.json"


def verify_frozen_sources() -> dict[str, str]:
    freeze = read_json(PRE_FREEZE)
    if freeze.get("status") != "PASS":
        raise RuntimeError("PI2M pre-evaluation freeze is not PASS")
    observed: dict[str, str] = {}
    for frozen_path, expected in freeze["sources_sha256"].items():
        path = Path(frozen_path.replace("$REPO_ROOT", str(ROOT)))
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"frozen evaluation source changed: {path}")
        observed[frozen_path] = actual
    return observed


def verify_completed_controls() -> dict[str, Any]:
    payloads: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for model in ("B1", "B_HVA"):
        path = raw_path(model)
        actual = sha256_file(path)
        if actual != EXPECTED_RAW_SHA256[model]:
            raise RuntimeError(f"{model} raw artifact changed")
        payload = read_json(path)
        if (
            payload.get("status") != "PASS"
            or payload.get("episodes") != 200
            or len(payload.get("episode_results", [])) != 200
            or payload.get("evaluator_seed") != 7
        ):
            raise RuntimeError(f"{model} raw artifact is incomplete")
        payloads[model] = payload
        hashes[model] = actual
    left = [row["reset_identity"] for row in payloads["B1"]["episode_results"]]
    right = [row["reset_identity"] for row in payloads["B_HVA"]["episode_results"]]
    if left != right:
        raise RuntimeError("B1/B_HVA reset identity sequences differ")
    return {
        "raw_sha256": hashes,
        "reset_identities": left,
        "reset_sequence_sha256": canonical_sha256(left),
    }


def verify_interrupted_boundary(reset_identities: list[str]) -> dict[str, Any]:
    if raw_path("B2").exists():
        raise RuntimeError("B2 raw artifact already exists; recovery must not overwrite it")
    output = CACHE / "b2"
    diagnostics = TMP / "b2_inference.jsonl"
    if not output.is_dir() or not diagnostics.is_file():
        raise RuntimeError("interrupted B2 cache or diagnostics is missing")
    parsed: list[tuple[int, str, str]] = []
    for path in sorted(entry for entry in output.iterdir() if entry.is_dir()):
        match = EPISODE_PATTERN.fullmatch(path.name)
        if match is None:
            raise RuntimeError(f"unexpected B2 episode directory: {path.name}")
        parsed.append((int(match.group(1)), match.group(2), path.name))
    complete = sorted(index for index, result, _ in parsed if result != "temp")
    active = sorted(index for index, result, _ in parsed if result == "temp")
    if complete != EXPECTED_COMPLETE or active != [EXPECTED_ACTIVE]:
        raise RuntimeError(
            f"unexpected interrupted boundary: complete={complete}, active={active}"
        )
    rows = [json.loads(line) for line in diagnostics.read_text().splitlines() if line]
    if not rows or {int(row["episode_index"]) for row in rows} != set(range(32)):
        raise RuntimeError("B2 diagnostics do not cover exactly interrupted episodes 0..31")
    if not all(
        row.get("type") == "action_chunk"
        and row.get("finite") is True
        and row.get("shape") == [30, 22]
        and row.get("contact_state_sent") is True
        and row.get("training_only_fields_sent") == []
        and row.get("reset_identity") == reset_identities[int(row["episode_index"])]
        for row in rows
    ):
        raise RuntimeError("interrupted B2 diagnostics failed frozen runtime gates")
    return {
        "complete_episode_indices": complete,
        "active_episode_index": active[0],
        "episode_directory_names_sha256": canonical_sha256(
            [name for _, _, name in parsed]
        ),
        "diagnostic_rows": len(rows),
        "diagnostics_sha256": sha256_file(diagnostics),
        "all_seen_reset_identities_match_controls": True,
        "all_action_chunks_finite_30x22": True,
        "contact_state_sent_for_every_action_chunk": True,
        "training_only_targets_never_sent": True,
    }


def tmux_alive() -> bool:
    return (
        subprocess.run(
            ["tmux", "has-session", "-t", SESSION], capture_output=True, check=False
        ).returncode
        == 0
    )


def process_alive(fragment: str) -> bool:
    result = subprocess.run(
        ["pgrep", "-af", fragment], capture_output=True, check=False, text=True
    )
    excluded = {os.getpid(), os.getppid()}
    for line in result.stdout.splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            continue
        pid_text, command = fields
        if (
            pid_text.isdigit()
            and int(pid_text) not in excluded
            and fragment in command
            and "pgrep -af" not in command
        ):
            return True
    return False


def move_into(path: Path, destination: Path) -> dict[str, Any]:
    before = tree_manifest(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    path.rename(destination)
    after = tree_manifest(destination)
    if (
        before["tree_sha256"] != after["tree_sha256"]
        or before["files"] != after["files"]
        or before["bytes"] != after["bytes"]
    ):
        raise RuntimeError(f"archive verification failed for {path}")
    return {"source": str(path), "archive": str(destination), **after}


def archive_interrupted_attempt(boundary: dict[str, Any]) -> dict[str, Any]:
    if RECOVERY_ROOT.exists():
        raise RuntimeError(f"recovery archive already exists: {RECOVERY_ROOT}")
    RECOVERY_ROOT.mkdir(parents=True, exist_ok=False)
    snapshots = RECOVERY_ROOT / "supervisor_snapshots"
    snapshots.mkdir()
    for path in (
        STATUS,
        ARTIFACTS / "evaluation_launch.json",
        ARTIFACTS / "evaluation_job_manifest.json",
        ARTIFACTS / "resume_state.json",
    ):
        if path.is_file():
            shutil.copy2(path, snapshots / path.name)
    moves: list[dict[str, Any]] = []
    moves.append(move_into(CACHE / "b2", RECOVERY_ROOT / "interrupted/cache/b2"))
    for path in sorted(LOGS.glob("b2_*")):
        moves.append(
            move_into(path, RECOVERY_ROOT / "interrupted/logs" / path.name)
        )
    for path in sorted(TMP.glob("b2_*")):
        moves.append(
            move_into(path, RECOVERY_ROOT / "interrupted/tmp" / path.name)
        )
    stale_status_tmp = STATUS.with_suffix(STATUS.suffix + ".tmp")
    if stale_status_tmp.exists():
        moves.append(
            move_into(
                stale_status_tmp,
                RECOVERY_ROOT / "interrupted/supervisor/evaluation_job_status.json.tmp",
            )
        )
    manifest = {
        "schema": "tactile3d-unit.s4-3-pi2m-b2-interruption-archive.v1",
        "status": "PASS",
        "archived_at": now(),
        "reason": "original supervisor disappeared during B2 episode 31 before its atomic final status/raw artifact write",
        "scientific_recovery_rule": "do not splice; replay only B2 end-to-end with unchanged frozen evaluator and seed7",
        "B1_B_HVA_relaunched": False,
        "B1_B_HVA_modified": False,
        "boundary": boundary,
        "moves": moves,
    }
    atomic_json(RECOVERY_ROOT / "interruption_archive_manifest.json", manifest)
    return manifest


def select_gpu(frozen: Any) -> tuple[int, str, list[dict[str, Any]]]:
    snapshots = [frozen.gpu_snapshot()]
    time.sleep(2)
    snapshots.append(frozen.gpu_snapshot())
    eligible = [
        gpu
        for gpu in range(4)
        if frozen.gpu_is_idle(gpu, snapshots[0])
        and frozen.gpu_is_idle(gpu, snapshots[1])
    ]
    if not eligible:
        raise RuntimeError("no genuinely idle GPU among physical GPU0-3")
    # Prefer the same physical GPU used by the interrupted B2 worker.
    gpu = 3 if 3 in eligible else eligible[0]
    inventory = {
        int(fields[0].strip()): fields[1].strip()
        for row in snapshots[1]["inventory"]
        if len(fields := row.split(",")) >= 2
    }
    probe = frozen.acquire_gpu_lock(gpu)
    probe.close()
    return gpu, inventory[gpu], snapshots


def launch() -> None:
    if tmux_alive() or process_alive(f"{Path(__file__).resolve()} supervise"):
        raise RuntimeError("B2 recovery supervisor is already running")
    if RECOVERY_LAUNCH.exists() or RECOVERY_COMPLETION.exists():
        raise RuntimeError("refusing to overwrite an existing B2 recovery record")
    sources = verify_frozen_sources()
    controls = verify_completed_controls()
    archive_path = RECOVERY_ROOT / "interruption_archive_manifest.json"
    if RECOVERY_ROOT.exists():
        if not archive_path.is_file():
            raise RuntimeError("incomplete B2 interruption archive requires manual audit")
        archive = read_json(archive_path)
        if archive.get("status") != "PASS":
            raise RuntimeError("B2 interruption archive is not PASS")
        boundary = archive["boundary"]
        archived_cache = RECOVERY_ROOT / "interrupted/cache/b2"
        if (
            boundary.get("complete_episode_indices") != EXPECTED_COMPLETE
            or boundary.get("active_episode_index") != EXPECTED_ACTIVE
            or not archived_cache.is_dir()
            or (CACHE / "b2").exists()
            or (TMP / "b2_inference.jsonl").exists()
        ):
            raise RuntimeError("archived B2 recovery boundary changed")
    else:
        boundary = verify_interrupted_boundary(controls["reset_identities"])
        archive = archive_interrupted_attempt(boundary)

    sys.path.insert(0, str(ROOT))
    from scripts.simulation import run_s4_3_pi2m_eval as formal

    frozen = formal.configure_frozen_runtime()
    try:
        gpu, uuid, snapshots = select_gpu(frozen)
    except RuntimeError as error:
        atomic_json(
            STATUS,
            {
                "schema": "tactile3d-unit.s4-3-pi2m-evaluation-job-status.v1",
                "state": "WAITING_FOR_GPU",
                "updated_at": now(),
                "recovery": "B2_ONLY_CLEAN_REPLAY",
                "message": str(error),
                "archive_manifest": str(
                    RECOVERY_ROOT / "interruption_archive_manifest.json"
                ),
            },
        )
        raise SystemExit(3) from error
    launch_payload = {
        "schema": "tactile3d-unit.s4-3-pi2m-b2-recovery-launch.v1",
        "status": "LAUNCHING",
        "launched_at": now(),
        "session": SESSION,
        "supervision": "B2_ONLY_CLEAN_REPLAY",
        "reason_full_B2_replay_required": "frozen evaluator emits the complete scientific raw artifact only after episode 199; interrupted in-memory episode summaries cannot be spliced",
        "prefix_replay_comparison_role": "integrity diagnostic only; not a selection or acceptance gate",
        "models_relaunched": ["B2"],
        "models_not_relaunched": ["B1", "B_HVA"],
        "evaluator_seed": 7,
        "episodes": 200,
        "physical_gpu_id": gpu,
        "physical_gpu_uuid": uuid,
        "snapshots": snapshots,
        "frozen_sources_sha256": sources,
        "protected_raw_sha256": controls["raw_sha256"],
        "interruption_archive_sha256": sha256_file(
            RECOVERY_ROOT / "interruption_archive_manifest.json"
        ),
        "interruption_archive": archive,
        "log": str(RECOVERY_LOG),
    }
    atomic_json(RECOVERY_LAUNCH, launch_payload)
    command = shlex.join(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "supervise",
            "--gpu",
            str(gpu),
            "--uuid",
            uuid,
        ]
    )
    result = subprocess.run(
        ["tmux", "new-session", "-d", "-s", SESSION, command],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "failed to create B2 recovery tmux")
    time.sleep(3)
    if not tmux_alive():
        state = read_json(STATUS).get("state") if STATUS.exists() else "MISSING"
        if state not in {"DONE", "FAILED"}:
            raise RuntimeError("B2 recovery tmux exited before stable status")
    print(json.dumps(read_json(STATUS), sort_keys=True))


def supervise(gpu: int, uuid: str) -> None:
    if gpu not in range(4):
        raise SystemExit("physical GPU must be in range 0-3")
    launch_payload = read_json(RECOVERY_LAUNCH)
    if (
        launch_payload.get("physical_gpu_id") != gpu
        or launch_payload.get("physical_gpu_uuid") != uuid
        or launch_payload.get("models_relaunched") != ["B2"]
    ):
        raise RuntimeError("B2 recovery launch identity mismatch")
    verify_frozen_sources()
    controls = verify_completed_controls()
    if raw_path("B2").exists() or (CACHE / "b2").exists() or (TMP / "b2_inference.jsonl").exists():
        raise RuntimeError("fresh B2 recovery destinations are not empty")

    sys.path.insert(0, str(ROOT))
    from scripts.simulation import run_s4_3_pi2m_eval as formal

    frozen = formal.configure_frozen_runtime()
    lock = frozen.acquire_gpu_lock(gpu)
    started = time.time()
    running = {
        "schema": "tactile3d-unit.s4-3-pi2m-evaluation-job-status.v1",
        "state": "RECOVERY_RUNNING",
        "updated_at": now(),
        "session": SESSION,
        "supervisor_pid": os.getpid(),
        "physical_gpu_ids": [gpu],
        "physical_gpu_uuids": [uuid],
        "models_complete": ["B1", "B_HVA"],
        "active_model": "B2",
        "recovery": "B2_ONLY_CLEAN_REPLAY",
        "message": "frozen seed7 B2 clean replay running; B1/B_HVA are preserved",
        "log": str(RECOVERY_LOG),
    }
    atomic_json(STATUS, running)
    RECOVERY_LOG.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] | None = None
    failure: str | None = None
    try:
        snapshot = frozen.gpu_snapshot()
        inventory = {
            int(fields[0].strip()): fields[1].strip()
            for value in snapshot["inventory"]
            if len(fields := value.split(",")) >= 2
        }
        if inventory.get(gpu) != uuid or not frozen.gpu_is_idle(gpu, snapshot):
            raise RuntimeError("selected recovery GPU changed identity or became busy")
        with RECOVERY_LOG.open("w") as output:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                row = formal.run_one("B2", gpu, 8382)
        b2 = read_json(raw_path("B2"))
        b2_resets = [item["reset_identity"] for item in b2["episode_results"]]
        if (
            b2.get("status") != "PASS"
            or b2.get("episodes") != 200
            or len(b2.get("episode_results", [])) != 200
            or b2.get("evaluator_seed") != 7
            or b2_resets != controls["reset_identities"]
        ):
            raise RuntimeError("recovered B2 raw artifact failed completeness/reset parity")
        archived_names = [
            path.name
            for path in (RECOVERY_ROOT / "interrupted/cache/b2").iterdir()
            if path.is_dir() and not path.name.endswith("_temp")
        ]
        recovered_names = {path.name for path in (CACHE / "b2").iterdir() if path.is_dir()}
        replay_matches_interrupted_prefix = all(name in recovered_names for name in archived_names)
        completion = {
            "schema": "tactile3d-unit.s4-3-pi2m-b2-recovery-completion.v1",
            "status": "PASS",
            "completed_at": now(),
            "recovery_method": "B2_ONLY_CLEAN_REPLAY",
            "models_relaunched": ["B2"],
            "models_not_relaunched": ["B1", "B_HVA"],
            "B1_B_HVA_raw_sha256": controls["raw_sha256"],
            "B2_raw_sha256": sha256_file(raw_path("B2")),
            "B2_cache": tree_manifest(CACHE / "b2"),
            "B2_diagnostics_sha256": sha256_file(TMP / "b2_inference.jsonl"),
            "episodes": 200,
            "evaluator_seed": 7,
            "reset_sequence_sha256": canonical_sha256(b2_resets),
            "reset_identity_parity_B1_B_HVA_B2": True,
            "interrupted_complete_prefix_replayed_identically": replay_matches_interrupted_prefix,
            "interrupted_complete_prefix_length": 31,
            "worker": row,
            "elapsed_seconds": time.time() - started,
            "PI2B_started": False,
        }
        atomic_json(RECOVERY_COMPLETION, completion)
        final = {
            **running,
            "state": "DONE",
            "exit_code": 0,
            "updated_at": now(),
            "message": "formal B1/B_HVA artifacts preserved and clean B2 seed7 replay completed",
            "episodes_complete": {"B1": 200, "B_HVA": 200, "B2": 200},
            "total_rollouts_complete": 600,
            "completion_artifact": str(RECOVERY_COMPLETION),
            "worker": row,
        }
        atomic_json(STATUS, final)
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        with RECOVERY_LOG.open("a") as output:
            traceback.print_exc(file=output)
        atomic_json(
            STATUS,
            {
                **running,
                "state": "FAILED",
                "exit_code": 1,
                "updated_at": now(),
                "message": "B2-only recovery failed; preserved outputs require audit before retry",
                "failure": failure,
            },
        )
        raise
    finally:
        lock.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("launch")
    supervise_parser = sub.add_parser("supervise")
    supervise_parser.add_argument("--gpu", type=int, required=True)
    supervise_parser.add_argument("--uuid", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "launch":
        launch()
    else:
        supervise(args.gpu, args.uuid)


if __name__ == "__main__":
    main()
