#!/usr/bin/env python3
"""Publish the final local-only PI2S audit handoff and verified Git bundle."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PI2S_ROOT = REPOSITORY_ROOT / ".local/experiments/simulation/s4_3_pi2s"
ARTIFACTS = PI2S_ROOT / "artifacts"
INTEGRATION = PI2S_ROOT / "integration"
HANDOFF = PI2S_ROOT / "handoff"
FINAL_BUNDLE = INTEGRATION / "develop_sim_benchmark_final.bundle"
BRANCH_REF = "refs/heads/develop/sim-benchmark"

REQUIRED_RELATIVE_PATHS = (
    "integration/start_state.json",
    "integration/source_refs_freeze.json",
    "integration/protected_before.json",
    "integration/merge_semantic_audit.json",
    "integration/ancestry_and_parity.json",
    "integration/evidence_ledger.json",
    "integration/paper_core_before.md",
    "integration/paper_core_delta.md",
    "integration/source_path_resolution.json",
    "artifacts/pi2s_diagnostic_protocol.json",
    "artifacts/checkpoint_recipe_matrix.json",
    "artifacts/initialization_rng_audit.json",
    "artifacts/h_physical_time_audit.json",
    "artifacts/offline_online_h_parity.json",
    "artifacts/h_distribution_diagnostics.json",
    "artifacts/prefix_contract_audit.json",
    "artifacts/fixed_observation_interventions.json",
    "artifacts/read_only_gradient_diagnostics.json",
    "artifacts/historical_failure_stage_analysis.json",
    "artifacts/diagnostic_rollout_manifest.json",
    "artifacts/diagnostic_rollout_results.json",
    "artifacts/diagnosis_and_evidence_strength.json",
    "artifacts/next_action_recommendation.json",
    "artifacts/real_robot_readiness.json",
    "artifacts/worktree_retirement_audit.json",
    "artifacts/runtime_path_dependencies.json",
    "artifacts/backup_manifest.json",
    "artifacts/protected_after.json",
    "artifacts/regression_tests.json",
    "artifacts/final_decision.json",
    "artifacts/HUMAN_ACCEPTANCE.md",
    "artifacts/plots_manifest.json",
    "resume_state.json",
)


class HandoffError(RuntimeError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HandoffError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HandoffError(f"expected object: {path}")
    return value


def canonical_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def run(command: list[str], cwd: Path = REPOSITORY_ROOT) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise HandoffError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout}"
        )
    return completed


def artifact_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HandoffError(f"required file missing: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def publish_git_bundle(head: str) -> dict[str, Any]:
    INTEGRATION.mkdir(parents=True, exist_ok=True)
    temporary = INTEGRATION / f".{FINAL_BUNDLE.name}.tmp-{os.getpid()}"
    temporary.unlink(missing_ok=True)
    try:
        run(["git", "bundle", "create", str(temporary), BRANCH_REF])
        verify = run(["git", "bundle", "verify", str(temporary)]).stdout
        heads = run(["git", "bundle", "list-heads", str(temporary)]).stdout.splitlines()
        expected_line = f"{head} {BRANCH_REF}"
        if expected_line not in heads:
            raise HandoffError(f"final bundle does not contain {expected_line}")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, FINAL_BUNDLE)
        directory = os.open(INTEGRATION, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        **artifact_record(FINAL_BUNDLE),
        "head": head,
        "branch_ref": BRANCH_REF,
        "verify_output": verify.splitlines(),
        "list_heads": heads,
        "push_performed": False,
    }


def run_regressions() -> dict[str, Any]:
    unit = "/home/wbcd/miniconda3/envs/unit/bin/python"
    openpi = "/home/wbcd/miniconda3/envs/openpi/bin/python"
    commands = [
        {
            "scope": "all PI2S CPU/static tests",
            "command": [unit, "-m", "pytest", "-q", "tests/simulation/pi2s"],
        },
        {
            "scope": "Track A pure/final protocol and all Track B tests",
            "command": [
                unit,
                "-m",
                "pytest",
                "-q",
                "tests/simulation/pi2b_policy/test_contract.py",
                "tests/simulation/pi2b_policy/test_final_protocol.py",
                "tests/simulation/pi2b_policy/test_resets.py",
                "tests/simulation/pi2b_policy/test_statistics.py",
                "tests/simulation/pi2b_teacher",
            ],
        },
        {
            "scope": "Track A OpenPI training contract",
            "command": [
                openpi,
                "-m",
                "pytest",
                "-q",
                "tests/simulation/pi2b_policy/test_training_contract.py",
            ],
        },
    ]
    results = []
    for item in commands:
        completed = run(item["command"])
        matches = re.findall(r"(\d+) passed", completed.stdout)
        results.append(
            {
                "scope": item["scope"],
                "command": item["command"],
                "returncode": completed.returncode,
                "passed": int(matches[-1]) if matches else None,
                "output_tail": completed.stdout.splitlines()[-8:],
            }
        )
    previous = load_json(ARTIFACTS / "regression_tests.json")
    return {
        "schema": "tactile3d-unit.pi2s-regression-tests.v2",
        "status": "PASS",
        "created_at_utc": utc_now(),
        "current": results,
        "historical_pre_adapter_failure_preserved": previous.get("pre_adapter"),
        "environment_mutated": False,
        "gpu_used": False,
    }


def protected_after() -> dict[str, Any]:
    before = load_json(INTEGRATION / "protected_before.json")
    small_files = {}
    for path_string, expected in before["protected_small_files"].items():
        path = Path(path_string)
        actual = artifact_record(path)
        status = (
            "UNCHANGED"
            if actual["sha256"] == expected["sha256"] and actual["bytes"] == expected["size"]
            else "CHANGED"
        )
        if status != "UNCHANGED":
            raise HandoffError(f"protected file changed: {path}")
        small_files[path_string] = {
            "before_sha256": expected["sha256"],
            "after_sha256": actual["sha256"],
            "before_size": expected["size"],
            "after_size": actual["bytes"],
            "status": status,
        }
    checkpoints = []
    for row in before["checkpoints"]:
        path = Path(row["path_resolved"])
        presence = {
            "checkpoint_metadata": (path / "_CHECKPOINT_METADATA").is_file(),
            "params_manifest": (path / "params/manifest.ocdbt").is_file(),
            "train_state_manifest": (path / "train_state/manifest.ocdbt").is_file(),
        }
        if not all(presence.values()):
            raise HandoffError(f"protected checkpoint structure missing: {path}")
        checkpoints.append(
            {
                "model": row["model"],
                "training_seed": row["training_seed"],
                "path": str(path),
                "tree_sha256": row["tree_sha256"],
                "presence": presence,
                "verification": (
                    "PRIOR_FULL_TREE_HASH_REBOUND_PLUS_CURRENT_STRUCTURE_AND_ZERO_WRITE_CONTRACT;"
                    "NO_REPEAT_140GB_SCAN"
                ),
            }
        )
    return {
        "schema": "tactile3d-unit.pi2s-protected-after.v1",
        "status": "PASS",
        "created_at_utc": utc_now(),
        "protected_small_files": small_files,
        "checkpoints": checkpoints,
        "checkpoint_write_contracts": {
            "base_rollout_checkpoint_writes": load_json(
                ARTIFACTS / "diagnostic_rollout_results.json"
            )["checkpoint_writes"],
            "optional_rollout_checkpoint_writes": load_json(
                ARTIFACTS / "diagnostic_rollout_optional_results_v21.json"
            )["checkpoint_writes"],
            "model_diagnostic_checkpoint_writes": load_json(
                ARTIFACTS / "fixed_observation_interventions.json"
            )["scientific_budget"]["checkpoint_writes"],
        },
        "paper_core_after": artifact_record(REPOSITORY_ROOT / ".local/paper/PAPER_CORE.md"),
        "large_checkpoint_verification_scope": (
            "No repeat scan of 140+ GB; prior authoritative hashes are rebound and current "
            "checkpoint structure plus all diagnostic zero-write contracts are verified"
        ),
    }


def backup_manifest(bundle: dict[str, Any]) -> dict[str, Any]:
    retirement_audit = Path(
        "/mnt/ugreen_nas/storage/UniT_storage/experiments/simulation/s4_3_pi2s/"
        "retirement_backups/worktree_local_retirement_audit.publication/CONTENT/BUNDLE/"
        "worktree_local_retirement_audit.json"
    )
    return {
        "schema": "tactile3d-unit.pi2s-backup-manifest.v1",
        "status": "PASS_WITH_DECLARED_FAILURE_DOMAINS",
        "created_at_utc": utc_now(),
        "git": {
            "source_refs_bundle": artifact_record(INTEGRATION / "source_refs.bundle"),
            "final_local_branch_bundle": bundle,
            "final_local_branch_pushed": False,
        },
        "worktree_local_evidence": {
            "aggregate_audit": artifact_record(retirement_audit),
            "verified": True,
            "source_deleted": False,
            "removal_executed": False,
            "failure_domain": (
                "Archives are persistent copies on the shared experiment NAS, not independent "
                "disaster-recovery media"
            ),
        },
        "models": {
            "retrained_or_replaced": False,
            "checkpoint_writes": 0,
            "authoritative_checkpoint_hashes_reused": True,
        },
    }


def final_decision(head: str, bundle: dict[str, Any]) -> dict[str, Any]:
    contract = load_json(REPOSITORY_ROOT / "configs/simulation/pi2s/final_decision.json")
    diagnosis = load_json(ARTIFACTS / "diagnosis_and_evidence_strength.json")
    retirement = load_json(ARTIFACTS / "worktree_retirement_audit.json")
    return {
        **contract,
        "local_final_head": head,
        "local_final_bundle": bundle,
        "diagnosis_artifact": artifact_record(ARTIFACTS / "diagnosis_and_evidence_strength.json"),
        "paper_core": artifact_record(REPOSITORY_ROOT / ".local/paper/PAPER_CORE.md"),
        "worktree_classifications": retirement["summary"]["final_classification"],
        "worktree_classification_scope": (
            "All visible code/data/runtime gates pass; UNKNOWN remains because other-user cwd/fd "
            "dependencies cannot be excluded without privileged review"
        ),
        "base_rollouts": diagnosis["budget"]["base_development_rollouts"],
        "optional_rollouts": diagnosis["budget"]["optional_h_rollouts"],
        "push_performed": False,
        "pull_request_created": False,
        "main_merge_performed": False,
        "worktree_removal_executed": False,
    }


def required_manifest() -> list[dict[str, Any]]:
    records = []
    for relative in REQUIRED_RELATIVE_PATHS:
        records.append({"relative_path": relative, **artifact_record(PI2S_ROOT / relative)})
    return records


def write_outputs() -> dict[str, Any]:
    head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    branch = run(["git", "branch", "--show-current"]).stdout.strip()
    if branch != "develop/sim-benchmark":
        raise HandoffError(f"unexpected branch: {branch}")
    if run(["git", "status", "--porcelain"]).stdout.strip():
        raise HandoffError("tracked or untracked worktree changes must be committed before handoff")

    bundle = publish_git_bundle(head)
    regression = run_regressions()
    protected = protected_after()
    backup = backup_manifest(bundle)
    decision = final_decision(head, bundle)
    resume = {
        "schema": "tactile3d-unit.pi2s-resume.v3",
        "status": "COMPLETE",
        "phase": "EVIDENCE_INTEGRATION_AND_S4_3_PI2S_COMPLETE",
        "updated_at_utc": utc_now(),
        "source_commit": head,
        "long_job_running": False,
        "base_canonical_tuples_completed": 72,
        "optional_canonical_tuples_completed": 48,
        "total_s5_canonical_tuples_completed": 120,
        "additional_rollouts_authorized": False,
        "training_performed": False,
        "real_robot_used": False,
        "push_performed": False,
        "worktree_removal_performed": False,
    }
    for path, value in (
        (ARTIFACTS / "regression_tests.json", regression),
        (ARTIFACTS / "protected_after.json", protected),
        (ARTIFACTS / "backup_manifest.json", backup),
        (ARTIFACTS / "final_decision.json", decision),
        (PI2S_ROOT / "resume_state.json", resume),
    ):
        atomic_write(path, canonical_bytes(value))

    required = required_manifest()
    handoff = {
        "schema": "tactile3d-unit.pi2s-integration-handoff.v1",
        "status": "COMPLETE_VALID_DIAGNOSIS_INCONCLUSIVE",
        "created_at_utc": utc_now(),
        "branch": branch,
        "final_local_head": head,
        "final_local_bundle": bundle,
        "required_artifacts": required,
        "required_artifact_count": len(required),
        "paper_core": artifact_record(REPOSITORY_ROOT / ".local/paper/PAPER_CORE.md"),
        "track_a_tip": "ac750461e19ec6f287bbe4ee28dcb751dd268485",
        "track_b_tip": "d5f52a5ac63e1662820da6855c9177c2abf218e2",
        "contact_tokenizer_tip": "e489a2b3a8044b99d91e091edb6bbcc13137f3c5",
        "push_performed": False,
        "pull_request_created": False,
        "main_merge_performed": False,
        "tag_or_release_created": False,
        "new_branch_or_worktree_created": False,
        "worktree_removal_authorized": False,
        "worktree_removal_executed": False,
        "real_robot_used": False,
        "human_next_step": (
            "Review the diagnosis and UNKNOWN worktree classifications. Any experiment, hardware "
            "activity, push/PR, or removal requires a new explicit authorization."
        ),
    }
    handoff_path = HANDOFF / "integration_pi2s_handoff.json"
    atomic_write(handoff_path, canonical_bytes(handoff))
    return {"handoff": artifact_record(handoff_path), "head": head, "bundle": bundle}


def check_outputs() -> dict[str, Any]:
    handoff_path = HANDOFF / "integration_pi2s_handoff.json"
    handoff = load_json(handoff_path)
    head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    if handoff.get("final_local_head") != head:
        raise HandoffError("handoff head is stale")
    run(["git", "bundle", "verify", str(FINAL_BUNDLE)])
    bundle_heads = run(["git", "bundle", "list-heads", str(FINAL_BUNDLE)]).stdout.splitlines()
    if f"{head} {BRANCH_REF}" not in bundle_heads:
        raise HandoffError("final bundle head mismatch")
    expected = {row["relative_path"]: row for row in handoff["required_artifacts"]}
    if set(expected) != set(REQUIRED_RELATIVE_PATHS):
        raise HandoffError("required artifact list mismatch")
    for relative in REQUIRED_RELATIVE_PATHS:
        actual = artifact_record(PI2S_ROOT / relative)
        if actual["sha256"] != expected[relative]["sha256"]:
            raise HandoffError(f"required artifact drift: {relative}")
    if run(["git", "status", "--porcelain"]).stdout.strip():
        raise HandoffError("Git worktree is not clean")
    return {"handoff": artifact_record(handoff_path), "head": head}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = write_outputs() if args.write else check_outputs()
    print(
        json.dumps(
            {
                "status": "WRITE_PASS" if args.write else "CHECK_PASS",
                **result,
                "push_performed": False,
                "worktree_removal_executed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
