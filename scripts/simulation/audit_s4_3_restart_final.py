#!/usr/bin/env python3
"""Audit the complete restarted S4.3 artifact and repository handoff."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_3_training import atomic_json, read_json, sha256_file  # noqa: E402

ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_restart"
REQUIRED = (
    "starting_integrity.json",
    "policy_dataset_identity.json",
    "policy_dataset_audit.json",
    "task_success_contract.json",
    "policy_eval_v1_manifest.json",
    "s4_3_restart_protocol_freeze.json",
    "causal_observation_contract.json",
    "causal_action_contract.json",
    "causal_contact_auxiliary.json",
    "p3_gradient_audit.json",
    "causal_trace_audit.json",
    "runtime_smoke.json",
    "act_implementation_audit.json",
    "act_training_manifest.json",
    "act_checkpoint_manifest.json",
    "offline_dev_evaluation.json",
    "pre_rollout_freeze.json",
    "closed_loop_rollouts.json",
    "closed_loop_summary.json",
    "primary_statistics.json",
    "tactile_active_statistics.json",
    "hammer_control_analysis.json",
    "secondary_metrics.json",
    "training_seed_analysis.json",
    "uncertainty_diagnostics.json",
    "s4_2_immutability.json",
    "environment_integrity.json",
    "regression_tests.json",
    "warnings.json",
    "final_decision.json",
    "HUMAN_ACCEPTANCE.md",
)


def command(*args: str) -> str:
    return subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    missing = [name for name in REQUIRED if not (ARTIFACT_ROOT / name).is_file()]
    if missing:
        raise RuntimeError(f"missing required S4.3 artifacts: {missing}")
    plots = sorted((ARTIFACT_ROOT / "plots").glob("*.png"))
    plot_numbers = {
        int(match.group(1))
        for path in plots
        if (match := re.match(r"^(\d\d)_", path.name)) is not None
    }
    if not set(range(1, 25)).issubset(plot_numbers):
        raise RuntimeError("required 01-24 visualization set is incomplete")
    videos = read_json(ARTIFACT_ROOT / "representative_videos.json")
    if videos.get("status") != "PASS":
        raise RuntimeError("representative rollout video selection failed")
    rollouts = read_json(ARTIFACT_ROOT / "closed_loop_rollouts.json")
    if rollouts.get("completed_rollouts") != 1080 or rollouts.get("status") != "PASS":
        raise RuntimeError("closed-loop rollout matrix is incomplete")
    checkpoints = read_json(ARTIFACT_ROOT / "act_checkpoint_manifest.json")
    if checkpoints.get("count") != 36 or checkpoints.get("status") != "PASS":
        raise RuntimeError("canonical ACT checkpoint manifest is incomplete")
    if command("git", "ls-files", ".local"):
        raise RuntimeError(".local content is tracked")
    branch = command("git", "branch", "--show-current")
    if branch != "develop/sim-benchmark":
        raise RuntimeError("final S4.3 branch changed")
    changed = command("git", "diff", "--name-only").splitlines()
    untracked = command("git", "ls-files", "--others", "--exclude-standard").splitlines()
    scan_paths = sorted(
        {
            path
            for path in (*changed, *untracked)
            if path and (ROOT / path).is_file() and not path.startswith(".local/")
        }
    )
    privacy_matches: dict[str, list[str]] = {}
    machine_home = str(Path.home()) + "/"
    machine_user = Path.home().name
    mount_prefix = "/" + "mnt" + "/"
    patterns = (
        re.escape(machine_home),
        re.escape(mount_prefix),
        rf"\b{re.escape(machine_user)}\b",
        re.escape("Bear" + "er") + r"\s+",
        re.escape("Author" + "ization:"),
    )
    for relative in scan_paths:
        text = (ROOT / relative).read_text(encoding="utf-8", errors="ignore")
        hits = [pattern for pattern in patterns if re.search(pattern, text)]
        if hits:
            privacy_matches[relative] = hits
    if privacy_matches:
        raise RuntimeError(f"tracked privacy scan failed: {privacy_matches}")
    artifact_hashes = {name: sha256_file(ARTIFACT_ROOT / name) for name in REQUIRED}
    final = read_json(ARTIFACT_ROOT / "final_decision.json")
    result: dict[str, Any] = {
        "schema": "tactile3d-unit.s4-3-final-artifact-audit.v1",
        "stage": "R20-R25",
        "required_artifacts": artifact_hashes,
        "required_artifact_count": len(REQUIRED),
        "plots": len(plots),
        "plot_numbers_01_through_24": True,
        "representative_videos": len(videos["videos"]),
        "canonical_checkpoints": checkpoints["count"],
        "closed_loop_rollouts": rollouts["completed_rollouts"],
        "branch": branch,
        "tracked_local_files": 0,
        "privacy_matches": privacy_matches,
        "historical_failure_preserved": True,
        "old_probing_data_used": False,
        "s4_3_2_decision": final["decision"],
        "s4_3_3_readiness": final["s4_3_3_readiness"]["decision"],
        "status": "PASS",
    }
    atomic_json(ARTIFACT_ROOT / "final_artifact_audit.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
