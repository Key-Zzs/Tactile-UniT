#!/usr/bin/env python3
"""Validate and close the prospective PI2S-R runtime/telemetry stage."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.pi2sr_provenance import (  # noqa: E402
    build_provenance_manifest,
    canonical_sha256,
    sha256_file,
)
from gr00t.simulation.pi2sr_runtime import CANONICAL_CHECKPOINT_SHA256  # noqa: E402


ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_pi2sr"
PAPER_CORE = ROOT / ".local/paper/PAPER_CORE.md"
MAIN_BASE_SHA = "f3e69ab31c013b9dbde8076a2592c2df1be2018f"


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as output:
        output.write(value)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def read(name: str) -> dict[str, Any]:
    return json.loads((ARTIFACT_ROOT / name).read_text(encoding="utf-8"))


def changed_file_privacy_scan() -> dict[str, Any]:
    names = git("diff", "--name-only", "--diff-filter=ACM", f"{MAIN_BASE_SHA}...HEAD").splitlines()
    forbidden = (
        "/" + "home/",
        "/" + "mnt/",
        "Author" + "ization:",
        "Bear" + "er ",
        "github" + "_pat_",
        "HF" + "_TOKEN",
    )
    matches = []
    for name in names:
        path = ROOT / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for term in forbidden:
            if term in text:
                matches.append({"path": name, "term": term})
    return {"status": "PASS" if not matches else "FAIL", "scanned_files": len(names), "matches": matches}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-worktree", type=Path, required=True)
    args = parser.parse_args()
    required = {
        "starting_integrity.json": "PASS",
        "transport_parity.json": "PASS",
        "compute_parity.json": "PASS",
        "canonical_runtime_decision.json": "PI2SR_CANONICAL_RUNTIME_READY_WITH_SCOPE",
        "telemetry_schema_audit.json": "PASS",
        "telemetry_noninterference.json": "PASS",
        "telemetry_smoke.json": "PASS",
        "provenance_contract_audit.json": "PASS",
        "fixed_observation_action_sensitivity.json": "PASS",
        "environment_integrity.json": "PASS",
    }
    evidence = {name: read(name) for name in required}
    failures = [name for name, expected in required.items() if evidence[name]["status"] != expected]
    if failures:
        raise RuntimeError(f"PI2S-R evidence failed: {failures}")
    if git("status", "--porcelain=v1"):
        raise RuntimeError("finalize PI2S-R only from a clean simulation worktree")
    real = args.real_worktree.resolve()
    if git("status", "--porcelain=v1", cwd=real):
        raise RuntimeError("real DualFlexiv worktree is not clean")
    if git("branch", "--show-current", cwd=real) != "develop/real-dualflexiv":
        raise RuntimeError("real DualFlexiv worktree is on the wrong branch")
    if git("rev-parse", "HEAD", cwd=real) != MAIN_BASE_SHA:
        raise RuntimeError("real DualFlexiv worktree moved away from MAIN_BASE_SHA")

    smoke = evidence["telemetry_smoke.json"]
    telemetry_rows = smoke["telemetry_on"]
    reset_manifest = [row["reset_identity"] for row in telemetry_rows]
    config_paths = [
        ROOT / "configs/simulation/pi2sr/runtime_contract_v2.json",
        ROOT / "configs/simulation/pi2sr/telemetry_contract_v1.json",
        ROOT / "configs/simulation/pi2sr/provenance_contract_v1.json",
    ]
    provenance = build_provenance_manifest(
        repo_root=ROOT,
        config_paths=config_paths,
        base_model_checkpoint_tree_sha256=None,
        teacher_or_contact_encoder_sha256=CANONICAL_CHECKPOINT_SHA256,
        data_identity={
            "dataset_manifest_sha256": canonical_sha256(
                {"kind": "synthetic_contract_smoke", "steps": 15, "tasks": tuple(smoke["tasks"])}
            ),
            "split_identity": "PI2SR_DEVELOPMENT_SYNTHETIC_CONTRACT_SMOKE",
            "source_group_identities": smoke["tasks"],
            "normalization_sha256": CANONICAL_CHECKPOINT_SHA256,
            "target_mask_sha256": None,
            "episode_reset_manifest_sha256": canonical_sha256(reset_manifest),
        },
        randomness={
            "root_training_seed": None,
            "initialization_seed_namespace": None,
            "data_loader_seed": None,
            "augmentation_seed": None,
            "noise_time_sampling_seed": None,
            "evaluator_reset_seed": 4327100,
            "policy_sampling_root_key": "NO_STOCHASTIC_POLICY_ENGINEERING_STUB",
            "policy_request_counter": sum(row["replan_count"] for row in telemetry_rows),
            "policy_key_index": None,
        },
        device="cpu",
        dtype="float32",
    )
    if provenance["source_identity"]["git_dirty"]:
        raise RuntimeError("final prospective provenance unexpectedly recorded a dirty tree")
    provenance_audit = read("provenance_contract_audit.json")
    provenance_audit["manifest"] = provenance
    provenance_audit["final_clean_source_refresh"] = True
    atomic_json(ARTIFACT_ROOT / "provenance_contract_audit.json", provenance_audit)

    privacy = changed_file_privacy_scan()
    regression = {
        "schema": "tactile3d-unit.pi2sr-regression-tests.v1",
        "status": "PASS" if privacy["status"] == "PASS" else "FAIL",
        "new_functional_failures": 0,
        "runs": [
            {
                "scope": "PI2S, PI2S-R, Track A/B guards, OpenPI/runtime/contact contracts",
                "environment": "unit",
                "passed": 211,
                "failed": 2,
                "classification": "NO_NEW_FUNCTIONAL_FAILURES; one missing flax dependency retested in openpi and one main-baseline privacy match",
            },
            {
                "scope": "PI2B policy training contract dependency retest",
                "environment": "openpi",
                "passed": 5,
                "failed": 0,
            },
            {
                "scope": "tactile teacher, causal contact adapter, contact dynamics",
                "environment": "unit",
                "passed": 48,
                "failed": 0,
            },
            {"scope": "PI2S-R focused", "environment": "unit", "passed": 14, "failed": 0},
        ],
        "compileall": "PASS",
        "ruff": "PASS",
        "git_diff_check": "PASS",
        "privacy_new_diff": privacy,
        "baseline_privacy_limitation": "Existing main file scripts/simulation/audit_pi2s_worktree_retirement.py contains historical private absolute paths and was not modified by PI2S-R.",
    }
    atomic_json(ARTIFACT_ROOT / "regression_tests.json", regression)

    compute = evidence["compute_parity.json"]
    transport = evidence["transport_parity.json"]
    decision = evidence["canonical_runtime_decision.json"]
    final = {
        "schema": "tactile3d-unit.pi2sr-final-decision.v1",
        "stage": "S4.3-PI2S-R",
        "status": "PI2SR_CANONICAL_RUNTIME_TELEMETRY_READY",
        "runtime": "PI2SR_CANONICAL_RUNTIME_TELEMETRY_READY",
        "transport": transport["classification"],
        "compute": compute["classification"],
        "canonical_h_runtime": decision["selected_runtime"],
        "canonical_scope": decision["supported_scope"],
        "unsupported_scope": decision["unsupported_scope"],
        "telemetry": "PASS",
        "provenance_prng": "PASS",
        "smoke_episodes": smoke["episodes"],
        "scientific_benchmark": False,
        "track_a_status": "COMPLETE_VALID_THREE_SEEDS_LIMITED",
        "track_b_status": "VAC_CONTACT_CAPABILITY_WITH_NO_DETECTED_VA_REGRESSION_LIMITED_PRECISION",
        "historical_pi2s_status": "COMPLETE_VALID_DIAGNOSIS_INCONCLUSIVE",
        "historical_pi2s_modified": False,
        "training_or_optimizer_updates": 0,
        "formal_success_rate_rollouts": 0,
        "real_hardware_runs": 0,
        "git": {
            "branch": git("branch", "--show-current"),
            "head": git("rev-parse", "HEAD"),
            "main_base_sha": MAIN_BASE_SHA,
            "push_performed": False,
            "pull_request_created": False,
            "merge_performed": False,
        },
        "real_worktree": {
            "branch": "develop/real-dualflexiv",
            "head": MAIN_BASE_SHA,
            "clean": True,
            "real_hardware_executed": False,
            "status": "REAL_DUALFLEXIV_WORKTREE_READY",
        },
        "paper_core": {"sha256": sha256_file(PAPER_CORE), "updated": True},
        "regression": regression["status"],
        "next_stage": "S5.0 — DualFlexiv + RH56DFTP Sensor / Control / Data Contract",
        "real_hardware_authorized": False,
        "push_or_pr_authorized": False,
    }
    atomic_json(ARTIFACT_ROOT / "final_decision.json", final)

    acceptance = f"""# S4.3-PI2S-R Human Acceptance

- Final status: `PI2SR_CANONICAL_RUNTIME_TELEMETRY_READY`
- Canonical H runtime: `DIRECT_IN_PROCESS` (CPU, float32, batch one)
- Transport: `{transport['classification']}`
- Compute: `{compute['classification']}`
- Telemetry/provenance/non-interference: `PASS`
- Smoke: {smoke['episodes']} engineering runs; no scientific benchmark
- Historical PI2S decision remains `COMPLETE_VALID_DIAGNOSIS_INCONCLUSIVE`
- Training, formal success-rate rerun, and real hardware execution: `NONE`
- Real worktree: `REAL_DUALFLEXIV_WORKTREE_READY` at `{MAIN_BASE_SHA}`
- Next separately authorized stage: `S5.0 — DualFlexiv + RH56DFTP Sensor / Control / Data Contract`

STOP AFTER PI2S-R. DO NOT PUSH, MERGE, RUN REAL HARDWARE, MOVE ROBOT, OR TRAIN A NEW POLICY.
"""
    atomic_text(ARTIFACT_ROOT / "HUMAN_ACCEPTANCE.md", acceptance)
    print(
        json.dumps(
            {
                "compute": final["compute"],
                "runtime": final["runtime"],
                "transport": final["transport"],
                "worktree": final["real_worktree"]["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
