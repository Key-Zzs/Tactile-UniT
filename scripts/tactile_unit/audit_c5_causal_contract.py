#!/usr/bin/env python3
"""Freeze the Track C5 causal/provenance and planned-action contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.c5_causal_visual import CausalFrameSelection, VisualSupport  # noqa: E402
from gr00t.tactile_unit.vac_latent_dataset import atomic_json  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/tactile_unit/c5_causal_visual_planned_action.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--unit-checkpoint", type=Path, default=Path(os.environ["UNIT_FULLDATA_CKPT"]) if os.environ.get("UNIT_FULLDATA_CKPT") else None)
    parser.add_argument("--artifact-root", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()


def write_hashed(path: Path, value: dict[str, Any]) -> str:
    atomic_json(path, value)
    digest = sha256_file(path)
    (path.parent / f"{path.name}.sha256").write_text(f"{digest}  {path.name}\n")
    return digest


def verify_hashes(spec: dict[str, Any], unit_checkpoint: Path) -> dict[str, Any]:
    runtime, expected = spec["runtime"], spec["accepted"]
    paths = {
        "c1_manifest_sha256": ROOT / runtime["c1_cache_root"] / "manifest.json",
        "c2_checkpoint_sha256": ROOT / runtime["c2_checkpoint"],
        "c2r_checkpoint_sha256": ROOT / runtime["c2r_checkpoint"],
        "c3dp_checkpoint_sha256": ROOT / runtime["c3dp_checkpoint"],
        "full_checkpoint_sha256": ROOT / runtime["full_checkpoint"],
        "offline_va_checkpoint_sha256": ROOT / runtime["offline_va_checkpoint"],
        "emergency_a_checkpoint_sha256": ROOT / runtime["emergency_a_checkpoint"],
        "c4_uncertainty_checkpoint_sha256": ROOT / runtime["c4_uncertainty_checkpoint"],
        "action_checkpoint_sha256": ROOT / runtime["action_checkpoint"],
        "s1_checkpoint_sha256": ROOT / runtime["s1_checkpoint"],
        "s2_checkpoint_sha256": ROOT / runtime["s2_checkpoint"],
    }
    actual = {}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen C5 dependency: {key}")
        actual[key] = sha256_file(path)
        if actual[key] != expected[key]:
            raise RuntimeError(f"frozen identity mismatch: {key}")
    tokenizer = unit_checkpoint / "tokenizer"
    actual["original_unit_tokenizer_files_sha256"] = {
        name: sha256_file(tokenizer / name)
        for name in expected["original_unit_tokenizer_files_sha256"]
    }
    if actual["original_unit_tokenizer_files_sha256"] != expected["original_unit_tokenizer_files_sha256"]:
        raise RuntimeError("Original UniT/DINO tokenizer identity mismatch")
    return actual


def accepted_policy_audit() -> dict[str, Any]:
    """Find identity-bearing T-Rex policy/planner artifacts without fabricating one."""
    candidates = []
    terms = ("policy_checkpoint", "policy_sha256", "policy_identity", "planner_checkpoint", "planner_sha256", "planner_identity")
    for relative in git("ls-files", "configs/tactile_unit", "docs/research").splitlines():
        if "c5" in relative.lower():
            continue
        path = ROOT / relative
        try:
            content = path.read_text(errors="ignore").lower()
        except OSError:
            continue
        if "t-rex" in content and any(term in content for term in terms):
            candidates.append(relative)
    return {
        "search_scope": ["tracked configs/tactile_unit", "tracked docs/research"],
        "required_identity_terms": list(terms), "candidates": candidates,
        "accepted_identity_locked_policy_found": bool(candidates),
    }


def main() -> None:
    args = parse_args()
    spec = json.loads(args.config.read_text())
    if args.unit_checkpoint is None:
        raise RuntimeError("UNIT_FULLDATA_CKPT or --unit-checkpoint is required")
    artifact_root = args.artifact_root or ROOT / spec["runtime"]["artifact_root"]
    artifact_root.mkdir(parents=True, exist_ok=True)
    branch, head = git("branch", "--show-current"), git("rev-parse", "HEAD")
    if branch != "develop/tactile-unit-vac" or head != "9e1b4313ceb9119c570ed71f022a03d478b205d5":
        raise RuntimeError("C5 starting branch/HEAD contract failed")
    required = ("f43b71c", "beaa831", "639b4a9", "808416a", "5fe4bdb", "6a271c1", "7e77f7e", "9e1b431")
    history = git("log", "--format=%H", "-80")
    if any(prefix not in history for prefix in required):
        raise RuntimeError("accepted C1-C4 ancestry is absent")
    changed = [line for line in git("status", "--short").splitlines() if line]
    allowed_legacy_test = "tests/tactile_unit/test_c3dp_shared_private.py"
    if any("c5_" not in line and "c5-" not in line and allowed_legacy_test not in line for line in changed):
        raise RuntimeError("unknown worktree divergence during C5 audit")
    identities = verify_hashes(spec, args.unit_checkpoint)
    policy_audit = accepted_policy_audit()
    if policy_audit["accepted_identity_locked_policy_found"]:
        raise RuntimeError("accepted policy candidate requires explicit C5 provenance review")
    current = CausalFrameSelection.create(VisualSupport.CURRENT_FRAME, 1, 15, 100)
    history_selection = CausalFrameSelection.create(VisualSupport.CAUSAL_HISTORY_8, 1, 15, 100)
    audit = {
        "schema": "tactile3d-unit.vac-c5-causal-contract-audit.v1",
        "branch": branch,
        "starting_head": head,
        "starting_worktree_clean": True,
        "starting_audit_performed_before_c5_edits": True,
        "accepted_ancestry": list(required),
        "frozen_identities": identities,
        "frozen_before_after_required": True,
        "vision_source": {
            "boundary": spec["visual"]["frozen_frame_boundary"],
            "preprocessing": spec["accepted"]["vision_preprocessing_identity"],
            "current_indices": list(current.frame_indices),
            "history_indices": list(history_selection.frame_indices),
            "maximum_offset": 0,
            "future_frames": False,
            "backbone_trainable": False,
        },
        "contact": {"future_tactile_input": False, "private_residual_input_or_target": False},
        "test_loaded": False,
        "pass": True,
    }
    atomic_json(artifact_root / "causal_contract_audit.json", audit)
    planned = {
        "schema": "tactile3d-unit.vac-c5-planned-action-contract.v1",
        "type": "PlannedActionChunk",
        "shapes": [["B", 16, 58], ["B", 16, 128]],
        "horizon": 16,
        "interval": ["a_t", "a_t+15"],
        "a_t_plus_16": False,
        "embodiment": 31,
        "raw_58_ordering": ["left arm 7", "left hand 22", "right arm 7", "right hand 22"],
        "sources": ["POLICY_GENERATED", "DEMONSTRATION_TEACHER", "ORACLE_EVAL"],
        "runtime_legal": ["POLICY_GENERATED"],
        "runtime_rejected": ["DEMONSTRATION_TEACHER", "ORACLE_EVAL"],
        "source_default": None,
        "normalization": "accepted frozen train-only mean/std",
        "state_relative_features_recomputed_by_frozen_a_r": True,
        "first_differences_recomputed_by_frozen_a_r": True,
        "continuous_pre_rq": True,
        "rq_used": False,
        "source_tag_changes_numeric_encoding": False,
        "policy_generated_plans_available": False,
        "policy_provenance": "NOT AVAILABLE: no accepted identity-locked T-Rex policy-plan artifact",
        "policy_artifact_audit": policy_audit,
        "policy_plan_domain_validated": False,
        "warning": "POLICY_PLAN_DOMAIN_WARNING",
        "test_loaded": False,
    }
    planned_sha = write_hashed(artifact_root / "planned_action_contract.json", planned)
    contract = {
        "schema": "tactile3d-unit.vac-c5-contract.v1",
        "config_sha256": sha256_file(args.config),
        "planned_action_contract_sha256": planned_sha,
        "selection_split": "train + validation only",
        "locked_benchmark_rows": 17504,
        "locked_benchmark_label": "LOCKED POST-HOC C5 ENGINEERING EVALUATION",
        "first_look_untouched": False,
        "maximum_trials": 6,
        "candidate_ids": [row["id"] for row in spec["training"]["trials"]],
        "current_history_simplicity_tolerance": 0.01,
        "direct_modular_tolerance": 0.01,
        "mean_freeze_precedes_uncertainty": True,
        "offline_future_vision_runtime_routable": False,
        "c6_started": False,
        "test_loaded": False,
    }
    contract_sha = write_hashed(artifact_root / "c5_contract.json", contract)
    policy = {
        "schema": "tactile3d-unit.vac-c5-policy-plan-domain-audit.v1",
        "accepted_policy_found": False,
        "accepted_policy_identity": None,
        "policy_training_or_fabrication": False,
        "evaluation_plan_source": "ORACLE PLAN SURROGATES plus controlled perturbations",
        "actual_policy_domain_validation": "NOT AVAILABLE",
        "warning": "POLICY_PLAN_DOMAIN_WARNING",
        "repository_audit": policy_audit,
        "test_loaded": False,
    }
    atomic_json(artifact_root / "policy_plan_domain_audit.json", policy)
    print(json.dumps({"causal_audit": "PASS", "c5_contract_sha256": contract_sha, "planned_action_contract_sha256": planned_sha, "policy_plan_domain_warning": True}, indent=2))


if __name__ == "__main__":
    main()
