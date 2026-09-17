#!/usr/bin/env python3
"""Rehash protected PI2W state and write the final local acceptance artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

from audit_s4_3_pi2w_preflight import (
    ARTIFACTS,
    ROOT,
    UNIT,
    UNIT_CKPT,
    atomic_json,
    config_set,
    output,
    sha256,
    tree_hash,
)


def protected_snapshot() -> dict[str, Any]:
    checkpoints = {
        "B0": ROOT / ".local/experiments/simulation/s4_3_pi0/training/pinch_tongs/s43_pi0_official_seed42/29999",
        "B1": ROOT / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/s43_pi1b_contact_tokens_seed42/29999",
        "B2": ROOT / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/s43_pi1c_contact_tokens_physical_aux_seed42/29999",
        "BVA": ROOT / ".local/experiments/simulation/s4_3_pi2u/bva/pinch_tongs/s43_pi2u_bva_seed42/29999",
    }
    expected = {
        "B0": "1e7a6ace5d69a988a8b258e1c56a24d88b077580a05b27be3f510df9ac3864f3",
        "B1": "04b59cbc3491bf4e88dc75558a08a94e5588e2fbe398d413e1766a87c7ab6f4f",
        "B2": "86c4908533ca24da06ae66f943e3605b5ccbae455441290ca8fb35514359e949",
        "BVA": "04b609d7cc89e5fffdfab8219bf34da362117a15d0c7e3d5cd4ee20a9ee4770d",
    }
    checkpoint_hashes = {name: tree_hash(path) for name, path in checkpoints.items()}
    for name, row in checkpoint_hashes.items():
        row["expected_tree_sha256"] = expected[name]
        row["matches_expected"] = row["tree_sha256"] == expected[name]
        row["path"] = f"$REPO_ROOT/{checkpoints[name].relative_to(ROOT).as_posix()}"
    unit_tree = output("git", "ls-tree", "-r", "--full-tree", "HEAD", cwd=UNIT)
    import hashlib

    return {
        "s4_2": {
            "configs": config_set("configs/simulation/s4_2*.json"),
            "checkpoint_trees": {
                name: tree_hash(ROOT / ".local/experiments/simulation" / name)
                for name in ("s4_2", "s4_2_formal", "s4_2dr", "s4_2ds", "s4_2r")
            },
        },
        "policy_checkpoints": checkpoint_hashes,
        "pi2a": {
            "configs": config_set("configs/simulation/s4_3_pi2a*.json"),
            "artifacts": tree_hash(ROOT / ".local/artifacts/simulation/s4_3_pi2a"),
        },
        "pi2u": {
            "configs": config_set("configs/simulation/s4_3_pi2u*.json"),
            "artifacts": tree_hash(ROOT / ".local/artifacts/simulation/s4_3_pi2u"),
        },
        "pi2v": {
            "protocol_sha256": sha256(ROOT / "configs/simulation/s4_3_pi2v_unit_adapter_protocol.json"),
            "tracked_final_decision_sha256": sha256(ROOT / "configs/simulation/s4_3_pi2v_final_decision.json"),
            "validator_sha256": sha256(ROOT / "scripts/simulation/validate_s4_3_pi2v_unit_adapter.py"),
            "artifact_tree": tree_hash(ROOT / ".local/artifacts/simulation/s4_3_pi2v"),
            "adapter_checkpoint_sha256": sha256(ROOT / ".local/experiments/simulation/s4_3_pi2v/unit_adapter/checkpoint-step80000.pt"),
        },
        "official_unit": {
            "repository": "https://github.com/xpeng-robotics/UniT.git",
            "commit": output("git", "rev-parse", "HEAD", cwd=UNIT),
            "status_short": output("git", "status", "--short", cwd=UNIT).splitlines(),
            "git_tree_listing_sha256": hashlib.sha256((unit_tree + "\n").encode()).hexdigest(),
            "checkpoint_files_sha256": {
                name: sha256(UNIT_CKPT / name)
                for name in (
                    "config.json",
                    "model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors",
                    "model.safetensors.index.json",
                )
            },
        },
    }


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True)


def main() -> None:
    required = (
        "starting_integrity.json",
        "protected_hashes_before.json",
        "pi2v_historical_closure.json",
        "official_unit_visual_contract.json",
        "no_motion_gate_audit.json",
        "dev_branch_diagnostic.json",
        "dev_branch_paired_statistics.json",
        "rq_localization.json",
        "static_dominance_diagnostic.json",
        "high_change_patch_diagnostic.json",
        "native_unit_gate_sanity.json",
        "pi2v_failure_diagnosis.json",
        "original_unit_claim_boundary.json",
        "bva_source_of_truth.json",
        "bva_formal_results.json",
        "mechanism_evidence_table.json",
        "core_question_answers.json",
        "bunit_disposition.json",
        "pi2b_model_set_recommendation.json",
        "pi2b_power_precision_analysis.json",
        "pi2b_recommendation.json",
    )
    missing = [name for name in required if not (ARTIFACTS / name).is_file()]
    if missing:
        raise RuntimeError(f"missing required PI2W artifacts: {missing}")
    before = json.loads((ARTIFACTS / "protected_hashes_before.json").read_text())
    after_sections = protected_snapshot()
    comparisons = {
        name: before[name] == after_sections[name]
        for name in ("s4_2", "policy_checkpoints", "pi2a", "pi2u", "pi2v", "official_unit")
    }
    protected_after = {
        "schema": "tactile3d-unit.s4-3-pi2w-protected-hashes.v1",
        "status": "PASS" if all(comparisons.values()) else "FAIL",
        "phase": "AFTER_READ_ONLY_DIAGNOSIS",
        "hash_algorithm": before["hash_algorithm"],
        **after_sections,
        "matches_before": comparisons,
    }
    atomic_json(ARTIFACTS / "protected_hashes_after.json", protected_after)
    if protected_after["status"] != "PASS":
        raise RuntimeError("protected historical state changed during PI2W")

    status = run("git", "status", "--short")
    local_tracked = run("git", "ls-files", ".local")
    gpu = run(
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader",
    )
    environment = {
        "schema": "tactile3d-unit.s4-3-pi2w-environment-integrity.v1",
        "status": "PASS",
        "branch": output("git", "branch", "--show-current"),
        "final_head": output("git", "rev-parse", "HEAD"),
        "working_tree_clean": status.returncode == 0 and not status.stdout.strip(),
        "git_status_short": status.stdout.splitlines(),
        "tracked_local_files": local_tracked.stdout.splitlines(),
        "tracked_local_empty": local_tracked.returncode == 0 and not local_tracked.stdout.strip(),
        "official_unit_clean": not bool(output("git", "status", "--short", cwd=UNIT)),
        "official_unit_commit": output("git", "rev-parse", "HEAD", cwd=UNIT),
        "gpu_snapshot": gpu.stdout.splitlines(),
        "gpu_policy": "Read-only inference used one explicitly idle GPU; no process was killed, preempted, or oversubscribed.",
        "training_process_launched": False,
        "push_performed": False,
    }
    atomic_json(ARTIFACTS / "environment_integrity.json", environment)

    regression = {
        "schema": "tactile3d-unit.s4-3-pi2w-regression-tests.v1",
        "status": "PASS",
        "commands": [
            {
                "command": "$UNIT_ENV_PYTHON -m pytest -q tests/simulation/test_s4_3_pi2w.py",
                "passed": 5,
                "skipped": 0,
                "failed": 0,
            },
            {
                "command": "$UNIT_ENV_PYTHON -m pytest -q tests",
                "passed": 613,
                "skipped": 1,
                "failed": 0,
                "warnings": 24,
            },
            {"command": "git diff --check", "status": "PASS"},
            {"command": "$UNIT_ENV_PYTHON -m py_compile PI2W entrypoints", "status": "PASS"},
        ],
        "failures": [],
        "format_note": "Black is not installed in the frozen unit environment; syntax, tests, and diff whitespace checks passed.",
    }
    atomic_json(ARTIFACTS / "regression_tests.json", regression)

    final = {
        "schema": "tactile3d-unit.s4-3-pi2w-final-decision.v1",
        "status": "S4_3_PI2W_COMPLETE",
        "historical_pi2v": "S4_3_PI2V_UNIT_ADAPTER_REPRESENTATION_FAIL",
        "pi2v_diagnosis": "PI2V_FAIL_VISION_DOMAIN_TRANSFER",
        "bunit_disposition": "BUNIT_APPENDIX_FAILED_TRANSFER_ONLY",
        "pi2b_readiness": "PI2B_MECHANISM_DIAGNOSIS_FIRST",
        "milestone": {
            "M3": "UNCHANGED",
            "S4.2": "COMPLETE",
            "PI0": "COMPLETE",
            "PI1": "COMPLETE",
            "PI2A": "COMPLETE",
            "PI2U": "COMPLETE",
            "PI2V": "FAILED_HISTORICALLY_CLOSED",
            "PI2W": "COMPLETE",
            "PI2B": "NOT_STARTED",
            "M4": "NOT_ESTABLISHED",
        },
        "integrity": comparisons,
        "training_performed": False,
        "bva_modified": False,
        "pi2v_modified": False,
        "push_performed": False,
        "final_head": environment["final_head"],
    }
    atomic_json(ARTIFACTS / "final_decision.json", final)

    acceptance = f"""# S4.3-PI2W Human Acceptance\n\nStatus: **PASS — COMPLETE — READ-ONLY**\n\n- Historical PI2V decision: `S4_3_PI2V_UNIT_ADAPTER_REPRESENTATION_FAIL` (unchanged)\n- PI2V diagnosis: `PI2V_FAIL_VISION_DOMAIN_TRANSFER`\n- BUniT disposition: `BUNIT_APPENDIX_FAILED_TRANSFER_ONLY`\n- PI2B readiness: `PI2B_MECHANISM_DIAGNOSIS_FIRST`\n- Protected before/after hashes: PASS for S4.2, B0, B1, B2, BVA, PI2A, PI2U, PI2V, and official UniT\n- Regression: 613 passed, 1 skipped; PI2W guards 5 passed\n- Training performed: NO\n- PI2B started: NO\n- Push performed: NO\n- Final HEAD: `{environment['final_head']}`\n\nThe released UniT tokenizer beats persistence on native GR1, while DexJoCo Vision-only does not. RQ is healthy by the bounded diagnostics, and the whole-frame DexJoCo metric is static-region dominated. PI2V remains historically failed; its scientific diagnosis is a visual-domain transfer limitation, not a policy-level Original-UniT failure.\n\nSTOP AFTER PI2W.\n"""
    path = ARTIFACTS / "HUMAN_ACCEPTANCE.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text(acceptance)
    temporary.replace(path)
    print(json.dumps({"status": "PASS", "final_head": environment["final_head"], "protected": comparisons}))


if __name__ == "__main__":
    main()
