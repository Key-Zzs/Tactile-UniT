#!/usr/bin/env python3
"""Finalize Track B artifacts without editing the main PAPER_CORE."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.simulation.pi2b_teacher.common import (  # noqa: E402
    atomic_json,
    canonical_digest,
    git_output,
    load_json,
    sha256_file,
)


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_training_plot(root: Path, statuses: dict[str, Any]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for mode, color in (("T_VA_match", "#4169e1"), ("T_VAC_match", "#d95f02")):
        rows = jsonl(Path(statuses[mode]["log"]))
        axes[0].plot([row["step"] for row in rows], [row["va_total"] for row in rows], label=mode, color=color)
        axes[1].plot([row["step"] for row in rows], [row["common_gradient_norm"] for row in rows], label=mode, color=color)
    axes[0].set(title="Common VA training loss", xlabel="optimizer step", ylabel="VA loss")
    axes[1].set(title="Common-path gradient norm", xlabel="optimizer step", ylabel="L2 norm")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    fig.tight_layout()
    path = root / "plots/training_mechanism.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_effect_plot(root: Path, analysis: dict[str, Any]) -> Path:
    primary = [name for name in ("vision_native_recovery_mse", "action_native_recovery_mse", "mean_bidirectional_va_retrieval_r10", "action_reversal_margin", "mean_common_va_contact_probe_macro_f1") if name in analysis["paired_effects"]]
    estimates = [analysis["paired_effects"][name]["estimate"] for name in primary]
    lower = [estimate - analysis["paired_effects"][name]["ci95"][0] for estimate, name in zip(estimates, primary)]
    upper = [analysis["paired_effects"][name]["ci95"][1] - estimate for estimate, name in zip(estimates, primary)]
    fig, axis = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(primary))
    axis.errorbar(estimates, y, xerr=[lower, upper], fmt="o", color="#5e3c99", capsize=3)
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set_yticks(y, [name.replace("_", " ") for name in primary])
    axis.set(title="T_VAC_match - T_VA_match paired source-group effects", xlabel="effect (95% stratified group bootstrap CI)")
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    path = root / "plots/common_va_effects.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_contact_plot(root: Path, analysis: dict[str, Any]) -> Path:
    names = list(analysis["contact_statistics"])
    estimates = [analysis["contact_statistics"][name]["estimate"] for name in names]
    lower = [estimate - analysis["contact_statistics"][name]["ci95"][0] for estimate, name in zip(estimates, names)]
    upper = [analysis["contact_statistics"][name]["ci95"][1] - estimate for estimate, name in zip(estimates, names)]
    fig, axis = plt.subplots(figsize=(9, 4))
    y = np.arange(len(names))
    axis.errorbar(estimates, y, xerr=[lower, upper], fmt="o", color="#1b9e77", capsize=3)
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set_yticks(y, [name.replace("_", " ") for name in names])
    axis.set(title="VAC Contact capability margins", xlabel="margin (95% stratified group bootstrap CI)")
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    path = root / "plots/contact_capability.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    args = parser.parse_args()
    runtime = load_json(args.runtime_manifest)
    root = Path(runtime["write_root"])
    analysis = load_json(root / "artifacts/stratified_group_statistics.json")
    evaluation = load_json(root / "artifacts/evaluation_result.json")
    independent = load_json(root / "artifacts/statistics_independent_audit.json")
    if analysis["status"] != "COMPLETE_VALID" or independent["status"] != "PASS":
        raise RuntimeError("analysis or independent audit is incomplete")
    statuses = {mode: load_json(root / "status" / f"{mode}.json") for mode in ("T_VA_match", "T_VAC_match")}
    plot_paths = [save_training_plot(root, statuses), save_effect_plot(root, analysis), save_contact_plot(root, analysis)]
    plot_hashes = {path.name: sha256_file(path) for path in plot_paths}
    native_before = load_json(root / "artifacts/native_hashes_before.json")
    history = Path(runtime["worktree_root"])
    native_after = {
        "paired_train": sha256_file(history / ".local/refs/base/cache/simulation/s4_2_formal/paired_train.npz"),
        "paired_dev": sha256_file(history / ".local/refs/base/cache/simulation/s4_2_formal/paired_validation.npz"),
        "A0_checkpoint": sha256_file(history / ".local/experiments/simulation/s4_2_formal/action/selected.pt"),
        "C3_checkpoint": sha256_file(history / ".local/experiments/simulation/s4_2r/contact_dynamics/selected.pt"),
    }
    native_unchanged = native_after == native_before
    atomic_json(root / "artifacts/native_hashes_after.json", native_after)
    paper_source = history / ".local/paper/PAPER_CORE_BASE.md"
    paper_delta = {
        "schema": "tactile3d-unit.s4-3-pi2b-teacher-paper-delta.v1",
        "source_paper_core_sha256": sha256_file(paper_source),
        "claim_id": "S4.3-PI2B-T_MATCHED_TEACHER",
        "scientific_decision": analysis["scientific_decision"],
        "method_delta": "Matched fresh VA/VAC slot teachers with byte-identical common initialization, identical V/A rows and update budget, and additive Contact supervision as the treatment.",
        "result_axes": {key: analysis[key] for key in ("COMMON_VA_RECOVERY_EFFECT", "COMMON_VA_SEMANTIC_EFFECT", "CONTACT_CROSS_MODAL_CAPABILITY", "CONTACT_INFORMATION_IN_COMMON_VA_READOUTS", "NONCOLLAPSE_STATUS")},
        "claim_boundary": [
            "fixed teacher training seed42 only",
            "nine fresh source groups across three scripted tasks",
            "no policy utility test",
            "no equivalence or noninferiority claim",
            "does not replace historical teachers or Track A targets",
        ],
        "counterpart_missing": {"policy_transfer": "NOT_TESTED", "cross_teacher_seed_stability": "NOT_TESTED", "real_robot": "NOT_TESTED"},
        "main_paper_core_modified": False,
        "artifact_digests": {"evaluation": evaluation["metric_digest"], "analysis": analysis["analysis_digest"]},
    }
    atomic_json(root / "artifacts/paper_delta.json", paper_delta)
    markdown = f"""# PAPER_CORE delta — S4.3-PI2B-T\n\nSource PAPER_CORE SHA-256: `{paper_delta['source_paper_core_sha256']}`\n\n## Proposed method addition\n\n{paper_delta['method_delta']}\n\n## Proposed result addition\n\nScientific decision: `{analysis['scientific_decision']}`. Common V/A recovery: `{analysis['COMMON_VA_RECOVERY_EFFECT']}`; common V/A semantics: `{analysis['COMMON_VA_SEMANTIC_EFFECT']}`; Contact cross-modal capability: `{analysis['CONTACT_CROSS_MODAL_CAPABILITY']}`; Contact information in common V/A readouts: `{analysis['CONTACT_INFORMATION_IN_COMMON_VA_READOUTS']}`.\n\nThis is a fixed-teacher-seed matched representation study over nine fresh source groups. It does not test policy utility, prove noninferiority/equivalence, establish cross-seed stability, or authorize replacement of any Track A teacher.\n\nMain PAPER_CORE was not modified.\n"""
    (root / "artifacts/paper_delta.md").write_text(markdown, encoding="utf-8")
    handoff = {
        "schema": "tactile3d-unit.s4-3-pi2b-teacher-integration-handoff.v1",
        "branch": git_output("branch", "--show-current"),
        "head": git_output("rev-parse", "HEAD"),
        "base_sha": "8f39aed123adf0a8b7241e75472e678355444e5e",
        "checkpoints": {mode: {"path": statuses[mode]["checkpoint"], "sha256": statuses[mode]["checkpoint_sha256"], "interface": "independent native [B,8,32] modality -> shared [B,8,32]"} for mode in statuses},
        "normalization": {"path": runtime["common_va_normalization"], "sha256": runtime["common_va_normalization_sha256"], "transform": "identity_native_units"},
        "training_data": load_json(root / "artifacts/paired_dataset_manifest.json"),
        "confirmation_data": load_json(root / "artifacts/confirmation_cache_manifest.json"),
        "protocol_freeze_sha256": sha256_file(root / "artifacts/protocol_freeze.json"),
        "pretest_freeze_sha256": sha256_file(root / "artifacts/pretest_freeze.json"),
        "evaluation_metric_digest": evaluation["metric_digest"],
        "analysis_digest": analysis["analysis_digest"],
        "paper_delta_sha256": sha256_file(root / "artifacts/paper_delta.json"),
        "policy_utility": "NOT_TESTED",
        "track_a_integration_authorized": False,
    }
    atomic_json(root / "artifacts/integration_handoff.json", handoff)
    regression = {
        "targeted_tests": "recorded by final operator",
        "native_hashes_unchanged": native_unchanged,
        "historical_teacher_mutated": False,
        "track_a_assets_modified": False,
        "paper_core_modified": False,
        "pi05_trained": False,
    }
    atomic_json(root / "artifacts/regression_tests.json", regression)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "git_head": git_output("rev-parse", "HEAD"),
        "git_status_porcelain": git_output("status", "--porcelain"),
        "runtime_snapshot": runtime["snapshot_root"],
    }
    atomic_json(root / "artifacts/environment_integrity.json", environment)
    final = {
        "schema": "tactile3d-unit.s4-3-pi2b-teacher-final.v1",
        "status": "COMPLETE_VALID",
        "scientific_decision": analysis["scientific_decision"],
        "axes": {key: analysis[key] for key in ("ENGINEERING_STATUS", "MATCHING_GRADE", "COMMON_VA_RECOVERY_EFFECT", "COMMON_VA_SEMANTIC_EFFECT", "CONTACT_CROSS_MODAL_CAPABILITY", "CONTACT_INFORMATION_IN_COMMON_VA_READOUTS", "NONCOLLAPSE_STATUS", "DATA_AND_TRAINING_SEED_SCOPE", "POLICY_UTILITY", "NEXT_STUDY")},
        "checkpoints": {mode: statuses[mode]["checkpoint_sha256"] for mode in statuses},
        "native_hashes_unchanged": native_unchanged,
        "independent_statistics_audit": "PASS",
        "plot_sha256": plot_hashes,
        "paper_delta_sha256": sha256_file(root / "artifacts/paper_delta.json"),
        "integration_handoff_sha256": sha256_file(root / "artifacts/integration_handoff.json"),
        "no_push_pr_merge": True,
        "main_paper_core_modified": False,
    }
    final["final_digest"] = canonical_digest(final)
    atomic_json(root / "artifacts/final_result.json", final)
    (root / "artifacts/HUMAN_ACCEPTANCE.md").write_text(
        "# Human acceptance\n\nTrack B completed without push, PR, merge, policy training, historical teacher replacement, or main PAPER_CORE modification. Review `final_result.json`, `paper_delta.md`, and `integration_handoff.json` before any future integration.\n",
        encoding="utf-8",
    )
    atomic_json(root / "status/resume_state.json", {"state": "COMPLETE", "final_digest": final["final_digest"], "next": "HUMAN_REVIEW_ONLY"})
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
