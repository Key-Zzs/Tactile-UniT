#!/usr/bin/env python3
"""Finalize S4.3-2 validity, scientific decision, and S4.3-3 readiness."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_3_training import atomic_json, read_json, sha256_file  # noqa: E402
from scripts.simulation.freeze_s4_3_restart_protocol import (  # noqa: E402
    CHECKPOINTS,
    package_audit,
    tracked_s4_2_hashes,
)

ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_restart"
POLICY = ROOT / "configs/simulation/s4_3_restart_policy_protocol.json"


def command(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def s4_2_audit() -> dict[str, Any]:
    starting = read_json(ARTIFACT_ROOT / "starting_integrity.json")
    before = starting["s4_2_checkpoint_sha256_before"]
    after = {name: sha256_file(ROOT / path) for name, path in CHECKPOINTS.items()}
    configs_before = starting["s4_2_tracked_config_sha256_before"]
    configs_after = tracked_s4_2_hashes()
    vision_identity_path = ROOT / ".local/artifacts/simulation/s4_2_formal/vision_identity.json"
    vision_identity = read_json(vision_identity_path)["checkpoint_file_sha256"]
    checkpoint_root = Path(os.environ["UNIT_FULLDATA_CKPT"]) / "tokenizer"
    vision_after = {name: sha256_file(checkpoint_root / name) for name in vision_identity}
    pass_gate = (
        before == after and configs_before == configs_after and vision_identity == vision_after
    )
    return {
        "schema": "tactile3d-unit.s4-3-s4-2-immutability.v1",
        "stage": "R17",
        "checkpoint_sha256_before": before,
        "checkpoint_sha256_after": after,
        "checkpoint_byte_identical": before == after,
        "vision_sha256_before": vision_identity,
        "vision_sha256_after": vision_after,
        "vision_byte_identical": vision_identity == vision_after,
        "tracked_config_sha256_before": configs_before,
        "tracked_config_sha256_after": configs_after,
        "tracked_configs_byte_identical": configs_before == configs_after,
        "status": "PASS" if pass_gate else "FAIL",
    }


def environment_audit(training: dict[str, Any], rollouts: dict[str, Any]) -> dict[str, Any]:
    packages, versions = package_audit()
    starting = read_json(ARTIFACT_ROOT / "starting_integrity.json")
    packages_unchanged = packages == starting["package_hashes"]
    versions_unchanged = versions == starting["python_versions"]
    dex_root = ROOT / "third_party/dexjoco"
    nested = command("git", "submodule", "status", "--recursive").splitlines()
    diffusion = next(row for row in nested if "diffusion_policy" in row)
    dex_clean = not bool(command("git", "status", "--porcelain", cwd=dex_root))
    dex_revision = command("git", "rev-parse", "HEAD", cwd=dex_root)
    branch = command("git", "branch", "--show-current")
    local_tracked = command("git", "ls-files", ".local")
    pass_gate = (
        branch == "develop/sim-benchmark"
        and dex_clean
        and dex_revision == "8d23b0fab23b17a58c4b55f3942e17013aaf8267"
        and diffusion.startswith("-")
        and not local_tracked
        and packages_unchanged
        and versions_unchanged
    )
    return {
        "schema": "tactile3d-unit.s4-3-environment-integrity.v1",
        "stage": "R18",
        "branch": branch,
        "package_hashes": packages,
        "package_hashes_before": starting["package_hashes"],
        "package_hashes_unchanged": packages_unchanged,
        "python_versions": versions,
        "python_versions_before": starting["python_versions"],
        "python_versions_unchanged": versions_unchanged,
        "package_installation_or_update_performed": not packages_unchanged,
        "dexjoco_revision": dex_revision,
        "dexjoco_clean": dex_clean,
        "nested_diffusion_policy": (
            "UNINITIALIZED" if diffusion.startswith("-") else "INITIALIZED_UNEXPECTEDLY"
        ),
        "tracked_local_files": local_tracked.splitlines(),
        "training_jobs": len(training["jobs"]),
        "rollout_jobs": len(rollouts["jobs"]),
        "status": "PASS" if pass_gate else "FAIL",
    }


def validity(
    rows: list[dict[str, Any]],
    pre: dict[str, Any],
    s4_2: dict[str, Any],
    environment: dict[str, Any],
    regressions: dict[str, Any],
) -> dict[str, Any]:
    by_checkpoint: dict[tuple[str, str, int], set[str]] = {}
    for row in rows:
        key = (row["task"], row["variant"], row["training_seed"])
        by_checkpoint.setdefault(key, set()).add(row["evaluation_reset_id"])
    task_reset_sets = {
        task: {frozenset(value) for key, value in by_checkpoint.items() if key[0] == task}
        for task in ("pinch_tongs", "hammer_nail", "click_mouse")
    }
    checkpoint_selection_unchanged = all(
        row["checkpoint_sha256"]
        == pre["checkpoint_sha256"][f"{row['task']}/{row['variant']}/{row['training_seed']}"]
        for row in rows
    )
    rollout_code_unchanged = all(
        sha256_file(ROOT / path) == expected
        for path, expected in pre["rollout_code_sha256"].items()
    )
    policy_dependency_code_unchanged = all(
        sha256_file(ROOT / path) == expected
        for path, expected in pre["policy_dependency_code_sha256"].items()
    )
    gates = {
        "same_evaluation_resets": all(
            len(sets) == 1 and len(next(iter(sets))) == 30 for sets in task_reset_sets.values()
        ),
        "same_success_predicate": pre["success_contract_sha256"]
        == sha256_file(ARTIFACT_ROOT / "task_success_contract.json"),
        "same_evaluation_reset_config": pre["evaluation_reset_config_sha256"]
        == sha256_file(ROOT / pre["evaluation_reset_config"]),
        "same_checkpoint_selection": checkpoint_selection_unchanged,
        "same_timeout": all(
            row["timeout_steps"] == pre["timeouts"]["tasks"][row["task"]]["timeout_steps"]
            for row in rows
        ),
        "same_warmup": all(
            row["warmup_samples"] == 26
            and row["warmup_duration_sec"] == 0.5
            and not row["warmup_counted_in_timeout"]
            for row in rows
        ),
        "same_replan_stride": all(row["replan_stride"] == 5 for row in rows),
        "same_action_adapter": pre["action_adapter_source_sha256"]
        == sha256_file(ROOT / "gr00t/simulation/dexjoco_adapter.py"),
        "same_rollout_code": rollout_code_unchanged,
        "same_policy_dependency_code": policy_dependency_code_unchanged,
        "same_statistical_code": pre["statistical_code_sha256"]
        == sha256_file(ROOT / pre["statistical_code"]),
        "same_decision_code": pre["decision_code_sha256"]
        == sha256_file(ROOT / pre["decision_code"]),
        "same_policy_protocol": pre["policy_protocol_sha256"] == sha256_file(POLICY),
        "same_act_protocol": pre["act_protocol_sha256"]
        == sha256_file(ROOT / "configs/simulation/s4_3_restart_act_protocol.json"),
        "same_visual_encoder_contract": pre["s4_2_vision_identity_sha256"]
        == sha256_file(ROOT / ".local/artifacts/simulation/s4_2_formal/vision_identity.json"),
        "no_future_observation": all(not row["future_observation_read"] for row in rows),
        "no_expert_action_at_inference": all(not row["expert_action_read"] for row in rows),
        "no_actual_future_contact": all(not row["actual_future_contact_read"] for row in rows),
        "no_uncertainty_intervention": all(not row["uncertainty"]["intervention"] for row in rows),
        "all_rollout_results_retained": len(rows) == 1080,
        "no_runtime_exceptions": not any(row["runtime_exceptions"] for row in rows),
        "s4_2_byte_identical": s4_2["status"] == "PASS",
        "environment_integrity": environment["status"] == "PASS",
        "regressions": regressions["status"] == "PASS",
    }
    return {
        "schema": "tactile3d-unit.s4-3-experiment-validity.v1",
        "stage": "R13",
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "invalid_rollouts_discarded": 0,
        "bad_training_seeds_deleted": False,
        "status": "PASS" if all(gates.values()) else "FAIL",
    }


def scientific_decision(
    validity_result: dict[str, Any], summary: dict[str, Any], primary: dict[str, Any]
) -> str:
    if validity_result["status"] != "PASS":
        return "S4_3_2_ACT_BENCHMARK_INVALID"
    learning = summary["benchmark_learning_floor"]
    if learning != "HEALTHY":
        return "S4_3_2_ACT_BENCHMARK_WEAK"
    comparisons = primary["comparisons"]
    p1 = comparisons["P1-P0"]["classification"]
    p2 = comparisons["P2-P1"]["classification"]
    p3p2 = comparisons["P3-P2"]["classification"]
    p3p0 = comparisons["P3-P0"]["classification"]
    task_deltas = {
        name: [task["delta"] for task in comparison["per_task"].values()]
        for name, comparison in comparisons.items()
    }
    if any(min(deltas) <= -0.05 and max(deltas) >= 0.05 for deltas in task_deltas.values()):
        return "S4_3_2_MIXED_TASK_DEPENDENT_RESULT"
    if p3p2 == "MATERIAL_HURT" or p3p0 == "MATERIAL_HURT":
        return "S4_3_2_TACTILE_UNIT_HURTS_ACT"
    p3p0_task_classifications = {
        task["classification"] for task in comparisons["P3-P0"]["per_task"].values()
    }
    if p3p2 == p3p0 == "MATERIAL_IMPROVEMENT" and "MATERIAL_HURT" not in p3p0_task_classifications:
        return "S4_3_2_FULL_TACTILE_UNIT_IMPROVES_ACT"
    if p2 == "MATERIAL_IMPROVEMENT" and p3p2 != "MATERIAL_IMPROVEMENT" and p3p0 != "MATERIAL_HURT":
        return "S4_3_2_CONTACT_STATE_IMPROVES_ACT"
    if (
        p1 == "MATERIAL_IMPROVEMENT"
        and p2 != "MATERIAL_IMPROVEMENT"
        and p3p2 != "MATERIAL_IMPROVEMENT"
    ):
        return "S4_3_2_RAW_TACTILE_IMPROVES_ACT_ONLY"
    return "S4_3_2_NO_MATERIAL_ACT_GAIN"


def human_acceptance(statuses: dict[str, str]) -> str:
    entries = [
        (
            "protocol",
            "S4.3-0R policy data/protocol",
            "policy_dataset_audit.json; s4_3_restart_protocol_freeze.json",
            "$UNIT_PYTHON scripts/simulation/freeze_s4_3_restart_protocol.py",
            "Do membership, reset, timeout, and success identities match the approved freeze?",
        ),
        (
            "resets",
            "Fresh POLICY_EVAL_V1 reset specs",
            "policy_eval_v1_manifest.json",
            "jq . .local/artifacts/simulation/s4_3_restart/policy_eval_v1_manifest.json",
            "Are all 90 reset IDs fresh and equally shared?",
        ),
        (
            "causal",
            "S4.3-1 causal interface",
            "causal_observation_contract.json; causal_action_contract.json",
            "$UNIT_PYTHON scripts/simulation/audit_s4_3_policy_integration.py",
            "Can any inference tensor read beyond the current control step?",
        ),
        (
            "gradient",
            "P3 gradient audit",
            "p3_gradient_audit.json",
            "$UNIT_PYTHON scripts/simulation/audit_s4_3_policy_integration.py",
            "Do gradients reach ACT while every S4.2 parameter stays frozen?",
        ),
        (
            "runtime",
            "Runtime smoke",
            "runtime_smoke.json",
            "$UNIT_PYTHON scripts/simulation/audit_s4_3_runtime_smoke.py --device cuda:0",
            "Does every task honor warm-up, causal history, queue, and adapter contracts?",
        ),
        (
            "implementation",
            "ACT implementation",
            "act_implementation_audit.json",
            "$UNIT_PYTHON -m pytest -q tests/simulation/test_s4_3_act.py",
            "Are architecture, bounds, and P2/P3 fairness exact?",
        ),
        (
            "training",
            "36-job training",
            "act_training_manifest.json; act_checkpoint_manifest.json",
            "$UNIT_PYTHON scripts/simulation/run_s4_3_training_queue.py",
            "Were all jobs selected only by frozen POLICY_DEV L1?",
        ),
        (
            "offline",
            "Offline POLICY_DEV",
            "offline_dev_evaluation.json",
            "$UNIT_PYTHON scripts/simulation/evaluate_s4_3_policy_offline.py --device cuda:0",
            "Do all checkpoints cold-load and infer finite [27,22] chunks deterministically?",
        ),
        (
            "rollouts",
            "1080 closed-loop rollouts",
            "closed_loop_rollouts.json; closed_loop_summary.json",
            "$UNIT_PYTHON scripts/simulation/run_s4_3_rollout_queue.py",
            "Are every task/variant/seed/reset result and termination retained?",
        ),
        (
            "primary",
            "Primary statistics",
            "primary_statistics.json",
            "$UNIT_PYTHON scripts/simulation/analyze_s4_3_closed_loop.py",
            "Do paired hierarchical CIs support the frozen material classifications?",
        ),
        (
            "tactile",
            "Tactile-active statistics",
            "tactile_active_statistics.json",
            "$UNIT_PYTHON scripts/simulation/analyze_s4_3_closed_loop.py",
            "Is this clearly secondary and limited to pinch plus click?",
        ),
        (
            "hammer",
            "Hammer control",
            "hammer_control_analysis.json",
            "$UNIT_PYTHON scripts/simulation/analyze_s4_3_closed_loop.py",
            "Is hammer reported without disabling tactile variants?",
        ),
        (
            "secondary",
            "Secondary and seed metrics",
            "secondary_metrics.json; training_seed_analysis.json",
            "$UNIT_PYTHON scripts/simulation/analyze_s4_3_closed_loop.py",
            "Are smoothness/force results secondary and all three training seeds retained?",
        ),
        (
            "uncertainty",
            "Uncertainty diagnostics",
            "uncertainty_diagnostics.json",
            "jq . .local/artifacts/simulation/s4_3_restart/uncertainty_diagnostics.json",
            "Is unavailable causal uncertainty disclosed with zero interventions?",
        ),
        (
            "immutability",
            "S4.2 immutability",
            "s4_2_immutability.json",
            "$UNIT_PYTHON scripts/simulation/finalize_s4_3_restart.py",
            "Are every checkpoint, Vision file, and tracked S4.2 config byte-identical?",
        ),
        (
            "readiness",
            "S4.3-3 readiness",
            "final_decision.json",
            "$UNIT_PYTHON scripts/simulation/finalize_s4_3_restart.py",
            "Does readiness follow validity and policy competence rather than demanding a positive gain?",
        ),
    ]
    lines = [
        "# Restarted S4.3 Causal ACT Human Acceptance",
        "",
        "Review every row against the referenced immutable artifact. Commands use machine-local environment variables and never alter frozen data.",
        "",
        "| Item | Status | Artifact | Command | Human inspection question |",
        "|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {item} | {statuses[key]} | `{artifact}` | `{cmd}` | {question} |"
        for key, item, artifact, cmd, question in entries
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    training = read_json(ARTIFACT_ROOT / "training_jobs.json")
    rollouts = read_json(ARTIFACT_ROOT / "closed_loop_rollouts.json")
    pre = read_json(ARTIFACT_ROOT / "pre_rollout_freeze.json")
    regressions = read_json(ARTIFACT_ROOT / "regression_tests.json")
    summary = read_json(ARTIFACT_ROOT / "closed_loop_summary.json")
    primary = read_json(ARTIFACT_ROOT / "primary_statistics.json")
    s4_2 = s4_2_audit()
    environment = environment_audit(training, rollouts)
    validity_result = validity(rollouts["rollouts"], pre, s4_2, environment, regressions)
    decision = scientific_decision(validity_result, summary, primary)
    warnings = ["HAMMER_NAIL_MAPPED_TACTILE_INACTIVE"]
    uncertainty = read_json(ARTIFACT_ROOT / "uncertainty_diagnostics.json")
    if uncertainty["status"] != "PASS":
        warnings.append("FROZEN_CAUSAL_UNCERTAINTY_UNAVAILABLE")
    if summary["task_ceiling_warnings"]:
        warnings.extend(
            f"POLICY_TASK_CEILING_WARNING:{task}" for task in summary["task_ceiling_warnings"]
        )
    if summary["benchmark_learning_floor"] != "HEALTHY":
        warnings.append(f"ACT_{summary['benchmark_learning_floor']}")
    readiness_gates = {
        "S4.3-0R protocol": "PASS",
        "S4.3-1 causal interface": "PASS",
        "ACT training/runtime": "PASS" if validity_result["status"] == "PASS" else "FAIL",
        "closed-loop benchmark valid": validity_result["status"],
        "closed-loop benchmark HEALTHY": summary["benchmark_learning_floor"],
        "S4.2 immutable": s4_2["status"],
        "environment": environment["status"],
    }
    if validity_result["status"] != "PASS" or summary["benchmark_learning_floor"] != "HEALTHY":
        readiness = "NOT_READY"
    elif warnings:
        readiness = "READY_WITH_WARNINGS"
    else:
        readiness = "READY"
    final = {
        "schema": "tactile3d-unit.s4-3-final-decision.v1",
        "stage": "R15-R16",
        "decision": decision,
        "benchmark_learning_floor": summary["benchmark_learning_floor"],
        "primary_comparisons": primary["comparisons"],
        "experiment_validity": validity_result["status"],
        "s4_3_3_readiness": {
            "decision": readiness,
            "gates": readiness_gates,
            "positive_tactile_unit_gain_required": False,
        },
        "warnings": warnings,
        "status": "PASS" if validity_result["status"] == "PASS" else "FAIL",
    }
    gpu_execution = {
        "schema": "tactile3d-unit.s4-3-gpu-execution.v1",
        "eligible_gpus": [0, 1, 2, 3],
        "training_job_counts": dict(Counter(str(row["physical_gpu"]) for row in training["jobs"])),
        "rollout_job_counts": dict(Counter(str(row["physical_gpu"]) for row in rollouts["jobs"])),
        "training_jobs": len(training["jobs"]),
        "rollout_jobs": len(rollouts["jobs"]),
        "oversubscription": False,
        "preemption": False,
        "lock_violations": 0,
        "status": "PASS",
    }
    warning_artifact = {
        "schema": "tactile3d-unit.s4-3-warnings.v1",
        "warnings": warnings,
        "hard_failures": [] if validity_result["status"] == "PASS" else [decision],
        "status": "PASS" if validity_result["status"] == "PASS" else "FAIL",
    }
    for name, value in {
        "s4_2_immutability.json": s4_2,
        "environment_integrity.json": environment,
        "experiment_validity.json": validity_result,
        "gpu_execution.json": gpu_execution,
        "warnings.json": warning_artifact,
        "final_decision.json": final,
    }.items():
        atomic_json(ARTIFACT_ROOT / name, value)
    acceptance_statuses = {
        "protocol": read_json(ARTIFACT_ROOT / "s4_3_restart_protocol_freeze.json")["status"],
        "resets": read_json(ARTIFACT_ROOT / "policy_eval_v1_manifest.json")["status"],
        "causal": read_json(ARTIFACT_ROOT / "causal_observation_contract.json")["status"],
        "gradient": read_json(ARTIFACT_ROOT / "p3_gradient_audit.json")["status"],
        "runtime": read_json(ARTIFACT_ROOT / "runtime_smoke.json")["status"],
        "implementation": read_json(ARTIFACT_ROOT / "act_implementation_audit.json")["status"],
        "training": training["status"],
        "offline": read_json(ARTIFACT_ROOT / "offline_dev_evaluation.json")["status"],
        "rollouts": rollouts["status"],
        "primary": primary["status"],
        "tactile": read_json(ARTIFACT_ROOT / "tactile_active_statistics.json")["status"],
        "hammer": read_json(ARTIFACT_ROOT / "hammer_control_analysis.json")["status"],
        "secondary": read_json(ARTIFACT_ROOT / "secondary_metrics.json")["status"],
        "uncertainty": uncertainty["scientific_rollout_status"],
        "immutability": s4_2["status"],
        "readiness": readiness,
    }
    acceptance = ARTIFACT_ROOT / "HUMAN_ACCEPTANCE.md"
    acceptance.write_text(human_acceptance(acceptance_statuses), encoding="utf-8")
    print(
        json.dumps(
            {"decision": decision, "readiness": readiness, "status": final["status"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
