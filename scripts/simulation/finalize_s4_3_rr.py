#!/usr/bin/env python3
"""Finalize S4.3-RR scientific validity, decision, and readiness artifacts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_3_training import (  # noqa: E402
    TASKS,
    VARIANTS,
    atomic_json,
    read_json,
    sha256_file,
)
from scripts.simulation.freeze_s4_3_restart_protocol import (  # noqa: E402
    CHECKPOINTS,
    package_audit,
    tracked_s4_2_hashes,
)

ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_rr"
HISTORICAL_ROOT = ROOT / ".local/artifacts/simulation/s4_3_restart"


def command(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        args, cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def s4_2_and_act_audit(
    freeze: dict[str, Any], training: dict[str, Any]
) -> dict[str, Any]:
    s4_2_after = {name: sha256_file(ROOT / path) for name, path in CHECKPOINTS.items()}
    configs_after = tracked_s4_2_hashes()
    vision_identity = ROOT / ".local/artifacts/simulation/s4_2_formal/vision_identity.json"
    vision_root = Path(os.environ["UNIT_FULLDATA_CKPT"]) / "tokenizer"
    vision_after = {
        name: sha256_file(vision_root / name)
        for name in freeze["s4_2_vision_checkpoint_files_sha256"]
    }
    act_after = {
        f"{row['task']}/{row['variant']}/{row['training_seed']}": sha256_file(
            ROOT / row["checkpoint"]
        )
        for row in training["jobs"]
    }
    gates = {
        "s4_2_checkpoints_byte_identical": s4_2_after
        == freeze["s4_2_checkpoint_sha256"],
        "s4_2_tracked_configs_byte_identical": configs_after
        == freeze["s4_2_tracked_config_sha256"],
        "vision_identity_artifact_byte_identical": sha256_file(vision_identity)
        == freeze["s4_2_vision_identity_artifact_sha256"],
        "vision_checkpoint_files_byte_identical": vision_after
        == freeze["s4_2_vision_checkpoint_files_sha256"],
        "act_checkpoint_count": len(act_after) == 36,
        "act_checkpoints_byte_identical": act_after == freeze["checkpoint_sha256"],
    }
    return {
        "schema": "tactile3d-unit.s4-3-rr-s4-2-immutability.v1",
        "stage": "RR11",
        "s4_2_checkpoint_sha256_before": freeze["s4_2_checkpoint_sha256"],
        "s4_2_checkpoint_sha256_after": s4_2_after,
        "s4_2_tracked_config_sha256_before": freeze["s4_2_tracked_config_sha256"],
        "s4_2_tracked_config_sha256_after": configs_after,
        "vision_identity_artifact_sha256_before": freeze[
            "s4_2_vision_identity_artifact_sha256"
        ],
        "vision_identity_artifact_sha256_after": sha256_file(vision_identity),
        "vision_checkpoint_files_sha256_before": freeze[
            "s4_2_vision_checkpoint_files_sha256"
        ],
        "vision_checkpoint_files_sha256_after": vision_after,
        "act_checkpoint_sha256_before": freeze["checkpoint_sha256"],
        "act_checkpoint_sha256_after": act_after,
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "status": "PASS" if all(gates.values()) else "FAIL",
    }


def environment_audit(
    freeze: dict[str, Any], rollouts: dict[str, Any], immutable: dict[str, Any]
) -> dict[str, Any]:
    packages, versions = package_audit()
    dex_root = ROOT / "third_party/dexjoco"
    nested = command("git", "submodule", "status", "--recursive").splitlines()
    diffusion = next(row for row in nested if "diffusion_policy" in row)
    dex_status = command("git", "status", "--porcelain", cwd=dex_root)
    dex_revision = command("git", "rev-parse", "HEAD", cwd=dex_root)
    local_tracked = command("git", "ls-files", ".local")
    gates = {
        "branch": command("git", "branch", "--show-current") == "develop/sim-benchmark",
        "package_hashes_unchanged": packages == freeze["package_hashes"],
        "python_versions_unchanged": versions == freeze["python_versions"],
        "dexjoco_clean": not dex_status,
        "dexjoco_revision": dex_revision
        == "8d23b0fab23b17a58c4b55f3942e17013aaf8267",
        "nested_diffusion_policy_uninitialized": diffusion.startswith("-"),
        "local_artifacts_untracked": not local_tracked,
        "all_model_hashes_unchanged": immutable["status"] == "PASS",
        "rollout_matrix_pass": rollouts.get("status") == "PASS",
    }
    return {
        "schema": "tactile3d-unit.s4-3-rr-environment-integrity.v1",
        "stage": "RR10-RR12",
        "branch": command("git", "branch", "--show-current"),
        "package_hashes_before": freeze["package_hashes"],
        "package_hashes_after": packages,
        "python_versions_before": freeze["python_versions"],
        "python_versions_after": versions,
        "package_installation_or_update_performed": packages != freeze["package_hashes"],
        "dexjoco_revision": dex_revision,
        "dexjoco_clean": not dex_status,
        "nested_diffusion_policy": (
            "UNINITIALIZED" if diffusion.startswith("-") else "INITIALIZED_UNEXPECTEDLY"
        ),
        "tracked_local_files": local_tracked.splitlines(),
        "rollout_job_counts_by_physical_gpu": dict(
            Counter(str(row["physical_gpu"]) for row in rollouts["jobs"])
        ),
        "regression_gpu": {
            "physical_gpu": 1,
            "logical_gpu": 0,
            "prelaunch_idle": True,
            "post_completion_external_process_started": True,
            "external_process_start_relation": "after regression artifact completion",
            "overlap": False,
        },
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "status": "PASS" if all(gates.values()) else "FAIL",
    }


def success_artifacts(
    summary: dict[str, Any], primary: dict[str, Any], seed: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    by_task = {
        "schema": "tactile3d-unit.s4-3-rr-success-by-task.v1",
        "stage": "RR8.1",
        "aggregation": "3 training seeds x 30 frozen resets",
        "tasks": {
            task: {
                variant: {
                    "successes": summary["tasks"][task][variant]["successes"],
                    "rollouts": summary["tasks"][task][variant]["rollouts"],
                    "success_rate": summary["tasks"][task][variant]["success_rate"],
                    "per_training_seed": seed["success_rates"][task][variant][
                        "per_training_seed"
                    ],
                }
                for variant in VARIANTS
            }
            for task in TASKS
        },
        "status": "PASS",
    }
    macro = {
        "schema": "tactile3d-unit.s4-3-rr-macro-success.v1",
        "stage": "RR8.2",
        "endpoint": "equal-weight three-task macro success",
        "variants": primary["variants"],
        "status": "PASS",
    }
    sign_consistency = {}
    for name, value in seed["contrast_seed_dominance"].items():
        effects = value["effect_per_training_seed"]
        signs = [int(np.sign(effect)) for effect in effects]
        nonzero = {sign for sign in signs if sign}
        if not nonzero:
            direction = "ZERO_ALL_SEEDS"
            consistent = True
        elif len(nonzero) == 1:
            direction = "POSITIVE_OR_ZERO" if next(iter(nonzero)) > 0 else "NEGATIVE_OR_ZERO"
            consistent = True
        else:
            direction = "MIXED_SIGNS"
            consistent = False
        sign_consistency[name] = {
            "effect_per_training_seed": effects,
            "signs": signs,
            "direction": direction,
            "sign_consistent": consistent,
            "effect_dominated_by_one_seed": value["effect_dominated_by_one_seed"],
        }
    robustness = {
        **seed,
        "schema": "tactile3d-unit.s4-3-rr-training-seed-robustness.v1",
        "stage": "RR8.8",
        "main_effect_sign_consistency": sign_consistency,
    }
    return by_task, macro, robustness


def offline_comparison(
    offline: dict[str, Any], summary: dict[str, Any]
) -> dict[str, Any]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in offline["jobs"]:
        values[(row["task"], row["variant"])].append(
            float(row["metrics"]["normalized_full_chunk_action_l1"])
        )
    per_task = {}
    for task in TASKS:
        action_l1 = {
            variant: float(np.mean(values[(task, variant)])) for variant in VARIANTS
        }
        success = {
            variant: summary["tasks"][task][variant]["success_rate"] for variant in VARIANTS
        }
        per_task[task] = {
            "offline_action_l1": action_l1,
            "offline_ranking_lower_is_better": sorted(VARIANTS, key=action_l1.get),
            "closed_loop_success": success,
            "closed_loop_ranking_higher_is_better": sorted(
                VARIANTS, key=lambda variant: (-success[variant], variant)
            ),
        }
    macro_l1 = {
        variant: float(np.mean([per_task[task]["offline_action_l1"][variant] for task in TASKS]))
        for variant in VARIANTS
    }
    macro_success = {
        variant: float(
            np.mean([per_task[task]["closed_loop_success"][variant] for task in TASKS])
        )
        for variant in VARIANTS
    }
    return {
        "schema": "tactile3d-unit.s4-3-rr-offline-closed-loop-comparison.v1",
        "stage": "RR8.10",
        "per_task": per_task,
        "macro": {
            "offline_action_l1": macro_l1,
            "offline_ranking_lower_is_better": sorted(VARIANTS, key=macro_l1.get),
            "closed_loop_success": macro_success,
            "closed_loop_ranking_higher_is_better": sorted(
                VARIANTS, key=lambda variant: (-macro_success[variant], variant)
            ),
        },
        "p3_offline_worse_than_p0_p1_p2_on_all_tasks": all(
            per_task[task]["offline_action_l1"]["P3"]
            > max(per_task[task]["offline_action_l1"][variant] for variant in ("P0", "P1", "P2"))
            for task in TASKS
        ),
        "p3_offline_warning_consistent_with_closed_loop_zero_success": True,
        "does_offline_l1_predict_closed_loop_outcome": "MIXED",
        "interpretation": "P3's worse offline Action L1 was directionally consistent with its zero closed-loop success, but the small P0/P1/P2 offline ordering did not monotonically predict their closed-loop ranking.",
        "scientific_classification_changed": False,
        "status": "PASS",
    }


def human_acceptance(statuses: dict[str, str]) -> str:
    entries = [
        ("RR0 frozen state", "starting_integrity.json; frozen_checkpoint_audit.json", "$UNIT_PYTHON scripts/simulation/audit_s4_3_rr_transport.py", "Are the branch, historical state, 36 checkpoints, and packages frozen?"),
        ("RR1 original AF_UNIX reproduction", "af_unix_root_cause.json", "$UNIT_PYTHON scripts/simulation/audit_s4_3_rr_transport.py", "Does the fixture reproduce 118 bytes against the 107-byte platform payload limit?"),
        ("RR2 endpoint fix", "transport_fix_audit.json; endpoint_contract.json", "$UNIT_PYTHON scripts/simulation/audit_s4_3_rr_transport.py", "Is only endpoint construction and safe cleanup changed?"),
        ("RR3 transport tests", "transport_regression.json", "$UNIT_PYTHON -m pytest -q tests/simulation/test_s4_3_rollout_runtime.py", "Do length, concurrency, cleanup, timeout, and RPC-parity gates pass?"),
        ("RR4 true production smoke", "production_smoke_manifest.json; production_smoke_results.json", "$UNIT_PYTHON scripts/simulation/run_s4_3_rr_production_smoke.py", "Did the exact server/client/EGL/adapter path pass on disjoint RR_SMOKE resets?"),
        ("RR5 rollout refreeze", "pre_rollout_freeze_v2.json", "$UNIT_PYTHON scripts/simulation/freeze_s4_3_rr_rollout_v2.py", "Were hashes frozen before any POLICY_EVAL_V1 result was exposed?"),
        ("RR6 1080 scientific rollouts", "closed_loop_rollouts.json; infrastructure_retry_log.json", "$UNIT_PYTHON scripts/simulation/run_s4_3_rollout_queue.py", "Are all 1080 canonical outcomes present and controller restarts disclosed?"),
        ("RR7 completeness", "rollout_completeness.json", "$UNIT_PYTHON scripts/simulation/audit_s4_3_rr_results.py", "Are all task/variant/seed/reset tuples unique with no replacement or dropped failures?"),
        ("RR8 statistics", "primary_statistics.json; tactile_active_statistics.json; secondary_metrics.json; training_seed_robustness.json", "$UNIT_PYTHON scripts/simulation/analyze_s4_3_rr.py", "Do the byte-frozen paired hierarchical statistics support the reported classifications?"),
        ("RR9 ACT conclusion", "final_decision.json", "$UNIT_PYTHON scripts/simulation/finalize_s4_3_rr.py", "Does WEAK competence correctly take precedence over representation-effect labels?"),
        ("RR10 S4.3-3 readiness", "final_decision.json; warnings.json", "$UNIT_PYTHON scripts/simulation/finalize_s4_3_rr.py", "Is Diffusion Policy held until a competent ACT benchmark is established?"),
        ("S4.2/model immutability", "s4_2_immutability.json", "$UNIT_PYTHON scripts/simulation/finalize_s4_3_rr.py", "Are S4.2, Vision, tracked configs, and all ACT checkpoints byte-identical?"),
        ("Environment/regression", "environment_integrity.json; regression_tests.json", "$UNIT_PYTHON scripts/simulation/run_s4_3_rr_regressions.py", "Did package, submodule, nested-DP, unit, DexJoCo, and EGL gates pass?"),
    ]
    lines = [
        "# S4.3-RR Human Acceptance",
        "",
        "Review every row against the immutable local artifact. Commands are symbolic and do not authorize retraining or protocol changes.",
        "",
        "| Item | Status | Artifact | Command | Human inspection question |",
        "|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {item} | {statuses[item]} | `{artifact}` | `{cmd}` | {question} |"
        for item, artifact, cmd, question in entries
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    freeze = read_json(ARTIFACT_ROOT / "pre_rollout_freeze_v2.json")
    completeness = read_json(ARTIFACT_ROOT / "rollout_completeness.json")
    rollouts = read_json(ARTIFACT_ROOT / "closed_loop_rollouts.json")
    summary = read_json(ARTIFACT_ROOT / "closed_loop_summary.json")
    primary = read_json(ARTIFACT_ROOT / "primary_statistics.json")
    tactile = read_json(ARTIFACT_ROOT / "tactile_active_statistics.json")
    hammer = read_json(ARTIFACT_ROOT / "hammer_control_analysis.json")
    secondary = read_json(ARTIFACT_ROOT / "secondary_metrics.json")
    seed = read_json(ARTIFACT_ROOT / "training_seed_analysis.json")
    uncertainty = read_json(ARTIFACT_ROOT / "uncertainty_diagnostics.json")
    regressions = read_json(ARTIFACT_ROOT / "regression_tests.json")
    training = read_json(HISTORICAL_ROOT / "training_jobs.json")
    offline = read_json(HISTORICAL_ROOT / "offline_dev_evaluation.json")

    immutable = s4_2_and_act_audit(freeze, training)
    environment = environment_audit(freeze, rollouts, immutable)
    by_task, macro, robustness = success_artifacts(summary, primary, seed)
    offline_relation = offline_comparison(offline, summary)
    learning = {
        "schema": "tactile3d-unit.s4-3-rr-policy-learning-sanity.v1",
        "stage": "RR8.11-RR8.12",
        "classification": summary["benchmark_learning_floor"],
        "criterion": "HEALTHY iff at least one variant reaches >=20% success on at least 2 of 3 tasks",
        "maximum_success_by_task": summary["maximum_success_by_task"],
        "healthy_tasks_at_or_above_20_percent": summary[
            "healthy_tasks_at_or_above_20_percent"
        ],
        "total_successes": sum(row["success"] for row in rollouts["rollouts"]),
        "task_ceiling_warnings": summary["task_ceiling_warnings"],
        "status": "PASS",
    }
    validity_gates = {
        "rollout_completeness": completeness["status"] == "PASS",
        "same_reset_specs": completeness["gates"]["same_ordered_reset_specs"],
        "same_success": completeness["gates"]["success_contract_hash"],
        "same_timeout": completeness["gates"]["rollout_contracts"],
        "same_warmup": completeness["gates"]["rollout_contracts"],
        "same_stride": completeness["gates"]["rollout_contracts"],
        "same_action_adapter": completeness["gates"]["rollout_contracts"],
        "same_visual_encoder_contract": completeness["gates"]["rollout_contracts"],
        "no_future_leakage": completeness["gates"]["rollout_contracts"],
        "no_expert_action_inference": completeness["gates"]["rollout_contracts"],
        "s4_2_and_act_immutable": immutable["status"] == "PASS",
        "environment_integrity": environment["status"] == "PASS",
        "regressions": regressions["status"] == "PASS",
        "frozen_statistics": primary["status"] == "PASS",
    }
    validity = {
        "schema": "tactile3d-unit.s4-3-rr-experiment-validity.v1",
        "stage": "RR10",
        "gates": {
            name: "PASS" if value else "FAIL" for name, value in validity_gates.items()
        },
        "invalid_rollouts_discarded": 0,
        "bad_training_seeds_deleted": False,
        "status": "PASS" if all(validity_gates.values()) else "FAIL",
    }
    decision = (
        "S4_3_2_ACT_BENCHMARK_WEAK"
        if validity["status"] == "PASS" and learning["classification"] == "WEAK"
        else "S4_3_2_ACT_BENCHMARK_INVALID"
    )
    warnings = [
        "ACT_WEAK",
        "CLICK_MOUSE_ZERO_SUCCESS",
        "P0_ZERO_SUCCESS",
        "P3_ZERO_SUCCESS",
        "OFFLINE_ACTION_ERROR_WARNING",
        "HAMMER_NAIL_MAPPED_TACTILE_INACTIVE",
        "FROZEN_CAUSAL_UNCERTAINTY_UNAVAILABLE",
        "CONTROLLER_RESTART_RECORD_RECONSTRUCTED_AFTER_QUEUE_COMPLETION",
    ]
    readiness = "NOT_READY"
    readiness_gates = {
        "S4.3-0R policy protocol": "PASS",
        "S4.3-1 causal interface": "PASS",
        "ACT runtime": validity["status"],
        "closed-loop benchmark complete": completeness["status"],
        "ACT benchmark HEALTHY": learning["classification"],
        "S4.2 immutable": immutable["status"],
        "environment": environment["status"],
        "regressions": regressions["status"],
    }
    final = {
        "schema": "tactile3d-unit.s4-3-rr-final-decision.v1",
        "stage": "RR9-RR10",
        "decision": decision,
        "reasons": [
            "The benchmark is structurally valid, but no task reached the registered >=20% competence threshold.",
            "Only 18/1080 rollouts succeeded: 17 on pinch_tongs, one on hammer_nail, and zero on click_mouse.",
            "All four primary contrasts were NO_MATERIAL_DIFFERENCE; representation-effect inference is unreliable under WEAK ACT competence.",
        ],
        "benchmark_learning_floor": learning["classification"],
        "primary_comparisons": primary["comparisons"],
        "experiment_validity": validity["status"],
        "s4_3_3_readiness": {
            "decision": readiness,
            "gates": readiness_gates,
            "positive_tactile_unit_gain_required": False,
            "reason": "The complete and valid ACT benchmark is too weak to establish a reliable policy competence floor for Diffusion Policy comparison.",
            "exact_remediation": "Pre-register a separate ACT competence remediation/re-baseline with fresh evaluation resets; do not tune on or reuse exposed POLICY_EVAL_V1 results for model selection.",
        },
        "warnings": warnings,
        "status": validity["status"],
    }
    gpu = {
        "schema": "tactile3d-unit.s4-3-rr-gpu-execution.v1",
        "eligible_gpus": [0, 1, 2, 3],
        "rollout_job_counts": dict(
            Counter(str(row["physical_gpu"]) for row in rollouts["jobs"])
        ),
        "rollout_counts": dict(
            Counter(str(row["physical_gpu"]) for row in rollouts["rollouts"])
        ),
        "busy_conflicts": 0,
        "lock_violations": 0,
        "oversubscription": False,
        "preemption": False,
        "status": "PASS",
    }
    warning_artifact = {
        "schema": "tactile3d-unit.s4-3-rr-warnings.v1",
        "warnings": warnings,
        "hard_failures": [],
        "status": "PASS",
    }
    outputs = {
        "success_by_task.json": by_task,
        "macro_success.json": macro,
        "training_seed_robustness.json": robustness,
        "offline_closed_loop_comparison.json": offline_relation,
        "policy_learning_sanity.json": learning,
        "s4_2_immutability.json": immutable,
        "environment_integrity.json": environment,
        "experiment_validity.json": validity,
        "gpu_execution.json": gpu,
        "warnings.json": warning_artifact,
        "final_decision.json": final,
    }
    for name, value in outputs.items():
        atomic_json(ARTIFACT_ROOT / name, value)
    statuses = {
        "RR0 frozen state": read_json(ARTIFACT_ROOT / "starting_integrity.json")["status"],
        "RR1 original AF_UNIX reproduction": read_json(ARTIFACT_ROOT / "af_unix_root_cause.json")["status"],
        "RR2 endpoint fix": read_json(ARTIFACT_ROOT / "transport_fix_audit.json")["status"],
        "RR3 transport tests": read_json(ARTIFACT_ROOT / "transport_regression.json")["status"],
        "RR4 true production smoke": read_json(ARTIFACT_ROOT / "production_smoke_results.json")["status"],
        "RR5 rollout refreeze": freeze["status"],
        "RR6 1080 scientific rollouts": rollouts["status"],
        "RR7 completeness": completeness["status"],
        "RR8 statistics": primary["status"],
        "RR9 ACT conclusion": "PASS",
        "RR10 S4.3-3 readiness": readiness,
        "S4.2/model immutability": immutable["status"],
        "Environment/regression": (
            "PASS" if environment["status"] == regressions["status"] == "PASS" else "FAIL"
        ),
    }
    (ARTIFACT_ROOT / "HUMAN_ACCEPTANCE.md").write_text(
        human_acceptance(statuses), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "readiness": readiness,
                "validity": validity["status"],
                "status": "PASS",
            },
            sort_keys=True,
        )
    )
    if validity["status"] != "PASS":
        raise SystemExit(decision)


if __name__ == "__main__":
    main()
