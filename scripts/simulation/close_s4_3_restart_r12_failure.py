#!/usr/bin/env python3
"""Close restarted S4.3 after the frozen R12 runtime environment failure."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_3_training import (  # noqa: E402
    atomic_json,
    read_json,
    sha256_file,
)
from scripts.simulation.finalize_s4_3_restart import (  # noqa: E402
    environment_audit,
    s4_2_audit,
)

ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_restart"
LOG_ROOT = ROOT / ".local/logs/simulation/s4_3_restart"
PLOTS = ARTIFACT_ROOT / "plots"
VIDEOS = ARTIFACT_ROOT / "videos"
DECISION = "S4_3_2_ENVIRONMENT_FAIL"
BLOCKED_STATUS = "NOT_RUN_STRUCTURAL_FAILURE"
FAILED_JOB = ("click_mouse", "P0", 0)
FAILED_JOB_ID = "click_mouse_P0_seed0"
FAILURE_MESSAGE = "AF_UNIX path too long"


def command(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_frozen_code(pre: dict[str, Any]) -> dict[str, bool]:
    groups = {
        "rollout": pre["rollout_code_sha256"],
        "policy_dependency": pre["policy_dependency_code_sha256"],
        "statistical": {pre["statistical_code"]: pre["statistical_code_sha256"]},
        "decision": {pre["decision_code"]: pre["decision_code_sha256"]},
    }
    results: dict[str, bool] = {}
    for group, values in groups.items():
        results[group] = all(sha256_file(ROOT / path) == digest for path, digest in values.items())
    require(all(results.values()), "pre-rollout frozen code changed after R11")
    return results


def verify_training(training: dict[str, Any], pre: dict[str, Any]) -> dict[str, Any]:
    jobs = training["jobs"]
    identities = {(row["task"], row["variant"], row["training_seed"]) for row in jobs}
    expected = {
        (task, variant, seed)
        for task in ("pinch_tongs", "hammer_nail", "click_mouse")
        for variant in ("P0", "P1", "P2", "P3")
        for seed in (0, 1, 2)
    }
    require(training["status"] == "PASS", "training manifest is not PASS")
    require(len(jobs) == 36 and identities == expected, "training matrix is not exactly 36 jobs")
    actual: dict[str, str] = {}
    for row in jobs:
        key = f"{row['task']}/{row['variant']}/{row['training_seed']}"
        digest = sha256_file(ROOT / row["checkpoint"])
        require(digest == row["checkpoint_sha256"], f"checkpoint digest changed: {key}")
        require(digest == pre["checkpoint_sha256"][key], f"R11 checkpoint mismatch: {key}")
        require(sha256_file(ROOT / row["trace"]) == row["trace_sha256"], f"trace changed: {key}")
        actual[key] = digest
    return {
        "jobs": len(jobs),
        "identities_exact": identities == expected,
        "checkpoint_hashes_verified": len(actual),
        "trace_hashes_verified": len(jobs),
        "status": "PASS",
    }


def record_failure(pre: dict[str, Any], training: dict[str, Any], frozen_code: dict[str, bool]) -> dict[str, Any]:
    rollouts_path = ARTIFACT_ROOT / "closed_loop_rollouts.json"
    launch = read_json(rollouts_path)
    server_log = LOG_ROOT / "rollout_jobs" / f"{FAILED_JOB_ID}_server.log"
    job_log = LOG_ROOT / "rollout_queue" / f"{FAILED_JOB_ID}.log"
    queue_log = LOG_ROOT / "rollout_queue.log"
    server_text = server_log.read_text(encoding="utf-8")
    job_text = job_log.read_text(encoding="utf-8")
    require(FAILURE_MESSAGE in server_text, "frozen server log lacks the AF_UNIX failure")
    require(
        "policy server exited before socket readiness" in job_text,
        "rollout worker log lacks the readiness failure",
    )
    metadata = sorted((LOG_ROOT / "closed_loop").glob("**/metadata.json"))
    require(not metadata, "scientific rollout metadata exists despite pre-client failure")
    require(launch["completed_rollouts"] == 0, "rollout summary is not at zero")
    require(launch["rollout_performance_seen"] is False, "rollout performance was observed")

    launch_snapshot = ARTIFACT_ROOT / "closed_loop_rollouts_launch_snapshot.json"
    atomic_json(launch_snapshot, launch)
    socket_relative = f".local/tmp/simulation/s4_3_restart/rollout_sockets/{FAILED_JOB_ID}.sock"
    socket_bytes = len(os.fsencode(str(ROOT / socket_relative)))
    failure = {
        "schema": "tactile3d-unit.s4-3-r12-structural-failure.v1",
        "stage": "R12",
        "classification": DECISION,
        "structural_hard_failure": True,
        "dependent_stages_stopped": True,
        "failed_job": {
            "task": FAILED_JOB[0],
            "variant": FAILED_JOB[1],
            "training_seed": FAILED_JOB[2],
            "attempt": 1,
        },
        "failure_type": "OSError",
        "failure_message": FAILURE_MESSAGE,
        "failure_boundary": "policy server AF_UNIX listener bind before socket readiness",
        "socket_endpoint": socket_relative,
        "socket_path_bytes_in_current_workspace": socket_bytes,
        "linux_af_unix_payload_limit_bytes": 107,
        "path_exceeds_limit": socket_bytes > 107,
        "policy_worker_launched": True,
        "policy_server_ready": False,
        "dexjoco_client_started": False,
        "scientific_rollouts_started": 0,
        "scientific_rollouts_completed": 0,
        "rollout_performance_seen": False,
        "retuning_performed": False,
        "rollout_code_changed_after_freeze": False,
        "frozen_code_verification": frozen_code,
        "pre_rollout_freeze_sha256": sha256_file(ARTIFACT_ROOT / "pre_rollout_freeze.json"),
        "checkpoint_set_sha256": pre["checkpoint_set_sha256"],
        "launch_snapshot": str(launch_snapshot.relative_to(ROOT)),
        "launch_snapshot_sha256": sha256_file(launch_snapshot),
        "evidence": {
            "server_log": str(server_log.relative_to(ROOT)),
            "server_log_sha256": sha256_file(server_log),
            "job_log": str(job_log.relative_to(ROOT)),
            "job_log_sha256": sha256_file(job_log),
            "queue_log": str(queue_log.relative_to(ROOT)),
            "queue_log_sha256": sha256_file(queue_log),
        },
        "training_jobs_complete": len(training["jobs"]) == 36,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": "FAIL",
    }
    atomic_json(ARTIFACT_ROOT / "r12_structural_failure.json", failure)

    closed = dict(launch)
    closed.update(
        {
            "status": "FAIL",
            "end_time": failure["recorded_at"],
            "completed_jobs": 0,
            "completed_rollouts": 0,
            "invalid_rollouts": 0,
            "rollout_performance_seen": False,
            "running": [],
            "failed": [failure["failed_job"]],
            "failure_classification": DECISION,
            "structural_failure_artifact": "r12_structural_failure.json",
            "dependent_stages_stopped": True,
        }
    )
    atomic_json(rollouts_path, closed)
    return failure


def blocked_artifact(schema: str, stage: str, reason: str) -> dict[str, Any]:
    return {
        "schema": schema,
        "stage": stage,
        "blocked_by": "r12_structural_failure.json",
        "reason": reason,
        "scientific_results_fabricated": False,
        "status": BLOCKED_STATUS,
    }


def write_downstream_artifacts(failure: dict[str, Any]) -> None:
    reason = (
        "R12 stopped before the first scientific POLICY_EVAL_V1 rollout because the "
        "frozen policy server could not bind its AF_UNIX socket."
    )
    summary = blocked_artifact("tactile3d-unit.s4-3-closed-loop-summary.v1", "R12", reason)
    summary.update(
        {
            "expected_jobs": 36,
            "expected_rollouts": 1080,
            "completed_jobs": 0,
            "completed_rollouts": 0,
            "invalid_rollouts": 0,
            "failure_classification": DECISION,
            "rollout_performance_seen": False,
        }
    )
    atomic_json(ARTIFACT_ROOT / "closed_loop_summary.json", summary)

    outputs = {
        "primary_statistics.json": (
            "tactile3d-unit.s4-3-primary-statistics.v1",
            "R13",
        ),
        "tactile_active_statistics.json": (
            "tactile3d-unit.s4-3-tactile-active-statistics.v1",
            "R13",
        ),
        "hammer_control_analysis.json": (
            "tactile3d-unit.s4-3-hammer-control-analysis.v1",
            "R13",
        ),
        "secondary_metrics.json": (
            "tactile3d-unit.s4-3-secondary-metrics.v1",
            "R13",
        ),
        "training_seed_analysis.json": (
            "tactile3d-unit.s4-3-training-seed-analysis.v1",
            "R13",
        ),
    }
    for filename, (schema, stage) in outputs.items():
        atomic_json(ARTIFACT_ROOT / filename, blocked_artifact(schema, stage, reason))
    uncertainty = blocked_artifact("tactile3d-unit.s4-3-uncertainty-diagnostics.v1", "R13", reason)
    uncertainty.update(
        {
            "scientific_status": "DIAGNOSTIC ONLY",
            "invocations": 0,
            "interventions": 0,
            "causal_runtime_availability": "UNAVAILABLE_CAUSAL_INPUT_MISMATCH",
        }
    )
    atomic_json(ARTIFACT_ROOT / "uncertainty_diagnostics.json", uncertainty)

    warnings = {
        "schema": "tactile3d-unit.s4-3-warnings.v1",
        "stage": "R13-R16",
        "warnings": [
            "HAMMER_NAIL_MAPPED_TACTILE_INACTIVE",
            "R12_AF_UNIX_ENDPOINT_EXCEEDS_PLATFORM_LIMIT",
            "NO_CLOSED_LOOP_SCIENTIFIC_RESULTS",
        ],
        "failure_classification": DECISION,
        "status": "PASS",
    }
    atomic_json(ARTIFACT_ROOT / "warnings.json", warnings)

    decision = {
        "schema": "tactile3d-unit.s4-3-final-decision.v1",
        "stage": "R14-R15",
        "decision": DECISION,
        "exactly_one_classification": True,
        "benchmark_status": "FAIL",
        "scientific_effect_classification_performed": False,
        "reasons": [
            failure["failure_message"],
            "The frozen policy service exited before socket readiness and before the DexJoCo client started.",
            "Zero of 1080 POLICY_EVAL_V1 rollouts completed, so representation effects are not estimable.",
        ],
        "s4_3_3_readiness": {
            "decision": "NOT_READY",
            "act_positive_tactile_unit_result_required": False,
            "reason": "Runtime execution failure is a frozen NOT_READY structural blocker.",
            "exact_remediation": (
                "Shorten the production AF_UNIX socket endpoint below the platform limit, "
                "repeat the production-path runtime integration smoke, re-freeze R11 with "
                "the corrected code hashes, and start a new clean S4.3-2 rollout execution."
            ),
            "s4_3_3_implemented": False,
        },
        "status": "PASS",
    }
    atomic_json(ARTIFACT_ROOT / "final_decision.json", decision)


def human_acceptance() -> str:
    entries = [
        (
            "PASS",
            "S4.3-0R policy data/protocol",
            "policy_dataset_audit.json; s4_3_restart_protocol_freeze.json",
            "$UNIT_PYTHON scripts/simulation/freeze_s4_3_restart_protocol.py",
            "Do immutable dataset membership and the frozen protocol match the approved identities?",
        ),
        (
            "PASS",
            "Fresh POLICY_EVAL_V1 reset specs",
            "policy_eval_v1_manifest.json",
            "jq . .local/artifacts/simulation/s4_3_restart/policy_eval_v1_manifest.json",
            "Are all 90 reset specifications fresh and frozen?",
        ),
        (
            "PASS",
            "Success/timeout contract",
            "task_success_contract.json; pre_rollout_freeze.json",
            "jq '{success_predicates,timeouts}' .local/artifacts/simulation/s4_3_restart/pre_rollout_freeze.json",
            "Are task predicates and timeout values frozen before rollout?",
        ),
        (
            "PASS",
            "S4.3-1 causal interface",
            "causal_observation_contract.json; causal_action_contract.json",
            "$UNIT_PYTHON scripts/simulation/audit_s4_3_policy_integration.py",
            "Can any inference input read beyond the current control step?",
        ),
        (
            "PASS",
            "P3 gradient audit",
            "p3_gradient_audit.json",
            "jq . .local/artifacts/simulation/s4_3_restart/p3_gradient_audit.json",
            "Do gradients reach ACT while all S4.2 parameters stay frozen?",
        ),
        (
            "PASS",
            "Runtime causal smoke",
            "runtime_smoke.json",
            "jq . .local/artifacts/simulation/s4_3_restart/runtime_smoke.json",
            "Did the bounded smoke cover all tasks, warm-up, queue, adapter, and EGL?",
        ),
        (
            "PASS",
            "ACT implementation",
            "act_implementation_audit.json",
            "$UNIT_PYTHON -m pytest -q tests/simulation/test_s4_3_act.py",
            "Are architecture, bounds, and P2/P3 fairness exact?",
        ),
        (
            "PASS",
            "36-job training",
            "act_training_manifest.json; act_checkpoint_manifest.json",
            "jq '{status,expected_jobs,completed_jobs}' .local/artifacts/simulation/s4_3_restart/act_training_manifest.json",
            "Are all 36 jobs present with frozen POLICY_DEV selection?",
        ),
        (
            "PASS",
            "Offline POLICY_DEV",
            "offline_dev_evaluation.json",
            "jq '{status,hard_sanity}' .local/artifacts/simulation/s4_3_restart/offline_dev_evaluation.json",
            "Did all 36 checkpoints cold-load and pass hard sanity?",
        ),
        (
            "FAIL",
            "1080 closed-loop rollouts",
            "closed_loop_rollouts.json; r12_structural_failure.json",
            "jq . .local/artifacts/simulation/s4_3_restart/r12_structural_failure.json",
            "Does the evidence prove the AF_UNIX failure occurred before any scientific rollout?",
        ),
        (
            BLOCKED_STATUS,
            "Primary statistics",
            "primary_statistics.json",
            "jq . .local/artifacts/simulation/s4_3_restart/primary_statistics.json",
            "Is the analysis correctly withheld rather than fabricated?",
        ),
        (
            BLOCKED_STATUS,
            "Tactile-active statistics",
            "tactile_active_statistics.json",
            "jq . .local/artifacts/simulation/s4_3_restart/tactile_active_statistics.json",
            "Is the secondary result correctly withheld?",
        ),
        (
            BLOCKED_STATUS,
            "Hammer control analysis",
            "hammer_control_analysis.json",
            "jq . .local/artifacts/simulation/s4_3_restart/hammer_control_analysis.json",
            "Is the control result correctly withheld?",
        ),
        (
            BLOCKED_STATUS,
            "Secondary metrics",
            "secondary_metrics.json; training_seed_analysis.json",
            "jq . .local/artifacts/simulation/s4_3_restart/secondary_metrics.json",
            "Are rollout-derived metrics absent rather than inferred from offline data?",
        ),
        (
            BLOCKED_STATUS,
            "Uncertainty diagnostics",
            "uncertainty_diagnostics.json",
            "jq . .local/artifacts/simulation/s4_3_restart/uncertainty_diagnostics.json",
            "Are zero invocations/interventions and DIAGNOSTIC ONLY status explicit?",
        ),
        (
            "PASS",
            "S4.2 immutability",
            "s4_2_immutability.json",
            "jq . .local/artifacts/simulation/s4_3_restart/s4_2_immutability.json",
            "Are checkpoints, Vision files, and tracked configs byte-identical?",
        ),
        (
            "NOT_READY",
            "S4.3-3 readiness",
            "final_decision.json",
            "jq . .local/artifacts/simulation/s4_3_restart/final_decision.json",
            "Does readiness stop on runtime execution failure without demanding a positive ACT result?",
        ),
    ]
    lines = ["# Restarted S4.3 Causal ACT Human Acceptance", ""]
    for index, (status, title, artifact, cmd, question) in enumerate(entries, 1):
        lines.extend(
            [
                f"## {index}. {title}",
                "",
                f"Status: {status}",
                "",
                f"Artifact: {artifact}",
                "",
                f"Command: `{cmd}`",
                "",
                f"Human inspection question: {question}",
                "",
            ]
        )
    return "\n".join(lines)


def status_panel(number: int, title: str, detail: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.axis("off")
    ax.text(0.5, 0.66, title, ha="center", va="center", fontsize=20, weight="bold")
    ax.text(0.5, 0.42, detail, ha="center", va="center", fontsize=14, color="#8b1e1e", wrap=True)
    ax.text(0.5, 0.17, "No rollout result was imputed or fabricated.", ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS / f"{number:02d}_{title.lower().replace(' ', '_').replace('/', '_')}.png", dpi=150)
    plt.close(fig)


def make_plots(training: dict[str, Any], offline: dict[str, Any]) -> dict[str, Any]:
    PLOTS.mkdir(parents=True, exist_ok=True)
    # 1: truthful stage dataflow.
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.axis("off")
    stages = ["Frozen data", "Causal ACT", "36 training jobs", "Offline dev", "R12 rollout"]
    colors = ["#2d7d46"] * 4 + ["#a32626"]
    for index, (stage, color) in enumerate(zip(stages, colors)):
        x = 0.1 + index * 0.2
        ax.text(
            x,
            0.55,
            stage,
            ha="center",
            va="center",
            color="white",
            weight="bold",
            bbox={"boxstyle": "round,pad=0.7", "facecolor": color, "edgecolor": "none"},
        )
        if index < len(stages) - 1:
            ax.annotate("", xy=(x + 0.13, 0.55), xytext=(x + 0.07, 0.55), arrowprops={"arrowstyle": "->", "lw": 2})
    ax.text(
        0.9,
        0.25,
        "STOP: AF_UNIX endpoint too long\n0/1080 scientific rollouts",
        ha="center",
        color="#a32626",
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(PLOTS / "01_restarted_policy_benchmark_dataflow.png", dpi=150)
    plt.close(fig)

    # 2: causal timestamp graph.
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(-26, 28)
    ax.set_ylim(-0.2, 1.3)
    ax.axvline(0, color="black", lw=2)
    ax.hlines(0.8, -25, 0, color="#2d7d46", lw=8, label="Allowed tactile history")
    ax.scatter([0], [0.55], s=140, color="#1f77b4", label="Current RGB/proprio")
    ax.hlines(0.3, 0, 26, color="#d28c19", lw=8, label="Predicted action chunk")
    ax.axvspan(0.01, 27, color="#a32626", alpha=0.08, label="No future observation")
    ax.set_xlabel("Control timestamp relative to t")
    ax.set_yticks([])
    ax.set_title("Strict causal timestamp contract")
    ax.legend(loc="upper center", ncol=4)
    fig.tight_layout()
    fig.savefig(PLOTS / "02_strict_causal_timestamp_graph.png", dpi=150)
    plt.close(fig)

    dataset = read_json(ARTIFACT_ROOT / "policy_dataset_audit.json")
    counts = dataset["membership_counts"]
    tasks = ["pinch_tongs", "hammer_nail", "click_mouse"]
    train_counts = [counts[f"POLICY_TRAIN|{task}|True"] for task in tasks]
    dev_counts = [counts[f"POLICY_DEV|{task}|True"] for task in tasks]
    x = np.arange(len(tasks))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - 0.18, train_counts, 0.36, label="TRAIN")
    ax.bar(x + 0.18, dev_counts, 0.36, label="DEV")
    ax.set_xticks(x, tasks)
    ax.set_ylabel("Episodes")
    ax.set_title("Frozen BC-eligible POLICY_EXPERT membership")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "03_policy_expert_train_dev_counts.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(tasks, [30, 30, 30], color="#1f77b4")
    ax.set_ylim(0, 35)
    ax.set_ylabel("Frozen resets")
    ax.set_title("POLICY_EVAL_V1 reset distribution (90 total)")
    fig.tight_layout()
    fig.savefig(PLOTS / "04_policy_evaluation_reset_distribution.png", dpi=150)
    plt.close(fig)

    variants = ["P0", "P1", "P2", "P3"]
    trace_last: dict[tuple[str, str], list[float]] = {}
    p3_pairs: dict[str, list[tuple[float, float]]] = {task: [] for task in tasks}
    for row in training["jobs"]:
        trace_rows = [
            json.loads(line)
            for line in (ROOT / row["trace"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        last = trace_rows[-1]
        trace_last.setdefault((row["task"], row["variant"]), []).append(last["act_total"])
        if row["variant"] == "P3":
            p3_pairs[row["task"]].append((last["act_total"], last["contact_mse"]))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.18
    for idx, variant in enumerate(variants):
        values = [np.mean(trace_last[(task, variant)]) for task in tasks]
        ax.bar(x + (idx - 1.5) * width, values, width, label=variant)
    ax.set_xticks(x, tasks)
    ax.set_ylabel("Mean terminal ACT training loss")
    ax.set_title("ACT training loss by task and variant")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "05_act_training_loss_by_task_variant.png", dpi=150)
    plt.close(fig)

    offline_values: dict[tuple[str, str], list[float]] = {}
    for row in offline["jobs"]:
        offline_values.setdefault((row["task"], row["variant"]), []).append(
            row["metrics"]["normalized_full_chunk_action_l1"]
        )
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for idx, variant in enumerate(variants):
        values = [np.mean(offline_values[(task, variant)]) for task in tasks]
        ax.bar(x + (idx - 1.5) * width, values, width, label=variant)
    ax.set_xticks(x, tasks)
    ax.set_ylabel("POLICY_DEV normalized full-chunk L1")
    ax.set_title("ACT offline dev Action error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "06_act_dev_action_error.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, task in zip(axes, tasks):
        pairs = p3_pairs[task]
        ax.scatter([pair[0] for pair in pairs], [pair[1] for pair in pairs], s=70)
        for seed, pair in enumerate(pairs):
            ax.annotate(f"seed {seed}", pair)
        ax.set_title(task)
        ax.set_xlabel("terminal ACT loss")
        ax.set_ylabel("terminal Contact auxiliary MSE")
    fig.suptitle("P3 ACT loss vs frozen Contact auxiliary loss")
    fig.tight_layout()
    fig.savefig(PLOTS / "07_p3_act_vs_contact_auxiliary_loss.png", dpi=150)
    plt.close(fig)

    blocked_titles = [
        "Closed-loop success by task and variant",
        "All-task macro success with 95% CI",
        "P1-P0 primary contrast",
        "P2-P1 primary contrast",
        "P3-P2 primary contrast",
        "P3-P0 primary contrast",
        "Tactile-active secondary macro result",
        "Hammer-nail control-task result",
        "Success by training seed",
        "Time-to-success",
        "Force metrics",
        "Action smoothness",
        "Termination and failure distribution",
        "Uncertainty success-vs-failure",
        "Representative rollout timeline",
    ]
    detail = "NOT RUN — R12 structural environment failure before the first scientific rollout (0/1080)."
    for number, title in enumerate(blocked_titles, 8):
        status_panel(number, title, detail)
    status_panel(23, "Final S4.3-2 decision matrix", f"Selected exactly once: {DECISION}")
    status_panel(
        24,
        "S4.3-3 readiness matrix",
        "NOT_READY — runtime execution failure requires a fresh corrected R11 freeze.",
    )

    plots = sorted(PLOTS.glob("*.png"))
    manifest = {
        "schema": "tactile3d-unit.s4-3-visualization-manifest.v1",
        "stage": "R21",
        "plots": [{"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)} for path in plots],
        "plot_count": len(plots),
        "rollout_derived_plots": BLOCKED_STATUS,
        "fabricated_results": False,
        "status": "PASS" if len(plots) == 24 else "FAIL",
    }
    atomic_json(PLOTS / "visualization_manifest.json", manifest)
    require(manifest["status"] == "PASS", "visualization set is not exactly 24 plots")
    return manifest


def main() -> None:
    pre = read_json(ARTIFACT_ROOT / "pre_rollout_freeze.json")
    training = read_json(ARTIFACT_ROOT / "act_training_manifest.json")
    offline = read_json(ARTIFACT_ROOT / "offline_dev_evaluation.json")
    regressions = read_json(ARTIFACT_ROOT / "regression_tests.json")
    require(pre["status"] == "PASS" and pre["rollout_performance_seen"] is False, "invalid R11 freeze")
    require(offline["status"] == "PASS" and len(offline["jobs"]) == 36, "offline gate incomplete")
    require(regressions["status"] == "PASS", "regressions must pass before failure closeout")
    frozen_code = verify_frozen_code(pre)
    training_verification = verify_training(training, pre)
    failure = record_failure(pre, training, frozen_code)
    write_downstream_artifacts(failure)

    s4_2 = s4_2_audit()
    atomic_json(ARTIFACT_ROOT / "s4_2_immutability.json", s4_2)
    environment = environment_audit(training, {"jobs": []})
    environment.update(
        {
            "rollout_jobs": 0,
            "rollout_environment_failure": True,
            "r12_failure_classification": DECISION,
        }
    )
    atomic_json(ARTIFACT_ROOT / "environment_integrity.json", environment)
    require(s4_2["status"] == "PASS", "S4.2 immutability failed")
    require(environment["status"] == "PASS", "environment integrity changed")

    VIDEOS.mkdir(parents=True, exist_ok=True)
    (VIDEOS / "README.md").write_text(
        "# Representative rollout videos\n\n"
        "Status: NOT_AVAILABLE_STRUCTURAL_FAILURE\n\n"
        "R12 stopped before the first scientific rollout, so no representative video exists.\n",
        encoding="utf-8",
    )
    atomic_json(
        ARTIFACT_ROOT / "representative_videos.json",
        {
            "schema": "tactile3d-unit.s4-3-representative-videos.v1",
            "stage": "R12-R21",
            "videos": [],
            "reason": "Zero scientific rollouts completed before the R12 structural failure.",
            "status": "NOT_AVAILABLE_STRUCTURAL_FAILURE",
        },
    )
    (ARTIFACT_ROOT / "HUMAN_ACCEPTANCE.md").write_text(human_acceptance(), encoding="utf-8")
    visualizations = make_plots(training, offline)
    result = {
        "classification": DECISION,
        "training_verification": training_verification,
        "offline_jobs": len(offline["jobs"]),
        "scientific_rollouts": 0,
        "s4_2_immutability": s4_2["status"],
        "environment_integrity": environment["status"],
        "regressions": regressions["status"],
        "plots": visualizations["plot_count"],
        "status": "PASS",
    }
    atomic_json(ARTIFACT_ROOT / "r12_failure_closeout.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
