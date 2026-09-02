#!/usr/bin/env python3
"""Create the frozen S4.3-2 result plots and representative video set."""

from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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

ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_restart"
PLOT_ROOT = ARTIFACT_ROOT / "plots"
VIDEO_ROOT = ARTIFACT_ROOT / "videos"
COLORS = {"P0": "#546e7a", "P1": "#1976d2", "P2": "#00897b", "P3": "#ef6c00"}


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / name, dpi=180)
    plt.close(fig)


def training_plots(training: dict[str, Any], offline: dict[str, Any]) -> None:
    traces: dict[tuple[str, str], list[list[dict[str, Any]]]] = defaultdict(list)
    for row in training["jobs"]:
        values = [
            json.loads(line)
            for line in (ROOT / row["trace"]).read_text(encoding="utf-8").splitlines()
        ]
        traces[(row["task"], row["variant"])].append(values)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for axis, task in zip(axes, TASKS):
        for variant in VARIANTS:
            runs = traces[(task, variant)]
            for run in runs:
                axis.plot(
                    [row["step"] for row in run],
                    [row["action_l1"] for row in run],
                    color=COLORS[variant],
                    alpha=0.2,
                )
            common = sorted(set.intersection(*[{row["step"] for row in run} for run in runs]))
            mean = [
                np.mean(
                    [next(row["action_l1"] for row in run if row["step"] == step) for run in runs]
                )
                for step in common
            ]
            axis.plot(common, mean, color=COLORS[variant], label=variant, linewidth=2)
        axis.set_title(task)
        axis.set_xlabel("optimizer step")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("training normalized Action L1")
    axes[-1].legend()
    save(fig, "05_act_training_loss_by_task_variant.png")

    values = defaultdict(list)
    for row in offline["jobs"]:
        values[(row["task"], row["variant"])].append(
            row["metrics"]["normalized_full_chunk_action_l1"]
        )
    x = np.arange(len(TASKS))
    width = 0.2
    fig, axis = plt.subplots(figsize=(10, 5))
    for index, variant in enumerate(VARIANTS):
        means = [np.mean(values[(task, variant)]) for task in TASKS]
        errors = [np.std(values[(task, variant)]) for task in TASKS]
        axis.bar(
            x + (index - 1.5) * width,
            means,
            width,
            yerr=errors,
            color=COLORS[variant],
            label=variant,
        )
    axis.set_xticks(x, TASKS)
    axis.set_ylabel("POLICY_DEV normalized full-chunk Action L1")
    axis.set_title("Selected checkpoint offline Action error (mean ± seed SD)")
    axis.legend()
    save(fig, "06_act_dev_action_error.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, task in zip(axes, TASKS):
        for run in traces[(task, "P3")]:
            axis.plot(
                [row["step"] for row in run],
                [row["action_l1"] for row in run],
                color="#1976d2",
                alpha=0.35,
            )
            axis.plot(
                [row["step"] for row in run],
                [0.1 * row["contact_mse"] for row in run],
                color="#ef6c00",
                alpha=0.35,
            )
        axis.set_title(task)
        axis.set_xlabel("optimizer step")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("loss contribution")
    axes[-1].plot([], [], color="#1976d2", label="ACT Action L1")
    axes[-1].plot([], [], color="#ef6c00", label="0.1 × Contact MSE")
    axes[-1].legend()
    save(fig, "07_p3_act_vs_contact_auxiliary_loss.png")


def result_plots(
    rollouts: dict[str, Any],
    summary: dict[str, Any],
    primary: dict[str, Any],
    tactile: dict[str, Any],
    hammer: dict[str, Any],
    seed: dict[str, Any],
    uncertainty: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    x = np.arange(len(TASKS))
    width = 0.2
    fig, axis = plt.subplots(figsize=(10, 5))
    for index, variant in enumerate(VARIANTS):
        rates = [summary["tasks"][task][variant]["success_rate"] for task in TASKS]
        axis.bar(x + (index - 1.5) * width, rates, width, label=variant, color=COLORS[variant])
    axis.set_xticks(x, TASKS)
    axis.set_ylim(0, 1)
    axis.set_ylabel("closed-loop success rate")
    axis.set_title("POLICY_EVAL_V1 success by task and policy")
    axis.legend()
    save(fig, "08_closed_loop_success_by_task_variant.png")

    rows = rollouts["rollouts"]
    macro = [primary["variants"][variant]["macro_success"] for variant in VARIANTS]
    intervals = [primary["variants"][variant]["ci95"] for variant in VARIANTS]
    fig, axis = plt.subplots(figsize=(8, 5))
    error = np.asarray([[m - ci[0], ci[1] - m] for m, ci in zip(macro, intervals)]).T
    axis.bar(VARIANTS, macro, color=[COLORS[v] for v in VARIANTS], yerr=error, capsize=5)
    axis.set_ylim(0, 1)
    axis.set_ylabel("equal-weight 3-task macro success")
    axis.set_title("All-task macro success with hierarchical 95% CI")
    save(fig, "09_all_task_macro_success_ci.png")

    for number, name in enumerate(("P1-P0", "P2-P1", "P3-P2", "P3-P0"), start=10):
        result = primary["comparisons"][name]
        labels = [*TASKS, "macro"]
        deltas = [result["per_task"][task]["delta"] for task in TASKS] + [result["delta"]]
        cis = [result["per_task"][task]["ci95"] for task in TASKS] + [result["ci95"]]
        errors = np.asarray([[delta - ci[0], ci[1] - delta] for delta, ci in zip(deltas, cis)]).T
        fig, axis = plt.subplots(figsize=(8, 5))
        axis.errorbar(labels, deltas, yerr=errors, fmt="o", capsize=5, color="#283593")
        axis.axhline(0, color="black", linewidth=1)
        axis.axhline(0.05, color="#2e7d32", linestyle="--", alpha=0.6)
        axis.axhline(-0.05, color="#c62828", linestyle="--", alpha=0.6)
        axis.set_ylabel("paired success-rate delta")
        axis.set_title(f"{name} — {result['classification']}")
        save(fig, f"{number:02d}_{name.lower().replace('-', '_minus_')}_primary_contrast.png")

    labels = list(tactile["comparisons"])
    deltas = [tactile["comparisons"][name]["delta"] for name in labels]
    cis = [tactile["comparisons"][name]["ci95"] for name in labels]
    errors = np.asarray([[delta - ci[0], ci[1] - delta] for delta, ci in zip(deltas, cis)]).T
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.errorbar(labels, deltas, yerr=errors, fmt="o", capsize=5, color="#00897b")
    axis.axhline(0, color="black")
    axis.set_ylabel("pinch + click macro success delta")
    axis.set_title("SECONDARY_TACTILE_ACTIVE_ANALYSIS")
    save(fig, "14_tactile_active_secondary_macro.png")

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(
        VARIANTS,
        [hammer["success_rates"][v] for v in VARIANTS],
        color=[COLORS[v] for v in VARIANTS],
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("hammer_nail success rate")
    axis.set_title("MAPPED_TACTILE_INACTIVE_CONTROL_TASK")
    save(fig, "15_hammer_nail_control_task.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for axis, task in zip(axes, TASKS):
        for index, variant in enumerate(VARIANTS):
            values = seed["success_rates"][task][variant]["per_training_seed"]
            axis.plot(range(3), values, marker="o", label=variant, color=COLORS[variant])
        axis.set_title(task)
        axis.set_xticks(range(3))
        axis.set_xlabel("training seed")
    axes[0].set_ylabel("success rate")
    axes[-1].legend()
    save(fig, "16_success_by_training_seed.png")

    fig, axis = plt.subplots(figsize=(10, 5))
    for index, variant in enumerate(VARIANTS):
        values = [
            summary["tasks"][task][variant]["time_to_success_successes_mean_sec"] for task in TASKS
        ]
        axis.bar(
            x + (index - 1.5) * width,
            [np.nan if value is None else value for value in values],
            width,
            label=variant,
            color=COLORS[variant],
        )
    axis.set_xticks(x, TASKS)
    axis.set_ylabel("mean time-to-success among successes (sec)")
    axis.set_title("Closed-loop time-to-success")
    axis.legend()
    save(fig, "17_time_to_success.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    force_names = (
        ("mean_peak_normal_force", "peak normal force"),
        ("mean_integrated_normal_force", "integrated normal force"),
        ("mean_peak_tangential_force", "peak tangential force"),
    )
    for axis, (field, title) in zip(axes, force_names):
        for index, variant in enumerate(VARIANTS):
            axis.bar(
                x + (index - 1.5) * width,
                [summary["tasks"][task][variant][field] for task in TASKS],
                width,
                color=COLORS[variant],
                label=variant,
            )
        axis.set_xticks(x, TASKS, rotation=15)
        axis.set_title(title)
    axes[-1].legend()
    save(fig, "18_force_metrics.png")

    metric_names = ("action_step_norm", "action_acceleration", "tcp_jerk", "hand_total_variation")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for axis, metric in zip(axes.flat, metric_names):
        for index, variant in enumerate(VARIANTS):
            values = []
            for task in TASKS:
                scoped = [
                    row["action_metrics"][metric]
                    for row in rows
                    if row["task"] == task and row["variant"] == variant
                ]
                values.append(np.mean(scoped))
            axis.bar(x + (index - 1.5) * width, values, width, color=COLORS[variant], label=variant)
        axis.set_xticks(x, TASKS, rotation=15)
        axis.set_title(metric.replace("_", " "))
    axes.flat[-1].legend()
    save(fig, "19_action_smoothness.png")

    reasons = sorted({row["termination_reason"] for row in rows})
    labels = [f"{task}\n{variant}" for task in TASKS for variant in VARIANTS]
    bottom = np.zeros(len(labels))
    fig, axis = plt.subplots(figsize=(14, 6))
    for reason in reasons:
        counts = []
        for task in TASKS:
            for variant in VARIANTS:
                counts.append(
                    sum(
                        row["termination_reason"] == reason
                        for row in rows
                        if row["task"] == task and row["variant"] == variant
                    )
                )
        axis.bar(labels, counts, bottom=bottom, label=reason)
        bottom += counts
    axis.set_ylabel("rollouts")
    axis.set_title("Termination and failure distribution")
    axis.legend()
    save(fig, "20_termination_failure_distribution.png")

    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.axis("off")
    axis.text(
        0.5,
        0.55,
        "Frozen uncertainty diagnostic unavailable\nfor legal causal S4.3 inputs",
        ha="center",
        va="center",
        fontsize=16,
    )
    axis.text(
        0.5,
        0.25,
        f"invocations={uncertainty['invocations']} · interventions={uncertainty['interventions']}\nNo correlations fabricated",
        ha="center",
        va="center",
        fontsize=11,
    )
    save(fig, "21_uncertainty_success_vs_failure.png")

    representative = next(
        (row for row in rows if row["variant"] == "P3" and row["success"]), rows[0]
    )
    with np.load(ROOT / representative["trace"], allow_pickle=False) as trace:
        normal = trace["total_normal_force"]
        tangent = trace["total_tangential_force"]
        actions = trace["policy_action"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(normal, label="normal")
    axes[0].plot(tangent, label="tangential")
    axes[0].legend()
    axes[1].plot(np.linalg.norm(actions[:, :6], axis=1))
    axes[1].set_ylabel("TCP Action norm")
    axes[2].plot(np.linalg.norm(actions[:, 6:22], axis=1))
    axes[2].set_ylabel("hand Action norm")
    axes[2].set_xlabel("policy control step")
    fig.suptitle(f"Representative rollout: {representative['rollout_id']}")
    save(fig, "22_representative_rollout_timeline.png")

    fig, axis = plt.subplots(figsize=(11, 4.5))
    axis.axis("off")
    lines = [f"Final: {decision['decision']}", f"Learning: {decision['benchmark_learning_floor']}"]
    lines += [
        f"{name}: {value['classification']} ({value['delta']:+.3f})"
        for name, value in primary["comparisons"].items()
    ]
    axis.text(0.03, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=12)
    save(fig, "23_final_s4_3_2_decision_matrix.png")

    fig, axis = plt.subplots(figsize=(11, 4.5))
    axis.axis("off")
    readiness = decision["s4_3_3_readiness"]
    lines = [f"S4.3-3: {readiness['decision']}"] + [
        f"{name}: {status}" for name, status in readiness["gates"].items()
    ]
    axis.text(0.03, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=12)
    save(fig, "24_s4_3_3_readiness_matrix.png")


def representative_videos(rows: list[dict[str, Any]]) -> None:
    selected = []
    for task in TASKS:
        for variant in VARIANTS:
            scoped = sorted(
                [row for row in rows if row["task"] == task and row["variant"] == variant],
                key=lambda row: (row["training_seed"], row["reset_index"]),
            )
            for outcome, candidate in (
                ("success", next((row for row in scoped if row["success"]), None)),
                ("failure", next((row for row in scoped if not row["success"]), None)),
            ):
                if candidate is None:
                    continue
                source = ROOT / candidate["raw_video"]
                destination = VIDEO_ROOT / task / variant / f"{outcome}.mp4"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                selected.append(
                    {
                        "task": task,
                        "variant": variant,
                        "outcome": outcome,
                        "rollout_id": candidate["rollout_id"],
                        "video": str(destination.relative_to(ROOT)),
                        "sha256": sha256_file(destination),
                    }
                )
    atomic_json(
        ARTIFACT_ROOT / "representative_videos.json",
        {
            "schema": "tactile3d-unit.s4-3-representative-videos.v1",
            "selection_after_raw_rollout_identity_freeze": True,
            "videos": selected,
            "status": "PASS",
        },
    )


def main() -> None:
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    training = read_json(ARTIFACT_ROOT / "training_jobs.json")
    offline = read_json(ARTIFACT_ROOT / "offline_dev_evaluation.json")
    rollouts = read_json(ARTIFACT_ROOT / "closed_loop_rollouts.json")
    summary = read_json(ARTIFACT_ROOT / "closed_loop_summary.json")
    primary = read_json(ARTIFACT_ROOT / "primary_statistics.json")
    tactile = read_json(ARTIFACT_ROOT / "tactile_active_statistics.json")
    hammer = read_json(ARTIFACT_ROOT / "hammer_control_analysis.json")
    seed = read_json(ARTIFACT_ROOT / "training_seed_analysis.json")
    uncertainty = read_json(ARTIFACT_ROOT / "uncertainty_diagnostics.json")
    decision = read_json(ARTIFACT_ROOT / "final_decision.json")
    training_plots(training, offline)
    result_plots(rollouts, summary, primary, tactile, hammer, seed, uncertainty, decision)
    representative_videos(rollouts["rollouts"])
    expected = [PLOT_ROOT / f"{index:02d}_" for index in range(1, 25)]
    missing = [
        index
        for index, prefix in enumerate(expected, start=1)
        if not any(PLOT_ROOT.glob(prefix.name + "*.png"))
    ]
    if missing:
        raise RuntimeError(f"required S4.3 visualization missing: {missing}")
    print(json.dumps({"plots": 24, "status": "PASS"}, sort_keys=True))


if __name__ == "__main__":
    main()
