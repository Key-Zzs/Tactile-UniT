#!/usr/bin/env python3
"""Generate the registered S4.3-RR plots and representative video set."""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_3_training import (  # noqa: E402
    TASKS,
    VARIANTS,
    atomic_json,
    read_json,
    sha256_file,
)

ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_rr"
PLOT_ROOT = ARTIFACT_ROOT / "plots"
VIDEO_ROOT = ARTIFACT_ROOT / "videos"
COLORS = {"P0": "#546e7a", "P1": "#1976d2", "P2": "#00897b", "P3": "#ef6c00"}


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / name, dpi=180)
    plt.close(fig)


def effect_plot(result: dict[str, Any], name: str, title: str) -> None:
    labels = [*TASKS, "macro"]
    deltas = [result["per_task"][task]["delta"] for task in TASKS] + [result["delta"]]
    intervals = [result["per_task"][task]["ci95"] for task in TASKS] + [result["ci95"]]
    errors = np.asarray(
        [[delta - interval[0], interval[1] - delta] for delta, interval in zip(deltas, intervals)]
    ).T
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.errorbar(labels, deltas, yerr=errors, fmt="o", capsize=5, color="#283593")
    axis.axhline(0, color="black", linewidth=1)
    axis.axhline(0.05, color="#2e7d32", linestyle="--", alpha=0.65)
    axis.axhline(-0.05, color="#c62828", linestyle="--", alpha=0.65)
    axis.set_ylabel("paired success-rate delta")
    axis.set_title(f"{title} — {result['classification']}")
    axis.grid(axis="y", alpha=0.2)
    save(fig, name)


def representative_videos(rows: list[dict[str, Any]]) -> None:
    selected = []
    missing = []
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
                    missing.append({"task": task, "variant": variant, "outcome": outcome})
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
            "schema": "tactile3d-unit.s4-3-rr-representative-videos.v1",
            "stage": "RR15",
            "selection_after_raw_results_immutable": True,
            "scientific_statistics_affected": False,
            "videos": selected,
            "unavailable_outcomes": missing,
            "status": "PASS",
        },
    )


def main() -> None:
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    VIDEO_ROOT.mkdir(parents=True, exist_ok=True)
    rollouts = read_json(ARTIFACT_ROOT / "closed_loop_rollouts.json")
    completeness = read_json(ARTIFACT_ROOT / "rollout_completeness.json")
    summary = read_json(ARTIFACT_ROOT / "closed_loop_summary.json")
    primary = read_json(ARTIFACT_ROOT / "primary_statistics.json")
    tactile = read_json(ARTIFACT_ROOT / "tactile_active_statistics.json")
    hammer = read_json(ARTIFACT_ROOT / "hammer_control_analysis.json")
    seed = read_json(ARTIFACT_ROOT / "training_seed_robustness.json")
    offline = read_json(ARTIFACT_ROOT / "offline_closed_loop_comparison.json")
    learning = read_json(ARTIFACT_ROOT / "policy_learning_sanity.json")
    decision = read_json(ARTIFACT_ROOT / "final_decision.json")
    rows = rollouts["rollouts"]
    if completeness["status"] != "PASS" or decision["experiment_validity"] != "PASS":
        raise SystemExit("S4_3_RR_ROLLOUT_COMPLETENESS_FAIL")

    job_labels = [f"{task}/{variant}/s{training_seed}" for task in TASKS for variant in VARIANTS for training_seed in range(3)]
    matrix = np.zeros((36, 30), dtype=np.int8)
    label_index = {label: index for index, label in enumerate(job_labels)}
    for row in rows:
        label = f"{row['task']}/{row['variant']}/s{row['training_seed']}"
        matrix[label_index[label], row["reset_index"]] += 1
    fig, axis = plt.subplots(figsize=(13, 11))
    image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="Greens")
    axis.set_yticks(range(36), job_labels, fontsize=7)
    axis.set_xticks(range(0, 30, 5))
    axis.set_xlabel("POLICY_EVAL_V1 reset index")
    axis.set_ylabel("task / variant / training seed")
    axis.set_title("1080 canonical rollout completeness (all cells = 1)")
    fig.colorbar(image, ax=axis, label="canonical outcomes")
    save(fig, "05_rollout_completeness_matrix.png")

    x = np.arange(len(TASKS))
    width = 0.2
    fig, axis = plt.subplots(figsize=(10, 5))
    for index, variant in enumerate(VARIANTS):
        axis.bar(
            x + (index - 1.5) * width,
            [summary["tasks"][task][variant]["success_rate"] for task in TASKS],
            width,
            label=variant,
            color=COLORS[variant],
        )
    axis.set_xticks(x, TASKS)
    axis.set_ylim(0, 0.22)
    axis.set_ylabel("success rate")
    axis.set_title("POLICY_EVAL_V1 success by task and variant")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    save(fig, "06_success_rate_by_task_variant.png")

    macro = [primary["variants"][variant]["macro_success"] for variant in VARIANTS]
    intervals = [primary["variants"][variant]["ci95"] for variant in VARIANTS]
    errors = np.asarray([[value - ci[0], ci[1] - value] for value, ci in zip(macro, intervals)]).T
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(VARIANTS, macro, color=[COLORS[v] for v in VARIANTS], yerr=errors, capsize=5)
    axis.set_ylim(0, 0.15)
    axis.set_ylabel("equal-weight 3-task macro success")
    axis.set_title("Primary macro success with hierarchical 95% CI")
    axis.grid(axis="y", alpha=0.2)
    save(fig, "07_three_task_macro_success_ci.png")

    effect_plot(primary["comparisons"]["P1-P0"], "08_p1_minus_p0_raw_tactile_effect.png", "P1-P0 raw tactile effect")
    effect_plot(primary["comparisons"]["P2-P1"], "09_p2_minus_p1_contact_state_effect.png", "P2-P1 Contact-State effect")
    effect_plot(primary["comparisons"]["P3-P2"], "10_p3_minus_p2_tactile_unit_auxiliary_effect.png", "P3-P2 Tactile-UniT auxiliary effect")
    effect_plot(primary["comparisons"]["P3-P0"], "11_p3_minus_p0_full_method_effect.png", "P3-P0 full method effect")

    contrast_labels = list(tactile["comparisons"])
    deltas = [tactile["comparisons"][name]["delta"] for name in contrast_labels]
    cis = [tactile["comparisons"][name]["ci95"] for name in contrast_labels]
    errors = np.asarray([[value - ci[0], ci[1] - value] for value, ci in zip(deltas, cis)]).T
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.errorbar(contrast_labels, deltas, yerr=errors, fmt="o", capsize=5, color="#00897b")
    axis.axhline(0, color="black", linewidth=1)
    axis.set_ylabel("pinch + click macro success delta")
    axis.set_title("SECONDARY_TACTILE_ACTIVE_ANALYSIS")
    axis.grid(axis="y", alpha=0.2)
    save(fig, "12_tactile_active_secondary_analysis.png")

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(VARIANTS, [hammer["success_rates"][variant] for variant in VARIANTS], color=[COLORS[v] for v in VARIANTS])
    axis.set_ylim(0, 0.05)
    axis.set_ylabel("hammer_nail success rate")
    axis.set_title("MAPPED_TACTILE_INACTIVE_CONTROL_TASK")
    axis.grid(axis="y", alpha=0.2)
    save(fig, "13_hammer_nail_control_analysis.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for axis, task in zip(axes, TASKS):
        for variant in VARIANTS:
            axis.plot(range(3), seed["success_rates"][task][variant]["per_training_seed"], marker="o", label=variant, color=COLORS[variant])
        axis.set_title(task)
        axis.set_xticks(range(3))
        axis.set_xlabel("training seed")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("success rate")
    axes[-1].legend()
    save(fig, "14_training_seed_success_variation.png")

    fig, axis = plt.subplots(figsize=(10, 5))
    for index, variant in enumerate(VARIANTS):
        values = [summary["tasks"][task][variant]["time_to_success_successes_mean_sec"] for task in TASKS]
        plotted = [np.nan if value is None else value for value in values]
        axis.bar(x + (index - 1.5) * width, plotted, width, label=variant, color=COLORS[variant])
    axis.set_xticks(x, TASKS)
    axis.set_ylabel("mean time-to-success among successes (sec)")
    axis.set_title("Time-to-success (no bar where no success exists)")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    save(fig, "15_time_to_success.png")

    reasons = sorted({row["termination_reason"] for row in rows})
    labels = [f"{task}\n{variant}" for task in TASKS for variant in VARIANTS]
    bottom = np.zeros(len(labels))
    fig, axis = plt.subplots(figsize=(14, 6))
    for reason in reasons:
        counts = [sum(row["termination_reason"] == reason for row in rows if row["task"] == task and row["variant"] == variant) for task in TASKS for variant in VARIANTS]
        axis.bar(labels, counts, bottom=bottom, label=reason)
        bottom += counts
    axis.set_ylabel("rollouts")
    axis.set_title("Timeout and scientific outcome classes")
    axis.legend()
    save(fig, "16_timeout_failure_classes.png")

    force_fields = (("peak_normal_force", "peak normal force"), ("integrated_normal_force", "integrated normal force"), ("peak_tangential_force", "peak tangential force"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, (field, title) in zip(axes, force_fields):
        for index, variant in enumerate(VARIANTS):
            values = [np.mean([row["force_metrics"][field] for row in rows if row["task"] == task and row["variant"] == variant]) for task in TASKS]
            axis.bar(x + (index - 1.5) * width, values, width, label=variant, color=COLORS[variant])
        axis.set_xticks(x, TASKS, rotation=15)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[-1].legend()
    save(fig, "17_peak_integrated_tangential_force.png")

    smooth_fields = ("action_step_norm", "action_acceleration", "tcp_jerk", "hand_total_variation")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, field in zip(axes.flat, smooth_fields):
        for index, variant in enumerate(VARIANTS):
            values = [np.mean([row["action_metrics"][field] for row in rows if row["task"] == task and row["variant"] == variant]) for task in TASKS]
            axis.bar(x + (index - 1.5) * width, values, width, label=variant, color=COLORS[variant])
        axis.set_xticks(x, TASKS, rotation=15)
        axis.set_title(field.replace("_", " "))
        axis.grid(axis="y", alpha=0.2)
    axes.flat[-1].legend()
    save(fig, "18_action_smoothness.png")

    fig, axis = plt.subplots(figsize=(9, 6))
    markers = {"pinch_tongs": "o", "hammer_nail": "s", "click_mouse": "^"}
    for task in TASKS:
        for variant in VARIANTS:
            xv = offline["per_task"][task]["offline_action_l1"][variant]
            yv = offline["per_task"][task]["closed_loop_success"][variant]
            axis.scatter(xv, yv, marker=markers[task], color=COLORS[variant], s=65)
    variant_legend = [
        Line2D([0], [0], marker="o", linestyle="none", color=COLORS[variant], label=variant)
        for variant in VARIANTS
    ]
    task_legend = [
        Line2D([0], [0], marker=markers[task], linestyle="none", color="#424242", label=task)
        for task in TASKS
    ]
    first_legend = axis.legend(handles=variant_legend, title="variant", loc="upper right")
    axis.add_artist(first_legend)
    axis.legend(handles=task_legend, title="task", loc="center right")
    axis.set_xlabel("POLICY_DEV normalized full-chunk Action L1 (lower is better)")
    axis.set_ylabel("closed-loop success rate")
    axis.set_title("Offline Action L1 vs closed-loop success — relation MIXED")
    axis.grid(alpha=0.2)
    save(fig, "19_offline_l1_vs_closed_loop_success.png")

    maxima = [learning["maximum_success_by_task"][task] for task in TASKS]
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(TASKS, maxima, color="#546e7a")
    axis.axhline(0.2, color="#c62828", linestyle="--", label="HEALTHY task threshold (20%)")
    axis.set_ylim(0, 0.25)
    axis.set_ylabel("maximum variant success rate")
    axis.set_title(f"Policy learning sanity: {learning['classification']}")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    save(fig, "20_policy_learning_sanity.png")

    fig, axis = plt.subplots(figsize=(12, 5))
    axis.axis("off")
    lines = [f"Final: {decision['decision']}", f"Learning: {decision['benchmark_learning_floor']}", f"Validity: {decision['experiment_validity']}"]
    lines += [f"{name}: {value['classification']} ({value['delta']:+.3f})" for name, value in primary["comparisons"].items()]
    axis.text(0.03, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=12)
    save(fig, "21_final_s4_3_2_decision_matrix.png")

    fig, axis = plt.subplots(figsize=(12, 5))
    axis.axis("off")
    readiness = decision["s4_3_3_readiness"]
    lines = [f"S4.3-3: {readiness['decision']}"] + [f"{name}: {status}" for name, status in readiness["gates"].items()]
    axis.text(0.03, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=12)
    save(fig, "22_s4_3_3_readiness_matrix.png")

    representative_videos(rows)
    plots = []
    for index in range(1, 23):
        matches = sorted(PLOT_ROOT.glob(f"{index:02d}_*.png"))
        if len(matches) != 1:
            raise RuntimeError(f"required S4.3-RR visualization {index:02d} count={len(matches)}")
        plots.append(
            {
                "index": index,
                "path": str(matches[0].relative_to(ROOT)),
                "sha256": sha256_file(matches[0]),
            }
        )
    atomic_json(
        ARTIFACT_ROOT / "visualization_manifest.json",
        {
            "schema": "tactile3d-unit.s4-3-rr-visualization-manifest.v1",
            "stage": "RR14",
            "plots": plots,
            "plot_count": len(plots),
            "scientific_statistics_recomputed": False,
            "status": "PASS",
        },
    )
    print(json.dumps({"plots": len(plots), "status": "PASS"}, sort_keys=True))


if __name__ == "__main__":
    main()
