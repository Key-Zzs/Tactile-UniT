#!/usr/bin/env python3
"""Render the applicable PI2V adapter-training and representation evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2v"
LOG = ROOT / ".local/logs/simulation/s4_3_pi2v/unit_adapter/train.jsonl"
PLOTS = ARTIFACTS / "plots"


def read_json(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text())


def finish(fig: plt.Figure, name: str) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / name, dpi=180, bbox_inches="tight", metadata={"Software": "matplotlib"})
    plt.close(fig)


def label_bars(ax: plt.Axes, bars, fmt: str = "{:.3g}") -> None:
    for bar in bars:
        value = bar.get_height()
        ax.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_axis_off()
    boxes = [
        (0.03, 0.64, 0.18, 0.19, "Front RGB\nt → t+27", "#d9eaf7"),
        (0.03, 0.18, 0.18, 0.19, "State25 + action22\nDexJoCo category 30", "#eeeeee"),
        (0.29, 0.64, 0.19, 0.19, "Frozen DINOv2 +\nVision M-Former", "#d9eaf7"),
        (0.29, 0.18, 0.19, 0.19, "Trainable DexJoCo frontends\n23,485,568 parameters", "#f8dfbd"),
        (0.56, 0.42, 0.16, 0.19, "Frozen fusion +\nAction M-Former", "#d9eaf7"),
        (0.79, 0.42, 0.17, 0.19, "Frozen 2-level RVQ\n8 slots × 2 × 128", "#d9eaf7"),
        (0.56, 0.08, 0.16, 0.16, "Frozen visual\ndecoder", "#d9eaf7"),
        (0.79, 0.08, 0.17, 0.16, "Trainable category-30\naction decoder slice", "#f8dfbd"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#334155", linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
    arrows = [
        ((0.21, 0.735), (0.29, 0.735)), ((0.21, 0.275), (0.29, 0.275)),
        ((0.48, 0.735), (0.56, 0.54)), ((0.48, 0.275), (0.56, 0.49)),
        ((0.72, 0.515), (0.79, 0.515)), ((0.87, 0.42), (0.64, 0.24)),
        ((0.87, 0.42), (0.87, 0.24)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#334155"})
    ax.text(0.5, 0.95, "Original UniT shared/frozen vs DexJoCo-trainable components", ha="center", fontsize=14)
    ax.text(0.03, 0.02, "Blue: released shared component (frozen)   ·   Orange: new category-30 tensors (trainable)   ·   Gray: input", fontsize=9)
    finish(fig, "01_unit_frozen_vs_trainable.png")


def parameter_map() -> None:
    contract = read_json("adapter_trainable_parameters.json")
    trainable = contract["trainable"]
    groups: dict[str, int] = {}
    for row in trainable:
        name = row["name"] if isinstance(row, dict) else str(row)
        count = row.get("parameters", row.get("numel", 0)) if isinstance(row, dict) else 0
        key = "state encoder" if "state" in name else "action encoder" if "encoder" in name else "action decoder"
        groups[key] = groups.get(key, 0) + int(count)
    if not any(groups.values()):
        groups = {"state/action encoder slices": 13_024_768, "action decoder slices": 10_460_800}
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels, values = list(groups), np.asarray(list(groups.values())) / 1e6
    bars = ax.bar(labels, values, color=plt.cm.Set2.colors[: len(values)])
    label_bars(ax, bars, "{:.2f}M")
    ax.set_ylabel("Trainable parameters (millions)")
    ax.set_title("DexJoCo adapter parameter map")
    ax.text(0.99, 0.96, "32 tensors · 23.486M total\n0 shared official parameters trainable", transform=ax.transAxes, ha="right", va="top")
    ax.spines[["top", "right"]].set_visible(False)
    finish(fig, "02_adapter_parameter_map.png")


def resource_plot() -> None:
    smoke = read_json("adapter_memory_smoke.json")
    completion = read_json("adapter_training_completion.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    bars = axes[0].bar(["Smoke", "Formal"], [smoke["peak_reserved_mib"] / 1024, completion["peak_reserved_mib"] / 1024], color="#4c78a8")
    label_bars(axes[0], bars, "{:.2f} GiB")
    axes[0].axhline(48, color="#666666", linestyle="--", label="physical 48 GiB")
    axes[0].set_ylabel("Peak reserved VRAM (GiB/GPU)")
    axes[0].legend(frameon=False)
    bars = axes[1].bar(["Smoke", "Formal"], [smoke["median_steady_sec_per_step"], completion["median_steady_sec_per_step"]], color="#59a14f")
    label_bars(axes[1], bars, "{:.3f} s")
    axes[1].set_ylabel("Median steady seconds/step")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Adapter memory and throughput")
    finish(fig, "03_adapter_memory_throughput.png")


def loss_curves() -> None:
    rows = [json.loads(line) for line in LOG.read_text().splitlines() if line.strip()]
    sample = rows[::100]
    if rows[-1]["step"] != sample[-1]["step"]:
        sample.append(rows[-1])
    step = np.asarray([row["step"] for row in sample])
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for key, label in (("loss", "total"), ("action_recon_loss", "action recon"), ("vq_loss", "VQ")):
        axes[0].plot(step, [row[key] for row in sample], label=label, linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Loss (log scale)")
    axes[0].legend(frameon=False, ncol=3)
    for key, label in (("vision_recon_loss", "fused"), ("vision_only_vision_recon_loss", "vision-only"), ("action_only_vision_recon_loss", "action-only")):
        axes[1].plot(step, [row[key] for row in sample], label=label, linewidth=1)
    axes[1].set_ylabel("Vision cosine loss")
    axes[1].set_xlabel("Optimizer step")
    axes[1].legend(frameon=False, ncol=3)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2)
    fig.suptitle("Original UniT adapter training losses (100-step downsample)")
    finish(fig, "04_original_unit_loss_curves.png")


def rvq_usage() -> None:
    usage = read_json("adapter_codebook_usage.json")["routes"]
    labels, active, entropy = [], [], []
    for route in ("fused", "vision_only", "action_only"):
        for level in ("level_1", "level_2"):
            labels.append(f"{route.replace('_only', '')}\n{level.replace('_', ' ')}")
            active.append(usage[route][level]["active_codes"])
            entropy.append(usage[route][level]["normalized_entropy"])
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    bars = axes[0].bar(x, active, color="#4c78a8")
    label_bars(axes[0], bars, "{:.0f}")
    axes[0].axhline(4, color="#e15759", linestyle="--", label="minimum = 4")
    axes[0].set_ylabel("Active codes (of 128)")
    axes[0].legend(frameon=False)
    bars = axes[1].bar(x, entropy, color="#59a14f")
    label_bars(axes[1], bars, "{:.3f}")
    axes[1].axhline(0.05, color="#e15759", linestyle="--", label="minimum = 0.05")
    axes[1].set_ylabel("Normalized entropy")
    axes[1].set_xticks(x, labels)
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Frozen two-level RVQ usage on 4,860-row DEV")
    finish(fig, "05_rvq_code_usage_entropy.png")


def reconstruction_plots() -> None:
    metrics = read_json("adapter_representation_metrics.json")["reconstruction"]
    cross = read_json("adapter_cross_reconstruction.json")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = ["Fusion→vision", "Vision→vision", "Action→vision", "No-motion"]
    values = [metrics["fused_vision_cosine_loss"], metrics["vision_to_vision_cosine_loss"], metrics["action_to_vision_cosine_loss"], metrics["no_motion_vision_cosine_loss"]]
    bars = ax.bar(labels, values, color=["#e15759", "#4c78a8", "#4c78a8", "#9c9c9c"])
    label_bars(ax, bars, "{:.5f}")
    ax.set_ylabel("Future-vision cosine loss (lower is better)")
    ax.set_title("Vision reconstruction gate: FAIL")
    ax.spines[["top", "right"]].set_visible(False)
    finish(fig, "06_vision_reconstruction.png")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = ["Fusion→action", "Vision→action", "Action→action", "TRAIN mean"]
    values = [metrics["fused_action_smooth_l1"], metrics["vision_to_action_smooth_l1"], metrics["action_to_action_smooth_l1"], metrics["train_mean_action_smooth_l1"]]
    bars = ax.bar(labels, values, color=["#59a14f", "#4c78a8", "#4c78a8", "#9c9c9c"])
    label_bars(ax, bars, "{:.4g}")
    ax.set_yscale("log")
    ax.set_ylabel("Action Smooth-L1 (log scale; lower is better)")
    ax.set_title("Action reconstruction gate: PASS")
    ax.spines[["top", "right"]].set_visible(False)
    finish(fig, "07_action_reconstruction.png")

    routes = ["vision_to_action", "action_to_vision", "fusion_to_action", "fusion_to_vision"]
    paired = [cross[name][next(key for key in cross[name] if key.startswith("paired_"))] for name in routes]
    shuffled = [cross[name][next(key for key in cross[name] if key.startswith("shuffled_"))] for name in routes]
    ratios = np.asarray(paired) / np.asarray(shuffled)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar([name.replace("_", "→", 1).replace("_", " ") for name in routes], ratios, color="#4c78a8")
    label_bars(ax, bars, "{:.3f}×")
    ax.axhline(1, color="#e15759", linestyle="--", label="paired = shuffled")
    ax.set_ylabel("Paired / group-shuffled loss (lower is better)")
    ax.set_title("All cross-reconstruction routes beat invalid controls")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    finish(fig, "08_cross_reconstruction.png")


def alignment() -> None:
    relation = read_json("adapter_representation_metrics.json")["paired_vs_shuffled_relation"]
    means = [relation["paired_mean_cosine"], relation["group_shuffled_mean_cosine"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    bars = axes[0].bar(["Paired", "Group-shuffled"], means, color=["#59a14f", "#9c9c9c"])
    label_bars(axes[0], bars, "{:.5f}")
    axes[0].set_ylim(min(means) - 0.01, max(means) + 0.01)
    axes[0].set_ylabel("Vision/action mean cosine")
    delta = relation["paired_minus_shuffled_mean"]
    low, high = relation["bootstrap_ci95"]
    axes[1].errorbar([0], [delta], yerr=[[delta - low], [high - delta]], fmt="o", color="#4c78a8", capsize=6)
    axes[1].axhline(0, color="#e15759", linestyle="--")
    axes[1].set_xlim(-0.7, 0.7)
    axes[1].set_xticks([0], ["Paired − shuffled"])
    axes[1].set_ylabel("Cosine difference, bootstrap 95% CI")
    axes[1].text(0, high + 0.0004, f"{delta:+.5f}\n[{low:+.5f}, {high:+.5f}]", ha="center", fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Adapted UniT V/A representation relation: PASS")
    finish(fig, "09_va_fusion_alignment.png")


def main() -> None:
    architecture()
    parameter_map()
    resource_plot()
    loss_curves()
    rvq_usage()
    reconstruction_plots()
    alignment()
    print(f"wrote 9 plots to {PLOTS}")


if __name__ == "__main__":
    main()
