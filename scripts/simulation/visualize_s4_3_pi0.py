#!/usr/bin/env python3
"""Render available S4.3-PI0 official-pipeline audit visuals."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_pi0"
PLOT_ROOT = ARTIFACT_ROOT / "plots"
TRAIN_LOG = ROOT / ".local/logs/simulation/s4_3_pi0/train_official_seed42.log"


def plot_pipeline(output: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    boxes = [
        ("Official LeRobot\npinch_tongs", "front + wrist RGB\nstate[23], action[22]"),
        ("Official transforms", "224x224 resize\nquantile normalize + pad"),
        ("pi0.5 LoRA", "Gemma 2B\n300M action expert"),
        ("Policy chunk", "30 x 22\nxyz + rotvec + hand"),
        ("DexJoCo wrapper", "rotvec -> quaternion\nenvironment action[23]"),
    ]
    fig, axis = plt.subplots(figsize=(14, 4.5))
    axis.set_xlim(0, len(boxes) * 3)
    axis.set_ylim(0, 4)
    axis.axis("off")
    for index, (title, body) in enumerate(boxes):
        x = index * 3 + 0.25
        patch = FancyBboxPatch(
            (x, 1.0), 2.3, 2.0, boxstyle="round,pad=0.08", facecolor="#e8f1fb", edgecolor="#235789"
        )
        axis.add_patch(patch)
        axis.text(x + 1.15, 2.35, title, ha="center", va="center", weight="bold", fontsize=10)
        axis.text(x + 1.15, 1.55, body, ha="center", va="center", fontsize=9)
        if index < len(boxes) - 1:
            axis.add_patch(
                FancyArrowPatch((x + 2.35, 2), (x + 2.95, 2), arrowstyle="-|>", mutation_scale=15, color="#444")
            )
    axis.set_title("S4.3-PI0 official pi0.5 dataflow (no tactile input)", fontsize=15, pad=12)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_contract(output: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [
        ("Policy state [23]", [("TCP xyz", 3), ("quaternion", 4), ("Allegro hand", 16)]),
        ("Policy action [22]", [("TCP xyz", 3), ("rotation vector", 3), ("Allegro hand", 16)]),
        ("Environment action [23]", [("TCP xyz", 3), ("quaternion", 4), ("Allegro hand", 16)]),
    ]
    colors = ["#3b82f6", "#f59e0b", "#10b981"]
    fig, axis = plt.subplots(figsize=(11, 4.8))
    for row, (label, segments) in enumerate(rows):
        left = 0
        for index, (name, width) in enumerate(segments):
            axis.barh(row, width, left=left, height=0.55, color=colors[index], edgecolor="white")
            axis.text(
                left + width / 2, row, f"{name}\n{width}D", ha="center", va="center", color="white", weight="bold"
            )
            left += width
    axis.set_yticks(range(len(rows)), [row[0] for row in rows])
    axis.invert_yaxis()
    axis.set_xlim(0, 23)
    axis.set_xlabel("Dimensions")
    axis.set_title("Official DexJoCo single-arm state/action contract")
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_official_rollout(output: Path) -> None:
    import matplotlib.pyplot as plt

    evaluation = json.loads((ARTIFACT_ROOT / "official_checkpoint_eval.json").read_text(encoding="utf-8"))
    success = evaluation["successes"]
    episodes = evaluation["episodes"]
    rate = evaluation["success_rate"] * 100
    ci_low, ci_high = [value * 100 for value in evaluation["wilson_95ci"]]
    fig, axis = plt.subplots(figsize=(7.6, 5.2))
    axis.bar(["Official released\ncheckpoint"], [rate], color="#2563eb", width=0.52)
    axis.errorbar(
        [0], [rate], yerr=[[rate - ci_low], [ci_high - rate]], fmt="none", ecolor="#111827", capsize=8, linewidth=2
    )
    axis.axhline(24.0, color="#dc2626", linestyle="--", label="Paper rand-obj mean: 24.0%")
    axis.text(0, rate + 2, f"{success}/{episodes} = {rate:.1f}%", ha="center", weight="bold")
    axis.set_ylim(0, 55)
    axis.set_ylabel("Success rate (%)")
    axis.set_title("pinch_tongs, seed 0, 20 official rollouts")
    axis.legend(loc="upper right")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_training_curves(output: Path) -> None:
    import matplotlib.pyplot as plt

    pattern = re.compile(
        r"Step (\d+): grad_norm=([-+0-9.eE]+), loss=([-+0-9.eE]+), "
        r"param_norm=([-+0-9.eE]+)"
    )
    records = [
        (int(step), float(grad), float(loss), float(param))
        for step, grad, loss, param in pattern.findall(TRAIN_LOG.read_text(encoding="utf-8", errors="replace"))
    ]
    steps = [row[0] for row in records]
    gradients = [row[1] for row in records]
    losses = [row[2] for row in records]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(steps, losses, color="#2563eb", linewidth=1.2)
    axes[0].set_ylabel("Training loss")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.2)
    axes[1].plot(steps, gradients, color="#d97706", linewidth=1.2)
    axes[1].set_ylabel("Gradient norm")
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.2)
    fig.suptitle("Official pi0.5 LoRA training diagnostics (300 logged points)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_checkpoint_comparison(output: Path) -> None:
    import matplotlib.pyplot as plt

    comparison = json.loads((ARTIFACT_ROOT / "official_vs_reproduced_comparison.json").read_text(encoding="utf-8"))
    checkpoints = [comparison["official_checkpoint"], comparison["reproduced_checkpoint"]]
    labels = ["Official released", "Local 30k LoRA"]
    rates = [item["success_rate"] * 100 for item in checkpoints]
    lows = [item["wilson_95ci"][0] * 100 for item in checkpoints]
    highs = [item["wilson_95ci"][1] * 100 for item in checkpoints]
    fig, axis = plt.subplots(figsize=(8.4, 5.5))
    axis.bar(labels, rates, color=["#2563eb", "#059669"], width=0.55)
    axis.errorbar(
        range(2),
        rates,
        yerr=[
            [rate - low for rate, low in zip(rates, lows, strict=True)],
            [high - rate for rate, high in zip(rates, highs, strict=True)],
        ],
        fmt="none",
        ecolor="#111827",
        capsize=8,
        linewidth=2,
    )
    paper = comparison["published_reference"]
    axis.axhline(
        paper["success_rate_mean"] * 100,
        color="#dc2626",
        linestyle="--",
        label="Published rand-obj reference: 24.0%",
    )
    for index, (rate, item) in enumerate(zip(rates, checkpoints, strict=True)):
        axis.text(
            index,
            rate + 2,
            f"{item['successes']}/{item['episodes']} = {rate:.1f}%",
            ha="center",
            weight="bold",
        )
    axis.set_ylim(0, max(55, max(highs) + 8))
    axis.set_ylabel("Success rate (%)")
    axis.set_title("pinch_tongs: paired seed-0 official evaluation")
    axis.legend(loc="upper right")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    plot_pipeline(PLOT_ROOT / "official_pi05_pipeline_dataflow.png")
    plot_contract(PLOT_ROOT / "official_state_action_contract.png")
    plot_official_rollout(PLOT_ROOT / "official_checkpoint_rollout_summary.png")
    plot_training_curves(PLOT_ROOT / "reproduced_training_curves.png")
    if (ARTIFACT_ROOT / "official_vs_reproduced_comparison.json").is_file():
        plot_checkpoint_comparison(PLOT_ROOT / "official_vs_reproduced_rollout_comparison.png")
    print(PLOT_ROOT)


if __name__ == "__main__":
    main()
