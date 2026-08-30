#!/usr/bin/env python3
"""Create the compact C1/C2 human-acceptance plot set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c1-baseline", type=Path, default=ROOT / ".local/artifacts/tactile_unit/vac_c1/native_pairwise_baseline.json")
    parser.add_argument("--training", type=Path, default=ROOT / ".local/experiments/tactile_unit/vac_c2/training_summary.json")
    parser.add_argument("--evaluation", type=Path, default=ROOT / ".local/artifacts/tactile_unit/vac_c2/evaluation.json")
    parser.add_argument("--output", type=Path, default=ROOT / ".local/artifacts/tactile_unit/vac_c2/plots")
    return parser.parse_args()


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    c1 = json.loads(args.c1_baseline.read_text())
    training = json.loads(args.training.read_text())
    evaluation = json.loads(args.evaluation.read_text())
    output = args.output
    pairs = ["V-A", "V-C", "A-C"]
    paths = []

    native = [c1["pairs"][name]["all"]["paired_minus_shuffled_margin"] for name in pairs]
    shared = [evaluation["alignment"][name]["all"]["paired_minus_shuffled_margin"] for name in pairs]
    fig, ax = plt.subplots(figsize=(6, 4)); x = np.arange(3); width = 0.36
    ax.bar(x - width / 2, native, width, label="native C1")
    ax.bar(x + width / 2, shared, width, label="shared C2")
    ax.axhline(0, color="black", linewidth=0.8); ax.set_xticks(x, pairs)
    ax.set_ylabel("paired - shuffled cosine"); ax.legend(); ax.set_title("Native vs shared alignment")
    path = output / "native_vs_shared_margin.png"; save(fig, path); paths.append(path)

    labels, multipliers = [], []
    for name in pairs:
        for direction, arrow in (("forward", "→"), ("reverse", "←")):
            value = evaluation["alignment"][name]["all"]["retrieval"][direction]
            labels.append(f"{name} {arrow}")
            multipliers.append(value["recall_at_10"] / value["chance"]["recall_at_10"])
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(labels, multipliers)
    ax.axhline(1.0, color="black", label="chance"); ax.axhline(1.5, color="red", linestyle="--", label="gate")
    ax.set_ylabel("R@10 / chance"); ax.tick_params(axis="x", rotation=35); ax.legend(); ax.set_title("Six-direction independent retrieval")
    path = output / "retrieval_vs_chance.png"; save(fig, path); paths.append(path)

    trial_labels = ["C0"] + [f"T{row['trial_id']} {row['trial']['candidate']}" for row in training["trials"]]
    scores = [training["C0_native"]["summary"]["comprehensive_score"]] + [row["best"]["score"] for row in training["trials"]]
    fig, ax = plt.subplots(figsize=(9, 4)); ax.bar(trial_labels, scores)
    ax.tick_params(axis="x", rotation=35); ax.set_ylabel("validation utility"); ax.set_title("Preregistered candidate comparison")
    path = output / "candidate_comparison.png"; save(fig, path); paths.append(path)

    contact = evaluation["contact"]["probes"]
    fig, ax = plt.subplots(figsize=(6, 4)); names = ["transition", "force trend"]
    values = [contact["contact_transition"]["retention"], contact["force_trend"]["retention"]]
    ax.bar(names, values); ax.axhline(0.9, color="red", linestyle="--", label="hard gate")
    ax.set_ylim(0, max(1.1, max(values) * 1.1)); ax.set_ylabel("semantic retention"); ax.legend(); ax.set_title("Contact preservation")
    path = output / "contact_semantic_retention.png"; save(fig, path); paths.append(path)

    temporal = evaluation["action"]["shared_temporal"]["dynamic"]
    names = ["reversed", "shuffled", "different episode", "zero"]
    values = [temporal[f"{name.replace(' ', '_')}_over_correct"] for name in names]
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(names, values); ax.axhline(1.05, color="red", linestyle="--")
    ax.set_ylabel("error / correct"); ax.set_title("Recovered Action temporal retention")
    path = output / "action_temporal_retention.png"; save(fig, path); paths.append(path)

    geometry = evaluation["geometry"]["modalities"]
    fig, ax = plt.subplots(figsize=(6, 4)); names = ["Vision", "Action", "Contact"]
    values = [geometry[key]["effective_rank"] for key in ("vision", "action", "contact")]
    ax.bar(names, values); ax.set_ylabel("effective rank"); ax.set_title("Shared-space non-collapse")
    path = output / "effective_rank.png"; save(fig, path); paths.append(path)

    labels, values = [], []
    for pair in pairs:
        for subset in ("dynamic", "rare_boundary"):
            labels.append(f"{pair}\n{subset}")
            values.append(evaluation["alignment"][pair][subset].get("paired_minus_shuffled_margin", np.nan))
    fig, ax = plt.subplots(figsize=(9, 4)); ax.bar(labels, values); ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("paired - shuffled"); ax.set_title("Dynamic and rare-boundary alignment")
    path = output / "dynamic_boundary_alignment.png"; save(fig, path); paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 4)); ax.axis("off")
    for y, (native_name, projector, shared_name) in enumerate((("z_v", "P_v", "u_v"), ("z_a", "P_a", "u_a"), ("z_c", "P_c", "u_c"))):
        ypos = 0.8 - y * 0.3
        ax.text(0.08, ypos, native_name, bbox={"boxstyle": "round", "facecolor": "#d9eaf7"}, ha="center")
        ax.annotate("", (0.38, ypos), (0.15, ypos), arrowprops={"arrowstyle": "->"})
        ax.text(0.45, ypos, projector, bbox={"boxstyle": "round", "facecolor": "#f5dfb3"}, ha="center")
        ax.annotate("", (0.72, ypos), (0.52, ypos), arrowprops={"arrowstyle": "->"})
        ax.text(0.8, ypos, shared_name, bbox={"boxstyle": "round", "facecolor": "#d9f0d3"}, ha="center")
    ax.text(0.45, 0.02, "Each row is independently computable; no paired counterpart input", ha="center", weight="bold")
    path = output / "independent_encoding_schematic.png"; save(fig, path); paths.append(path)

    summary = {"schema": "tactile3d-unit.vac-c2-visualization.v1", "plots": [str(path.relative_to(ROOT)) for path in paths]}
    (output.parent / "visualization_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    acceptance = output.parent / "HUMAN_ACCEPTANCE.md"
    with acceptance.open("a") as handle:
        handle.write("## Plots\n\n")
        for path in paths:
            handle.write(f"- `{path.relative_to(ROOT)}`\n")
        handle.write("\n## Exact inspection commands\n\n")
        handle.write("```bash\n")
        handle.write("python -m json.tool .local/artifacts/tactile_unit/vac_c2/evaluation.json\n")
        handle.write("python -m json.tool .local/experiments/tactile_unit/vac_c2/training_summary.json\n")
        handle.write("```\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
