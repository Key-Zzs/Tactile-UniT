#!/usr/bin/env python3
"""Create the standalone S3.3 continuous-action and frozen-RQ diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = ROOT / ".local/artifacts/tactile_unit/s3_3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = json.loads((args.artifacts / "held_out_evaluation.json").read_text())
    values = np.load(args.artifacts / "visualization_data.npz")
    plot_root = args.artifacts / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)

    continuous = values["test_z"].reshape(len(values["test_z"]), -1)
    quantized = values["quantized_test_z"].reshape(len(values["quantized_test_z"]), -1)
    combined = np.concatenate((continuous, quantized))
    embedding = PCA(n_components=2, random_state=3301).fit_transform(combined)
    count = len(continuous)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    scatter = axes[0].scatter(
        embedding[:count, 0],
        embedding[:count, 1],
        c=values["magnitude"],
        s=5,
        alpha=0.55,
        cmap="viridis",
    )
    axes[0].set_title("Continuous T-Rex Action L2 · color=action magnitude")
    axes[0].set_xlabel("PCA 1")
    axes[0].set_ylabel("PCA 2")
    figure.colorbar(scatter, ax=axes[0], label="normalized RMS delta")
    axes[1].scatter(
        embedding[:count, 0], embedding[:count, 1], s=4, alpha=0.25, label="continuous"
    )
    axes[1].scatter(
        embedding[count:, 0], embedding[count:, 1], s=4, alpha=0.25, label="frozen RQ"
    )
    axes[1].set_title("Frozen Original-UniT RQ displacement")
    axes[1].set_xlabel("PCA 1")
    axes[1].set_ylabel("PCA 2")
    axes[1].legend()
    figure.savefig(plot_root / "continuous_and_frozen_rq_pca.png", dpi=180)
    plt.close(figure)

    temporal = result["temporal_controls"]
    names = ["correct", "reversed", "shuffled", "different_episode"]
    values_temporal = [temporal[name] for name in names]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bars = axis.bar(names, values_temporal, color=["#4c78a8", "#f58518", "#e45756", "#72b7b2"])
    axis.bar_label(bars, fmt="%.4f", padding=3)
    axis.set_ylabel("held-out normalized reconstruction/matching MSE")
    axis.set_title("Temporal controls (lower is better)")
    figure.savefig(plot_root / "temporal_controls.png", dpi=180)
    plt.close(figure)

    reconstruction = result["reconstruction"]
    groups = ["all", "left_arm", "left_hand", "right_arm", "right_hand"]
    all_mse = [reconstruction[name]["normalized_mse"] for name in groups]
    dynamic_mse = [
        reconstruction["dynamic"]["normalized_mse"],
        *[reconstruction[f"dynamic_{name}"]["normalized_mse"] for name in groups[1:]],
    ]
    positions = np.arange(len(groups))
    width = 0.38
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.bar(positions - width / 2, all_mse, width=width, label="all windows")
    axis.bar(positions + width / 2, dynamic_mse, width=width, label="dynamic windows")
    axis.set_xticks(positions, groups)
    axis.set_ylabel("normalized MSE")
    axis.set_title("T-Rex action reconstruction by anatomical segment")
    axis.legend()
    figure.savefig(plot_root / "reconstruction_segments.png", dpi=180)
    plt.close(figure)

    summary = {
        "schema": "tactile3d-unit.s3-3-visualization-summary.v1",
        "decision": result["decision"],
        "points": count,
        "plots": [
            "plots/continuous_and_frozen_rq_pca.png",
            "plots/temporal_controls.png",
            "plots/reconstruction_segments.png",
        ],
        "qualitative_only": ["continuous_and_frozen_rq_pca.png"],
        "quantitative_sources": [
            "held_out_evaluation.json",
            "visualization_data.npz",
        ],
    }
    (args.artifacts / "visualization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
