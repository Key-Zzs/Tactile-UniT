#!/usr/bin/env python3
"""Render the standalone A-R diagnostics and human-acceptance summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = ROOT / ".local/artifacts/tactile_unit/s3_3_r"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_root = args.artifact_root
    evaluation = json.loads((artifact_root / "held_out_evaluation.json").read_text())
    diagnosis = json.loads((artifact_root / "a_r0_diagnosis.json").read_text())
    training = json.loads((artifact_root / "training_summary.json").read_text())
    plots = artifact_root / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    temporal = evaluation["temporal_controls"]
    names = ["correct", "reversed", "shuffled", "different_episode", "zero", "mean"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, subset in zip(axes, ("all", "dynamic")):
        values = [temporal[subset][name] for name in names]
        axis.bar(names, values, color=["#4c78a8", "#f58518", "#e45756", "#72b7b2", "#b279a2", "#ff9da6"])
        axis.set_title(f"{subset.title()} temporal controls")
        axis.set_ylabel("normalized MSE")
        axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(plots / "temporal_controls.png", dpi=180)
    plt.close(figure)

    ablation = evaluation["feature_ablation"]["normalized_mse"]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(list(ablation), list(ablation.values()), color="#4c78a8")
    axis.set_ylabel("validation normalized MSE")
    axis.set_title("Transition-centered feature ablation")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(plots / "transition_feature_ablation.png", dpi=180)
    plt.close(figure)

    reconstruction = evaluation["reconstruction"]
    segment_names = ["left_arm", "left_hand", "right_arm", "right_hand"]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(segment_names))
    axis.bar(x - 0.18, [reconstruction[name]["normalized_mse"] for name in segment_names], 0.36, label="all")
    axis.bar(x + 0.18, [reconstruction[f"dynamic_{name}"]["normalized_mse"] for name in segment_names], 0.36, label="dynamic")
    axis.set_xticks(x, segment_names)
    axis.set_ylabel("normalized MSE")
    axis.set_title("Action reconstruction by anatomical group")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "reconstruction_segments.png", dpi=180)
    plt.close(figure)

    values = np.load(artifact_root / "visualization_data.npz")
    continuous = values["test_z"].reshape(len(values["test_z"]), -1)
    quantized = values["quantized_test_z"].reshape(len(values["quantized_test_z"]), -1)
    count = min(4096, len(continuous))
    combined = np.concatenate((continuous[:count], quantized[:count]))
    embedding = PCA(n_components=2, random_state=3371).fit_transform(combined)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    axes[0].scatter(embedding[:count, 0], embedding[:count, 1], c=values["magnitude"][:count], s=4, cmap="viridis")
    axes[0].set_title("continuous z_a")
    axes[1].scatter(embedding[count:, 0], embedding[count:, 1], c=values["magnitude"][:count], s=4, cmap="viridis")
    axes[1].set_title("frozen-RQ z_a")
    figure.tight_layout()
    figure.savefig(plots / "continuous_and_frozen_rq_pca.png", dpi=180)
    plt.close(figure)

    raw_test = diagnosis["raw_negative_strength"]["test"]["controls"]
    selected = evaluation["selected_architecture"]
    noncollapse = evaluation["noncollapse"]
    acceptance = f"""# Human Acceptance — Route A-R

## 1. Raw correct/reversed/shuffled distances

- Correct normalized MSE distance: `{raw_test['correct']['all']['normalized_mse']:.8f}`.
- Reversed: all `{raw_test['reversed']['all']['normalized_mse']:.8f}`, dynamic `{raw_test['reversed']['dynamic']['normalized_mse']:.8f}`.
- Shuffled: all `{raw_test['shuffled']['all']['normalized_mse']:.8f}`, dynamic `{raw_test['shuffled']['dynamic']['normalized_mse']:.8f}`.

## 2. Decoder token necessity

- Zero/full ratio: `{temporal['paired_bootstrap']['all']['zero']['ratio']:.4f}`.
- Mean/full ratio: `{temporal['paired_bootstrap']['all']['mean']['ratio']:.4f}`.
- State-only is the explicit zero-token decoder path and is reported separately in the evaluation JSON.

## 3. Temporal control comparison

- Dynamic reversed/correct: `{temporal['paired_bootstrap']['dynamic']['reversed']['ratio']:.4f}`; 95% CI `[{temporal['paired_bootstrap']['dynamic']['reversed']['ci95_lower']:.4f}, {temporal['paired_bootstrap']['dynamic']['reversed']['ci95_upper']:.4f}]`.
- Dynamic shuffled/correct: `{temporal['paired_bootstrap']['dynamic']['shuffled']['ratio']:.4f}`; 95% CI `[{temporal['paired_bootstrap']['dynamic']['shuffled']['ci95_lower']:.4f}, {temporal['paired_bootstrap']['dynamic']['shuffled']['ci95_upper']:.4f}]`.
- Different-episode/correct: `{temporal['paired_bootstrap']['all']['different_episode']['ratio']:.4f}`.

![Temporal controls](plots/temporal_controls.png)

## 4. Transition-centered feature ablation

`{json.dumps(evaluation['feature_ablation'], sort_keys=True)}`

![Feature ablation](plots/transition_feature_ablation.png)

## 5. Shared vs native encoder

- R1-P gate: `{training['r1_p']['gate_passed']}`.
- R1-N executed: `{training['r1_n']['executed']}`; gate: `{training['r1_n'].get('gate_passed')}`.
- Selected: `{selected['candidate']}` / `{selected['encoder_type']}`.

## 6. Dynamic action examples

The dynamic subset is frozen by the train-derived threshold. The temporal and segment plots use the untouched test distribution without filtering or resampling it.

![Segment reconstruction](plots/reconstruction_segments.png)

## 7. Non-collapse

- Effective rank: `{noncollapse['effective_rank']:.4f}`.
- Collapsed-query fraction: `{noncollapse['collapsed_query_fraction']:.6f}`.
- Mean query cosine distance: `{noncollapse['query_diversity']['mean_cosine_distance']:.4f}`.

![Continuous and frozen-RQ PCA](plots/continuous_and_frozen_rq_pca.png)

## 8. Final decision

`{evaluation['decision']}`
"""
    (artifact_root / "HUMAN_ACCEPTANCE.md").write_text(acceptance)
    summary = {
        "schema": "tactile3d-unit.s3-3-r-visualization-summary.v1",
        "plots": sorted(path.name for path in plots.glob("*.png")),
        "human_acceptance": "HUMAN_ACCEPTANCE.md",
        "decision": evaluation["decision"],
    }
    (artifact_root / "visualization_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
