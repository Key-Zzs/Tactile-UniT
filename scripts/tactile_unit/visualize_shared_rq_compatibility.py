#!/usr/bin/env python3
"""Render deterministic visual diagnostics for the completed S3.0 audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from umap import UMAP  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / ".local/artifacts/tactile_unit/s3_0"
COLORS = ("#4C78A8", "#F58518", "#54A24B", "#E45756")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def normalized_pooled(values: np.ndarray) -> np.ndarray:
    pooled = np.asarray(values, dtype=np.float64).mean(axis=1)
    return pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)


def scatter_embedding(
    embedding: np.ndarray,
    labels: tuple[str, ...],
    count: int,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for index, label in enumerate(labels):
        values = embedding[index * count : (index + 1) * count]
        ax.scatter(
            values[:, 0],
            values[:, 1],
            s=8,
            alpha=0.48,
            color=COLORS[index],
            label=label,
            rasterized=True,
        )
    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.legend(frameon=False, markerscale=2)
    save(fig, path)


def main() -> int:
    args = parse_args()
    data_path = args.output_dir / "visualization_data.npz"
    summary_path = args.output_dir / "s3_0_summary.json"
    if not data_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("run the S3.0 audit before visualization")
    summary = json.loads(summary_path.read_text())
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    with np.load(data_path, allow_pickle=False) as data:
        labels = tuple(data["modality"].tolist())
        reference = np.asarray(data["reference_l2"])
        contact = np.asarray(data["contact_continuous"])
        relative = np.asarray(data["relative_distortion"])
        frequencies = np.asarray(data["code_frequency"])
    if labels != ("vision", "action", "multimodal", "contact"):
        raise RuntimeError("unexpected visualization modality identity")
    values = (reference[:, 0], reference[:, 1], reference[:, 2], contact)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(
        min(np.linalg.norm(value, axis=2).min() for value in values),
        max(np.linalg.norm(value, axis=2).max() for value in values),
        70,
    )
    for label, value, color in zip(labels, values, COLORS):
        ax.hist(
            np.linalg.norm(value, axis=2).ravel(),
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            label=label,
            color=color,
        )
    ax.set_title("Original UniT L2 and contact token norms")
    ax.set_xlabel("token L2 norm")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    save(fig, plot_dir / "latent_norms.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(
        [relative[index] for index in range(len(labels))],
        tick_labels=labels,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#D8E6F3"},
        medianprops={"color": "black"},
    )
    ax.set_yscale("log")
    ax.set_title("Per-sample energy-normalized quantization distortion")
    ax.set_ylabel("relative distortion (log scale)")
    save(fig, plot_dir / "quantization_distortion.png")

    usage = [
        row
        for row in summary["codebook_usage"]["per_stage"]
        if row["query"] == "all"
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    width = 0.36
    for stage in range(2):
        values_by_modality = [
            next(
                float(row["perplexity"])
                for row in usage
                if row["modality"] == label and int(row["stage"]) == stage
            )
            for label in labels
        ]
        ax.bar(x + (stage - 0.5) * width, values_by_modality, width, label=f"stage {stage}")
    ax.set_xticks(x, labels)
    ax.set_ylabel("perplexity")
    ax.set_title("Frozen-RQ codebook perplexity")
    ax.legend(frameon=False)
    save(fig, plot_dir / "codebook_perplexity.png")

    fig, ax = plt.subplots(figsize=(13, 5))
    matrix = frequencies.reshape(len(labels) * 2, frequencies.shape[-1])
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_yticks(
        np.arange(len(labels) * 2),
        [f"{label} / stage {stage}" for label in labels for stage in range(2)],
    )
    ax.set_xlabel("code index")
    ax.set_title("Empirical frozen-RQ code frequencies")
    fig.colorbar(image, ax=ax, label="frequency")
    save(fig, plot_dir / "code_usage_heatmap.png")

    reconstruction = summary["reconstruction_retention"]["conditions"]
    conditions = ("continuous", "quantized", "zero", "shuffled")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(conditions))
    width = 0.36
    all_mse = [reconstruction[name]["all"]["future_mse"] for name in conditions]
    dynamic_mse = [reconstruction[name]["dynamic"]["future_mse"] for name in conditions]
    ax.bar(x - width / 2, all_mse, width, label="all windows")
    ax.bar(x + width / 2, dynamic_mse, width, label="dynamic windows")
    ax.set_xticks(x, conditions)
    ax.set_ylabel("future contact-latent MSE")
    ax.set_title("Continuous vs frozen-RQ contact reconstruction")
    ax.legend(frameon=False)
    save(fig, plot_dir / "contact_quantized_reconstruction.png")

    pooled = [normalized_pooled(value) for value in values]
    joint = np.concatenate(pooled, axis=0)
    pca = PCA(n_components=2, random_state=42).fit_transform(joint)
    scatter_embedding(
        pca,
        labels,
        len(contact),
        "Joint PCA of mean-pooled L2 representations",
        plot_dir / "joint_pca.png",
    )
    umap = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        n_jobs=1,
    ).fit_transform(joint)
    scatter_embedding(
        umap,
        labels,
        len(contact),
        "Joint UMAP (qualitative; GR1 and T-Rex are unpaired)",
        plot_dir / "joint_umap.png",
    )

    plots = sorted(str(path) for path in plot_dir.glob("*.png"))
    visualization_summary = {
        "schema": "tactile3d-unit.s3-0-visualization.v1",
        "status": "PASS" if len(plots) == 7 else "FAIL",
        "joint_fit": True,
        "pooling": "mean across 8 queries then L2 normalize",
        "unpaired_interpretation": "qualitative distribution/codebook diagnostic only",
        "plots": plots,
    }
    (args.output_dir / "visualization_summary.json").write_text(
        json.dumps(visualization_summary, indent=2, sort_keys=True) + "\n"
    )
    summary["artifacts"]["plots"] = plots
    summary["visualization"] = visualization_summary
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(visualization_summary, indent=2))
    return 0 if visualization_summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
