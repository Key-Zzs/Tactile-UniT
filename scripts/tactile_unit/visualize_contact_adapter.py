#!/usr/bin/env python3
"""Render deterministic S3.2 contact-adaptor diagnostics."""

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
DEFAULT_OUTPUT = ROOT / ".local/artifacts/tactile_unit/s3_2"
COLORS = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#72B7B2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def pooled_normalized(values: np.ndarray) -> np.ndarray:
    pooled = np.asarray(values, dtype=np.float64).mean(axis=1)
    return pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)


def scatter(
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
            alpha=0.45,
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
    summary_path = args.output_dir / "s3_2_summary.json"
    data_path = args.output_dir / "visualization_data.npz"
    if not summary_path.is_file() or not data_path.is_file():
        raise FileNotFoundError("run S3.2 evaluation before visualization")
    summary = json.loads(summary_path.read_text())
    selected = summary["validation_selection"]["selected"]
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    with np.load(data_path, allow_pickle=False) as data:
        reference = np.asarray(data["reference_l2"])
        identity_contact = np.asarray(data["contact_identity"])
        adapted_contact = np.asarray(data["selected_adapted"])
        frequencies = np.asarray(data["code_frequency"])
        frequency_models = tuple(data["model"].tolist())

    quant = summary["quantization"]
    references = summary["original_unit_quantization_references"]
    distortion_labels = ("vision", "action", "multimodal", "identity", "P1", "P2")
    distortion_values = [
        references["vision"]["relative_distortion"],
        references["action"]["relative_distortion"],
        references["multimodal"]["relative_distortion"],
        quant["identity"]["relative_distortion"],
        quant["affine"]["relative_distortion"],
        quant["mlp"]["relative_distortion"],
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(distortion_labels, distortion_values, color=COLORS)
    ax.set_yscale("log")
    ax.set_ylabel("energy-normalized relative distortion")
    ax.set_title("Frozen-RQ quantization compatibility")
    save(fig, plot_dir / "distortion_comparison.png")

    usage = summary["codebook_usage"]
    models = ("identity", "affine", "mlp")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    width = 0.36
    for stage in range(2):
        values = [usage[name][stage]["perplexity"] for name in models]
        ax.bar(x + (stage - 0.5) * width, values, width, label=f"stage {stage}")
    ax.axhspan(80, 106, color="#BBBBBB", alpha=0.2, label="Original UniT envelope")
    ax.set_xticks(x, models)
    ax.set_ylabel("perplexity")
    ax.set_title("Frozen-RQ codebook usage")
    ax.legend(frameon=False)
    save(fig, plot_dir / "codebook_perplexity.png")

    fig, ax = plt.subplots(figsize=(13, 4.8))
    image = ax.imshow(
        frequencies.reshape(len(frequency_models) * 2, frequencies.shape[-1]),
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
    )
    ax.set_yticks(
        np.arange(len(frequency_models) * 2),
        [f"{name} / stage {stage}" for name in frequency_models for stage in range(2)],
    )
    ax.set_xlabel("code index")
    ax.set_title("Contact code-frequency heatmap")
    fig.colorbar(image, ax=ax, label="frequency")
    save(fig, plot_dir / "code_usage_heatmap.png")

    reconstruction = summary["reconstruction"]
    reconstruction_names = (
        "continuous",
        "identity_quantized",
        f"{selected}_quantized",
        "zero",
        "shuffled",
    )
    labels = ("continuous", "identity RQ", "adapted RQ", "zero", "shuffled")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(
        x - width / 2,
        [reconstruction[name]["all"]["future_mse"] for name in reconstruction_names],
        width,
        label="all",
    )
    ax.bar(
        x + width / 2,
        [reconstruction[name]["dynamic"]["future_mse"] for name in reconstruction_names],
        width,
        label="dynamic",
    )
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylabel("future latent MSE")
    ax.set_title("Frozen S2 decoder reconstruction retention")
    ax.legend(frameon=False)
    save(fig, plot_dir / "reconstruction_comparison.png")

    norm_values = (
        reference[:, 0],
        reference[:, 1],
        reference[:, 2],
        identity_contact,
        adapted_contact,
    )
    norm_labels = ("vision", "action", "multimodal", "identity contact", "adapted contact")
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(
        min(np.linalg.norm(value, axis=2).min() for value in norm_values),
        max(np.linalg.norm(value, axis=2).max() for value in norm_values),
        70,
    )
    for label, value, color in zip(norm_labels, norm_values, COLORS):
        ax.hist(
            np.linalg.norm(value, axis=2).ravel(),
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            label=label,
            color=color,
        )
    ax.set_xlabel("token L2 norm")
    ax.set_ylabel("density")
    ax.set_title("Original UniT and adapted Contact token norms")
    ax.legend(frameon=False)
    save(fig, plot_dir / "token_norms.png")

    embedding_values = (*norm_values[:3], identity_contact, adapted_contact)
    embedding_labels = ("vision", "action", "multimodal", "identity contact", "adapted contact")
    joint = np.concatenate([pooled_normalized(value) for value in embedding_values], axis=0)
    pca = PCA(n_components=2, random_state=42).fit_transform(joint)
    scatter(
        pca,
        embedding_labels,
        len(identity_contact),
        "Joint PCA of Original UniT and Contact L2",
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
    scatter(
        umap,
        embedding_labels,
        len(identity_contact),
        "Joint UMAP (cross-dataset distribution diagnostic; unpaired)",
        plot_dir / "joint_umap.png",
    )

    semantic = summary["dynamic_semantic_retention"]
    probe_names = ("contact_transition", "force_trend")
    representations = ("continuous", "identity_quantized", f"{selected}_quantized")
    representation_labels = ("continuous", "identity RQ", "adapted RQ")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(probe_names))
    width = 0.24
    for index, (representation, label) in enumerate(zip(representations, representation_labels)):
        ax.bar(
            x + (index - 1) * width,
            [semantic[representation][probe]["macro_f1"] for probe in probe_names],
            width,
            label=label,
        )
    ax.set_xticks(x, ("contact transition", "force trend"))
    ax.set_ylim(0, 1)
    ax.set_ylabel("macro-F1")
    ax.set_title("Dynamic semantic retention")
    ax.legend(frameon=False)
    save(fig, plot_dir / "semantic_retention.png")

    diversity = summary["query_diversity_and_noncollapse"]
    diversity_names = ("identity_continuous", f"{selected}_continuous", f"{selected}_quantized")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(diversity_names))
    ax.bar(
        x,
        [diversity[name]["mean_distance_from_sample_token_mean"] for name in diversity_names],
        color=COLORS[:3],
    )
    ax.set_xticks(x, ("identity", "adapted continuous", "adapted quantized"))
    ax.set_ylabel("mean distance from sample token mean")
    ax.set_title("Query diversity (collapsed fraction is zero when healthy)")
    save(fig, plot_dir / "query_diversity.png")

    plots = sorted(str(path) for path in plot_dir.glob("*.png"))
    visualization = {
        "schema": "tactile3d-unit.s3-2-contact-adaptor-visualization.v1",
        "status": "PASS" if len(plots) == 9 else "FAIL",
        "selected_architecture": selected,
        "joint_fit": True,
        "unpaired_interpretation": (
            "GR1 Original UniT and T-Rex Contact plots are distribution diagnostics, not paired alignment"
        ),
        "plots": plots,
    }
    write_text = json.dumps(visualization, indent=2, sort_keys=True) + "\n"
    (args.output_dir / "visualization_summary.json").write_text(write_text)
    summary["visualizations"] = visualization
    summary["artifacts"]["plots"] = plots
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(visualization, indent=2))
    return 0 if visualization["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
