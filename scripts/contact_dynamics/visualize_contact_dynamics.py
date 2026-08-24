#!/usr/bin/env python3
"""Generate the qualitative S2/M2 contact-transition acceptance figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


CLASS_NAMES = ("free→free", "free→contact", "contact→contact", "contact→free")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(".local/artifacts/contact_dynamics/s2_4/s2_evaluation_summary.json"),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(".local/artifacts/contact_dynamics/s2_4/visualization_data.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".local/artifacts/contact_dynamics/s2_4/plots"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def scatter_embedding(
    embedding: np.ndarray,
    labels: np.ndarray,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7))
    for label, name in enumerate(CLASS_NAMES):
        mask = labels == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=7,
            alpha=0.55,
            label=f"{name} (n={int(mask.sum())})",
        )
    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.legend(markerscale=2, fontsize=8)
    ax.grid(alpha=0.15)
    save(fig, path)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.summary.read_text())
    data = np.load(args.data)
    code = np.asarray(data["code"], dtype=np.float32).reshape(len(data["code"]), -1)
    labels = np.asarray(data["contact_transition"], dtype=np.int64)
    current = np.asarray(data["current"], dtype=np.float32)
    target = np.asarray(data["target"], dtype=np.float32)
    latent_magnitude = np.linalg.norm(target - current, axis=1)
    force_delta = np.asarray(data["force_delta"], dtype=np.float32)
    outputs = []

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(latent_magnitude, bins=70, color="C0", alpha=0.85)
    axes[0].set_title("Teacher-latent transition magnitude")
    axes[0].set_xlabel(r"$\|h_{t+16}^c-h_t^c\|_2$")
    axes[0].set_ylabel("pairs")
    axes[1].hist(force_delta, bins=70, color="C1", alpha=0.85)
    axes[1].set_title("Physical max-fingertip force delta")
    axes[1].set_xlabel("future − current (public sensor units)")
    for ax in axes:
        ax.grid(alpha=0.2)
    path = args.output_dir / "transition_magnitude_distribution.png"
    save(fig, path)
    outputs.append(path)

    counts = np.bincount(labels, minlength=4)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.bar(CLASS_NAMES, counts, color=["C0", "C1", "C2", "C3"])
    ax.bar_label(bars)
    ax.set_title("Contact-transition class distribution (derived labels)")
    ax.set_ylabel("sampled test pairs")
    ax.grid(axis="y", alpha=0.2)
    path = args.output_dir / "contact_transition_distribution.png"
    save(fig, path)
    outputs.append(path)

    conditions = ("full", "zero", "shuffled_code", "reversed_transition", "shuffled_future")
    all_mse = [summary["ablations"][name]["all"]["future_mse"] for name in conditions]
    dynamic_mse = [summary["ablations"][name]["dynamic"]["future_mse"] for name in conditions]
    x = np.arange(len(conditions))
    fig, ax = plt.subplots(figsize=(10.5, 5))
    ax.bar(x - 0.18, all_mse, 0.36, label="all windows")
    ax.bar(x + 0.18, dynamic_mse, 0.36, label="dynamic windows")
    ax.set_xticks(x, [name.replace("_", "\n") for name in conditions])
    ax.set_yscale("log")
    ax.set_ylabel("future Teacher-latent MSE (log scale)")
    ax.set_title("Transition-code necessity and temporal/pairing controls")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    path = args.output_dir / "transition_code_ablation.png"
    save(fig, path)
    outputs.append(path)

    full = np.asarray(data["full"])
    zero = np.asarray(data["zero"])
    shuffled = np.asarray(data["shuffled"])
    reversed_prediction = np.asarray(data["reversed"])
    dynamic_indices = np.flatnonzero(data["dynamic"])
    order = dynamic_indices[np.argsort(latent_magnitude[dynamic_indices])]
    example_indices = order[np.linspace(0, len(order) - 1, 4).astype(int)]
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    dimensions = np.arange(64)
    for ax, index in zip(axes, example_indices):
        ax.plot(dimensions, target[index, :64], color="black", lw=1.8, label="target")
        ax.plot(dimensions, full[index, :64], color="C0", lw=1.2, label="full")
        ax.plot(dimensions, zero[index, :64], color="C1", alpha=0.8, label="zero")
        ax.plot(dimensions, shuffled[index, :64], color="C2", alpha=0.8, label="shuffled")
        ax.plot(dimensions, reversed_prediction[index, :64], color="C3", alpha=0.8, label="reversed")
        ax.set_ylabel("latent value")
        ax.set_title(
            f"dynamic example {int(data['index'][index])}: "
            f"|Δh|={latent_magnitude[index]:.3f}, Δforce={force_delta[index]:.3f}"
        )
        ax.grid(alpha=0.15)
    axes[0].legend(ncol=5, fontsize=8)
    axes[-1].set_xlabel("first 64 Teacher-latent dimensions")
    path = args.output_dir / "future_latent_reconstruction_examples.png"
    save(fig, path)
    outputs.append(path)

    pca = PCA(n_components=2, random_state=args.seed).fit_transform(code)
    path = args.output_dir / "latent_pca.png"
    scatter_embedding(pca, labels, "S2 transition code PCA (qualitative)", path)
    outputs.append(path)

    rng = np.random.default_rng(args.seed)
    tsne_count = min(2500, len(code))
    tsne_index = np.sort(rng.choice(len(code), tsne_count, replace=False))
    tsne = TSNE(
        n_components=2,
        perplexity=35,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
    ).fit_transform(code[tsne_index])
    path = args.output_dir / "latent_tsne.png"
    scatter_embedding(tsne, labels[tsne_index], "S2 transition code t-SNE (qualitative)", path)
    outputs.append(path)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=25,
        min_dist=0.15,
        metric="euclidean",
        random_state=args.seed,
    )
    umap_embedding = reducer.fit_transform(code)
    path = args.output_dir / "latent_umap.png"
    scatter_embedding(umap_embedding, labels, "S2 transition code UMAP (qualitative)", path)
    outputs.append(path)

    eigenvalues = np.asarray(summary["collapse"]["flattened_8x32"]["eigenvalues"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.semilogy(np.arange(1, len(eigenvalues) + 1), np.maximum(eigenvalues, 1e-12))
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("covariance eigenvalue (log scale)")
    ax.set_title(
        "Transition-code covariance spectrum — "
        f"effective rank {summary['collapse']['flattened_8x32']['effective_rank']:.2f}"
    )
    ax.grid(alpha=0.2)
    path = args.output_dir / "latent_covariance_spectrum.png"
    save(fig, path)
    outputs.append(path)

    token_mean = np.asarray(data["code"], dtype=np.float64).mean(axis=0)
    token_mean /= np.maximum(np.linalg.norm(token_mean, axis=1, keepdims=True), 1e-12)
    cosine = token_mean @ token_mean.T
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(cosine, vmin=-1, vmax=1, cmap="coolwarm")
    fig.colorbar(image, ax=ax, label="cosine")
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xlabel("query position")
    ax.set_ylabel("query position")
    ax.set_title("Mean query-position cosine matrix")
    path = args.output_dir / "query_diversity.png"
    save(fig, path)
    outputs.append(path)

    horizons = (8, 16, 24)
    persistence = [summary["horizon_audit"][str(k)]["persistence_future_mse"] for k in horizons]
    transition_mean = [
        summary["horizon_audit"][str(k)]["transition_l2"]["mean"] for k in horizons
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar([str(k) for k in horizons], persistence)
    axes[0].set_title("Persistence baseline by horizon")
    axes[0].set_xlabel("k (frames)")
    axes[0].set_ylabel("future latent MSE")
    axes[1].bar([str(k) for k in horizons], transition_mean, color="C2")
    axes[1].set_title("Mean latent transition magnitude")
    axes[1].set_xlabel("k (frames)")
    axes[1].set_ylabel(r"mean $\|Δh\|_2$")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    path = args.output_dir / "horizon_comparison.png"
    save(fig, path)
    outputs.append(path)

    result = {
        "schema": "tactile3d-unit.s2-visualization.v1",
        "status": "PASS",
        "sample_count": len(code),
        "tsne_sample_count": tsne_count,
        "seed": args.seed,
        "qualitative_only": ["PCA", "t-SNE", "UMAP"],
        "files": [str(path) for path in outputs],
    }
    (args.output_dir / "visualization_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
