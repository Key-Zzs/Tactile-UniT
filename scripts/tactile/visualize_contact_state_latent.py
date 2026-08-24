#!/usr/bin/env python3
"""Generate qualitative PCA, t-SNE, and UMAP views of frozen teacher latents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            ".local/artifacts/tactile_teacher/s1_4/teacher_latent_visualization_data.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".local/artifacts/tactile_teacher/s1_4"),
    )
    parser.add_argument("--tsne-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def plot_embedding(
    embedding: np.ndarray,
    labels: dict[str, np.ndarray],
    method: str,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    specifications = (
        ("contact", "derived contact/free", "coolwarm"),
        ("force_trend", "derived force trend", "viridis"),
        ("primitive", "actual motor primitive", "tab20"),
        ("object", "actual object (207 classes)", "turbo"),
    )
    for ax, (key, title, color_map) in zip(axes.flat, specifications):
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels[key],
            cmap=color_map,
            s=3,
            alpha=0.55,
            rasterized=True,
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        if key in ("contact", "force_trend"):
            fig.colorbar(scatter, ax=ax, shrink=0.72)
    fig.suptitle(f"Frozen continuous contact-state latent — {method} (qualitative)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from umap import UMAP

    args = parse_args()
    payload = np.load(args.input)
    latent = StandardScaler().fit_transform(payload["latent"])
    labels = {key: payload[key] for key in ("contact", "force_trend", "primitive", "object")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    pca_model = PCA(n_components=2, random_state=args.seed)
    pca = pca_model.fit_transform(latent)
    plot_embedding(pca, labels, "PCA", args.output_dir / "latent_pca.png")
    summary["pca"] = {
        "samples": len(pca),
        "explained_variance_ratio": pca_model.explained_variance_ratio_.tolist(),
    }

    rng = np.random.default_rng(args.seed)
    selected = np.sort(
        rng.choice(len(latent), size=min(args.tsne_samples, len(latent)), replace=False)
    )
    tsne = TSNE(
        n_components=2,
        perplexity=40,
        learning_rate="auto",
        init="pca",
        max_iter=1000,
        random_state=args.seed,
    ).fit_transform(latent[selected])
    plot_embedding(
        tsne,
        {key: value[selected] for key, value in labels.items()},
        "t-SNE",
        args.output_dir / "latent_tsne.png",
    )
    summary["tsne"] = {"samples": len(tsne), "perplexity": 40, "max_iter": 1000}

    umap = UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric="cosine",
        random_state=args.seed,
    ).fit_transform(latent)
    plot_embedding(umap, labels, "UMAP", args.output_dir / "latent_umap.png")
    summary["umap"] = {
        "samples": len(umap),
        "n_neighbors": 30,
        "min_dist": 0.1,
        "metric": "cosine",
    }
    (args.output_dir / "latent_visualization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
