#!/usr/bin/env python3
"""Joint-fit PCA, t-SNE, UMAP, and interactive views for T4 features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from unit_representation_metrics import mean_query_pool


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / ".local/artifacts/reproduction/t4"


def fit_tsne(values: np.ndarray, random_state: int) -> np.ndarray:
    from sklearn.manifold import TSNE

    perplexity = min(30.0, max(2.0, (len(values) - 1) / 3.0))
    kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "random_state": random_state,
        "init": "pca",
        "learning_rate": "auto",
        "method": "barnes_hut",
    }
    try:
        return TSNE(max_iter=1000, **kwargs).fit_transform(values)
    except TypeError:
        return TSNE(n_iter=1000, **kwargs).fit_transform(values)


def fit_umap(values: np.ndarray, random_state: int) -> np.ndarray:
    import umap

    return umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=random_state,
        transform_seed=random_state,
    ).fit_transform(values)


def save_static_plot(path: Path, coordinates: np.ndarray, tasks: np.ndarray, modalities: np.ndarray, title: str, color_by: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 9), dpi=160)
    if color_by == "modality":
        labels = modalities
        unique = ("vision", "action", "multimodal")
        cmap = {"vision": "#2563eb", "action": "#dc2626", "multimodal": "#16a34a"}
    else:
        labels = tasks
        unique = sorted(set(tasks.tolist()))
        colors = plt.cm.turbo(np.linspace(0.0, 1.0, len(unique)))
        cmap = dict(zip(unique, colors))
    for label in unique:
        mask = labels == label
        ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=10, alpha=0.65, label=str(label), c=[cmap[label]])
    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    if color_by == "modality":
        ax.legend(frameon=True)
    else:
        ax.legend(frameon=True, fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_interactive_html(
    path: Path,
    coordinates: np.ndarray,
    tasks: np.ndarray,
    episodes: np.ndarray,
    frames: np.ndarray,
    pair_ids: np.ndarray,
    modalities: np.ndarray,
    pair_subset: int,
) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    modality_colors = {"vision": "#2563eb", "action": "#dc2626", "multimodal": "#16a34a"}
    task_values = sorted(set(tasks.tolist()))
    task_colors = dict(zip(task_values, [f"hsl({int(360 * i / max(1, len(task_values)))},70%,45%)" for i in range(len(task_values))]))
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.1, subplot_titles=("UMAP — color by modality", "UMAP — color by task"))
    modality_trace_indices: list[int] = []
    task_trace_indices: dict[str, int] = {}
    custom_template = "modality=%{customdata[0]}<br>task=%{customdata[1]}<br>episode=%{customdata[2]}<br>frame=%{customdata[3]}<br>pair_id=%{customdata[4]}<extra></extra>"
    for modality in ("vision", "action", "multimodal"):
        mask = modalities == modality
        customdata = np.column_stack((modalities[mask], tasks[mask], episodes[mask], frames[mask], pair_ids[mask]))
        fig.add_trace(go.Scattergl(
            x=coordinates[mask, 0], y=coordinates[mask, 1], mode="markers", name=modality,
            marker={"size": 6, "color": modality_colors[modality]}, customdata=customdata,
            hovertemplate=custom_template, legendgroup=modality,
        ), row=1, col=1)
        modality_trace_indices.append(len(fig.data) - 1)

    # Restrained pair connections: only the first deterministic subset.
    unique_pairs = list(dict.fromkeys(pair_ids[modalities == "vision"].tolist()))[:pair_subset]
    pair_lookup = {pair: i for i, pair in enumerate(pair_ids.tolist())}
    for pair_id in unique_pairs:
        indices = [pair_lookup[pair_id + suffix] if pair_id + suffix in pair_lookup else None for suffix in ("", "", "")]
        base = pair_lookup[pair_id]
        coords = np.asarray([coordinates[base], coordinates[base + 1], coordinates[base + 2]])
        fig.add_trace(go.Scattergl(
            x=coords[:, 0], y=coords[:, 1], mode="lines", line={"color": "rgba(80,80,80,0.20)", "width": 0.7},
            name="paired V/A/M", showlegend=False, hoverinfo="skip",
        ), row=1, col=1)

    for task in task_values:
        mask = tasks == task
        customdata = np.column_stack((modalities[mask], tasks[mask], episodes[mask], frames[mask], pair_ids[mask]))
        fig.add_trace(go.Scattergl(
            x=coordinates[mask, 0], y=coordinates[mask, 1], mode="markers", name=str(task),
            marker={"size": 6, "color": task_colors[task]}, customdata=customdata,
            hovertemplate=custom_template, legendgroup=str(task),
        ), row=2, col=1)
        task_trace_indices[task] = len(fig.data) - 1

    all_visible = [True] * len(fig.data)
    buttons = [{"label": "All tasks", "method": "update", "args": [{"visible": all_visible}]}]
    for task in task_values:
        visible = [True] * len(fig.data)
        for other_task, index in task_trace_indices.items():
            visible[index] = other_task == task
        buttons.append({"label": str(task), "method": "update", "args": [{"visible": visible}]})
    fig.update_layout(
        title="Canonical Original UniT T4 Representation — L2 VQ Input",
        height=1300,
        hovermode="closest",
        legend={"groupclick": "togglegroup"},
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 1.02, "y": 0.98, "showactive": True}],
        margin={"l": 60, "r": 260, "t": 90, "b": 50},
    )
    fig.update_xaxes(title_text="UMAP 1", row=1, col=1)
    fig.update_yaxes(title_text="UMAP 2", row=1, col=1)
    fig.update_xaxes(title_text="UMAP 1", row=2, col=1)
    fig.update_yaxes(title_text="UMAP 2", row=2, col=1)
    fig.write_html(path, include_plotlyjs=True, full_html=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    feature_path = args.output_dir / "features" / "unit_representation_features.npz"
    if not feature_path.exists():
        raise FileNotFoundError("Run extract_unit_representation_baseline.py first")
    with np.load(feature_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    tasks = np.repeat(arrays["task"], 3)
    episodes = np.repeat(arrays["episode"], 3)
    frames = np.repeat(arrays["frame"], 3)
    pair_ids = np.repeat(arrays["pair_id"], 3)
    modality_names = np.asarray(["vision", "action", "multimodal"])
    modalities = np.tile(modality_names, len(arrays["task"]))
    visualization_dir = args.output_dir / "visualization"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {"seed": 42, "joint_fit": True, "layers": {}}

    for layer, label in (("l2", "pre_vq"), ("l3", "post_vq")):
        pooled = mean_query_pool(arrays[layer])
        combined = pooled.reshape(-1, pooled.shape[-1])
        pca_coordinates = __import__("sklearn.decomposition", fromlist=["PCA"]).PCA(n_components=2, random_state=42).fit_transform(combined)
        tsne_coordinates = fit_tsne(combined, 42)
        umap_coordinates = fit_umap(combined, 42)
        save_static_plot(visualization_dir / f"pca_{label}.png", pca_coordinates, tasks, modalities, f"PCA — {label}", "modality")
        save_static_plot(visualization_dir / f"tsne_{label}.png", tsne_coordinates, tasks, modalities, f"t-SNE — {label}", "modality")
        save_static_plot(visualization_dir / f"umap_{label}.png", umap_coordinates, tasks, modalities, f"UMAP — {label}", "modality")
        metadata["layers"][label] = {
            "input_shape": list(combined.shape),
            "pca_explained_variance_ratio": __import__("sklearn.decomposition", fromlist=["PCA"]).PCA(n_components=2, random_state=42).fit(combined).explained_variance_ratio_.tolist(),
            "tsne": {"perplexity": min(30.0, max(2.0, (len(combined) - 1) / 3.0)), "random_state": 42},
            "umap": {"n_neighbors": 15, "min_dist": 0.1, "metric": "cosine", "random_state": 42},
        }
        if layer == "l2":
            make_interactive_html(
                visualization_dir / "unit_representation_umap.html",
                umap_coordinates,
                tasks,
                episodes,
                frames,
                pair_ids,
                modalities,
                pair_subset=100,
            )

    summary = {
        "status": "PASS",
        "files": sorted(path.name for path in visualization_dir.iterdir()),
        "metadata": metadata,
        "interactive_html": str(visualization_dir / "unit_representation_umap.html"),
    }
    (args.output_dir / "visualization_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
