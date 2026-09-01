#!/usr/bin/env python3
"""Create decision-focused S4.2-DS visualizations from measured artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_2ds"
PLOT_ROOT = ARTIFACT_ROOT / "plots"


def load(name: str) -> dict[str, Any]:
    return json.loads(ARTIFACT_ROOT.joinpath(name).read_text(encoding="utf-8"))


def save(fig: plt.Figure, name: str) -> str:
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    path = PLOT_ROOT / name
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def bars(title: str, labels: list[str], series: dict[str, list[float]], ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    width = 0.8 / len(series)
    for index, (name, values) in enumerate(series.items()):
        ax.bar(x + (index - (len(series) - 1) / 2) * width, values, width, label=name)
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return fig


def main() -> None:
    utility = load("contact_representation_utility.json")
    action = load("action_pilot.json")
    bridge = {name: load(f"{name.lower()}_bridge_pilot.json") for name in ("C2", "C3")}
    comparison = load("candidate_comparison.json")
    canonical = load("canonical_contact_representation.json")
    confirmation = load("formal_validation_confirmation.json")
    files = []

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axis("off")
    boxes = [
        (0.02, 0.60, "C2 pair input\n[h, Δh]", "#D9EAF7"),
        (0.27, 0.60, "DeltaMLPEncoder\n498k params", "#D9EAF7"),
        (0.53, 0.60, "explicit zᶜ²\n8×32", "#B7E4C7"),
        (0.78, 0.60, "shared decoder\nfuture h", "#FDE2A7"),
        (0.02, 0.18, "C3 pair input\n[h, h′, Δh]", "#EADCF8"),
        (0.27, 0.18, "structured encoder\n725k params", "#EADCF8"),
        (0.53, 0.18, "explicit zᶜ³\n8×32", "#B7E4C7"),
        (0.78, 0.18, "shared decoder\nfuture h", "#FDE2A7"),
    ]
    for x, y, text, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.19, 0.22, color=color, ec="black"))
        ax.text(x + 0.095, y + 0.11, text, ha="center", va="center")
    for y in (0.71, 0.29):
        for x in (0.21, 0.47, 0.72):
            ax.annotate("", xy=(x + 0.05, y), xytext=(x, y), arrowprops={"arrowstyle": "->"})
    ax.set_title("C2/C3 frozen architecture and latent contract")
    files.append(save(fig, "01_c2_c3_architecture_latent_contract.png"))

    files.append(
        save(
            bars(
                "Contact semantic utility on DS-DEV",
                ["macro-F1", "balanced accuracy"],
                {
                    name: [
                        utility["candidates"][name]["probes"]["contact_transition"][key]
                        for key in ("macro_f1", "balanced_accuracy")
                    ]
                    for name in ("C2", "C3")
                },
                "score",
            ),
            "02_contact_semantic_comparison.png",
        )
    )
    files.append(
        save(
            bars(
                "Force/trend information on DS-DEV",
                ["trend F1", "future-force R²"],
                {
                    name: [
                        utility["candidates"][name]["probes"]["force_trend"]["macro_f1"],
                        utility["candidates"][name]["probes"]["future_force_magnitude"]["r2"],
                    ]
                    for name in ("C2", "C3")
                },
                "score",
            ),
            "03_contact_force_trend_comparison.png",
        )
    )
    controls = ("reversed", "different_episode", "mismatched_future")
    files.append(
        save(
            bars(
                "Temporal specificity: decoder MSE over correct",
                list(controls),
                {
                    name: [
                        utility["candidates"][name]["temporal_and_controls"]["decoder_recovery"][control]["over_full_ratio"]
                        for control in controls
                    ]
                    for name in ("C2", "C3")
                },
                "ratio",
            ),
            "04_temporal_specificity.png",
        )
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in ("C2", "C3"):
        spectrum = utility["candidates"][name]["geometry"]["spectrum_fraction"]
        ax.plot(np.arange(1, len(spectrum) + 1), np.cumsum(spectrum), label=name)
    ax.axhline(0.95, color="black", ls="--", lw=1)
    ax.set(xlabel="principal component", ylabel="cumulative variance", title="Representation rank spectra")
    ax.legend()
    files.append(save(fig, "05_representation_rank_spectra.png"))
    necessity = ("full", "zero", "shuffled", "different_episode", "gaussian_matched_variance")
    files.append(
        save(
            bars(
                "Transition-code information necessity",
                list(necessity),
                {
                    name: [
                        utility["candidates"][name]["temporal_and_controls"]["decoder_recovery"][control]["future_mse"]
                        for control in necessity
                    ]
                    for name in ("C2", "C3")
                },
                "future-state MSE",
            ),
            "06_transition_control_necessity.png",
        )
    )
    dynamic = action["metrics"]["dynamic"]
    files.append(
        save(
            bars(
                "Action pilot raw temporal controls",
                ["reversed", "shuffled", "different episode"],
                {
                    "MSE/correct": [
                        dynamic["reversed_over_correct"],
                        dynamic["shuffled_over_correct"],
                        dynamic["different_episode_over_correct"],
                    ]
                },
                "ratio",
            ),
            "07_action_pilot_temporal_controls.png",
        )
    )
    for number, pair, filename in (
        (8, "V-C", "08_c2_c3_vc_retrieval.png"),
        (9, "A-C", "09_c2_c3_ac_retrieval.png"),
    ):
        _ = number
        files.append(
            save(
                bars(
                    f"{pair} retrieval on DS-DEV",
                    ["forward R@10", "reverse R@10"],
                    {
                        name: [
                            bridge[name]["alignment"][pair]["retrieval"][direction]["recall_at_10"]
                            for direction in ("forward", "reverse")
                        ]
                        for name in ("C2", "C3")
                    },
                    "recall",
                ),
                filename,
            )
        )
    files.append(
        save(
            bars(
                "Contact semantic retention after bridge",
                ["contact", "force"],
                {name: [bridge[name]["contact_semantics"]["retention"][key] for key in ("contact", "force")] for name in ("C2", "C3")},
                "retention ratio",
            ),
            "10_contact_semantic_retention.png",
        )
    )
    files.append(
        save(
            bars(
                "Representation utility summary",
                ["contact F1", "force F1", "mean contact R@10"],
                {
                    name: [
                        utility["candidates"][name]["probes"]["contact_transition"]["macro_f1"],
                        utility["candidates"][name]["probes"]["force_trend"]["macro_f1"],
                        comparison["contact_bridge_mean_r10"][name],
                    ]
                    for name in ("C2", "C3")
                },
                "score",
            ),
            "11_representation_utility_summary.png",
        )
    )
    matrix = np.asarray(
        [
            [1, 1],
            [1, 1],
            [0, int(comparison["contact_semantic_advantage_C3"])],
            [0, int(comparison["bridge_advantage_C3"])],
            [0, int(comparison["frozen_future_reconstruction_advantage_C3"])],
            [0, int(canonical["candidate"] == "C3")],
        ]
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGn")
    ax.set_xticks([0, 1], ["C2", "C3"])
    ax.set_yticks(
        range(6),
        ["Contact gates", "Bridge gates", "Semantic advantage", "Bridge advantage", "Recovery advantage", "Selected"],
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, "YES" if matrix[row, column] else "NO", ha="center", va="center")
    ax.set_title("Frozen canonical-selection matrix")
    fig.colorbar(image, ax=ax, ticks=[0, 1])
    files.append(save(fig, "12_canonical_selection_matrix.png"))
    validation_alignment = confirmation["bridge"]["alignment"]
    files.append(
        save(
            bars(
                "Follow-up validation confirmation",
                ["contact F1", "force F1", "V-C margin", "A-C margin"],
                {
                    "selected C3": [
                        confirmation["contact"]["native_contact_transition"]["macro_f1"],
                        confirmation["contact"]["native_force_trend"]["macro_f1"],
                        validation_alignment["V-C"]["paired_minus_shuffled_margin"],
                        validation_alignment["A-C"]["paired_minus_shuffled_margin"],
                    ]
                },
                "measured value",
            ),
            "13_formal_validation_confirmation.png",
        )
    )
    summary = {
        "schema": "tactile3d-unit.s4-2ds-visualization-summary.v1",
        "plots": files,
        "count": len(files),
        "fabricated": False,
        "formal_test_loaded": False,
    }
    atomic = ARTIFACT_ROOT / "visualization_summary.json"
    atomic.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
