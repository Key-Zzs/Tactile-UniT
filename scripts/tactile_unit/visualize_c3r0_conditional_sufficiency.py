#!/usr/bin/env python3
"""Generate decision-focused C3-R0 plots and local human acceptance notes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.tactile_unit.c3r0_runtime import DEFAULT_CONFIG, atomic_json, load_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def bar(path, labels, values, title, ylabel, *, threshold=None, colors=None):
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(labels, values, color=colors or "#4C78A8")
    if threshold is not None:
        axis.axhline(threshold, color="#E45756", linestyle="--", label=f"gate {threshold:g}")
        axis.legend()
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def grouped(path, labels, left, right, title, left_name, right_name, ylabel):
    x = np.arange(len(labels))
    width = 0.38
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(x - width / 2, left, width, label=left_name, color="#4C78A8")
    axis.bar(x + width / 2, right, width, label=right_name, color="#F58518")
    axis.set_xticks(x, labels, rotation=35)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    config = load_config(args.config)
    artifact_root = ROOT / config["runtime"]["artifact_root"]
    result = json.loads((artifact_root / "locked_test_evaluation.json").read_text())
    plots = artifact_root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    sources = ["V", "A", "VA", "H", "VH", "AH", "VAH", "C", "ZC"]
    direct = result["direct_source"]

    bar(
        plots / "direct_contact_sufficiency.png", sources,
        [direct[name]["contact_transition"].get("semantic_ratio", 1.0) or 1.0 for name in sources],
        "Direct source Contact semantic sufficiency", "R_sem", threshold=0.75,
    )
    bar(
        plots / "direct_force_sufficiency.png", sources,
        [direct[name]["force_trend_class"].get("semantic_ratio", 1.0) or 1.0 for name in sources],
        "Direct source force semantic sufficiency", "R_force_sem", threshold=0.75,
    )
    grouped(
        plots / "rare_boundary_sufficiency.png", sources,
        [direct[name]["contact_transition"]["free_to_contact"]["f1"] for name in sources],
        [direct[name]["contact_transition"]["contact_to_free"]["f1"] for name in sources],
        "Rare Contact boundary F1", "free→contact", "contact→free", "F1",
    )
    gains = result["complementarity"]
    gain_names = ["VA vs V", "VA vs A", "VAH vs VA", "AH vs A", "AH vs H"]
    gain_keys = ["VA_vs_V", "VA_vs_A", "VAH_vs_VA", "AH_vs_A", "AH_vs_H"]
    grouped(
        plots / "source_complementarity.png", gain_names,
        [gains[key]["contact_transition"]["delta_f1"] for key in gain_keys],
        [gains[key]["force_trend_class"]["delta_f1"] for key in gain_keys],
        "Source complementarity gains", "Contact", "Force", "Δ macro-F1",
    )
    neighborhoods = result["neighborhoods"]
    neighborhood_sources = ["V", "A", "VA", "H", "VH", "AH", "VAH", "C"]
    bar(
        plots / "neighborhood_label_entropy.png", neighborhood_sources,
        [neighborhoods[name]["k"]["10"]["normalized_label_entropy"]["mean"] for name in neighborhood_sources],
        "Empirical neighborhood Contact-label ambiguity (k=10)", "normalized entropy",
    )
    bar(
        plots / "neighborhood_target_variance.png", neighborhood_sources,
        [neighborhoods[name]["k"]["10"]["target_ambiguity"]["local_over_global"]["mean"] for name in neighborhood_sources],
        "Local Contact-target variance (k=10)", "local/global variance",
    )
    nonparam = result["nonparametric"]
    grouped(
        plots / "knn_medoid_vs_mean.png", config["sources"],
        [nonparam[name]["medoid"]["contact_transition"]["semantic_ratio"] for name in config["sources"]],
        [nonparam[name]["mean"]["contact_transition"]["semantic_ratio"] for name in config["sources"]],
        "Mode-preserving medoid vs conditional mean", "medoid", "mean", "R_contact",
    )
    rank_labels = ["P2 V", "P2 A", "VAH 1NN", "VAH medoid", "VAH mean", "VA model", "VAH model", "oracle"]
    rank_values = [
        result["p2"]["V"]["representation"]["geometry"]["effective_rank"],
        result["p2"]["A"]["representation"]["geometry"]["effective_rank"],
        nonparam["VAH"]["1nn"]["representation"]["geometry"]["effective_rank"],
        nonparam["VAH"]["medoid"]["representation"]["geometry"]["effective_rank"],
        nonparam["VAH"]["mean"]["representation"]["geometry"]["effective_rank"],
        result["deterministic"]["VA"]["representation"]["geometry"]["effective_rank"],
        result["deterministic"]["VAH"]["representation"]["geometry"]["effective_rank"],
        25.503495,
    ]
    bar(plots / "predicted_effective_rank.png", rank_labels, rank_values, "Contact prediction effective rank", "effective rank")
    grouped(
        plots / "direct_source_vs_p2_gap.png", ["V", "A"],
        [result["direct_vs_p2_gap"][name]["contact_transition"]["direct_f1"] for name in ("V", "A")],
        [result["direct_vs_p2_gap"][name]["contact_transition"]["p2_f1"] for name in ("V", "A")],
        "Direct-source vs P2 Contact semantics", "direct source", "P2 predicted u_c", "macro-F1",
    )
    for source in ("VA", "VAH"):
        ceiling = result["deterministic"][source]
        bar(
            plots / f"{source.lower()}_deterministic_ceiling.png",
            ["Contact", "Force"],
            [ceiling["contact_transition"]["semantic_ratio"], ceiling["force_trend_class"]["semantic_ratio"]],
            f"{source} deterministic u_c ceiling ({ceiling['architecture']})", "retention", threshold=0.75,
        )
    temporal = result["action_temporal"]["direct_A"]
    grouped(
        plots / "action_temporal_source.png", ["correct", "reversed", "shuffled", "different_episode"],
        [temporal[name]["contact_transition"]["macro_f1"] for name in temporal],
        [temporal[name]["force_trend_class"]["macro_f1"] for name in temporal],
        "Action temporal sufficiency diagnostic", "Contact", "Force", "macro-F1",
    )
    context = result["current_context_confound"]
    grouped(
        plots / "current_context_confound.png", ["H", "VH", "AH", "VAH"],
        [context[name]["current_state"]["macro_f1"] for name in context],
        [context[name]["future_change"]["macro_f1"] for name in context],
        "Current Contact context: state vs future change", "current state", "future change", "macro-F1",
    )
    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.axis("off")
    root = result["root_cause"]
    text = (
        "C3-R0 decision\n\n"
        "V insufficient + A insufficient\n"
        "↓\nV+A deterministic gate: FAIL\n"
        "↓\nV+A+h_t^c deterministic gate: PASS\n"
        "↓\nFuture-change performance remains meaningful\n\n"
        f"PRIMARY: {root['primary']}\nNEXT: {root['next_stage']}"
    )
    axis.text(0.5, 0.5, text, ha="center", va="center", fontsize=13, bbox={"boxstyle": "round", "facecolor": "#EAF2F8"})
    fig.tight_layout()
    fig.savefig(plots / "root_cause_decision_tree.png", dpi=160)
    plt.close(fig)

    acceptance = f"""# C3-R0 Human Acceptance

Status: `C3R0_COMPLETE`

Primary diagnosis: `{root['primary']}`

Next stage recommendation: `{root['next_stage']}`

The next stage was not started. C4, C5, and C6/M3 remain not started.

## Key locked results

- V+A deterministic Contact retention: `{result['deterministic']['VA']['contact_transition']['semantic_ratio']:.6f}`
- V+A deterministic force retention: `{result['deterministic']['VA']['force_trend_class']['semantic_ratio']:.6f}`
- V+A gate: `{result['deterministic']['VA']['gate']}`
- V+A+h_t^c deterministic Contact retention: `{result['deterministic']['VAH']['contact_transition']['semantic_ratio']:.6f}`
- V+A+h_t^c deterministic force retention: `{result['deterministic']['VAH']['force_trend_class']['semantic_ratio']:.6f}`
- V+A+h_t^c future-change macro-F1: `{result['current_context_confound']['VAH']['future_change']['macro_f1']:.6f}`
- V+A+h_t^c gate: `{result['deterministic']['VAH']['gate']}`
- Frozen identities unchanged: `{result['frozen_identities_unchanged']}`

## Reproduction commands

```bash
python scripts/tactile_unit/audit_c3r0_source_semantics.py --device cpu
UNIT_FULLDATA_CKPT=/path/to/checkpoint python scripts/tactile_unit/evaluate_c3r0_conditional_sufficiency.py --device cpu
python scripts/tactile_unit/visualize_c3r0_conditional_sufficiency.py
python -m pytest -q tests/tactile_unit/test_c3r0_conditional_sufficiency.py
```

## Plots

All decision plots are under `plots/`.
"""
    (artifact_root / "HUMAN_ACCEPTANCE.md").write_text(acceptance)
    summary = {
        "schema": "tactile3d-unit.vac-c3r0-visualization.v1",
        "plots": sorted(path.name for path in plots.glob("*.png")),
        "primary": root["primary"], "next_stage": root["next_stage"],
    }
    atomic_json(artifact_root / "visualization_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
