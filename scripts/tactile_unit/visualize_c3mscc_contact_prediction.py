#!/usr/bin/env python3
"""Generate decision-focused C3-MS-CC acceptance plots and summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.tactile_unit.c3mscc_runtime import atomic_json, load_config  # noqa: E402


def save_bar(path: Path, title: str, labels, values, *, gate=None, ylabel=""):
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b", "#e45756"][:len(values)])
    if gate is not None:
        ax.axhline(gate, color="black", linestyle="--", linewidth=1, label=f"gate {gate:g}")
        ax.legend()
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    config = load_config()
    root = ROOT / config["runtime"]["artifact_root"]
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    evaluation = json.loads((root / "locked_test_evaluation.json").read_text())
    training = json.loads((root / "training_summary.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    ah, vah = evaluation["sources"]["AH"], evaluation["sources"]["VAH"]
    sources = ["A+H", "V+A+H"]
    files = []

    utilities = [row["best"]["utility"] for row in training["trials"]]
    labels = [row["trial"]["id"] for row in training["trials"]]
    save_bar(plots / "candidate_utility.png", "Validation-only candidate utility", labels, utilities, ylabel="utility")
    files.append("candidate_utility.png")
    save_bar(plots / "contact_retention.png", "Contact transition retention", sources, [ah["semantics"]["contact_transition"]["semantic_ratio"], vah["semantics"]["contact_transition"]["semantic_ratio"]], gate=0.75)
    files.append("contact_retention.png")
    save_bar(plots / "force_retention.png", "Force-trend retention", sources, [ah["semantics"]["force_trend_class"]["semantic_ratio"], vah["semantics"]["force_trend_class"]["semantic_ratio"]], gate=0.75)
    files.append("force_retention.png")
    save_bar(plots / "future_change.png", "Future-change and boundary F1", ["AH change", "VAH change", "AH free→contact", "VAH free→contact"], [ah["semantics"]["contact_transition"]["future_change"]["macro_f1"], vah["semantics"]["contact_transition"]["future_change"]["macro_f1"], ah["semantics"]["contact_transition"]["free_to_contact"]["f1"], vah["semantics"]["contact_transition"]["free_to_contact"]["f1"]])
    files.append("future_change.png")
    save_bar(plots / "shared_physics.png", "Shared Contact physics MSE", sources, [ah["physics"]["prediction_mse"], vah["physics"]["prediction_mse"]], ylabel="MSE")
    files.append("shared_physics.png")
    selected = evaluation["sources"][selection["source"]]
    h_values = {"correct": selected["h_context"]["correct_mse"], **selected["h_context"]["controls_mse"]}
    save_bar(plots / "h_context_controls.png", "Correct H vs invalid-H controls", list(h_values), list(h_values.values()), ylabel="shared target MSE")
    files.append("h_context_controls.png")
    action_values = {"correct": selected["action_temporal"]["correct_dynamic_mse"], **{name: value["dynamic_mse"] for name, value in selected["action_temporal"]["variants"].items()}}
    save_bar(plots / "action_temporal_controls.png", "Action temporal controls (dynamic)", list(action_values), list(action_values.values()), ylabel="shared target MSE")
    files.append("action_temporal_controls.png")
    increments = evaluation["vision_incremental"]
    save_bar(plots / "vision_incremental.png", "Vision incremental semantic effect", ["Contact F1", "Force F1"], [increments["contact_transition"]["f1_gain"], increments["force_trend_class"]["f1_gain"]], ylabel="V+A+H minus A+H")
    files.append("vision_incremental.png")
    save_bar(plots / "effective_rank.png", "Predicted Contact effective rank", ["C3-DP", "C3-R0 ceiling", "A+H", "V+A+H", "oracle"], [4.677386, 7.000549, ah["geometry"]["effective_rank"], vah["geometry"]["effective_rank"], 25.503495], ylabel="effective rank")
    files.append("effective_rank.png")
    save_bar(plots / "cosine_margin.png", "Paired-vs-shuffled cosine margin", sources, [ah["shared_target"]["cosine_margin"], vah["shared_target"]["cosine_margin"]])
    files.append("cosine_margin.png")
    save_bar(plots / "retrieval.png", "Target retrieval R@10 vs chance", ["AH", "VAH", "chance"], [ah["shared_target"]["retrieval"]["recall_at_10"], vah["shared_target"]["retrieval"]["recall_at_10"], vah["shared_target"]["retrieval"]["chance"]["recall_at_10"]])
    files.append("retrieval.png")
    save_bar(plots / "dynamic_prediction.png", "Dynamic shared-target MSE", sources, [ah["shared_target"]["dynamic_mse"], vah["shared_target"]["dynamic_mse"]], ylabel="MSE")
    files.append("dynamic_prediction.png")
    fig, ax = plt.subplots(figsize=(6, 4))
    for row in training["trials"]:
        metrics = row["best"]["validation"]
        ax.scatter(metrics["physics"]["prediction_mse"], metrics["semantic"]["contact_transition"]["semantic_ratio"], s=60)
        ax.annotate(row["trial"]["id"], (metrics["physics"]["prediction_mse"], metrics["semantic"]["contact_transition"]["semantic_ratio"]))
    ax.set_xlabel("validation physics MSE")
    ax.set_ylabel("Contact retention")
    ax.set_title("Trial Pareto: semantics vs physics")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "trial_pareto.png", dpi=160)
    plt.close(fig)
    files.append("trial_pareto.png")
    fig, ax = plt.subplots(figsize=(9, 2.2))
    ax.axis("off")
    ax.text(0.5, 0.65, evaluation["decision"], ha="center", va="center", fontsize=16, weight="bold")
    ax.text(0.5, 0.3, f"Selected: {selection['source']} / {selection['trial']} · C4 NOT READY", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(plots / "final_decision.png", dpi=160)
    plt.close(fig)
    files.append("final_decision.png")

    human = f"""# C3-MS-CC Human Acceptance

Evaluation: locked benchmark re-evaluation (17,504 rows; not first-look untouched)

Selected validation-only model: `{selection['trial']}` / `{selection['source']}`

Final decision: **{evaluation['decision']}**

Vision classification: **{evaluation['vision_classification']}**

## Gate summary

| Gate | A+H | V+A+H |
|---|---:|---:|
| Contact retention ≥0.75 | {ah['semantics']['contact_transition']['semantic_ratio']:.6f} | {vah['semantics']['contact_transition']['semantic_ratio']:.6f} |
| Force retention ≥0.75 | {ah['semantics']['force_trend_class']['semantic_ratio']:.6f} | {vah['semantics']['force_trend_class']['semantic_ratio']:.6f} |
| Shared latent | {'PASS' if ah['shared_target']['gate'] else 'FAIL'} | {'PASS' if vah['shared_target']['gate'] else 'FAIL'} |
| Shared physics | {'PASS' if ah['physics']['gate'] else 'FAIL'} | {'PASS' if vah['physics']['gate'] else 'FAIL'} |
| H usage | {'PASS' if ah['h_context']['gate'] else 'FAIL'} | {'PASS' if vah['h_context']['gate'] else 'FAIL'} |
| Exact A-R temporal usage | FAIL | FAIL |
| No collapse | {'PASS' if ah['noncollapse'] else 'FAIL'} | {'PASS' if vah['noncollapse'] else 'FAIL'} |

The Action temporal gate fails closed because the external immutable Original UniT tokenizer needed to recompute raw reversed-action A-R latents is unavailable. A shared-token reversal surrogate is reported but is not accepted as equivalent evidence. Selected V+A+H also fails the shared-physics control gate. C4 must not start.

## Reproduction

```bash
python scripts/tactile_unit/audit_c3mscc_contract.py
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<authorized-gpu> \\
  python scripts/tactile_unit/train_c3mscc_contact_prediction.py --device cuda:0
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<authorized-gpu> \\
  python scripts/tactile_unit/evaluate_c3mscc_contact_prediction.py --device cuda:0
python scripts/tactile_unit/visualize_c3mscc_contact_prediction.py
```

## Plots

""" + "\n".join(f"- `plots/{name}`" for name in files) + "\n"
    (root / "HUMAN_ACCEPTANCE.md").write_text(human)
    atomic_json(root / "visualization_summary.json", {
        "decision": evaluation["decision"], "selected_source": selection["source"],
        "plots": files, "human_acceptance": "HUMAN_ACCEPTANCE.md",
    })
    print(root / "HUMAN_ACCEPTANCE.md")


if __name__ == "__main__":
    main()
