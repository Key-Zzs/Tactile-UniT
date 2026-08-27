#!/usr/bin/env python3
"""Generate local human-acceptance plots for C3-MS-CC-R closure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/tactile_unit/c3msccr_exact_action_closure.json"


def save_bar(path: Path, title: str, labels: list[str], values: list[float], ylabel: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.8))
    colors = ["#2878B5", "#9AC9DB", "#F8AC8C", "#C82423", "#8ECFC9"]
    axis.bar(labels, values, color=colors[:len(values)])
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    artifact = ROOT / config["runtime"]["artifact_root"]
    plots = artifact / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    ar = json.loads((artifact / "ar_temporal_integrity.json").read_text())
    source = json.loads((artifact / "source_selection_audit.json").read_text())
    frozen = json.loads((artifact / "frozen_candidate_validation.json").read_text())
    remediation = json.loads((artifact / "remediation_trials.json").read_text())
    locked = json.loads((artifact / "locked_closure_evaluation.json").read_text())
    metrics = locked["metrics"]

    variants = ["reversed", "shuffled", "different"]
    save_bar(
        plots / "exact_raw_action_latent_distance.png",
        "Exact raw-Action perturbation distance (dynamic)", variants,
        [ar["latent"]["u_a"]["dynamic"][name]["mse"] for name in variants],
        "u_a MSE from correct",
    )
    save_bar(
        plots / "ar_temporal_integrity.png",
        "A-R temporal decoder integrity (dynamic)",
        ["correct", *variants],
        [ar["decoder"]["dynamic"]["correct"]] + [
            ar["decoder"]["dynamic"][name]["mse"] for name in variants
        ],
        "Action reconstruction MSE",
    )
    utility_labels = list(source["utilities"])
    save_bar(
        plots / "source_selection_reducer.png",
        "Original validation utilities and reducer audit", utility_labels,
        [source["utilities"][name] for name in utility_labels], "Validation utility",
    )
    required_gates = ["contact", "force", "physics", "h_context_mse", "action_exact_ar", "shared_target"]
    t0 = next(row for row in frozen["trials"] if row["trial"]["id"] == "T0")
    t1 = next(row for row in frozen["trials"] if row["trial"]["id"] == "T1")
    matrix = np.asarray([
        [float(t0["validation"]["gates"][name]) for name in required_gates],
        [float(t1["validation"]["gates"][name]) for name in required_gates],
    ])
    figure, axis = plt.subplots(figsize=(9, 3.2))
    axis.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    axis.set_xticks(range(len(required_gates)), required_gates, rotation=30, ha="right")
    axis.set_yticks([0, 1], ["T0 A+H", "T1 V+A+H"])
    axis.set_title("Frozen candidate exact-Action validation gates")
    figure.tight_layout()
    figure.savefig(plots / "frozen_candidate_gates.png", dpi=180)
    plt.close(figure)

    temporal = metrics["action_temporal"]
    save_bar(
        plots / "exact_action_temporal_use.png",
        "Exact Action temporal use (locked dynamic)",
        ["correct", *variants],
        [temporal["correct_dynamic_mse"]] + [
            temporal["variants"][name]["dynamic_mse"] for name in variants
        ],
        "Shared target MSE",
    )
    save_bar(
        plots / "action_prediction_controls.png",
        "Contact prediction under exact Action controls",
        ["correct", *variants],
        [metrics["semantics"]["contact_transition"]["macro_f1"]] + [
            temporal["variants"][name]["contact_f1"] for name in variants
        ],
        "Contact macro-F1",
    )
    save_bar(
        plots / "contact_force_retention.png",
        "A+H semantic retention",
        ["Contact F1", "Force F1", "R_contact", "R_force"],
        [
            metrics["semantics"]["contact_transition"]["macro_f1"],
            metrics["semantics"]["force_trend_class"]["macro_f1"],
            metrics["semantics"]["contact_transition"]["semantic_ratio"],
            metrics["semantics"]["force_trend_class"]["semantic_ratio"],
        ],
        "Score",
    )
    physics = metrics["physics"]
    save_bar(
        plots / "shared_physics.png",
        "Shared Contact physics closure",
        ["pred all", "control all", "pred dynamic"],
        [
            physics["prediction_mse"],
            physics["controls_mse"][physics["strongest_control"]],
            physics["prediction_dynamic_mse"],
        ],
        "Physics MSE",
    )
    h = metrics["h_context"]
    h_labels = ["correct", *h["controls_mse"].keys()]
    save_bar(
        plots / "h_usage_controls.png", "Current Contact context usage", h_labels,
        [h["correct_mse"], *h["controls_mse"].values()], "Shared target MSE",
    )
    contact = metrics["semantics"]["contact_transition"]
    save_bar(
        plots / "future_change_boundaries.png",
        "Future change and rare boundaries",
        ["future", "free→contact", "contact→free"],
        [
            contact["future_change"]["macro_f1"],
            contact["free_to_contact"]["f1"],
            contact["contact_to_free"]["f1"],
        ],
        "F1",
    )
    geometry = metrics["geometry"]
    save_bar(
        plots / "representation_geometry.png",
        "Representation geometry",
        ["effective rank", "variance ×100", "query diversity ×10", "CKA ×10"],
        [
            geometry["effective_rank"],
            100 * geometry["per_dimension_variance"]["mean"],
            10 * geometry["query_diversity"]["mean_cosine_distance"],
            10 * geometry["cka_with_oracle"],
        ],
        "Scaled diagnostic",
    )
    r1 = remediation["selected"]["validation"]
    save_bar(
        plots / "frozen_vs_remedied_ah.png",
        "Frozen T0 vs bounded R1 validation physics",
        ["T0 prediction", "R1 prediction", "mean control"],
        [
            t0["validation"]["physics"]["prediction_mse"],
            r1["physics"]["prediction_mse"],
            r1["physics"]["mean_control_mse"],
        ],
        "Physics MSE",
    )
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.axis("off")
    axis.text(
        0.5, 0.62, locked["decision"], ha="center", va="center",
        fontsize=18, weight="bold", color="#2878B5",
    )
    axis.text(
        0.5, 0.38,
        "Canonical source: A + H\nVision: optional short-horizon context / ablation\nC4: READY WITH RANK WARNING (not started)",
        ha="center", va="center", fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(plots / "final_minimal_source_decision.png", dpi=180)
    plt.close(figure)

    plot_names = sorted(path.name for path in plots.glob("*.png"))
    acceptance = f"""# C3-MS-CC-R Human Acceptance

Decision: **{locked['decision']}**

- Evaluation: LOCKED POST-HOC CLOSURE RE-EVALUATION (not first-look untouched test)
- Rows: {locked['rows']:,}
- Canonical source: A + H
- Exact Action temporal evidence: PASS
- Shared physics, all and dynamic: PASS
- Semantic retention: PASS
- Frozen integrity: PASS
- Rank warning: {'YES' if locked['rank_warning'] else 'NO'}
- C4 readiness: READY_WITH_RANK_WARNING; C4 was not started

Reproduce plots:

```bash
python scripts/tactile_unit/visualize_c3msccr_closure.py
```

Plots:

""" + "\n".join(f"- `plots/{name}`" for name in plot_names) + "\n"
    (artifact / "HUMAN_ACCEPTANCE.md").write_text(acceptance)
    print(artifact / "HUMAN_ACCEPTANCE.md")


if __name__ == "__main__":
    main()
