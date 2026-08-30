#!/usr/bin/env python3
"""Generate the local C2-R human-acceptance packet and plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / ".local/artifacts/tactile_unit/vac_c2r"
EXPERIMENT = ROOT / ".local/experiments/tactile_unit/vac_c2r"
PLOTS = ARTIFACT / "plots"


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(PLOTS / name, dpi=160)
    plt.close()


def main() -> None:
    result = json.loads((ARTIFACT / "locked_test_evaluation.json").read_text())
    training = json.loads((EXPERIMENT / "training_summary.json").read_text())
    PLOTS.mkdir(parents=True, exist_ok=True)
    accepted = json.loads((ROOT / ".local/artifacts/tactile_unit/vac_c2/evaluation.json").read_text())

    labels = ["R_contact", "R_force"]
    c2 = [
        accepted["contact"]["probes"]["contact_transition"]["retention"],
        accepted["contact"]["probes"]["force_trend"]["retention"],
    ]
    c2r = [result["probes"]["contact_transition"]["retention"], result["probes"]["force_trend"]["retention"]]
    x = np.arange(2)
    plt.bar(x - 0.18, c2, 0.36, label="C2")
    plt.bar(x + 0.18, c2r, 0.36, label="C2-R")
    plt.axhline(0.9, color="black", linestyle="--", label="hard gate")
    plt.xticks(x, labels); plt.ylabel("retention"); plt.legend()
    save("semantic_retention.png")

    names = ["native", "C2", "C2-R"]
    future = [result["physics"][key]["future_mse"] for key in ("native", "original_c2", "c2r")]
    dynamic = [result["physics"][key]["dynamic_mse"] for key in ("native", "original_c2", "c2r")]
    x = np.arange(3)
    plt.bar(x - 0.18, future, 0.36, label="future")
    plt.bar(x + 0.18, dynamic, 0.36, label="dynamic")
    plt.yscale("log"); plt.xticks(x, names); plt.ylabel("MSE (log scale)"); plt.legend()
    save("contact_physics.png")

    pairs = ["V-C", "A-C"]
    old = [accepted["alignment"][name]["all"]["paired_minus_shuffled_margin"] for name in pairs]
    new = [result["alignment"][name]["all"]["paired_minus_shuffled_margin"] for name in pairs]
    x = np.arange(2)
    plt.bar(x - 0.18, old, 0.36, label="C2")
    plt.bar(x + 0.18, new, 0.36, label="C2-R")
    plt.axhline(0, color="black"); plt.xticks(x, pairs); plt.ylabel("paired − shuffled margin"); plt.legend()
    save("alignment_nonregression.png")

    directions = ["V→A", "A→V", "V→C", "C→V", "A→C", "C→A"]
    values = []
    va = result["accepted_v_a_alignment"]["all"]["retrieval"]
    values.extend([va["forward"]["recall_at_10"] / va["forward"]["chance"]["recall_at_10"], va["reverse"]["recall_at_10"] / va["reverse"]["chance"]["recall_at_10"]])
    for pair in ("V-C", "A-C"):
        retrieval = result["alignment"][pair]["all"]["retrieval"]
        values.extend([retrieval["forward"]["recall_at_10"] / retrieval["forward"]["chance"]["recall_at_10"], retrieval["reverse"]["recall_at_10"] / retrieval["reverse"]["chance"]["recall_at_10"]])
    plt.bar(np.arange(6), values); plt.axhline(1.5, color="black", linestyle="--")
    plt.xticks(np.arange(6), directions, rotation=25); plt.ylabel("R@10 / chance")
    save("six_direction_retrieval.png")

    per_class = result["probes"]["contact_transition"]["shared"]["per_class"]
    boundary_labels = ["free→contact", "contact→free"]
    for metric, offset in (("precision", -0.25), ("recall", 0), ("f1", 0.25)):
        plt.bar(np.arange(2) + offset, [per_class[str(label)][metric] for label in (1, 2)], 0.25, label=metric)
    plt.xticks(np.arange(2), boundary_labels); plt.ylim(0, 1); plt.legend()
    save("rare_boundary_retention.png")

    subsets = ["dynamic", "rare_boundary", "free_to_contact", "contact_to_free"]
    for offset, pair in ((-0.18, "V-C"), (0.18, "A-C")):
        plt.bar(np.arange(4) + offset, [result["alignment"][pair][name]["paired_minus_shuffled_margin"] for name in subsets], 0.36, label=pair)
    plt.axhline(0, color="black"); plt.xticks(np.arange(4), subsets, rotation=20); plt.ylabel("margin"); plt.legend()
    save("dynamic_boundary_alignment.png")

    identity = result["frozen_output_identity"]
    plt.bar(["Vision", "Action", "native identities", "state boundary"], [float(identity["vision"]), float(identity["action"]), float(result["native_identities"]["pass"]), float(result["state_boundary"]["pass"])])
    plt.ylim(0, 1.1); plt.ylabel("pass = 1")
    save("frozen_identity.png")

    geometry = result["contact_geometry"]
    plt.bar(["effective rank", "query diversity", "mean variance", "pairwise distance"], [geometry["effective_rank"], geometry["query_diversity"]["mean_cosine_distance"], geometry["per_dimension_variance"]["mean"], geometry["pairwise_distance"]["mean_cosine_distance"]])
    plt.xticks(rotation=20); plt.ylabel("diagnostic value")
    save("contact_geometry.png")

    for row in training["trials"]:
        validation = row["best"]["validation"]
        plt.scatter(validation["contact_transition"]["retention"], -validation["physics"]["dynamic_mse"], label=f"T{row['trial_id']}")
    plt.xlabel("validation Contact retention"); plt.ylabel("− validation dynamic MSE"); plt.legend(ncol=2)
    save("bounded_trial_pareto.png")

    plt.axis("off")
    plt.text(0.5, 0.62, result["decision"], ha="center", va="center", fontsize=17, weight="bold")
    plt.text(0.5, 0.40, "STOP AFTER C2-R · M3 NOT ESTABLISHED", ha="center", va="center", fontsize=11)
    save("final_decision.png")

    contact = result["probes"]["contact_transition"]
    force = result["probes"]["force_trend"]
    acceptance = "\n".join([
        "# C2-R Contact Preservation Human Acceptance", "",
        f"Decision: **{result['decision']}**", "",
        "Evaluation: LOCKED RE-EVALUATION AFTER POST-C2 REMEDIATION.",
        "This is not an untouched first-look test.", "",
        f"Contact retention: {contact['retention']:.6f} (gate ≥ 0.90).",
        f"Force retention: {force['retention']:.6f} (gate ≥ 0.90).",
        f"Future MSE: {result['physics']['c2r']['future_mse']:.6g} (C2 {result['physics']['original_c2']['future_mse']:.6g}).",
        f"Dynamic MSE: {result['physics']['c2r']['dynamic_mse']:.6g} (C2 {result['physics']['original_c2']['dynamic_mse']:.6g}).",
        f"V-C alignment: {'PASS' if result['pair_gates']['V-C']['pass'] else 'FAIL'}.",
        f"A-C alignment: {'PASS' if result['pair_gates']['A-C']['pass'] else 'FAIL'}.",
        f"Vision/Action frozen identity: {'PASS' if all(result['frozen_output_identity'].values()) else 'FAIL'}.",
        f"Contact collapse: {'NO' if result['gates']['noncollapse'] else 'YES'}.", "",
        "Commands:", "",
        "```bash",
        "python scripts/tactile_unit/audit_c2r_metric_consistency.py --device cpu",
        "python scripts/tactile_unit/train_c2r_contact_preservation.py --device cuda:0",
        "python scripts/tactile_unit/evaluate_c2r_contact_preservation.py --device cuda:0",
        "python scripts/tactile_unit/visualize_c2r_contact_preservation.py",
        "```", "",
        "C3/C4/C5/C6: NOT STARTED. M3: NOT ESTABLISHED.",
        "STOP AFTER C2-R.", "",
    ])
    (ARTIFACT / "HUMAN_ACCEPTANCE.md").write_text(acceptance)
    summary = {
        "schema": "tactile3d-unit.vac-c2r-visualization.v1",
        "decision": result["decision"],
        "plots": sorted(path.name for path in PLOTS.glob("*.png")),
        "human_acceptance": "HUMAN_ACCEPTANCE.md",
    }
    (ARTIFACT / "visualization_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
