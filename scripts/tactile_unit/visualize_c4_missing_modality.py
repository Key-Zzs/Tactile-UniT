#!/usr/bin/env python3
"""Create decision-focused plots and the C4 human-acceptance record."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/tactile_unit/vac_c4"
PLOTS = ARTIFACTS / "plots"


def save(name, title, labels, values, ylabel="value", color="#4472c4"):
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar(np.arange(len(values)), values, color=color)
    axis.set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = PLOTS / name
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return name


def main():
    locked = json.loads((ARTIFACTS / "locked_test_evaluation.json").read_text())
    baseline = json.loads((ARTIFACTS / "availability_baseline.json").read_text())
    router = json.loads((ARTIFACTS / "router_contract.json").read_text())
    a, va, full = locked["fallbacks"]["A"], locked["fallbacks"]["VA"], locked["full"]
    plots = []
    plots.append(save("01_availability_mode_performance.png", "Availability mode shared error", ["FULL_AH", "FALLBACK_VA", "FALLBACK_A"], [full["shared_target"]["prediction_mse"], va["shared_target"]["prediction_mse"], a["shared_target"]["prediction_mse"]], "shared MSE"))
    plots.append(save("02_fallback_contact_retention.png", "Missing-H Contact retention", ["A", "V+A", "gate"], [a["semantics"]["contact_transition"]["semantic_ratio"], va["semantics"]["contact_transition"]["semantic_ratio"], 0.50], "R_contact"))
    plots.append(save("03_fallback_force_retention.png", "Missing-H force retention", ["A", "V+A", "gate"], [a["semantics"]["force_trend_class"]["semantic_ratio"], va["semantics"]["force_trend_class"]["semantic_ratio"], 0.65], "R_force"))
    contact = va["semantics"]["contact_transition"]
    plots.append(save("04_future_change_boundaries.png", "VA future change and boundaries", ["future change", "free→contact", "contact→free"], [contact["future_change"]["macro_f1"], contact["free_to_contact"]["f1"], contact["contact_to_free"]["f1"]], "F1"))
    controls = va["shared_target"]["controls_mse"]
    plots.append(save("05_fallback_shared_mse_controls.png", "VA shared prediction vs controls", ["VA"] + list(controls), [va["shared_target"]["prediction_mse"]] + list(controls.values()), "MSE"))
    physics = va["physics"]
    plots.append(save("06_fallback_dynamic_physics.png", "VA teacher-side physics", ["all", "dynamic"] + list(physics["controls_mse"]), [physics["prediction_mse"], physics["dynamic_mse"]] + list(physics["controls_mse"].values()), "MSE"))
    temporal = va["action_temporal"]
    plots.append(save("07_exact_action_temporal_use.png", "Exact Action temporal use", ["correct", "reversed", "shuffled", "different"], [temporal["correct_dynamic_mse"]] + [temporal["variants"][name]["dynamic_mse"] for name in ("reversed", "shuffled", "different")], "dynamic shared MSE"))
    vision = locked["vision_incremental"]
    plots.append(save("08_vision_missing_h_contribution.png", "Vision contribution when H is missing", ["Contact F1", "Force F1", "shared MSE gain"], [vision["contact_macro_f1"], vision["force_macro_f1"], vision["shared_mse_improvement"]], "VA minus A"))
    plots.append(save("09_full_vs_fallback_degradation.png", "Graceful degradation", ["FULL_AH", "FALLBACK_VA", "FALLBACK_A"], [full["semantics"]["contact_transition"]["semantic_ratio"], va["semantics"]["contact_transition"]["semantic_ratio"], a["semantics"]["contact_transition"]["semantic_ratio"]], "Contact retention"))
    unc = locked["uncertainty"]
    plots.append(save("10_uncertainty_error_association.png", "Uncertainty tracks actual shared error", list(unc), [unc[name]["spearman"] for name in unc], "Spearman ρ"))
    plots.append(save("11_uncertainty_calibration.png", "Calibrated NLL vs constant variance", [f"{name} learned" for name in unc] + [f"{name} constant" for name in unc], [unc[name]["nll"] for name in unc] + [unc[name]["constant_variance_nll"] for name in unc], "NLL"))
    plots.append(save("12_high_error_detection.png", "High-error detection", list(unc), [unc[name]["auroc"] for name in unc], "AUROC"))
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    for name, row in unc.items():
        risk = row["risk_coverage"]
        coverage = [1.0, 0.9, 0.8, 0.7, 0.5]
        axis.plot(coverage, [risk[str(value)] for value in coverage], marker="o", label=name)
    axis.set_title("Risk–coverage"); axis.set_xlabel("coverage"); axis.set_ylabel("shared MSE"); axis.legend(); axis.grid(alpha=0.25); fig.tight_layout(); fig.savefig(PLOTS / "13_risk_coverage.png", dpi=160); plt.close(fig); plots.append("13_risk_coverage.png")
    plots.append(save("14_full_fallback_uncertainty.png", "Availability-sensitive uncertainty", list(unc), [unc[name]["uncertainty_mean"] for name in unc], "mean calibrated variance"))
    boundary = locked["boundary_uncertainty"]["FALLBACK_VA"]
    plots.append(save("15_boundary_uncertainty.png", "VA static/dynamic/boundary uncertainty", list(boundary), list(boundary.values()), "mean calibrated variance"))
    plots.append(save("16_fallback_effective_rank.png", "Shared prediction effective rank", ["FULL_AH", "FALLBACK_VA", "FALLBACK_A", "oracle"], [full["geometry"]["effective_rank"], va["geometry"]["effective_rank"], a["geometry"]["effective_rank"], 25.503495], "effective rank"))
    fig, axis = plt.subplots(figsize=(8, 4.5)); axis.axis("off"); axis.text(0.02, 0.9, "Explicit availability router", fontsize=15, weight="bold"); y = 0.76
    for row in router["truth_table"]:
        axis.text(0.04, y, f"A={int(row['action_available'])} H={int(row['contact_context_available'])} V={int(row['vision_available'])}  →  {row['mode']}", family="monospace"); y -= 0.08
    fig.tight_layout(); fig.savefig(PLOTS / "17_router_schematic.png", dpi=160); plt.close(fig); plots.append("17_router_schematic.png")
    fig, axis = plt.subplots(figsize=(8, 4.5)); axis.axis("off"); axis.text(0.5, 0.62, locked["decision"], ha="center", fontsize=19, weight="bold", color="#2b7a2b"); axis.text(0.5, 0.42, f"C5 readiness: {locked['c5_readiness']}\nC5/C6 not started · M3 not established", ha="center", fontsize=13); fig.tight_layout(); fig.savefig(PLOTS / "18_final_decision.png", dpi=160); plt.close(fig); plots.append("18_final_decision.png")
    summary = {"schema": "tactile3d-unit.vac-c4-visualization.v1", "plots": plots, "decision": locked["decision"]}
    (ARTIFACTS / "visualization_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    acceptance = f"""# Full Track C — C4 Human Acceptance\n\nDecision: **{locked['decision']}**\n\n- Canonical full path: frozen A+H; Contact R={locked['full_nonregression']['contact_ratio']:.6f}, Force R={locked['full_nonregression']['force_ratio']:.6f}.\n- Missing-H fallback: V+A; Contact R={va['semantics']['contact_transition']['semantic_ratio']:.6f}, Force R={va['semantics']['force_trend_class']['semantic_ratio']:.6f}.\n- Vision classification: {vision['classification']}.\n- Calibrated VA uncertainty: Spearman={unc['FALLBACK_VA']['spearman']:.6f}, AUROC={unc['FALLBACK_VA']['auroc']:.6f}, top-20% removal reduction={unc['FALLBACK_VA']['risk_coverage']['top20_removal_reduction']:.6f}.\n- Router: explicit masks only; no Action means ABSTAIN_NO_ACTION.\n- Rank warning: retained.\n- C5 readiness: {locked['c5_readiness']}; C5 was not started.\n\nCommands:\n\n```bash\npython scripts/tactile_unit/audit_c4_availability_contract.py --device cuda:0\npython scripts/tactile_unit/train_c4_fallback.py --device cuda:0\npython scripts/tactile_unit/train_c4_uncertainty.py --device cuda:0\npython scripts/tactile_unit/evaluate_c4_missing_modality.py --device cuda:0\npython scripts/tactile_unit/visualize_c4_missing_modality.py\n```\n\nPlots: {len(plots)} decision-focused files under `plots/`.\n\nSTOP AFTER C4. Do not start C5 or C6; M3 is not established.\n"""
    (ARTIFACTS / "HUMAN_ACCEPTANCE.md").write_text(acceptance)


if __name__ == "__main__":
    PLOTS.mkdir(parents=True, exist_ok=True)
    main()
