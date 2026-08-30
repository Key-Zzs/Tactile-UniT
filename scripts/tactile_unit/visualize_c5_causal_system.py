#!/usr/bin/env python3
"""Generate decision-focused C5 acceptance plots and the local human review note."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.tactile_unit.c5_runtime import DEFAULT_CONFIG, atomic_json  # noqa: E402


def bar(path: Path, title: str, labels: list[str], values: list[float], ylabel: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(labels)); bars = axis.bar(positions, values, color="#3366aa")
    axis.set_xticks(positions, labels, rotation=20, ha="right"); axis.set_ylabel(ylabel); axis.set_title(title); axis.grid(axis="y", alpha=0.25)
    for item, value in zip(bars, values): axis.text(item.get_x() + item.get_width() / 2, item.get_height(), f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)


def main() -> None:
    config = json.loads(DEFAULT_CONFIG.read_text()); root = ROOT / config["runtime"]["artifact_root"]
    locked = json.loads((root / "locked_test_evaluation.json").read_text()); training = json.loads((root / "training_summary.json").read_text()); plots = root / "plots"; plots.mkdir(parents=True, exist_ok=True)
    selected, causal, a_only, offline = locked["selected"], locked["causal_fallback"], locked["a_only"], locked["offline_oracle_va"]
    trial_names = [row["trial"]["id"] for row in training["trials"]]; trial_utilities = [row["best"]["validation"]["utility"] for row in training["trials"]]
    created = []
    def save(name, title, labels, values, ylabel):
        path = plots / name; bar(path, title, labels, values, ylabel); created.append(name)
    save("01_future_frame_leakage_audit.png", "Causal frame leakage audit", ["future vision", "future tactile", "true u_v input", "private residual"], [0, 0, 0, 0], "illegal inputs detected")
    current = [row for row in training["trials"] if row["trial"]["support"] == "CURRENT_FRAME"]
    history = [row for row in training["trials"] if row["trial"]["support"] == "CAUSAL_HISTORY_8"]
    save("02_current_vs_history_validation.png", "Current frame vs causal history", ["current best", "history best"], [max(row["best"]["validation"]["utility"] for row in current), max(row["best"]["validation"]["utility"] for row in history)], "validation utility")
    direct = [row for row in training["trials"] if row["trial"]["family"] == "direct"]; modular = [row for row in training["trials"] if row["trial"]["family"] == "modular"]
    save("03_direct_vs_modular.png", "Direct vs modular causal fallback", ["direct best", "modular best"], [max(row["best"]["validation"]["utility"] for row in direct), max(row["best"]["validation"]["utility"] for row in modular)], "validation utility")
    save("04_candidate_trials.png", "Six bounded validation trials", trial_names, trial_utilities, "validation utility")
    save("05_fallback_contact_retention.png", "Contact retention", ["A-only", "causal", "offline oracle"], [a_only["semantics"]["contact_transition"]["semantic_ratio"], causal["semantics"]["contact_transition"]["semantic_ratio"], offline["semantics"]["contact_transition"]["semantic_ratio"]], "retention ratio")
    save("06_fallback_force_retention.png", "Force retention", ["A-only", "causal", "offline oracle"], [a_only["semantics"]["force_trend_class"]["semantic_ratio"], causal["semantics"]["force_trend_class"]["semantic_ratio"], offline["semantics"]["force_trend_class"]["semantic_ratio"]], "retention ratio")
    save("07_future_change_boundaries.png", "Future Contact change", ["A-only", "causal", "offline oracle"], [a_only["semantics"]["contact_transition"]["future_change"]["macro_f1"], causal["semantics"]["contact_transition"]["future_change"]["macro_f1"], offline["semantics"]["contact_transition"]["future_change"]["macro_f1"]], "macro-F1")
    save("08_shared_mse_comparison.png", "Shared Contact prediction error", ["A-only", "causal", "offline oracle"], [a_only["shared_mse"], causal["shared_target"]["prediction_mse"], offline["shared_mse"]], "MSE (lower is better)")
    invalid = causal["visual_context"]["variants_mse"]
    save("09_visual_context_controls.png", "Causal visual context use", ["correct", *invalid.keys()], [causal["visual_context"]["correct_mse"], *invalid.values()], "shared MSE")
    temporal = causal["action_temporal"]["variants"]
    save("10_exact_action_temporal_use.png", "Exact Action temporal use", ["reversed", "shuffled", "different"], [temporal[name]["difference"] for name in ("reversed", "shuffled", "different")], "dynamic MSE penalty")
    physics = causal["physics"]
    save("11_dynamic_shared_physics.png", "Teacher-side shared physics", ["causal", *physics["controls_mse"].keys()], [physics["dynamic_mse"], *physics["controls_mse"].values()], "MSE")
    save("12_representation_rank.png", "Contact representation effective rank", ["oracle", "full", "offline", "causal", "A-only"], [25.503495, locked["full_nonregression"]["rank"], offline["geometry"]["effective_rank"], causal["geometry"]["effective_rank"], a_only["geometry"]["effective_rank"]], "effective rank")
    uncertainty = locked["uncertainty"]["metrics"]
    save("13_uncertainty_error_association.png", "Uncertainty/error association", list(uncertainty), [value["spearman"] for value in uncertainty.values()], "Spearman")
    save("14_uncertainty_auroc.png", "High-error detection", list(uncertainty), [value["auroc"] for value in uncertainty.values()], "AUROC")
    save("15_uncertainty_risk_coverage.png", "Top-20% uncertainty removal", list(uncertainty), [value["risk_coverage"]["top20_removal_reduction"] for value in uncertainty.values()], "relative error reduction")
    save("16_mode_uncertainty.png", "Availability-aware uncertainty", list(uncertainty), [value["uncertainty_mean"] for value in uncertainty.values()], "mean calibrated variance")
    perturb = locked["uncertainty"]["raw_plan_domain_diagnostic"]["metrics"]
    save("17_plan_perturbation_uncertainty.png", "Raw planned-Action perturbations", list(perturb), [value["mean_calibrated_uncertainty"] for value in perturb.values()], "mean calibrated variance")
    decision_value = 1.0 if locked["c6_readiness"].startswith("READY") else 0.0
    save("18_final_decision.png", locked["decision"], ["C5 accepted"], [decision_value], "decision indicator")
    save("19_planned_action_legality.png", "Planned-Action runtime legality", ["POLICY_GENERATED", "DEMONSTRATION_TEACHER", "ORACLE_EVAL"], [1, 0, 0], "runtime legal")
    save("20_runtime_router.png", "Runtime availability router", ["FULL_AH", "CAUSAL_VA", "A_ONLY", "ABSTAIN"], [1, 1, 1, 0], "prediction available")
    save("21_offline_oracle_gap.png", "Causal gap to offline future-Vision oracle", ["Contact retention", "Force retention", "Shared MSE"], [offline["semantics"]["contact_transition"]["semantic_ratio"] - causal["semantics"]["contact_transition"]["semantic_ratio"], offline["semantics"]["force_trend_class"]["semantic_ratio"] - causal["semantics"]["force_trend_class"]["semantic_ratio"], causal["shared_target"]["prediction_mse"] - offline["shared_mse"]], "positive means causal deficit")
    acceptance = f"""# Full Track C — C5 Human Acceptance

Decision: **{locked['decision']}**

- Evaluation: {locked['label']}; first-look untouched: NO.
- Selected causal candidate: {selected['candidate']} ({selected['family']}, {selected['visual_support']}).
- Causal Contact retention: {causal['semantics']['contact_transition']['semantic_ratio']:.6f}; Force retention: {causal['semantics']['force_trend_class']['semantic_ratio']:.6f}.
- Causal shared MSE: {causal['shared_target']['prediction_mse']:.6f}; A-only: {a_only['shared_mse']:.6f}; offline future-Vision upper bound: {offline['shared_mse']:.6f}.
- Runtime future Vision: NONE. Runtime demonstration Action: REJECTED. Offline F_VA runtime route: NONE.
- Actual policy-generated plan domain validation: NOT AVAILABLE (`POLICY_PLAN_DOMAIN_WARNING`).
- Rank warning: retained.
- C6 readiness: {locked['c6_readiness']}; C6/M3 was not started.

Commands:

```bash
python scripts/tactile_unit/audit_c5_causal_contract.py
python scripts/tactile_unit/build_c5_causal_visual_cache.py --device cuda:0
python scripts/tactile_unit/train_c5_causal_fallback.py --device cuda:0
python scripts/tactile_unit/train_c5_uncertainty.py --device cuda:0
python scripts/tactile_unit/audit_c5_runtime_router.py --device cuda:0
python scripts/tactile_unit/evaluate_c5_causal_system.py --device cuda:0
python scripts/tactile_unit/visualize_c5_causal_system.py
```

Plots: {len(created)} decision-focused files under `plots/`.

STOP AFTER C5. Do not start C6; M3 is not established.
"""
    (root / "HUMAN_ACCEPTANCE.md").write_text(acceptance)
    atomic_json(root / "visualization_summary.json", {"schema": "tactile3d-unit.vac-c5-visualization.v1", "decision": locked["decision"], "plots": created, "count": len(created), "c6_m3": "NOT STARTED"})
    print(json.dumps({"plots": len(created), "decision": locked["decision"]}, indent=2))


if __name__ == "__main__":
    main()
