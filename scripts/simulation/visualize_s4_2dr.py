#!/usr/bin/env python3
"""Generate the required S4.2-DR diagnostic and decision plots."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.simulation.s4_2dr_common import (  # noqa: E402
    ARTIFACT_ROOT,
    atomic_json,
    load_json,
)

PLOTS = ARTIFACT_ROOT / "plots"


def save(fig: plt.Figure, name: str) -> str:
    path = PLOTS / name
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def bars(title: str, labels: list[str], values: list[float], ylabel: str, name: str, colors: list[str] | None = None) -> str:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values, color=colors)
    ax.set(title=title, ylabel=ylabel)
    ax.tick_params(axis="x", rotation=20)
    return save(fig, name)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    fairness = load_json(ARTIFACT_ROOT / "baseline_fairness.json")
    reproduction = load_json(ARTIFACT_ROOT / "baseline_reproduction.json")
    reference = load_json(ARTIFACT_ROOT / "empirical_predictability_reference.json")
    dispersion = load_json(ARTIFACT_ROOT / "conditional_dispersion.json")
    regimes = load_json(ARTIFACT_ROOT / "regime_decomposition.json")
    existing = load_json(ARTIFACT_ROOT / "existing_c3_followup.json")
    trials = load_json(ARTIFACT_ROOT / "remediation_trials.json")
    final = load_json(ARTIFACT_ROOT / "final_decision.json")
    files = []

    fairness_labels = ["same data", "same target", "same mask", "same metric", "no C2 oracle", "reproduction"]
    fairness_values = [
        fairness["data_identity"]["same_train_split"], fairness["data_identity"]["same_h_future_target"],
        fairness["data_identity"]["same_dynamic_q70_mask"], fairness["metric_fairness"]["same_evaluation_implementation"],
        not fairness["information_sets"]["C2_exclusive_future_or_oracle_advantage"], reproduction["gate"] == "PASS",
    ]
    files.append(bars("C2/C3 fairness audit", fairness_labels, [float(v) for v in fairness_values], "Pass (1=yes)", "01_c2_c3_fairness_summary.png"))

    dynamic = reference["metrics"]["dynamic"]
    baseline_names = ["persistence", "current_only", "linear", "ridge", "small_MLP", "kNN", "delta_C2", "C3"]
    files.append(bars("Simple predictors and transition autoencoders", baseline_names, [dynamic[name]["mse"] for name in baseline_names], "Dynamic validation MSE", "02_simple_baseline_dynamic_mse.png"))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(["E_ref (ridge)", "C2", "C3"], [reference["selected_reference"]["dynamic_mse"], reference["E_C2"], reference["E_C3"]], color=["C2", "C1", "C0"])
    ax.set_yscale("log")
    ax.set(ylabel="Dynamic validation MSE (log scale)", title="No empirical headroom reference below C2")
    ax.text(1.0, reference["E_C2"] * 1.4, "gap undefined", ha="center")
    files.append(save(fig, "03_empirical_reference_headroom.png"))

    dispersion_names = ["all", "dynamic", "boundary", "task:pinch_tongs", "task:hammer_nail", "task:click_mouse"]
    files.append(bars("TRAIN-neighbor local conditional dispersion", dispersion_names, [dispersion["regimes"][name]["mean"] for name in dispersion_names], "Mean local target variance", "04_local_conditional_dispersion.png"))

    task_names = ["pinch_tongs", "hammer_nail", "click_mouse"]
    x = np.arange(len(task_names)); width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, [regimes["comparisons"][name]["C2_mse"] for name in task_names], width, label="C2")
    ax.bar(x + width / 2, [regimes["comparisons"][name]["C3_mse"] for name in task_names], width, label="C3")
    ax.set_xticks(x, task_names); ax.set(ylabel="Validation MSE", title="C2 vs C3 by task"); ax.legend()
    files.append(save(fig, "05_c2_vs_c3_by_task.png"))

    transition_names = ["free_to_free", "free_to_contact", "contact_to_contact", "contact_to_free"]
    x = np.arange(len(transition_names))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, [regimes["comparisons"][name]["C2_mse"] for name in transition_names], width, label="C2")
    ax.bar(x + width / 2, [regimes["comparisons"][name]["C3_mse"] for name in transition_names], width, label="C3")
    ax.set_xticks(x, transition_names); ax.tick_params(axis="x", rotation=15); ax.set(ylabel="Validation MSE", title="C2 vs C3 by contact-transition class"); ax.legend()
    files.append(save(fig, "06_c2_vs_c3_contact_transition.png"))

    for number, regime, label in ((7, "high_force", "high-force"), (8, "high_tangential", "high-tangential")):
        row = regimes["comparisons"][regime]
        files.append(bars(f"C2 vs C3: {label} (N={row['samples']})", ["C2", "C3"], [row["C2_mse"], row["C3_mse"]], "Validation MSE", f"{number:02d}_c2_vs_c3_{regime}.png"))

    control_regimes = ["dynamic", "boundary", "high_force", "high_tangential"]
    control_names = ["zero", "shuffled", "reversed", "different_episode", "mismatch"]
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(control_regimes)); width = 0.14
    for offset, control in enumerate(control_names):
        ax.bar(x + (offset - 2) * width, [regimes["controls"][regime][control]["control_mse"] for regime in control_regimes], width, label=control)
    ax.set_xticks(x, control_regimes); ax.set_yscale("log"); ax.set(ylabel="Control MSE (log scale)", title="Transition-code controls by regime"); ax.legend(ncol=3)
    files.append(save(fig, "09_transition_code_controls_by_regime.png"))

    gate_names = list(existing["gates"])
    gate_values = [float(existing["gates"][name]) for name in gate_names]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(range(len(gate_names)), gate_values, color=["C2" if value else "C3" for value in gate_values])
    ax.set_yticks(range(len(gate_names)), gate_names); ax.set(xlim=(0, 1.05), xlabel="Pass (1=yes)", title="Route A gate matrix")
    files.append(save(fig, "10_route_a_gate_matrix.png"))

    fig, ax = plt.subplots(figsize=(9, 6))
    for row in trials["trials"]:
        ax.scatter(row["overall_mse"], row["dynamic_mse"], s=90, label=row["trial"])
        ax.annotate(row["trial"], (row["overall_mse"], row["dynamic_mse"]), xytext=(5, 5), textcoords="offset points")
    ax.axhline(reference["E_C2"] * 0.90, color="red", linestyle="--", label="10% Route B threshold")
    ax.set(xlabel="Overall validation MSE", ylabel="Dynamic validation MSE", title="Bounded remediation candidate Pareto")
    ax.legend()
    files.append(save(fig, "11_remediation_candidate_pareto.png"))

    labels = ["DR1", "DR2", "DR3", "DR4", "Route A", "R1", "R2", "R3", "Route B", "locked test"]
    values = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]
    colors = ["C2" if value else "C3" for value in values]
    files.append(bars(f"Final: {final['decision']}", labels, values, "Pass/preserved (1=yes)", "12_final_decision_matrix.png", colors))

    summary = {
        "schema": "tactile3d-unit.s4-2dr-visualization-summary.v1",
        "plots": files,
        "plot_count": len(files),
        "unexecuted_experiments_plotted": False,
        "decision": final["decision"],
        "test_loaded": False,
    }
    atomic_json(ARTIFACT_ROOT / "visualization_summary.json", summary)
    acceptance = """# S4.2-DR Human Acceptance

| Stage | Status | Artifact | Command | Human inspection question |
|---|---|---|---|---|
| DR1 fairness | PASS | `baseline_fairness.json`, `baseline_reproduction.json` | `python scripts/simulation/audit_s4_2dr_baseline_ceiling.py --phase fairness` | Do C2 and C3 use identical pair identities, targets, masks, and equivalent pair information? |
| DR2 empirical reference | NO_EMPIRICAL_HEADROOM_REFERENCE | `empirical_predictability_reference.json`, `conditional_dispersion.json` | `python scripts/simulation/audit_s4_2dr_baseline_ceiling.py --phase reference` | Is the reference described as empirical rather than an irreducible Bayes floor? |
| DR3 regime decomposition | PASS | `regime_decomposition.json`, `plots/05` through `plots/09` | `python scripts/simulation/evaluate_s4_2dr_regimes.py --evaluate` | Are gains positive across the critical regimes and controls necessary in every task? |
| DR4 protocol | FROZEN | `protocol_freeze.json` | `python scripts/simulation/freeze_s4_2dr_protocol.py` | Were both routes and TRAIN q90 thresholds frozen before DR5/training? |
| DR5 existing C3 | ROUTE A FAIL | `existing_c3_followup.json`, `plots/10_route_a_gate_matrix.png` | `python scripts/simulation/audit_s4_2dr_existing.py` | Are only the empirical-reference/headroom gates failing? |
| DR6 bounded remediation | 3 TRIALS, ROUTE B FAIL | `remediation_trials.json`, `plots/11_remediation_candidate_pareto.png` | `python scripts/simulation/train_s4_2dr_dynamics.py` | Did each trial stay within budget and fail only the unchanged 10% magnitude gate? |
| DR7 final decision | S4_2DR_DYNAMICS_REMEDIATION_FAIL | `dynamics_acceptance.json`, `final_decision.json`, `plots/12_final_decision_matrix.png` | `python scripts/simulation/audit_s4_2dr_final.py` | Is no checkpoint accepted, is S4.2-4 not ready, and is the locked model test preserved? |

STOP after S4.2-DR. Do not begin Action, Vision, VAC Bridge, or any later stage.
"""
    (ARTIFACT_ROOT / "HUMAN_ACCEPTANCE.md").write_text(acceptance, encoding="utf-8")
    print(json.dumps({"plot_count": len(files), "decision": final["decision"], "test_loaded": False}, indent=2))


if __name__ == "__main__":
    main()
