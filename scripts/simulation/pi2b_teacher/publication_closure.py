#!/usr/bin/env python3
"""Build the PI2B matched-teacher publication closure from frozen JSON only.

This program never imports or loads a teacher model.  It verifies the frozen
inputs, recomputes the registered endpoint summaries from saved evaluation
tables, and writes only beneath ``artifacts/posthoc_readonly_closure``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
LOGICAL_ROOT = "$PI2B_TEACHER_ROOT"
OUTPUT_RELATIVE = Path("artifacts/posthoc_readonly_closure")
TASKS = ("click_mouse", "hammer_nail", "pinch_tongs")
MODES = ("T_VA_match", "T_VAC_match")

EXPECTED_FROZEN_HASHES = {
    "T_VA_match_checkpoint": "91ae74d970917535e199822b3fb415997d88b36342dc7178b61efcd7a746c142",
    "T_VAC_match_checkpoint": "65ba855fda2175c51b23e58cd36e7889fa891da56a697ad442c3ea926951f698",
    "final_result.json": "81eb443367887ff8b64a8ffb7404ff7f6465d8787e0a869c8c5652c5e8b333da",
    "evaluation_result.json": "ee66cc8b3ea5837c0639ecc306b0eeb04374651253822085abded878218b5cbf",
    "stratified_group_statistics.json": "e939df390d7debaa01c8f68a9373271daad1d6fc0ccb1707c2f008702df56bca",
    "statistics_independent_audit.json": "1613d6a42202f5fedb903e0498f4a9b1a99de040071aac49468673688cf658db",
    "paper_delta.md": "6f90a3a540dd13880bea8523cc8dec2cda352f07271bd405e0fed971c7e1c975",
    "paper_delta.json": "e53e84df2463f31f900952ca728569f1bda4e522c3d3e53b328774d9745acaa3",
    "integration_handoff.json": "99ef6c413f118e05308af087db6de7f15f06dd65df302c9c322dc38776743ea3",
    "study_summary.png": "c80ea73b78f1a47b9333417249a30cb2709f2c6351f4f33b885b7cbdb398968d",
    "protocol_freeze.json": "e6044b9832b495170930075fc6d39fb38f061ade9d0ac873c3852446df795cb1",
    "pretest_freeze.json": "42dff7571a89a32eb5aeeab0fbcb7b3258e3c28ac6ed50a25524e015480eb127",
    "paired_train.npz": "73a0ab65fd430970cfc6d05e7fce71ea15a7bff46342dbd11f650076afe0c698",
    "paired_validation.npz": "37a7e5f9f2639f63f6923cb01f558c7606f1e837368735c67638200d666ddba9",
    "fresh_confirmation_native.npz": "66ac1dd14661876506e6e157f945a08b7f27b3cdfc08c4c77dd59837d577d88c",
    "A0_checkpoint": "9392154b5a87fbe0a62bae6842a7df84165bd79d0c345cb816812e939060dc58",
    "C3_checkpoint": "f67b8b0d6f519944adafecbb0a93bb4387213fa1a4f33f7f770a55938de05602",
    "Contact-State_checkpoint": "3d4519195e7a0d9f5af63399840a48121d5f614ea587c80b9461fc696a372a19",
    "Vision_config.json": "7a651f488c93521e0d507880fc250a475e6a08aa9307aa1349f9d3509844971e",
    "Vision_model-00001-of-00002.safetensors": "32d5c326f6c83d12185b6954d2a52511f66ad18b6fdf814aecc5726dd39c243c",
    "Vision_model-00002-of-00002.safetensors": "2f8093a900330e5111b63e44dc1687b3212bec343e5b3832bf2e40f2bf18a768",
    "Vision_model.safetensors.index.json": "3b6d73d2442ce694287c5cd8b93db1bb232909becf35f08ecabadb614b9a1b86",
    "PAPER_CORE_BASE.md": "a79a43e588b4a24b923dac2e06095b159b44c03c658d2149f264e704c6cd6467",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def resolve_root(argument: Path | None) -> Path:
    if argument is not None:
        return argument.resolve()
    environment = os.environ.get("PI2B_TEACHER_ROOT")
    if environment:
        return Path(environment).resolve()
    artifact_link = REPO_ROOT / ".local/artifacts/simulation/s4_3_pi2b_teacher"
    return artifact_link.resolve().parent


def frozen_paths(root: Path) -> dict[str, Path]:
    vision = REPO_ROOT / ".local/refs/base/external/s4_3_pi2u/unit_fulldata/VLA-UniT-3B-fulldata/tokenizer"
    return {
        "T_VA_match_checkpoint": root / "experiments/T_VA_match/seed42/final.pt",
        "T_VAC_match_checkpoint": root / "experiments/T_VAC_match/seed42/final.pt",
        "final_result.json": root / "artifacts/final_result.json",
        "evaluation_result.json": root / "artifacts/evaluation_result.json",
        "stratified_group_statistics.json": root / "artifacts/stratified_group_statistics.json",
        "statistics_independent_audit.json": root / "artifacts/statistics_independent_audit.json",
        "paper_delta.md": root / "artifacts/paper_delta.md",
        "paper_delta.json": root / "artifacts/paper_delta.json",
        "integration_handoff.json": root / "artifacts/integration_handoff.json",
        "study_summary.png": root / "plots/study_summary.png",
        "protocol_freeze.json": root / "artifacts/protocol_freeze.json",
        "pretest_freeze.json": root / "artifacts/pretest_freeze.json",
        "paired_train.npz": REPO_ROOT / ".local/refs/base/cache/simulation/s4_2_formal/paired_train.npz",
        "paired_validation.npz": REPO_ROOT / ".local/refs/base/cache/simulation/s4_2_formal/paired_validation.npz",
        "fresh_confirmation_native.npz": root / "cache/fresh_confirmation_native.npz",
        "A0_checkpoint": REPO_ROOT / ".local/experiments/simulation/s4_2_formal/action/selected.pt",
        "C3_checkpoint": REPO_ROOT / ".local/experiments/simulation/s4_2r/contact_dynamics/selected.pt",
        "Contact-State_checkpoint": REPO_ROOT / ".local/experiments/simulation/s4_2r/contact_state/accepted.pt",
        "Vision_config.json": vision / "config.json",
        "Vision_model-00001-of-00002.safetensors": vision / "model-00001-of-00002.safetensors",
        "Vision_model-00002-of-00002.safetensors": vision / "model-00002-of-00002.safetensors",
        "Vision_model.safetensors.index.json": vision / "model.safetensors.index.json",
        "PAPER_CORE_BASE.md": REPO_ROOT / ".local/paper/PAPER_CORE_BASE.md",
    }


def verify_frozen(root: Path) -> dict[str, Any]:
    actual = {name: sha256_file(path) for name, path in frozen_paths(root).items()}
    matches = {name: actual[name] == expected for name, expected in EXPECTED_FROZEN_HASHES.items()}
    result = {
        "schema": "tactile3d-unit.s4-3-pi2b-teacher-readonly-integrity.v2",
        "status": "PASS" if all(matches.values()) else "FAIL",
        "hashes": {
            name: {"expected_sha256": EXPECTED_FROZEN_HASHES[name], "actual_sha256": actual[name], "match": matches[name]}
            for name in sorted(actual)
        },
        "historical_artifacts_modified": False,
        "teacher_retraining": False,
        "teacher_inference": False,
        "policy_execution": False,
        "track_a_performance_read": False,
    }
    if result["status"] != "PASS":
        failed = [name for name, matched in matches.items() if not matched]
        raise RuntimeError(f"PI2B_TEACHER_CLOSURE_INTEGRITY_FAIL: {failed}")
    return result


def bidirectional(alignment: dict[str, Any], field: str) -> float:
    return 0.5 * (float(alignment["forward"][field]) + float(alignment["reverse"][field]))


def estimand_audit(evaluation: dict[str, Any], statistics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "tactile3d-unit.s4-3-pi2b-teacher-metric-scope-audit.v2",
        "status": "PASS",
        "classification": "MULTIPLE_DEFINED_ESTIMANDS",
        "mechanism": "CANDIDATE_POOL_DIFFERENCE",
        "implementation_bug": False,
        "changes_historical_decision": False,
        "code_paths": {
            "evaluation_result": "scripts/simulation/pi2b_teacher/evaluate.py::{alignment_report,group_common_metrics,main}",
            "retrieval_kernel": "gr00t/simulation/pi2b_teacher/evaluation.py::{retrieval_ranks,retrieval_metrics}",
            "registered_statistics": "scripts/simulation/pi2b_teacher/analyze.py::main",
            "independent_recalculation": "scripts/simulation/pi2b_teacher/independent_audit.py::main",
        },
        "rank_formula": "r_i = 1 + sum_j 1[cos(q_i,c_j) > cos(q_i,c_i)]",
        "r10_formula": "R@10 = N^-1 sum_i 1[r_i <= 10]",
        "mrr_formula": "MRR = N^-1 sum_i (1 / r_i)",
        "direction_formula": "bidirectional = 0.5 * (V-to-A + A-to-V)",
        "tie_rule": "Candidates strictly more similar than the paired positive increase rank; equal-similarity ties do not.",
        "self_and_pair_rule": "The paired positive c_i remains in the bank. No same-pair, same-episode, or overlapping-window negatives are excluded.",
        "window_rule": "All overlapping windows remain present and stay together inside their source-group resampling unit.",
        "formal_effect_formula": "Delta_group = (1/9) sum_g [m_g(VAC; C_g) - m_g(VA; C_g)]",
        "bootstrap": "Paired source-group effects; resample 3 groups with replacement inside each of 3 task strata; 10,000 draws.",
        "interpretation": "Pooled and source-group retrieval answer different questions. The sign difference is not a contradiction and neither estimand may replace the preregistered one.",
    }
    endpoints: dict[str, Any] = {}
    field_by_metric = {"r10": "recall_at_10", "mrr": "mrr"}
    group_field = {"r10": "mean_bidirectional_va_retrieval_r10", "mrr": "mean_bidirectional_va_mrr"}
    for metric, field in field_by_metric.items():
        pooled = {}
        task_restricted = {}
        group_macro = {}
        for mode in MODES:
            fresh = evaluation["models"][mode]["splits"]["fresh"]["va_alignment"]
            pooled[mode] = bidirectional(fresh["overall"], field)
            per_task = {task: bidirectional(fresh[f"task:{task}"], field) for task in TASKS}
            task_restricted[mode] = {
                "estimate": float(np.mean(list(per_task.values()))),
                "per_task": per_task,
            }
            values = [float(value[group_field[metric]]) for value in evaluation["group_metrics"][mode].values()]
            group_macro[mode] = float(np.mean(values))
        formal = statistics["paired_effects"][group_field[metric]]
        endpoints[metric] = {
            "pooled_descriptive": {
                "query_population": "all 4,860 fresh pairs",
                "candidate_bank": "all 4,860 fresh candidates across tasks and source groups",
                "query_weighting": "equal per query in each direction",
                "direction_weighting": "equal V-to-A and A-to-V",
                "candidate_count": 4860,
                "VA": pooled["T_VA_match"],
                "VAC": pooled["T_VAC_match"],
                "delta_VAC_minus_VA": pooled["T_VAC_match"] - pooled["T_VA_match"],
                "uncertainty": "descriptive only; no registered CI",
            },
            "source_group_formal": {
                "query_population": "all fresh pairs, evaluated separately within each source group",
                "candidate_bank": "540 candidates from the query's source group",
                "query_weighting": "equal within group",
                "group_weighting": "equal across 9 paired source groups",
                "task_weighting": "three groups per task; balanced equal weighting",
                "direction_weighting": "equal V-to-A and A-to-V within group",
                "candidate_count_per_group": 540,
                "VA_group_macro": group_macro["T_VA_match"],
                "VAC_group_macro": group_macro["T_VAC_match"],
                "delta_VAC_minus_VA": formal["estimate"],
                "ci95": formal["ci95"],
                "registered": True,
            },
            "task_restricted_macro_descriptive": {
                "query_population": "all fresh pairs, evaluated separately within task",
                "candidate_bank": "1,620 candidates from the query's task",
                "query_weighting": "equal within task",
                "task_weighting": "equal across 3 tasks",
                "direction_weighting": "equal V-to-A and A-to-V within task",
                "candidate_count_per_task": 1620,
                "VA": task_restricted["T_VA_match"]["estimate"],
                "VAC": task_restricted["T_VAC_match"]["estimate"],
                "delta_VAC_minus_VA": task_restricted["T_VAC_match"]["estimate"] - task_restricted["T_VA_match"]["estimate"],
                "per_task": {
                    task: {
                        "VA": task_restricted["T_VA_match"]["per_task"][task],
                        "VAC": task_restricted["T_VAC_match"]["per_task"][task],
                    }
                    for task in TASKS
                },
                "uncertainty": "descriptive only; no registered CI",
            },
        }
    result["estimands"] = endpoints
    result["weighting_diagnosis"] = {
        "all_source_groups_have_540_rows": True,
        "all_tasks_have_3_groups_and_1620_rows": True,
        "macro_micro_weight_difference_drives_sign_change": False,
        "candidate_pool_scope_can_change_teacher_specific_ranks": True,
    }
    return result


def effect_summary(statistics: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    paired = statistics["paired_effects"]
    contact = statistics["contact_statistics"]
    return {
        "contact_path_capability": {
            "status": "RELIABLE_CONTACT_CROSS_MODAL_CAPABILITY",
            "recovery": evaluation["models"]["T_VAC_match"]["contact"]["contact_recovery"],
            "vision_contact_paired_margin": contact["vision_contact_paired_margin"],
            "action_contact_paired_margin": contact["action_contact_paired_margin"],
            "vision_contact_temporal_margin": contact["vision_contact_temporal_margin"],
            "action_contact_temporal_margin": contact["action_contact_temporal_margin"],
            "scope": "Cross-modal pairing and one reversal-based temporal discrimination metric; not full state recovery or causal identification.",
        },
        "common_va_recovery": {
            "vision": paired["vision_native_recovery_mse"],
            "action": paired["action_native_recovery_mse"],
            "interpretation": "Vision recovery improved. Action recovery had an unfavorable point estimate and raw CI, but was not Holm-confirmed in the frozen family.",
        },
        "common_va_retrieval_and_semantics": {
            "r10": paired["mean_bidirectional_va_retrieval_r10"],
            "mrr": paired["mean_bidirectional_va_mrr"],
            "contact_probe": paired["mean_common_va_contact_probe_macro_f1"],
            "status": "INCONCLUSIVE",
        },
        "temporal_action_structure": {
            "action_reversal_margin": paired["action_reversal_margin"],
            "scope": "Confirmed improvement in one raw-Action reversal discrimination metric; not universal Action improvement.",
        },
    }


def task_heterogeneity(statistics: dict[str, Any]) -> dict[str, Any]:
    paired = statistics["paired_effects"]
    r10 = paired["mean_bidirectional_va_retrieval_r10"]["per_task"]
    mrr = paired["mean_bidirectional_va_mrr"]["per_task"]
    allowed = {
        "click_mouse": "The descriptive group-level R@10 point effect was negative for click_mouse in this frozen sample.",
        "hammer_nail": "The descriptive group-level R@10 point effect was positive for hammer_nail in this frozen sample.",
        "pinch_tongs": "The descriptive group-level R@10 point effect was approximately neutral for pinch_tongs in this frozen sample.",
    }
    result = {
        "schema": "tactile3d-unit.s4-3-pi2b-teacher-task-heterogeneity.v2",
        "status": "DESCRIPTIVE_ONLY",
        "sample_unit": "paired fresh source group",
        "groups_per_task": 3,
        "episodes_per_group": 5,
        "rows_per_group": 540,
        "tasks": {},
        "boundary": {
            "vision_recovery_VAC_minus_VA_mse": paired["boundary_vision_recovery_mse"],
            "action_recovery_VAC_minus_VA_mse": paired["boundary_action_recovery_mse"],
            "va_retrieval_r10_VAC_minus_VA": paired["boundary_mean_va_retrieval_r10"],
            "allowed_wording": "Boundary Vision recovery favored VAC, boundary Action recovery was unfavorable, and boundary retrieval was inconclusive in the frozen analysis.",
        },
        "dynamic_static": "No complete nine-group paired dynamic/static effect was saved; no closure claim is made.",
        "high_force": "No preregistered high-force paired effect was saved; unavailable rather than imputed.",
        "coverage_limits": [
            "Only three fresh source groups per task.",
            "The right_thumb region had zero fresh positive transitions.",
            "Region-transition probes are N/A because TRAIN lacks region-transition labels.",
        ],
        "forbidden_wording": "Do not present click harm, hammer benefit, or pinch neutrality as stable task laws.",
    }
    for task in TASKS:
        result["tasks"][task] = {
            "r10_effect_VAC_minus_VA": r10[task],
            "mrr_effect_VAC_minus_VA": mrr[task],
            "sample_unit": "paired fresh source group",
            "group_count": 3,
            "ci95": None,
            "ci_availability": "No task-specific bootstrap CI was frozen.",
            "allowed_wording": allowed[task],
            "forbidden_wording": "Do not generalize this point estimate into a stable task-specific treatment effect.",
        }
    return result


def style_axis(axis: plt.Axes) -> None:
    axis.axvline(0.0, color="#333333", linewidth=1.0)
    axis.grid(axis="x", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)


def errorbar_effect(axis: plt.Axes, labels: list[str], estimates: list[float], intervals: list[list[float]], color: str) -> None:
    y = np.arange(len(labels))
    lower = [estimate - interval[0] for estimate, interval in zip(estimates, intervals)]
    upper = [interval[1] - estimate for estimate, interval in zip(estimates, intervals)]
    axis.errorbar(estimates, y, xerr=[lower, upper], fmt="o", color=color, capsize=4, markersize=7)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    style_axis(axis)


def save_figure(path: Path, fig: plt.Figure) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight", metadata={"Software": "PI2B read-only publication closure"})
    plt.close(fig)


def make_plots(output: Path, metric_audit: dict[str, Any], statistics: dict[str, Any]) -> dict[str, str]:
    plot_root = output / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold", "figure.facecolor": "white"})
    paired = statistics["paired_effects"]

    vision = paired["vision_native_recovery_mse"]
    action = paired["action_native_recovery_mse"]
    estimates = [-vision["estimate"], -action["estimate"]]
    intervals = [[-vision["ci95"][1], -vision["ci95"][0]], [-action["ci95"][1], -action["ci95"][0]]]
    fig, axis = plt.subplots(figsize=(7.6, 3.2))
    errorbar_effect(axis, ["Vision native recovery", "Action native recovery"], estimates, intervals, "#3b6fb6")
    axis.set_xlabel("Improvement = MSE(VA) - MSE(VAC); positive favors VAC")
    axis.set_title("A. Common V/A recovery paired effects")
    axis.text(0.99, 0.03, "95% source-group bootstrap CI\nAction: not Holm-confirmed", transform=axis.transAxes, ha="right", va="bottom", fontsize=8)
    save_figure(plot_root / "figure_a_common_va_recovery.png", fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for axis, metric, title in zip(axes, ("r10", "mrr"), ("R@10", "MRR")):
        values = metric_audit["estimands"][metric]
        points = [
            values["pooled_descriptive"]["delta_VAC_minus_VA"],
            values["source_group_formal"]["delta_VAC_minus_VA"],
            values["task_restricted_macro_descriptive"]["delta_VAC_minus_VA"],
        ]
        labels = ["Pooled\ndescriptive", "Source-group\nformal", "Task-restricted\nmacro"]
        y = np.arange(3)
        axis.scatter(points, y, color=["#777777", "#7a3e9d", "#318f6b"], s=46, zorder=3)
        formal = values["source_group_formal"]
        axis.errorbar([formal["delta_VAC_minus_VA"]], [1], xerr=[[formal["delta_VAC_minus_VA"] - formal["ci95"][0]], [formal["ci95"][1] - formal["delta_VAC_minus_VA"]]], fmt="none", ecolor="#7a3e9d", capsize=4)
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        style_axis(axis)
        axis.set_xlabel("VAC - VA")
        axis.set_title(title)
    fig.suptitle("B. Different estimands — not contradictory measurements", fontweight="bold")
    fig.text(0.5, 0.01, "Only the source-group estimand has the preregistered 95% CI; candidate banks are 4,860 / 540 / 1,620.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    save_figure(plot_root / "figure_b_retrieval_estimands.png", fig)

    contact_names = [
        ("V-C paired", "vision_contact_paired_margin"),
        ("A-C paired", "action_contact_paired_margin"),
        ("V-C temporal", "vision_contact_temporal_margin"),
        ("A-C temporal", "action_contact_temporal_margin"),
    ]
    contact = statistics["contact_statistics"]
    fig, axis = plt.subplots(figsize=(7.6, 4.0))
    errorbar_effect(axis, [label for label, _ in contact_names], [contact[name]["estimate"] for _, name in contact_names], [contact[name]["ci95"] for _, name in contact_names], "#1b8a77")
    axis.set_xlabel("Paired or temporal cosine margin; positive is better")
    axis.set_title("C. Contact cross-modal capability")
    axis.text(0.01, 0.03, "95% stratified source-group bootstrap CI", transform=axis.transAxes, ha="left", va="bottom", fontsize=8)
    save_figure(plot_root / "figure_c_contact_capability.png", fig)

    probe = paired["mean_common_va_contact_probe_macro_f1"]
    fig, axis = plt.subplots(figsize=(7.6, 2.8))
    errorbar_effect(axis, ["Mean V/A Contact-probe macro-F1"], [probe["estimate"]], [probe["ci95"]], "#c06b2c")
    axis.set_xlabel("Paired effect (VAC - VA macro-F1)")
    axis.set_title("D. Common V/A Contact-probe increment")
    axis.text(0.99, 0.06, "INCONCLUSIVE; Holm p = 0.3012", transform=axis.transAxes, ha="right", va="bottom", fontsize=8)
    save_figure(plot_root / "figure_d_common_va_contact_probe.png", fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    for axis, metric, title in zip(axes, ("mean_bidirectional_va_retrieval_r10", "mean_bidirectional_va_mrr"), ("R@10", "MRR")):
        task_effects = paired[metric]["per_task"]
        y = np.arange(len(TASKS))
        axis.scatter([task_effects[task] for task in TASKS], y, color="#6c5a9a", s=48)
        axis.set_yticks(y, [task.replace("_", " ") for task in TASKS])
        axis.invert_yaxis()
        style_axis(axis)
        axis.set_xlabel("Group-level VAC - VA")
        axis.set_title(title)
    fig.suptitle("E. Descriptive task heterogeneity", fontweight="bold")
    fig.text(0.5, 0.01, "Three fresh source groups per task; points are descriptive and have no task-specific frozen CI.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    save_figure(plot_root / "figure_e_task_heterogeneity.png", fig)

    claims = [
        ("Contact path", "SUPPORTED", "#2f855a"),
        ("Vision recovery", "SUPPORTED improvement", "#2f855a"),
        ("Action recovery", "MIXED / NOT HOLM-CONFIRMED", "#b7791f"),
        ("V/A retrieval", "INCONCLUSIVE", "#805ad5"),
        ("Common V/A Contact readout", "INCONCLUSIVE", "#805ad5"),
        ("Policy utility", "NOT TESTED", "#718096"),
        ("Cross-teacher-seed stability", "NOT TESTED", "#718096"),
    ]
    fig, axis = plt.subplots(figsize=(10.2, 4.6))
    axis.axis("off")
    table = axis.table(cellText=[[claim, status] for claim, status, _ in claims], colLabels=["Claim", "Closure status"], cellLoc="left", colLoc="left", loc="center", colWidths=[0.48, 0.48])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for column in range(2):
        table[(0, column)].set_facecolor("#e8edf3")
        table[(0, column)].set_text_props(weight="bold")
    for row, (_, _, color) in enumerate(claims, start=1):
        table[(row, 1)].set_text_props(color=color, weight="bold")
    axis.set_title("F. Final claim matrix", fontweight="bold", pad=10)
    save_figure(plot_root / "figure_f_final_claim_matrix.png", fig)
    return {path.name: sha256_file(path) for path in sorted(plot_root.glob("*.png"))}


def figure_captions() -> str:
    return """# Publication-facing figure captions

**Figure A — Common V/A recovery paired effects.** Positive values denote lower MSE for VAC (`MSE_VA - MSE_VAC`). Vision recovery improved; Action recovery was unfavorable and was not significant after Holm correction. Error bars are 95% task-stratified source-group bootstrap intervals.

**Figure B — Retrieval estimand clarification.** Pooled descriptive, preregistered source-group, and task-restricted macro effects use candidate banks of 4,860, 540, and 1,620 respectively. Rank metrics depend on that bank, so the estimates answer different questions. Only the source-group effect carries the preregistered interval and inferential role.

**Figure C — Contact capability.** Vision–Contact and Action–Contact paired and temporal-reversal margins are positive with 95% task-stratified source-group bootstrap intervals above zero. These support Contact cross-modal capability, not complete physical-state recovery or causal dynamics identification.

**Figure D — Common V/A Contact-probe increment.** The paired macro-F1 increment is shown on an untruncated effect axis with its 95% source-group bootstrap interval. The interval crosses zero and the frozen Holm-adjusted result is inconclusive.

**Figure E — Task heterogeneity.** Per-task group-level R@10 and MRR point effects are descriptive. Each task has only three fresh source groups; no task-specific interval was frozen, so the apparent pattern must not be treated as a stable task law.

**Figure F — Final claim matrix.** Claim status distinguishes supported scoped findings, inconclusive outcomes, and outcomes not tested. `COMPLETE_VALID` is an artifact status, not an unrestricted scientific claim.
"""


def paper_delta_markdown(metric_audit: dict[str, Any]) -> str:
    r10 = metric_audit["estimands"]["r10"]
    mrr = metric_audit["estimands"]["mrr"]
    return f"""# PAPER_CORE delta v2 — S4.3-PI2B-T-C

This is a proposed integration delta. The main `PAPER_CORE` remains read-only.

## A. Proposed method addition

We compared fixed-seed matched `T_VA_match` and `T_VAC_match` teachers under byte-identical common Vision/Action initialization, identical ordered TRAIN pairs, batch schedule, 800 optimizer updates, sample exposure, and an unchanged additive V/A loss block. VAC added Contact-private modules and an additive Contact loss; total parameters and measured compute therefore differed. Fresh confirmation used 4,860 pairs from 45 episodes and nine source groups. Paired source groups were the inferential unit, resampled within task strata, with Holm adjustment over the preregistered primary family.

## B. Proposed representation-results addition

Adding Contact supervision produced a reliable Contact branch with strong Vision–Contact and Action–Contact paired and temporal relations. Effects on the common Vision/Action path were mixed: Vision recovery improved, while Action recovery showed an unfavorable point estimate that did not survive multiplicity correction; group-level retrieval and Contact-probe increments remained inconclusive.

The descriptive pooled R@10 effect was {r10['pooled_descriptive']['delta_VAC_minus_VA']:.6f}, whereas the formal source-group effect was {r10['source_group_formal']['delta_VAC_minus_VA']:.6f} (95% CI [{r10['source_group_formal']['ci95'][0]:.6f}, {r10['source_group_formal']['ci95'][1]:.6f}]). For MRR the corresponding effects were {mrr['pooled_descriptive']['delta_VAC_minus_VA']:.6f} and {mrr['source_group_formal']['delta_VAC_minus_VA']:.6f} ([{mrr['source_group_formal']['ci95'][0]:.6f}, {mrr['source_group_formal']['ci95'][1]:.6f}]). These are different candidate-pool estimands; pooled values do not replace the preregistered result.

## C. Claim-ledger modifications

- `SUPPORTED_WITH_SCOPE`: Under matched common-path initialization and data/update budget, adding Contact supervision learned a reliable Contact branch without a reliably detected degradation in the registered group-level V/A retrieval metrics. This is not a noninferiority claim.
- `SUPPORTED_WITH_SCOPE`: Contact supervision improved Vision native recovery in the tested matched-teacher setting.
- `SUPPORTED_WITH_SCOPE`: Effects on Action recovery and task-level retrieval were heterogeneous; the Action recovery point estimate was unfavorable and not Holm-confirmed.
- `INCONCLUSIVE`: No reliable increase in Contact information readable from the common V/A outputs was detected.
- `NOT_TESTED`: Policy utility and cross-teacher-training-seed stability remain untested.

## D. Figure and table proposals

Use Figures A–F in `artifacts/posthoc_readonly_closure/plots/` and the accompanying captions. Preserve `plots/study_summary.png` as the immutable historical figure.

## E. Exact limitations

The comparison uses one teacher training seed, three fresh source groups per task, overlapping windows clustered within source group, no equivalence or noninferiority margin, no policy evaluation, no cross-seed stability test, and no active right-thumb positives. Region-transition probes are unavailable because TRAIN lacks those labels. Parameter count and compute are not matched.

## F. Integration prerequisites

Wait for Track A final closure, then perform an evidence-integration task. Do not use Track A performance to revise this frozen Track B decision, do not replace Track A teachers automatically, and do not edit the main `PAPER_CORE` from this branch.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--final-commit", default="PENDING_CLOSURE_COMMIT")
    parser.add_argument("--regression-status", choices=("PASS", "FAIL", "NOT_RECORDED"), default="NOT_RECORDED")
    parser.add_argument("--privacy-status", choices=("PASS", "FAIL", "NOT_RECORDED"), default="NOT_RECORDED")
    parser.add_argument("--working-tree-status", choices=("CLEAN", "DIRTY", "NOT_RECORDED"), default="NOT_RECORDED")
    args = parser.parse_args()
    root = resolve_root(args.root)
    output = root / OUTPUT_RELATIVE
    output.mkdir(parents=True, exist_ok=True)

    integrity = verify_frozen(root)
    evaluation = load_json(root / "artifacts/evaluation_result.json")
    statistics = load_json(root / "artifacts/stratified_group_statistics.json")
    independent = load_json(root / "artifacts/statistics_independent_audit.json")
    final = load_json(root / "artifacts/final_result.json")
    if evaluation["track_a_performance_read"] or independent["status"] != "PASS":
        raise RuntimeError("frozen evaluation firewall or independent audit failed")
    if final["scientific_decision"] != "VAC_CONTACT_CAPABILITY_WITH_NO_DETECTED_VA_REGRESSION_LIMITED_PRECISION":
        raise RuntimeError("unexpected frozen scientific decision")

    metric_audit = estimand_audit(evaluation, statistics)
    effects = effect_summary(statistics, evaluation)
    heterogeneity = task_heterogeneity(statistics)
    atomic_json(output / "frozen_integrity_audit.json", integrity)
    atomic_json(output / "metric_scope_audit.json", metric_audit)
    atomic_json(output / "effect_size_claim_scope.json", effects)
    atomic_json(output / "task_heterogeneity_closure.json", heterogeneity)

    plot_hashes = make_plots(output, metric_audit, statistics)
    atomic_text(output / "figure_captions.md", figure_captions())
    plot_paths = [f"{LOGICAL_ROOT}/{OUTPUT_RELATIVE.as_posix()}/plots/{name}" for name in sorted(plot_hashes)]
    figure_manifest = {
        "schema": "tactile3d-unit.s4-3-pi2b-teacher-publication-figures.v2",
        "source": "Frozen evaluation_result.json and stratified_group_statistics.json only",
        "teacher_inference": False,
        "historical_study_summary_preserved": True,
        "plots": {path: plot_hashes[Path(path).name] for path in plot_paths},
        "captions": f"{LOGICAL_ROOT}/{OUTPUT_RELATIVE.as_posix()}/figure_captions.md",
    }
    atomic_json(output / "publication_figure_manifest.json", figure_manifest)

    delta_md = paper_delta_markdown(metric_audit)
    atomic_text(output / "paper_delta_v2.md", delta_md)
    paper_delta = {
        "schema": "tactile3d-unit.s4-3-pi2b-teacher-paper-delta.v2",
        "status": "PROPOSED_NOT_INTEGRATED",
        "source_paper_core_sha256": EXPECTED_FROZEN_HASHES["PAPER_CORE_BASE.md"],
        "main_paper_core_edited": False,
        "scientific_decision": final["scientific_decision"],
        "metric_scope_classification": metric_audit["classification"],
        "claim_status_vocabulary": ["SUPPORTED_WITH_SCOPE", "INCONCLUSIVE", "NOT_TESTED"],
        "proposed_claims": [
            {"status": "SUPPORTED_WITH_SCOPE", "claim": "Adding Contact supervision learned a reliable Contact branch without a reliably detected degradation in registered group-level V/A retrieval metrics; this is not noninferiority."},
            {"status": "SUPPORTED_WITH_SCOPE", "claim": "Contact supervision improved Vision native recovery in the tested matched-teacher setting."},
            {"status": "SUPPORTED_WITH_SCOPE", "claim": "Action recovery and task-level retrieval effects were heterogeneous, including an unfavorable non-Holm-confirmed Action recovery point estimate."},
            {"status": "INCONCLUSIVE", "claim": "No reliable increase in Contact information readable from common V/A outputs was detected."},
            {"status": "NOT_TESTED", "claim": "Policy utility and cross-teacher-seed stability remain untested."},
        ],
        "limitations": [
            "one teacher training seed42",
            "three fresh source groups per task",
            "no equivalence or noninferiority margin",
            "right_thumb has zero fresh positive transitions",
            "no policy evaluation",
            "total parameters and compute differ",
        ],
        "integration_prerequisite": "WAIT_FOR_TRACK_A_FINAL_THEN_EVIDENCE_INTEGRATION",
        "markdown_path": f"{LOGICAL_ROOT}/{OUTPUT_RELATIVE.as_posix()}/paper_delta_v2.md",
        "markdown_sha256": sha256_file(output / "paper_delta_v2.md"),
    }
    atomic_json(output / "paper_delta_v2.json", paper_delta)

    final_status = (
        "PI2B_T_CLOSURE_COMPLETE"
        if integrity["status"] == "PASS"
        and args.regression_status == "PASS"
        and args.privacy_status == "PASS"
        and args.working_tree_status == "CLEAN"
        and args.final_commit != "PENDING_CLOSURE_COMMIT"
        else "PI2B_T_CLOSURE_AUDIT_PENDING"
    )
    handoff = {
        "schema": "tactile3d-unit.s4-3-pi2b-teacher-integration-handoff.v2",
        "branch": "develop/pi2b-teacher",
        "base_commit": "8f39aed123adf0a8b7241e75472e678355444e5e",
        "starting_closure_commit": "e62913943bac9565b439800657a0d4488b35deb8",
        "final_closure_commit": args.final_commit,
        "teacher_checkpoint_sha256": {
            "T_VA_match": EXPECTED_FROZEN_HASHES["T_VA_match_checkpoint"],
            "T_VAC_match": EXPECTED_FROZEN_HASHES["T_VAC_match_checkpoint"],
        },
        "protocol_hashes": {
            "protocol_freeze": EXPECTED_FROZEN_HASHES["protocol_freeze.json"],
            "pretest_freeze": EXPECTED_FROZEN_HASHES["pretest_freeze.json"],
        },
        "data_identity": {
            "TRAIN_pairs": EXPECTED_FROZEN_HASHES["paired_train.npz"],
            "fresh_confirmation_pairs": EXPECTED_FROZEN_HASHES["fresh_confirmation_native.npz"],
            "fresh_pair_identity": "cde10e53396c98b521efaa242889901af495694377b4ed20fe6013f0ddc9773d",
            "rows": 4860,
            "episodes": 45,
            "source_groups": 9,
        },
        "final_decision": final["scientific_decision"],
        "closure_status": final_status,
        "metric_scope_clarification": {
            "classification": metric_audit["classification"],
            "mechanism": metric_audit["mechanism"],
            "changes_historical_decision": False,
            "path": f"{LOGICAL_ROOT}/{OUTPUT_RELATIVE.as_posix()}/metric_scope_audit.json",
        },
        "paper_delta_v2": {
            "markdown": f"{LOGICAL_ROOT}/{OUTPUT_RELATIVE.as_posix()}/paper_delta_v2.md",
            "json": f"{LOGICAL_ROOT}/{OUTPUT_RELATIVE.as_posix()}/paper_delta_v2.json",
        },
        "publication_draft": "docs/research/s4_3_pi2b_teacher_publication_closure.md",
        "new_figure_paths": plot_paths,
        "frozen_artifact_hashes": EXPECTED_FROZEN_HASHES,
        "protected_paper_core_sha256": EXPECTED_FROZEN_HASHES["PAPER_CORE_BASE.md"],
        "track_a_integration_status": "NOT_AUTHORIZED",
        "track_a_performance_read": False,
        "recommended_next_integration_point": ["WAIT_FOR_TRACK_A_FINAL", "THEN_EVIDENCE_INTEGRATION"],
        "push_pr_merge_tag_release": "NOT_PERFORMED",
    }
    atomic_json(output / "integration_handoff_v2.json", handoff)
    closure = {
        "schema": "tactile3d-unit.s4-3-pi2b-teacher-publication-closure.v2",
        "status": final_status,
        "historical_status": final["status"],
        "scientific_decision": final["scientific_decision"],
        "scientific_decision_modified": False,
        "metric_scope_classification": metric_audit["classification"],
        "integrity": integrity["status"],
        "regression": args.regression_status,
        "privacy": args.privacy_status,
        "working_tree": args.working_tree_status,
        "final_closure_commit": args.final_commit,
        "teacher_retraining": False,
        "teacher_inference": False,
        "policy_training_or_rollouts": False,
        "gpu_scientific_jobs": 0,
        "track_a_performance_read": False,
        "main_paper_core_edited": False,
        "historical_artifacts_modified": False,
    }
    atomic_json(output / "closure_result.json", closure)
    print(json.dumps({"status": final_status, "output": f"{LOGICAL_ROOT}/{OUTPUT_RELATIVE.as_posix()}", "plots": len(plot_hashes)}, indent=2))


if __name__ == "__main__":
    main()
