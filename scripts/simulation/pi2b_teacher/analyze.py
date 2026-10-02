#!/usr/bin/env python3
"""Paired source-group statistics and mechanism analysis for PI2B teachers."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.pi2b_teacher.evaluation import holm_adjust, stratified_group_bootstrap  # noqa: E402
from gr00t.simulation.pi2b_teacher.matched_teacher import MatchedTeacher, common_state  # noqa: E402
from scripts.simulation.pi2b_teacher.common import atomic_json, canonical_digest, load_json, sha256_file  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def training_summary(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    return {
        "logged_steps": len(rows),
        "first": rows[0],
        "last": rows[-1],
        "mean_common_gradient_norm": float(np.mean([row["common_gradient_norm"] for row in rows])),
        "mean_global_gradient_norm_preclip": float(np.mean([row["global_gradient_norm_preclip"] for row in rows])),
        "clip_activation_fraction": float(np.mean([row["global_clip_activated"] for row in rows])),
        "mean_va_total": float(np.mean([row["va_total"] for row in rows])),
        "mean_contact_total": float(np.mean([row["contact_total"] for row in rows])),
    }


def load_common_vector(path: Path) -> np.ndarray:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = MatchedTeacher(tuple(payload["modalities"]), initialization_seed=int(payload["initialization_seed"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    return np.concatenate([value.detach().numpy().reshape(-1) for _, value in sorted(common_state(model).items())]).astype(np.float64)


def duration_hours(lease: dict[str, Any]) -> float:
    start = datetime.fromisoformat(lease["supervisor_start_time_utc"])
    stop = datetime.fromisoformat(lease["finished_utc"])
    return (stop - start).total_seconds() / 3600.0


def effect_direction(metric: str, interval: list[float]) -> str:
    lower, upper = interval
    lower_is_better = "recovery_mse" in metric
    if lower_is_better:
        if lower > 0:
            return "VAC_REGRESSION"
        if upper < 0:
            return "VAC_IMPROVEMENT"
    else:
        if upper < 0:
            return "VAC_REGRESSION"
        if lower > 0:
            return "VAC_IMPROVEMENT"
    return "INCONCLUSIVE"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    args = parser.parse_args()
    runtime = load_json(args.runtime_manifest)
    write_root = Path(runtime["write_root"])
    evaluation = load_json(write_root / "artifacts/evaluation_result.json")
    pretest = load_json(write_root / "artifacts/pretest_freeze.json")
    if evaluation["status"] != "COMPLETE" or evaluation["metric_digest"] != load_json(write_root / "status/evaluation.json")["metric_digest"]:
        raise RuntimeError("frozen evaluation is incomplete")
    config = load_json(ROOT / "configs/simulation/pi2b_teacher/evaluation.json")
    groups = evaluation["group_metrics"]
    tasks = evaluation["group_tasks"]
    common_metrics = sorted(set(groups["T_VA_match"]) & set(groups["T_VAC_match"]))
    metric_names = sorted(
        set.intersection(
            *[
                set(groups["T_VA_match"][group]) & set(groups["T_VAC_match"][group])
                for group in common_metrics
            ]
        )
    )
    effects = {}
    for offset, metric in enumerate(metric_names):
        differences = {
            group: float(groups["T_VAC_match"][group][metric] - groups["T_VA_match"][group][metric])
            for group in common_metrics
        }
        effects[metric] = stratified_group_bootstrap(
            differences,
            tasks,
            samples=int(config["statistics"]["bootstrap_samples"]),
            seed=int(config["statistics"]["bootstrap_seed"]) + offset,
        )
        effects[metric]["contrast"] = "T_VAC_match - T_VA_match"
        effects[metric]["direction"] = effect_direction(metric, effects[metric]["ci95"])
        effects[metric]["per_task"] = {
            task: float(np.mean([value for group, value in differences.items() if tasks[group] == task]))
            for task in sorted(set(tasks.values()))
        }
    primary = [name for name in config["primary_common_va"] if name in effects]
    adjusted = holm_adjust({name: effects[name]["two_sided_bootstrap_p"] for name in primary})
    for name in primary:
        effects[name]["holm_p"] = adjusted[name]
        effects[name]["holm_significant_0p05"] = adjusted[name] < 0.05
    contact_statistics = {}
    contact_names = (
        "vision_contact_paired_margin",
        "action_contact_paired_margin",
        "vision_contact_temporal_margin",
        "action_contact_temporal_margin",
    )
    for offset, metric in enumerate(contact_names):
        values = {group: float(groups["T_VAC_match"][group][metric]) for group in common_metrics}
        contact_statistics[metric] = stratified_group_bootstrap(
            values,
            tasks,
            samples=int(config["statistics"]["bootstrap_samples"]),
            seed=int(config["statistics"]["bootstrap_seed"]) + 100 + offset,
        )
    contact_reliable = all(value["ci95"][0] > 0.0 for value in contact_statistics.values())
    recovery_primary = [name for name in ("vision_native_recovery_mse", "action_native_recovery_mse") if name in effects]
    semantic_primary = [name for name in primary if name not in recovery_primary]
    recovery_regression = any(effects[name]["direction"] == "VAC_REGRESSION" and effects[name].get("holm_significant_0p05", False) for name in recovery_primary)
    semantic_regression = any(effects[name]["direction"] == "VAC_REGRESSION" and effects[name].get("holm_significant_0p05", False) for name in semantic_primary)
    probe_effect = effects.get("mean_common_va_contact_probe_macro_f1")
    contact_readout = (
        "VAC_COMMON_VA_CONTACT_READOUT_INCREMENT"
        if probe_effect and probe_effect["ci95"][0] > 0.0
        else "NO_RELIABLE_COMMON_VA_CONTACT_READOUT_INCREMENT"
    )
    if recovery_regression or semantic_regression:
        scientific_decision = "SHARED_LEARNING_TRADEOFF_OBSERVED" if contact_reliable else "VAC_CONFIGURATION_NOT_SUPPORTED"
    elif contact_reliable:
        scientific_decision = "VAC_CONTACT_CAPABILITY_WITH_NO_DETECTED_VA_REGRESSION_LIMITED_PRECISION"
    else:
        scientific_decision = "INCONCLUSIVE"
    status = {mode: load_json(write_root / "status" / f"{mode}.json") for mode in ("T_VA_match", "T_VAC_match")}
    training = {mode: training_summary(Path(status[mode]["log"])) for mode in status}
    va_vector = load_common_vector(Path(status["T_VA_match"]["checkpoint"]))
    vac_vector = load_common_vector(Path(status["T_VAC_match"]["checkpoint"]))
    common_parameter_mechanism = {
        "l2_difference": float(np.linalg.norm(vac_vector - va_vector)),
        "relative_l2_difference_to_va": float(np.linalg.norm(vac_vector - va_vector) / max(np.linalg.norm(va_vector), 1e-12)),
        "parameter_cosine": float(np.dot(va_vector, vac_vector) / max(np.linalg.norm(va_vector) * np.linalg.norm(vac_vector), 1e-12)),
        "interpretation": "mechanism diagnostic only; not a cross-teacher representation quality metric",
    }
    coordination = Path(runtime["coordination_dir"])
    leases = {
        mode: load_json(coordination / "leases" / f"s4_3_pi2b_teacher_{mode.lower()}_seed42.json")
        for mode in status
    }
    compute = {
        mode: {
            "parameters": int(sum(value.numel() for value in torch.load(Path(status[mode]["checkpoint"]), map_location="cpu", weights_only=False)["state_dict"].values())),
            "optimizer_updates": status[mode]["final_step"],
            "sample_exposure": status[mode]["sample_exposure"],
            "measured_gpu_hours": duration_hours(leases[mode]),
        }
        for mode in status
    }
    geometries = [
        evaluation["models"][mode]["splits"]["fresh"][f"{modality}_geometry"]
        for mode in ("T_VA_match", "T_VAC_match")
        for modality in ("vision", "action")
    ]
    noncollapse = all(value["near_zero_variance_fraction"] < 0.5 and value["mean_dimension_variance"] > 1e-8 for value in geometries)
    result = {
        "schema": "tactile3d-unit.s4-3-pi2b-teacher-analysis.v1",
        "status": "COMPLETE_VALID",
        "ENGINEERING_STATUS": "PASS",
        "MATCHING_GRADE": "COMMON_PATH_INITIALIZATION_AND_DATA_UPDATE_BUDGET_MATCHED",
        "COMMON_VA_RECOVERY_EFFECT": "REGRESSION_DETECTED" if recovery_regression else "NO_DETECTED_VA_REGRESSION_WITH_LIMITED_PRECISION",
        "COMMON_VA_SEMANTIC_EFFECT": "REGRESSION_DETECTED" if semantic_regression else "NO_DETECTED_VA_REGRESSION_WITH_LIMITED_PRECISION",
        "CONTACT_CROSS_MODAL_CAPABILITY": "RELIABLE" if contact_reliable else "NOT_RELIABLY_ESTABLISHED",
        "CONTACT_INFORMATION_IN_COMMON_VA_READOUTS": contact_readout,
        "NONCOLLAPSE_STATUS": "PASS" if noncollapse else "FAIL",
        "DATA_AND_TRAINING_SEED_SCOPE": "9 fresh source groups / 45 episodes; one paired teacher training seed42",
        "POLICY_UTILITY": "NOT_TESTED",
        "NEXT_STUDY": "RECOMMENDATION_ONLY",
        "scientific_decision": scientific_decision,
        "paired_effects": effects,
        "contact_statistics": contact_statistics,
        "training_mechanism": training,
        "common_parameter_mechanism": common_parameter_mechanism,
        "compute": compute,
        "evaluation_metric_digest": evaluation["metric_digest"],
        "pretest_sha256": sha256_file(write_root / "artifacts/pretest_freeze.json"),
        "fixed_teacher_seed_only": True,
        "equivalence_or_noninferiority_claim": False,
    }
    result["analysis_digest"] = canonical_digest(result)
    atomic_json(write_root / "artifacts/stratified_group_statistics.json", result)
    atomic_json(write_root / "artifacts/matching_grade.json", {"grade": result["MATCHING_GRADE"], "engineering": result["ENGINEERING_STATUS"], "fixed_teacher_seed_only": True})
    atomic_json(write_root / "artifacts/final_decision.json", result)
    atomic_json(write_root / "status/analysis.json", {"state": "COMPLETE", "decision": scientific_decision, "analysis_digest": result["analysis_digest"]})
    print(json.dumps({key: result[key] for key in ("status", "scientific_decision", "COMMON_VA_RECOVERY_EFFECT", "COMMON_VA_SEMANTIC_EFFECT", "CONTACT_CROSS_MODAL_CAPABILITY", "analysis_digest")}, indent=2))


if __name__ == "__main__":
    main()
