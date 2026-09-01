#!/usr/bin/env python3
"""Audit S4.2 region mapping and train/validation tactile coverage."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_2_dataset import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    load_episode,
    references_for_split,
)

PAIR_ROOT = ROOT / ".local/cache/simulation/s4_2/pairs"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_2r"
REGION_CONFIG = ROOT / "configs/simulation/s4_2_dexjoco_contact_regions.json"
ORIGINAL_EVALUATION = ROOT / ".local/artifacts/simulation/s4_2/teacher_evaluation.json"
SPLITS = ("train", "validation")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def force_entropy(values: list[float], bins: int = 32) -> dict[str, Any]:
    positive = np.asarray([value for value in values if value > 0], dtype=np.float64)
    if not len(positive):
        return {"positive_samples": 0, "bits": 0.0, "normalized": 0.0}
    edges = np.linspace(float(positive.min()), float(positive.max()) + 1e-12, bins + 1)
    counts = np.histogram(positive, bins=edges)[0].astype(np.float64)
    probabilities = counts[counts > 0] / counts.sum()
    bits = float(-np.sum(probabilities * np.log2(probabilities)))
    return {
        "positive_samples": int(len(positive)),
        "bits": bits,
        "normalized": bits / np.log2(bins),
        "minimum": float(positive.min()),
        "maximum": float(positive.max()),
    }


def coverage_counts(pair_root: Path) -> tuple[dict[str, Any], float]:
    raw = {split: load_npz(pair_root / f"{split}.npz") for split in SPLITS}
    threshold = float(np.quantile(raw["train"]["force_delta_abs"], 0.70))
    result: dict[str, Any] = {}
    for split in SPLITS:
        rows = {}
        for task in sorted(np.unique(raw[split]["task"])):
            mask = raw[split]["task"] == task
            transitions = raw[split]["contact_transition"][mask]
            dynamic_mask = mask & (raw[split]["force_delta_abs"] > threshold)
            boundary_mask = mask & np.isin(raw[split]["contact_transition"], [1, 2])
            rows[str(task)] = {
                "anchors": int(mask.sum()),
                "dynamic_q70": int(dynamic_mask.sum()),
                "dynamic_q70_source_groups": int(
                    len(np.unique(raw[split]["source_trajectory_id"][dynamic_mask]))
                ),
                "boundary": int(np.sum(np.isin(transitions, [1, 2]))),
                "boundary_source_groups": int(
                    len(np.unique(raw[split]["source_trajectory_id"][boundary_mask]))
                ),
                "free_to_free": int(np.sum(transitions == 0)),
                "free_to_contact": int(np.sum(transitions == 1)),
                "contact_to_free": int(np.sum(transitions == 2)),
                "contact_to_contact": int(np.sum(transitions == 3)),
            }
        result[split] = rows
    return result, threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--pair-root", type=Path, default=PAIR_ROOT)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(REGION_CONFIG.read_text(encoding="utf-8"))
    region_names = [row["region_name"] for row in config["regions"]]
    task_names = sorted(config["tasks"])
    activity = {
        split: {task: np.zeros(5, dtype=np.int64) for task in task_names}
        for split in SPLITS
    }
    frames = {
        split: {task: 0 for task in task_names}
        for split in SPLITS
    }
    force_values: dict[str, dict[str, list[list[float]]]] = {
        split: {task: [[] for _ in range(5)] for task in task_names}
        for split in SPLITS
    }
    metadata_region_audits = []
    for split in SPLITS:
        for reference in references_for_split(split, args.dataset):
            values = load_episode(reference)
            tactile = values["sim_tactile"].reshape(-1, 5, 6)
            activity[split][reference.task] += np.sum(tactile[:, :, 0] > 0.5, axis=0)
            frames[split][reference.task] += len(tactile)
            for region in range(5):
                force_values[split][reference.task][region].extend(
                    tactile[:, region, 1].astype(float).tolist()
                )
            metadata_region_audits.append(reference.metadata["region_audit"])

    thumb_path = args.artifacts / "thumb_contact_audit.json"
    if not thumb_path.is_file():
        raise RuntimeError("run audit_s4_2r_thumb_contacts.py before the region coverage audit")
    thumb = json.loads(thumb_path.read_text(encoding="utf-8"))
    teacher = json.loads(ORIGINAL_EVALUATION.read_text(encoding="utf-8"))
    diversity, dynamic_threshold = coverage_counts(args.pair_root)
    overall_train = sum((activity["train"][task] for task in task_names), np.zeros(5, dtype=int))
    overall_validation = sum(
        (activity["validation"][task] for task in task_names), np.zeros(5, dtype=int)
    )
    meaningful = (overall_train > 0) & (overall_validation > 0)
    meaningful_count = int(meaningful.sum())
    mapping_metadata_pass = all(
        row["status"] == "PASS"
        and not row["unmapped_relevant_geoms"]
        and not row["overlapping_assignments"]
        and not row["missing_object_bodies"]
        for row in metadata_region_audits
    )
    runtime_thumb_geoms = {
        row["task"]: row["resolved_thumb_geoms"] for row in thumb["tasks"]
    }
    thumb_bodies = next(
        row["body_names"] for row in config["regions"] if row["region_name"] == "right_thumb"
    )
    thumb_bodies_resolved = all(
        {geom["body_name"] for geom in geoms} == set(thumb_bodies)
        for geoms in runtime_thumb_geoms.values()
    )
    task_diversity_pass = all(
        diversity[split][task][kind] >= 50
        and diversity[split][task][f"{kind}_source_groups"] >= 2
        for split in SPLITS
        for task in task_names
        for kind in ("dynamic_q70", "boundary")
    )
    semantic_gate_names = (
        "overall_baseline_improvement",
        "dynamic_improvement",
        "temporal_value",
        "contact_probe",
        "force_probe",
        "trend_probe",
    )
    semantic_temporal_pass = all(teacher["gates"][name] for name in semantic_gate_names)
    mapping_bug = bool(thumb["mapping_bug"])
    gates = {
        "metadata_region_mapping": mapping_metadata_pass,
        "thumb_bodies_resolve_to_collision_geoms": thumb_bodies_resolved,
        "no_duplicate_region_assignment": all(
            not row["overlapping_assignments"] for row in metadata_region_audits
        ),
        "no_raw_thumb_object_contact_hidden_by_zero_tactile": not mapping_bug,
        "at_least_four_of_five_regions_active_train_and_validation": meaningful_count >= 4,
        "three_task_dynamic_and_boundary_diversity": task_diversity_pass,
        "teacher_semantic_and_temporal_gates_strong": semantic_temporal_pass,
    }
    dataset_low_coverage = not (
        gates["at_least_four_of_five_regions_active_train_and_validation"]
        and gates["three_task_dynamic_and_boundary_diversity"]
        and gates["teacher_semantic_and_temporal_gates_strong"]
    )
    if mapping_bug:
        decision = "S4_2R_CONTACT_MAPPING_FAIL"
        status = "FAIL"
    elif dataset_low_coverage:
        decision = "S4_2R_DATA_COVERAGE_FAIL"
        status = "FAIL"
    else:
        decision = "THUMB_INACTIVITY_IS_DATA_COVERAGE"
        status = "PASS_WITH_WARNINGS"
    activity_rows = {
        split: {
            task: {
                region_names[index]: {
                    "active_steps": int(activity[split][task][index]),
                    "frames": frames[split][task],
                    "occupancy_rate": float(
                        activity[split][task][index] / max(frames[split][task], 1)
                    ),
                    "force_entropy": force_entropy(force_values[split][task][index]),
                }
                for index in range(5)
            }
            for task in task_names
        }
        for split in SPLITS
    }
    result = {
        "schema": "tactile3d-unit.s4-2r-region-coverage-audit.v1",
        "status": status,
        "decision": decision,
        "regions": region_names,
        "thumb_bodies": thumb_bodies,
        "runtime_resolved_thumb_geoms": runtime_thumb_geoms,
        "metadata_region_audits_checked": len(metadata_region_audits),
        "mapping_bug": mapping_bug,
        "raw_mujoco_thumb_object_contacts": thumb["raw_thumb_object_contact_steps"],
        "tactile_thumb_activity_replay": thumb["tactile_thumb_active_steps"],
        "meaningful_active_regions": [
            region_names[index] for index in np.flatnonzero(meaningful)
        ],
        "meaningful_active_region_count": meaningful_count,
        "overall_activity_steps": {
            "train": dict(zip(region_names, overall_train.astype(int).tolist())),
            "validation": dict(zip(region_names, overall_validation.astype(int).tolist())),
        },
        "activity_by_task_and_split": activity_rows,
        "dynamic_q70_threshold_newton": dynamic_threshold,
        "task_contact_regime_diversity": diversity,
        "coverage_diversity_definition": {
            "minimum_rows_per_task_split_and_regime": 50,
            "minimum_independent_source_groups": 2,
            "regimes": ["dynamic_q70", "boundary"],
            "purpose": "exclude sparse or single-source regime support",
        },
        "original_teacher_support": {
            "contact_macro_f1": teacher["probes"]["contact"]["macro_f1"],
            "force_r2": teacher["probes"]["force"]["r2"],
            "trend_macro_f1": teacher["probes"]["trend"]["macro_f1"],
            "temporal_controls": teacher["temporal_controls"],
        },
        "gates": gates,
        "warnings": ["THUMB_ZERO_ACTIVITY"],
        "test_loaded": False,
        "test_metadata_loaded": False,
        "selection_uses_test": False,
    }
    destination = args.artifacts / "region_coverage_audit.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    matrix = np.asarray(
        [
            [
                activity_rows[split][task][region]["occupancy_rate"]
                for region in region_names
            ]
            for split in SPLITS
            for task in task_names
        ]
    )
    labels = [f"{split}:{task}" for split in SPLITS for task in task_names]
    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(5), region_names, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_title("Per-region tactile occupancy coverage")
    fig.colorbar(image, ax=ax, label="Occupancy rate")
    fig.tight_layout()
    fig.savefig(args.artifacts / "per_region_activity_heatmap.png", dpi=180)
    plt.close(fig)

    thumb_labels = [row["task"] for row in thumb["tasks"]]
    all_contacts = [
        row["raw_thumb_contact_steps_all_counterparts"] for row in thumb["tasks"]
    ]
    object_contacts = [row["raw_thumb_object_contact_steps"] for row in thumb["tasks"]]
    tactile_contacts = [row["tactile_thumb_active_steps"] for row in thumb["tasks"]]
    x = np.arange(len(thumb_labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.25, all_contacts, 0.25, label="raw thumb/all")
    ax.bar(x, object_contacts, 0.25, label="raw thumb/object")
    ax.bar(x + 0.25, tactile_contacts, 0.25, label="frozen tactile thumb")
    ax.set_xticks(x, thumb_labels)
    ax.set(ylabel="Contact-active steps", title="Deterministic thumb contact audit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.artifacts / "thumb_contact_audit.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2))
    if status == "FAIL":
        raise SystemExit(decision)


if __name__ == "__main__":
    main()
