#!/usr/bin/env python3
"""Audit the formal S4.2 dataset without exposing test data to model selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_2_dataset import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    canonical_json_sha256,
    episode_references,
    load_episode,
    pair_id,
    sha256_file,
)

ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_2"


def rgb_tree_sha256(frames: Path) -> str:
    digest = hashlib.sha256()
    for frame in sorted(frames.glob("*.jpg")):
        digest.update(frame.name.encode("utf-8"))
        digest.update(hashlib.sha256(frame.read_bytes()).digest())
    return digest.hexdigest()


def scan(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    references = list(episode_references(root))
    episode_ids: set[str] = set()
    pair_ids: set[str] = set()
    groups_by_split: dict[str, set[tuple[str, str]]] = defaultdict(set)
    anchors_by_split: Counter[str] = Counter()
    episodes_by_split: Counter[str] = Counter()
    episodes_by_task: Counter[str] = Counter()
    regimes_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    task_regimes: dict[str, Counter[str]] = defaultdict(Counter)
    region_activity = np.zeros(5, dtype=np.int64)
    region_force: list[list[float]] = [[] for _ in range(5)]
    force_delta_by_split: dict[str, list[float]] = defaultdict(list)
    failures: list[str] = []
    episode_rows = []

    for reference in references:
        if reference.episode_id in episode_ids:
            failures.append(f"duplicate episode_id:{reference.episode_id}")
        episode_ids.add(reference.episode_id)
        episodes_by_split[reference.split] += 1
        episodes_by_task[reference.task] += 1
        groups_by_split[reference.split].add(
            (reference.task, reference.source_trajectory_id)
        )
        manifest = json.loads((reference.directory / "metadata.json").read_text())
        required_metadata = {
            "source_type",
            "source_trajectory_id",
            "task",
            "seed",
            "randomization_id",
            "randomization_parameters",
            "perturbation_id",
            "split",
        }
        missing = sorted(required_metadata - set(reference.metadata))
        if missing:
            failures.append(f"{reference.episode_id}:missing_metadata:{missing}")
        if sha256_file(reference.directory / "steps.npz") != manifest["checksums"][
            "steps_npz_sha256"
        ]:
            failures.append(f"{reference.episode_id}:numeric_checksum")
        if rgb_tree_sha256(reference.directory / "frames") != manifest["checksums"][
            "rgb_tree_sha256"
        ]:
            failures.append(f"{reference.episode_id}:rgb_checksum")
        values = load_episode(reference)
        length = len(values["control_step"])
        expected_shapes = {
            "policy_action": (length, 22),
            "env_action": (length, 23),
            "sim_tactile": (length, 30),
            "contact_count_by_region": (length, 5),
        }
        for name, shape in expected_shapes.items():
            if values[name].shape != shape:
                failures.append(f"{reference.episode_id}:{name}_shape:{values[name].shape}")
        if values["proprio"].ndim != 2 or values["proprio"].shape[0] != length or values[
            "proprio"
        ].shape[1] < 23:
            failures.append(f"{reference.episode_id}:proprio_shape")
        numeric = ("timestamp_sec", "proprio", "policy_action", "env_action", "sim_tactile")
        if any(not np.isfinite(values[name]).all() for name in numeric):
            failures.append(f"{reference.episode_id}:nonfinite")
        if not np.all(np.diff(values["timestamp_sec"]) > 0):
            failures.append(f"{reference.episode_id}:timestamp_nonmonotonic")
        if not np.allclose(np.diff(values["timestamp_sec"]), 0.02, atol=1e-9):
            failures.append(f"{reference.episode_id}:timestamp_spacing")
        if not np.array_equal(np.diff(values["control_step"]), np.ones(length - 1)):
            failures.append(f"{reference.episode_id}:control_discontinuity")
        if values["rgb_reference"].shape != (length,):
            failures.append(f"{reference.episode_id}:rgb_reference_shape")
        else:
            for rgb_reference in values["rgb_reference"]:
                decoded = cv2.imread(str(reference.directory / str(rgb_reference)))
                if decoded is None or decoded.shape != (640, 640, 3):
                    failures.append(f"{reference.episode_id}:rgb_decode:{rgb_reference}")
                    break

        tactile = values["sim_tactile"].reshape(length, 5, 6)
        region_activity += np.sum(tactile[:, :, 0] > 0, axis=0)
        for region in range(5):
            region_force[region].extend(tactile[:, region, 1].astype(float).tolist())
        occupancy = tactile[:, :, 0].sum(axis=1) > 0
        total_force = tactile[:, :, 1].sum(axis=1)
        episode_pairs = 0
        for anchor in range(25, length - 27):
            future = anchor + 27
            identifier = pair_id(
                reference.task, reference.episode_id, anchor, future, reference.split
            )
            if identifier in pair_ids:
                failures.append(f"duplicate_pair_id:{identifier}")
            pair_ids.add(identifier)
            current_contact = bool(occupancy[anchor])
            future_contact = bool(occupancy[future])
            regime = {
                (False, False): "free_to_free",
                (False, True): "free_to_contact",
                (True, True): "contact_to_contact",
                (True, False): "contact_to_free",
            }[(current_contact, future_contact)]
            regimes_by_split[reference.split][regime] += 1
            task_regimes[reference.task][regime] += 1
            force_delta_by_split[reference.split].append(
                abs(float(total_force[future] - total_force[anchor]))
            )
            episode_pairs += 1
        anchors_by_split[reference.split] += episode_pairs
        episode_rows.append(
            {
                "episode_id": reference.episode_id,
                "task": reference.task,
                "split": reference.split,
                "source_trajectory_id": reference.source_trajectory_id,
                "steps": length,
                "valid_anchors": episode_pairs,
                "steps_npz_sha256": manifest["checksums"]["steps_npz_sha256"],
                "rgb_tree_sha256": manifest["checksums"]["rgb_tree_sha256"],
            }
        )

    group_intersections = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = sorted(groups_by_split[left] & groups_by_split[right])
        group_intersections[f"{left}_vs_{right}"] = overlap
        if overlap:
            failures.append(f"source_group_leakage:{left}:{right}")

    train_delta = np.asarray(force_delta_by_split["train"], dtype=np.float64)
    thresholds = {f"q{int(q * 100)}": float(np.quantile(train_delta, q)) for q in (0.70, 0.75)}
    dynamic_fraction = {
        name: {
            split: (
                float(np.mean(np.asarray(force_delta_by_split[split]) > threshold))
                if force_delta_by_split[split]
                else 0.0
            )
            for split in ("train", "validation", "test")
        }
        for name, threshold in thresholds.items()
    }
    manifest = {
        "schema": "tactile3d-unit.s4-2-formal-dataset-manifest.v1",
        "root": str(root),
        "episodes": episode_rows,
        "counts": {
            "episodes": len(references),
            "episodes_by_task": dict(sorted(episodes_by_task.items())),
            "episodes_by_split": dict(sorted(episodes_by_split.items())),
            "anchors_by_split": dict(sorted(anchors_by_split.items())),
            "valid_anchors": int(sum(anchors_by_split.values())),
            "source_groups_by_split": {
                split: len(groups) for split, groups in sorted(groups_by_split.items())
            },
        },
        "source_group_intersections": group_intersections,
        "pair_id_count": len(pair_ids),
        "test_loaded": False,
        "test_quality_audit_only": True,
        "selection_uses_test": False,
        "training_uses_test": False,
    }
    quality = {
        "schema": "tactile3d-unit.s4-2-dataset-quality.v1",
        "regimes_by_split": {
            split: dict(sorted(counts.items())) for split, counts in regimes_by_split.items()
        },
        "regimes_by_task": {
            task: dict(sorted(counts.items())) for task, counts in task_regimes.items()
        },
        "region_activity_steps": region_activity.tolist(),
        "warnings": (
            [
                "INACTIVE_REGION_DIAGNOSTIC: "
                + ",".join(str(index) for index in np.flatnonzero(region_activity == 0))
            ]
            if np.any(region_activity == 0)
            else []
        ),
        "region_normal_force": [
            {
                "mean": float(np.mean(values)),
                "q95": float(np.quantile(values, 0.95)),
                "maximum": float(np.max(values)),
            }
            for values in region_force
        ],
        "train_only_dynamic_threshold_candidates": thresholds,
        "dynamic_fraction_candidates": dynamic_fraction,
        "ood_dynamics": "OOD_DYNAMICS_DEFERRED",
        "failures": failures,
    }
    gates = {
        "tasks_at_least_2": len(episodes_by_task) >= 2,
        "episodes_at_least_180": len(references) >= 180,
        "each_task_at_least_60": all(count >= 60 for count in episodes_by_task.values()),
        "anchors_at_least_20000": sum(anchors_by_split.values()) >= 20000,
        "train_anchors_at_least_12000": anchors_by_split["train"] >= 12000,
        "validation_anchors_at_least_2000": anchors_by_split["validation"] >= 2000,
        "test_anchors_at_least_2000": anchors_by_split["test"] >= 2000,
        "source_group_split": not any(group_intersections.values()),
        "test_free_to_contact_at_least_100": regimes_by_split["test"]["free_to_contact"] >= 100,
        "test_contact_to_free_at_least_100": regimes_by_split["test"]["contact_to_free"] >= 100,
        "dynamic_at_least_10_percent": all(
            dynamic_fraction[name][split] >= 0.10
            for name in dynamic_fraction
            for split in ("train", "validation", "test")
        ),
        "schema_and_integrity": not failures,
    }
    quality["gates"] = gates
    quality["status"] = "PASS" if all(gates.values()) else "FAIL"
    return manifest, quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, quality = scan(args.root)
    rebuilt_manifest, rebuilt_quality = scan(args.root)
    rebuild = {
        "manifest_sha256_first": canonical_json_sha256(manifest),
        "manifest_sha256_second": canonical_json_sha256(rebuilt_manifest),
        "quality_sha256_first": canonical_json_sha256(quality),
        "quality_sha256_second": canonical_json_sha256(rebuilt_quality),
    }
    rebuild["status"] = "PASS" if len(set(rebuild.values())) == 2 else "FAIL"
    manifest["canonical_sha256"] = rebuild["manifest_sha256_first"]
    quality["deterministic_manifest_rebuild"] = rebuild
    if rebuild["status"] != "PASS":
        quality["status"] = "FAIL"
    args.artifacts.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("dataset_manifest.json", manifest),
        ("dataset_quality.json", quality),
    ):
        (args.artifacts / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({"manifest": manifest["counts"], "quality": quality}, indent=2))
    if quality["status"] != "PASS":
        raise SystemExit("S4_2_DATASET_CONTRACT_FAIL")


if __name__ == "__main__":
    main()
