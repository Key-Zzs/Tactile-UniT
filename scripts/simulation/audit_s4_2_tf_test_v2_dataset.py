#!/usr/bin/env python3
"""Integrity-audit FORMAL_TEST_V2 after preserving audit-attempt-1's harness failure."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_2_dataset import (  # noqa: E402
    build_pair_arrays,
    episode_references,
    load_episode,
    sha256_file,
)
from scripts.simulation.audit_s4_2_dataset import rgb_tree_sha256  # noqa: E402
from scripts.simulation.manage_s4_2_tf import (  # noqa: E402
    ARTIFACT_ROOT,
    CONFIG,
    EVALUATOR_FREEZE,
    FORMAL_CACHE,
    PRETEST_V2,
    PREREGISTRATION,
    V1_ARTIFACT_ROOT,
    V2_CACHE,
    V2_DATASET,
    atomic_json,
    current_checkpoint_hashes,
    digest_strings,
    exact_v2_identities,
    load_json,
    verify_frozen_implementations,
)
from scripts.simulation.run_s4_2_8_locked_test import load_npz  # noqa: E402

ATTEMPT1 = ARTIFACT_ROOT / "test_v2_dataset_integrity_attempt1_structural_fail.json"
OUTPUT = ARTIFACT_ROOT / "test_v2_dataset_integrity.json"


def main() -> None:
    config = load_json(CONFIG)
    preregistration = load_json(PREREGISTRATION)
    attempt1 = load_json(ATTEMPT1)
    expected_attempt1_failures = {
        f"{episode}:control_steps" for episode in preregistration["episode_ids"]
    }
    if attempt1["status"] != "FAIL" or set(attempt1["failures"]) != expected_attempt1_failures:
        raise RuntimeError("unexpected first TEST_V2 integrity-audit failure")
    if any(
        attempt1[name]
        for name in ("pair_id_overlap_count",)
    ) or any(attempt1["overlap_with_prior"].values()):
        raise RuntimeError("first integrity attempt found a substantive dataset overlap")
    verify_frozen_implementations(preregistration["evaluator_implementation_sha256"])
    if current_checkpoint_hashes() != preregistration["checkpoint_sha256"]:
        raise RuntimeError("checkpoint changed before corrected TEST_V2 integrity audit")
    if V2_CACHE.exists() or PRETEST_V2.exists():
        raise RuntimeError("TEST_V2 performance access/pretest exists before corrected integrity audit")

    expected_episodes, expected_groups, expected_seeds = exact_v2_identities(config)
    expected_preregistration_sha = sha256_file(PREREGISTRATION)
    references = list(episode_references(V2_DATASET))
    failures: list[str] = []
    actual_episodes, actual_groups, actual_seeds = set(), set(), set()
    task_counts: Counter[str] = Counter()
    for reference in references:
        actual_episodes.add(reference.episode_id)
        actual_groups.add((reference.task, reference.source_trajectory_id))
        actual_seeds.add(int(reference.metadata["seed"]))
        task_counts[reference.task] += 1
        if reference.split != "test" or reference.metadata.get("formal_test_name") != "FORMAL_TEST_V2":
            failures.append(f"{reference.episode_id}:test_identity")
        if reference.metadata.get("generation_preregistration_sha256") != expected_preregistration_sha:
            failures.append(f"{reference.episode_id}:generation_preregistration")
        manifest = load_json(reference.directory / "metadata.json")
        if sha256_file(reference.directory / "steps.npz") != manifest["checksums"]["steps_npz_sha256"]:
            failures.append(f"{reference.episode_id}:steps_checksum")
        if rgb_tree_sha256(reference.directory / "frames") != manifest["checksums"]["rgb_tree_sha256"]:
            failures.append(f"{reference.episode_id}:rgb_checksum")
        values = load_episode(reference)
        length = len(values["control_step"])
        expected_shapes = {
            "policy_action": (160, 22),
            "env_action": (160, 23),
            "sim_tactile": (160, 30),
            "contact_count_by_region": (160, 5),
            "rgb_reference": (160,),
        }
        for name, shape in expected_shapes.items():
            if values[name].shape != shape:
                failures.append(f"{reference.episode_id}:{name}_shape")
        if values["proprio"].shape[0] != 160 or values["proprio"].shape[1] < 23:
            failures.append(f"{reference.episode_id}:proprio_shape")
        if any(
            not np.isfinite(values[name]).all()
            for name in ("timestamp_sec", "proprio", "policy_action", "env_action", "sim_tactile")
        ):
            failures.append(f"{reference.episode_id}:nonfinite")
        if not np.array_equal(np.diff(values["control_step"]), np.ones(length - 1)):
            failures.append(f"{reference.episode_id}:control_step_contiguity")
        if not np.allclose(np.diff(values["timestamp_sec"]), 0.02, atol=1e-9):
            failures.append(f"{reference.episode_id}:timestamps")
        for rgb_reference in values["rgb_reference"]:
            image = cv2.imread(str(reference.directory / str(rgb_reference)))
            if image is None or image.shape != (640, 640, 3):
                failures.append(f"{reference.episode_id}:rgb_decode")
                break
    if actual_episodes != expected_episodes:
        failures.append("episode_identity_set")
    if actual_groups != expected_groups:
        failures.append("source_group_identity_set")
    if actual_seeds != expected_seeds:
        failures.append("episode_seed_set")

    original_refs = list(episode_references(ROOT / ".local/datasets/simulation/s4_2"))
    original_episodes = {value.episode_id for value in original_refs}
    original_groups = {(value.task, value.source_trajectory_id) for value in original_refs}
    original_seeds = {int(value.metadata["seed"]) for value in original_refs}
    overlaps = {
        "episode_ids": sorted(actual_episodes & original_episodes),
        "source_groups": sorted([list(value) for value in actual_groups & original_groups]),
        "episode_seeds": sorted(actual_seeds & original_seeds),
    }
    if any(overlaps.values()):
        failures.append("overlap_with_train_dev_validation_or_test_v1")

    pairs = build_pair_arrays("test", V2_DATASET, purpose="dataset_quality")
    prior_pair_ids: set[str] = set()
    for path in (
        FORMAL_CACHE / "paired_train.npz",
        FORMAL_CACHE / "paired_validation.npz",
        FORMAL_CACHE / "paired_test.npz",
    ):
        prior_pair_ids |= set(load_npz(path)["pair_id"].tolist())
    pair_overlap = set(pairs["pair_id"].tolist()) & prior_pair_ids
    if pair_overlap:
        failures.append("pair_identity_overlap")
    transition_exact = bool(np.all(pairs["future_step"] - pairs["anchor_step"] == 27))
    if not transition_exact:
        failures.append("transition_offset")
    protocol = config["formal_test_v2"]
    counts_pass = (
        len(references) == protocol["episode_count"]
        and len(pairs["pair_id"]) == protocol["pair_count"]
        and len(actual_groups) == protocol["source_group_count"]
        and task_counts == Counter({task: 15 for task in protocol["tasks"]})
    )
    if not counts_pass:
        failures.append("preregistered_counts")
    integrity = {
        "schema": "tactile3d-unit.s4-2-tf-test-v2-dataset-integrity.v2",
        "status": "PASS" if not failures else "FAIL",
        "test_name": "FORMAL_TEST_V2",
        "audit_attempt_1": "STRUCTURAL_AUDIT_HARNESS_FAIL_PRESERVED",
        "audit_attempt_1_sha256": sha256_file(ATTEMPT1),
        "audit_remediation": "require contiguous +1 control steps; do not require a zero origin",
        "dataset_bytes_changed_after_generation": False,
        "episodes": len(references),
        "pairs": len(pairs["pair_id"]),
        "source_groups": len(actual_groups),
        "task_episode_counts": dict(sorted(task_counts.items())),
        "unique_episode_ids": len(actual_episodes),
        "unique_episode_seeds": len(actual_seeds),
        "unique_pair_ids": len(set(pairs["pair_id"].tolist())),
        "pair_identity_sha256": digest_strings(pairs["pair_id"]),
        "transition_offset_exact": transition_exact,
        "overlap_with_prior": overlaps,
        "pair_id_overlap_count": len(pair_overlap),
        "checksums_schema_finite_timing_rgb": "PASS" if not failures else "FAIL",
        "model_performance_metrics_loaded": False,
        "training_performed": False,
        "selection_performed": False,
        "failures": failures,
    }
    atomic_json(OUTPUT, integrity)
    if failures:
        raise SystemExit("S4_2_TF_TEST_V2_DATASET_INTEGRITY_FAIL")

    audit_relative = str(Path(__file__).resolve().relative_to(ROOT))
    implementations = dict(preregistration["evaluator_implementation_sha256"])
    implementations[audit_relative] = sha256_file(Path(__file__).resolve())
    pretest = {
        "schema": "tactile3d-unit.s4-2-tf-pretest-v2-freeze.v1",
        "stage": "S4.2-TF_PRETEST_V2_FREEZE",
        "status": "PASS",
        "test_name": "FORMAL_TEST_V2",
        "dataset_integrity_sha256": sha256_file(OUTPUT),
        "dataset_pair_identity_sha256": integrity["pair_identity_sha256"],
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "evaluator_freeze_sha256": sha256_file(EVALUATOR_FREEZE),
        "implementation_sha256": implementations,
        "checkpoint_sha256": preregistration["checkpoint_sha256"],
        "metric_thresholds_digest": preregistration["metric_thresholds_digest"],
        "selection_cache_sha256": load_json(
            V1_ARTIFACT_ROOT / "pretest_freeze.json"
        )["selection_cache_sha256"],
        "vision_checkpoint_file_sha256": load_json(
            V1_ARTIFACT_ROOT / "pretest_freeze.json"
        )["vision_checkpoint_file_sha256"],
        "training_complete": True,
        "selection_complete": True,
        "test_loaded": False,
        "formal_test_v2_model_metrics_loaded": False,
        "dataset_integrity_only_access": True,
        "locked_test_runs_allowed": 1,
        "deterministic_repeat_runs_allowed": 1,
        "post_test_changes_forbidden": config["post_test_v2_changes_forbidden"],
    }
    atomic_json(PRETEST_V2, pretest)
    print(json.dumps({"integrity": integrity, "pretest_v2": pretest}, indent=2))


if __name__ == "__main__":
    main()
