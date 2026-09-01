#!/usr/bin/env python3
"""Freeze, integrity-audit, and finalize S4.2-TF without training or selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

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
from scripts.simulation.run_s4_2_8_locked_test import (  # noqa: E402
    canonical_digest,
    load_npz,
    validate_evaluator_schema,
)

CONFIG = ROOT / "configs/simulation/s4_2_tf_locked_test_v2.json"
FORMAL_CONFIG = ROOT / "configs/simulation/s4_2_formal_downstream.json"
GENERATOR = ROOT / "scripts/simulation/generate_s4_2_test_v2.py"
EVALUATOR = ROOT / "scripts/simulation/run_s4_2_8_locked_test.py"
V2_RUNNER = ROOT / "scripts/simulation/run_s4_2_tf_locked_test_v2.py"
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_2_tf"
V1_ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_2_formal"
FORMAL_CACHE = ROOT / ".local/cache/simulation/s4_2_formal"
V2_CACHE = ROOT / ".local/cache/simulation/s4_2_tf/paired_test_v2.npz"
V2_DATASET = ROOT / ".local/datasets/simulation/s4_2_test_v2"
PREREGISTRATION = ARTIFACT_ROOT / "formal_test_v2_preregistration.json"
EVALUATOR_FREEZE = ARTIFACT_ROOT / "evaluator_remediation_freeze.json"
PRETEST_V2 = ARTIFACT_ROOT / "pretest_v2_freeze.json"

CHECKPOINT_PATHS = {
    "contact_state": ROOT / ".local/experiments/simulation/s4_2r/contact_state/accepted.pt",
    "contact_C3": ROOT / ".local/experiments/simulation/s4_2r/contact_dynamics/selected.pt",
    "action": ROOT / ".local/experiments/simulation/s4_2_formal/action/selected.pt",
    "bridge": ROOT / ".local/experiments/simulation/s4_2_formal/bridge/selected.pt",
    "shared_private": ROOT / ".local/experiments/simulation/s4_2_formal/s4_2_6/shared_private.pt",
    "rejected_contact_rq": ROOT / ".local/experiments/simulation/s4_2_formal/s4_2_6/contact_rq_selected.pt",
    "conditional_A_plus_H": ROOT / ".local/experiments/simulation/s4_2_formal/s4_2_7/A_plus_H.pt",
    "conditional_V_plus_A_plus_H": ROOT / ".local/experiments/simulation/s4_2_formal/s4_2_7/V_plus_A_plus_H.pt",
    "conditional_V_plus_A_missing_H": ROOT / ".local/experiments/simulation/s4_2_formal/s4_2_7/V_plus_A_missing_H.pt",
    "conditional_A_only_missing_H": ROOT / ".local/experiments/simulation/s4_2_formal/s4_2_7/A_only_missing_H.pt",
    "uncertainty_full": ROOT / ".local/experiments/simulation/s4_2_formal/s4_2_7/uncertainty_full.pt",
    "uncertainty_missing_H": ROOT / ".local/experiments/simulation/s4_2_formal/s4_2_7/uncertainty_missing_H.pt",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest_strings(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values.tolist():
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def current_checkpoint_hashes() -> dict[str, str]:
    return {name: sha256_file(path) for name, path in CHECKPOINT_PATHS.items()}


def git_identity() -> dict[str, str]:
    return {
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }


def implementation_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in (CONFIG, GENERATOR, EVALUATOR, V2_RUNNER, Path(__file__).resolve())
    }


def verify_frozen_implementations(expected: dict[str, str]) -> None:
    for relative, digest in expected.items():
        if sha256_file(ROOT / relative) != digest:
            raise RuntimeError(f"frozen S4.2-TF implementation changed: {relative}")


def exact_v2_identities(config: dict[str, Any]) -> tuple[set[str], set[tuple[str, str]], set[int]]:
    protocol = config["formal_test_v2"]
    episodes, groups, seeds = set(), set(), set()
    for task in protocol["tasks"]:
        for group in protocol["source_group_indices"]:
            groups.add((task, protocol["source_trajectory_id"].format(task=task, group=group)))
            for perturbation in range(protocol["perturbations_per_group"]):
                episodes.add(
                    protocol["episode_id"].format(
                        task=task, group=group, perturbation=perturbation
                    )
                )
                seeds.add(int(protocol["seed_bases"][task] + 10 * group + perturbation))
    return episodes, groups, seeds


def freeze_evaluator() -> None:
    config = load_json(CONFIG)
    formal = load_json(FORMAL_CONFIG)
    if V2_DATASET.exists() or V2_CACHE.exists() or PRETEST_V2.exists():
        raise RuntimeError("FORMAL_TEST_V2 material exists before evaluator/preregistration freeze")
    if config["status"] != "PREREGISTERED_BEFORE_FORMAL_TEST_V2_GENERATION":
        raise RuntimeError("invalid S4.2-TF preregistration status")
    checkpoints = current_checkpoint_hashes()
    if checkpoints != config["frozen_checkpoints_sha256"]:
        raise RuntimeError("frozen checkpoint identity changed during evaluator remediation")

    train = load_npz(FORMAL_CACHE / "paired_train.npz")
    shared_train = load_npz(FORMAL_CACHE / "shared_train.npz")
    validation = load_npz(FORMAL_CACHE / "paired_validation.npz")
    shared_validation = load_npz(FORMAL_CACHE / "shared_validation.npz")
    raw_validation = load_npz(ROOT / ".local/cache/simulation/s4_2/pairs/validation.npz")
    if not np.array_equal(validation["pair_id"], raw_validation["pair_id"]):
        raise RuntimeError("formal/raw validation pair identity mismatch")
    synthetic_test = validation | shared_validation | {
        "current_state": raw_validation["current_state"],
        "action_chunk": raw_validation["action_chunk"],
    }
    schema_audit = validate_evaluator_schema(train, shared_train, synthetic_test)
    regression = subprocess.check_output(
        [sys.executable, "-m", "pytest", "-q", "tests/simulation/test_s4_2_formal.py"],
        cwd=ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    )
    if "failed" in regression.lower():
        raise RuntimeError("S4.2 evaluator regression failed")

    v1_cache = load_npz(FORMAL_CACHE / "paired_test.npz")
    v1_pair_digest = digest_strings(v1_cache["pair_id"])
    if v1_pair_digest != config["test_v1"]["pair_identity_sha256"]:
        raise RuntimeError("TEST_V1 pair identity changed")
    v1 = {
        "schema": "tactile3d-unit.s4-2-tf-test-v1-exposed.v1",
        "classification": "TEST_V1_EXPOSED",
        "structural_fail": "PRESERVED",
        "scientific_decision_eligible": False,
        "pairs": len(v1_cache["pair_id"]),
        "unique_pair_ids": len(set(v1_cache["pair_id"].tolist())),
        "pair_identity_sha256": v1_pair_digest,
        "paired_cache_sha256": sha256_file(FORMAL_CACHE / "paired_test.npz"),
        "original_failure_artifact_sha256": sha256_file(V1_ARTIFACT_ROOT / "locked_test_failure.json"),
        "model_performance_result_available": False,
        "post_fix_diagnostic_run": False,
    }
    if v1["paired_cache_sha256"] != config["test_v1"]["paired_cache_sha256"]:
        raise RuntimeError("TEST_V1 cache bytes changed")
    atomic_json(ARTIFACT_ROOT / "test_v1_exposed.json", v1)

    expected_thresholds = {
        "action": {
            key: formal["s4_2_4"]["gates"][key]
            for key in (
                "reversed_over_correct_mse_min",
                "shuffled_over_correct_mse_min",
                "different_episode_over_correct_mse_min",
            )
        },
        "bridge": {
            key: formal["s4_2_5"]["gates"][key]
            for key in (
                "paired_margin_ci_lower_min",
                "contact_retrieval_r10_chance_multiplier_min",
                "contact_semantic_retention_min",
                "force_semantic_retention_min",
            )
        },
        "shared_private": {
            key: formal["s4_2_6"]["gates"][key]
            for key in ("private_incremental_recovery_min", "shared_cross_modal_margin_min")
        },
        "conditional": {
            key: formal["s4_2_7"]["gates"][key]
            for key in (
                "full_over_A_plus_H_relative_mse_improvement_min",
                "missing_VA_over_A_relative_mse_improvement_min",
                "missing_contact_macro_f1_retention_min",
            )
        },
        "uncertainty": {
            "nll_improvement_over_constant_min": formal["s4_2_7"]["gates"][
                "uncertainty_nll_improvement_over_constant_min"
            ],
            "interval_90_coverage_min": formal["s4_2_7"]["gates"]["interval_90_coverage_min"],
            "interval_90_coverage_max": formal["s4_2_7"]["gates"]["interval_90_coverage_max"],
        },
    }
    if expected_thresholds != config["metric_thresholds"]:
        raise RuntimeError("TEST_V2 thresholds differ from the frozen formal thresholds")
    implementations = implementation_hashes()
    freeze = {
        "schema": "tactile3d-unit.s4-2-tf-evaluator-remediation-freeze.v1",
        "status": "PASS",
        "bug_repaired": "train['u_c'] -> shared_train['u_c']",
        "schema_audit": schema_audit,
        "dry_run_split": "FORMAL_VALIDATION_ONLY",
        "formal_test_metrics_read": False,
        "regression": "PASS",
        "regression_summary": regression.strip().splitlines()[-1],
        "implementation_sha256": implementations,
        "checkpoint_sha256": checkpoints,
        "training_performed": False,
        "selection_performed": False,
        "threshold_change": False,
        "normalization_change": False,
        "split_change": False,
        "bridge_retraining": False,
        "predictor_retraining": False,
        "uncertainty_retraining": False,
        **git_identity(),
    }
    atomic_json(EVALUATOR_FREEZE, freeze)
    episodes, groups, seeds = exact_v2_identities(config)
    preregistration = {
        "schema": "tactile3d-unit.s4-2-tf-formal-test-v2-preregistration.v1",
        "status": "FROZEN_BEFORE_FORMAL_TEST_V2_GENERATION",
        "test_name": "FORMAL_TEST_V2",
        "config_sha256": sha256_file(CONFIG),
        "generation_implementation_sha256": sha256_file(GENERATOR),
        "evaluator_implementation_sha256": implementations,
        "evaluator_freeze_sha256": sha256_file(EVALUATOR_FREEZE),
        "checkpoint_sha256": checkpoints,
        "metric_thresholds": config["metric_thresholds"],
        "metric_thresholds_digest": canonical_digest(config["metric_thresholds"]),
        "tasks": config["formal_test_v2"]["tasks"],
        "episode_count": len(episodes),
        "pair_count": config["formal_test_v2"]["pair_count"],
        "source_groups": sorted([list(value) for value in groups]),
        "episode_ids": sorted(episodes),
        "episode_seeds": sorted(seeds),
        "generation_protocol": config["formal_test_v2"],
        "forbidden_overlap": config["formal_test_v2"]["required_disjoint_from"],
        "model_performance_metrics_loaded": False,
        "training_performed": False,
        "selection_performed": False,
        **git_identity(),
    }
    atomic_json(PREREGISTRATION, preregistration)
    print(json.dumps({"test_v1": v1, "evaluator": freeze, "v2": preregistration}, indent=2))


def audit_dataset() -> None:
    config = load_json(CONFIG)
    preregistration = load_json(PREREGISTRATION)
    freeze = load_json(EVALUATOR_FREEZE)
    verify_frozen_implementations(preregistration["evaluator_implementation_sha256"])
    if current_checkpoint_hashes() != preregistration["checkpoint_sha256"]:
        raise RuntimeError("checkpoint changed before TEST_V2 integrity audit")
    if V2_CACHE.exists():
        raise RuntimeError("TEST_V2 model-performance cache exists before pretest_v2_freeze")
    expected_episodes, expected_groups, expected_seeds = exact_v2_identities(config)
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
        if not np.array_equal(values["control_step"], np.arange(length)):
            failures.append(f"{reference.episode_id}:control_steps")
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
    pair_overlap = sorted(set(pairs["pair_id"].tolist()) & prior_pair_ids)
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
        "schema": "tactile3d-unit.s4-2-tf-test-v2-dataset-integrity.v1",
        "status": "PASS" if not failures else "FAIL",
        "test_name": "FORMAL_TEST_V2",
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
    atomic_json(ARTIFACT_ROOT / "test_v2_dataset_integrity.json", integrity)
    if failures:
        raise SystemExit("S4_2_TF_TEST_V2_DATASET_INTEGRITY_FAIL")
    pretest = {
        "schema": "tactile3d-unit.s4-2-tf-pretest-v2-freeze.v1",
        "stage": "S4.2-TF_PRETEST_V2_FREEZE",
        "status": "PASS",
        "test_name": "FORMAL_TEST_V2",
        "dataset_integrity_sha256": sha256_file(ARTIFACT_ROOT / "test_v2_dataset_integrity.json"),
        "dataset_pair_identity_sha256": integrity["pair_identity_sha256"],
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "evaluator_freeze_sha256": sha256_file(EVALUATOR_FREEZE),
        "implementation_sha256": preregistration["evaluator_implementation_sha256"],
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


def finalize() -> None:
    config = load_json(CONFIG)
    preregistration = load_json(PREREGISTRATION)
    pretest = load_json(PRETEST_V2)
    first = load_json(ARTIFACT_ROOT / "locked_test_v2.json")
    repeat = load_json(ARTIFACT_ROOT / "locked_test_v2_repeat.json")
    verify_frozen_implementations(pretest["implementation_sha256"])
    if current_checkpoint_hashes() != pretest["checkpoint_sha256"]:
        raise RuntimeError("checkpoint changed after FORMAL_TEST_V2")
    if repeat.get("deterministic_equal") is not True or repeat["metric_digest"] != first["metric_digest"]:
        raise RuntimeError("FORMAL_TEST_V2 deterministic equality failed")
    if first["test_name"] != "FORMAL_TEST_V2" or first["pairs"] != 4860:
        raise RuntimeError("FORMAL_TEST_V2 result identity mismatch")
    passed = first["overall"] == "PASS"
    decision = {
        "schema": "tactile3d-unit.s4-2-tf-final-decision.v1",
        "status": "COMPLETE",
        "test_v1": "TEST_V1_EXPOSED_STRUCTURAL_FAIL_PRESERVED",
        "test_v1_scientific_decision_eligible": False,
        "formal_test_v2": first["overall"],
        "formal_test_v2_metric_digest": first["metric_digest"],
        "deterministic_repeat": "PASS",
        "s4_2": "COMPLETE" if passed else "NOT_COMPLETE",
        "s4_3": "READY_WITH_WARNINGS" if passed else "NOT_READY",
        "warnings": (
            [
                "FROZEN_M3_ZERO_SHOT_TRANSFER_NOT_ESTABLISHED",
                "DEXJOCO_REQUIRES_SIM_SPECIFIC_CONTINUOUS_SHARED_MAPPING",
                "OOD_DYNAMICS_DEFERRED",
                "RIGHT_THUMB_INACTIVE_IN_FORMAL_SCRIPTED_CORPUS",
            ]
            if passed
            else ["FORMAL_TEST_V2_LOCKED_GATE_FAILURE"]
        ),
        "sections": {name: value["overall"] for name, value in first["sections"].items()},
        "historical_conclusions": config["scientific_boundary"],
        "training_after_test_v2": False,
        "checkpoint_change_after_test_v2": False,
        "bridge_change_after_test_v2": False,
        "predictor_change_after_test_v2": False,
        "uncertainty_change_after_test_v2": False,
        "threshold_change_after_test_v2": False,
        "normalization_change_after_test_v2": False,
        "split_change_after_test_v2": False,
        "locked_test_v2_runs": 1,
        "deterministic_repeat_runs": 1,
        "post_fix_test_v1_diagnostic_run": False,
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "pretest_v2_freeze_sha256": sha256_file(PRETEST_V2),
        "checkpoint_sha256": preregistration["checkpoint_sha256"],
    }
    atomic_json(ARTIFACT_ROOT / "final_decision.json", decision)
    acceptance = f"""# S4.2-TF Human Acceptance\n\n- TEST_V1: TEST_V1_EXPOSED; original STRUCTURAL_FAIL preserved; scientifically ineligible\n- Evaluator schema remediation and regression: PASS\n- FORMAL_TEST_V2 preregistration and dataset integrity: PASS\n- FORMAL_TEST_V2 frozen locked evaluation: {first['overall']}\n- Equality-only deterministic repeat: PASS (`{first['metric_digest']}`)\n- S4.2: **{decision['s4_2']}**\n- S4.3: **{decision['s4_3']}**\n\nNo training, selection, checkpoint, bridge, predictor, uncertainty, threshold, normalization, or split change occurred during or after TEST_V2.\n"""
    (ARTIFACT_ROOT / "HUMAN_ACCEPTANCE.md").write_text(acceptance, encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze-evaluator", "audit-dataset", "finalize"))
    args = parser.parse_args()
    {"freeze-evaluator": freeze_evaluator, "audit-dataset": audit_dataset, "finalize": finalize}[
        args.phase
    ]()


if __name__ == "__main__":
    main()
