#!/usr/bin/env python3
"""Exact-seed deterministic replay spot-check for two successes per task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_3_pd import TASKS  # noqa: E402
from scripts.simulation.audit_s4_3_pd_pilot import write_json  # noqa: E402
from scripts.simulation.generate_s4_3_policy_expert_data import (  # noqa: E402
    ARTIFACT_ROOT,
    FORMAL_ROOT,
    OFFICIAL_ROOT,
    formal_attempt_manifest,
    run_attempt,
    verify_frozen_protocol,
)

TMP_ROOT = ROOT / ".local/tmp/simulation/s4_3_pd/determinism"
TASK_RESULT_ROOT = ARTIFACT_ROOT / "determinism"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_attempts(original: Path, replay: Path) -> dict[str, Any]:
    import cv2

    original_metadata = read_json(original / "metadata.json")
    replay_metadata = read_json(replay / "metadata.json")
    with np.load(original / "steps.npz", allow_pickle=False) as source:
        original_arrays = {name: source[name] for name in source.files}
    with np.load(replay / "steps.npz", allow_pickle=False) as source:
        replay_arrays = {name: source[name] for name in source.files}
    same_fields = set(original_arrays) == set(replay_arrays)
    field_equality = {
        name: bool(np.array_equal(original_arrays[name], replay_arrays[name]))
        for name in sorted(set(original_arrays) & set(replay_arrays))
    }
    numeric_checksum_equality = (
        original_metadata["checksums"]["steps_npz_sha256"]
        == replay_metadata["checksums"]["steps_npz_sha256"]
    )
    rgb_checksum_equality = (
        original_metadata["checksums"]["rgb_tree_sha256"]
        == replay_metadata["checksums"]["rgb_tree_sha256"]
    )
    total_absolute_error = 0
    total_values = 0
    pixels_over_two = 0
    pixels_over_ten = 0
    max_absolute_error = 0
    original_frames = sorted((original / "frames").glob("*.jpg"))
    replay_frames = sorted((replay / "frames").glob("*.jpg"))
    if len(original_frames) != len(replay_frames):
        raise RuntimeError("deterministic replay frame count changed")
    for original_frame, replay_frame in zip(original_frames, replay_frames):
        first = cv2.imread(str(original_frame), cv2.IMREAD_COLOR)
        second = cv2.imread(str(replay_frame), cv2.IMREAD_COLOR)
        if first is None or second is None or first.shape != second.shape:
            raise RuntimeError("deterministic replay RGB frame is unreadable or changed shape")
        difference = np.abs(first.astype(np.int16) - second.astype(np.int16))
        total_absolute_error += int(difference.sum())
        total_values += int(difference.size)
        pixels_over_two += int(np.sum(difference > 2))
        pixels_over_ten += int(np.sum(difference > 10))
        max_absolute_error = max(max_absolute_error, int(difference.max()))
    rgb_pixel_metrics = {
        "decoded_values": total_values,
        "mean_absolute_error_8bit": total_absolute_error / max(total_values, 1),
        "max_absolute_error_8bit": max_absolute_error,
        "fraction_absolute_error_over_2": pixels_over_two / max(total_values, 1),
        "fraction_absolute_error_over_10": pixels_over_ten / max(total_values, 1),
    }
    renderer_variation_within_tolerance = (
        rgb_pixel_metrics["mean_absolute_error_8bit"] <= 0.01
        and rgb_pixel_metrics["max_absolute_error_8bit"] <= 32
        and rgb_pixel_metrics["fraction_absolute_error_over_2"] <= 1e-4
        and rgb_pixel_metrics["fraction_absolute_error_over_10"] <= 1e-6
    )
    success_equality = (
        original_metadata["success"] is True
        and replay_metadata["success"] is True
        and original_metadata["metadata"]["termination_reason"] == "NATIVE_SUCCESS"
        and replay_metadata["metadata"]["termination_reason"] == "NATIVE_SUCCESS"
    )
    passed = (
        same_fields
        and all(field_equality.values())
        and numeric_checksum_equality
        and (rgb_checksum_equality or renderer_variation_within_tolerance)
        and success_equality
    )
    return {
        "attempt_id": original_metadata["attempt_id"],
        "state_action_equality": all(
            field_equality.get(name, False)
            for name in (
                "proprio",
                "sim_tactile",
                "policy_action",
                "env_action",
                "native_task_state",
                "timestamp_sec",
                "success",
            )
        ),
        "all_array_fields_equal": same_fields and all(field_equality.values()),
        "field_equality": field_equality,
        "numeric_checksum_equality": numeric_checksum_equality,
        "rgb_checksum_equality": rgb_checksum_equality,
        "rgb_pixel_metrics": rgb_pixel_metrics,
        "renderer_variation_tolerance": {
            "mean_absolute_error_8bit_max": 0.01,
            "max_absolute_error_8bit_max": 32,
            "fraction_absolute_error_over_2_max": 0.0001,
            "fraction_absolute_error_over_10_max": 0.000001,
        },
        "renderer_variation_within_tolerance": renderer_variation_within_tolerance,
        "renderer_replay_classification": (
            "BYTE_IDENTICAL"
            if rgb_checksum_equality
            else "BOUNDED_EGL_PIXEL_NONDETERMINISM"
            if renderer_variation_within_tolerance
            else "RENDERER_REPLAY_MISMATCH"
        ),
        "success_equality": success_equality,
        "status": "PASS" if passed else "FAIL",
    }


def run_task(task: str) -> dict[str, Any]:
    manifest = formal_attempt_manifest(OFFICIAL_ROOT)
    freeze = verify_frozen_protocol(manifest, ARTIFACT_ROOT)
    dataset = read_json(ARTIFACT_ROOT / "policy_expert_dataset_manifest.json")
    selected_members = [
        row for row in dataset["successful_train"] if row["task"] == task
    ][:2]
    if len(selected_members) != 2:
        raise RuntimeError(f"{task} does not have two successful train episodes")
    by_id = {row["attempt_id"]: row for row in manifest["attempts"]}
    replay_root = TMP_ROOT / task / "attempts"
    failure_root = TMP_ROOT / task / "infrastructure_failures"
    rows = []
    for member in selected_members:
        expected = by_id[member["attempt_id"]]
        run_attempt(
            expected,
            official_root=OFFICIAL_ROOT,
            formal_root=replay_root,
            artifact_root=ARTIFACT_ROOT,
            failure_root=failure_root,
            freeze=freeze,
        )
        rows.append(
            compare_attempts(
                FORMAL_ROOT / member["attempt_id"],
                replay_root / member["attempt_id"],
            )
        )
    status = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
    result = {
        "schema": "tactile3d-unit.s4-3-pd-determinism-task.v1",
        "stage": "PD8",
        "task": task,
        "selection_rule": "first two successful frozen POLICY_TRAIN identities; non-selection audit",
        "episodes": rows,
        "exact_frozen_seed_and_config": True,
        "status": status,
    }
    write_json(TASK_RESULT_ROOT / f"{task}.json", result)
    return result


def combine() -> dict[str, Any]:
    task_results = {}
    for task in TASKS:
        path = TASK_RESULT_ROOT / f"{task}.json"
        if not path.is_file():
            raise RuntimeError(f"missing determinism task result: {task}")
        task_results[task] = read_json(path)
    episodes = [row for task in TASKS for row in task_results[task]["episodes"]]
    status = (
        "PASS"
        if len(episodes) == 6
        and all(task_results[task]["status"] == "PASS" for task in TASKS)
        else "FAIL"
    )
    result = {
        "schema": "tactile3d-unit.s4-3-pd-determinism-spotcheck.v1",
        "stage": "PD8",
        "episodes_per_task": 2,
        "total_episodes": len(episodes),
        "task_results": task_results,
        "state_action_equality": all(row["state_action_equality"] for row in episodes),
        "success_equality": all(row["success_equality"] for row in episodes),
        "numeric_checksum_equality": all(
            row["numeric_checksum_equality"] for row in episodes
        ),
        "rgb_checksum_equality": all(row["rgb_checksum_equality"] for row in episodes),
        "renderer_variation_within_tolerance": all(
            row["renderer_variation_within_tolerance"] for row in episodes
        ),
        "reproducibility_boundary": (
            "Physical state, Action, tactile, native success, timestamps, and numeric checksums must be exact. EGL JPEG bytes may differ only within the recorded decoded-pixel tolerance."
        ),
        "status": status,
    }
    write_json(ARTIFACT_ROOT / "determinism_spotcheck.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--combine-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.combine_only:
        result = combine()
    elif args.task:
        result = run_task(args.task)
    else:
        raise SystemExit("choose --task or --combine-only")
    print(json.dumps({"stage": "PD8", "status": result["status"]}, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
