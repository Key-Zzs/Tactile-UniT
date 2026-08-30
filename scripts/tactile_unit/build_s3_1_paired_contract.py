#!/usr/bin/env python3
"""Build and audit the canonical S3.1 paired V+A+C references."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.paired_contract import (  # noqa: E402
    ACTION_HORIZON,
    CANONICAL_FPS,
    CANONICAL_HORIZON,
    TREX_EMBODIMENT_ID,
    TREX_EMBODIMENT_TAG,
    VIDEO_KEY,
    cache_identity,
    data_relative_path,
    decode_rgb_frame_nearest,
    discover_dataset_revision,
    ffprobe_video,
    load_episode_video_pointers,
    load_info,
    load_transition_arrays,
    make_pair_record,
    normalize_and_pad_trex_state_action,
    pad_trex_state_action,
    preprocess_trex_rgb,
    sha256_file,
    sha256_json,
    validate_transition_anchor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--transition-cache", type=Path, default=ROOT / ".local/cache/contact_dynamics/s2_transition_pairs"
    )
    parser.add_argument(
        "--s3-0-manifest", type=Path, default=ROOT / ".local/artifacts/tactile_unit/s3_0/contact_manifest.json"
    )
    parser.add_argument(
        "--s1-split", type=Path, default=ROOT / ".local/artifacts/tactile_teacher/s1_0/split_manifest.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / ".local/artifacts/tactile_unit/s3_1"
    )
    parser.add_argument(
        "--s1-checkpoint", type=Path, default=ROOT / ".local/experiments/tactile_teacher/s1_teacher/best.pt"
    )
    parser.add_argument(
        "--s2-checkpoint", type=Path, default=ROOT / ".local/experiments/contact_dynamics/s2_models/proposed_best.pt"
    )
    parser.add_argument(
        "--contact-code-root", type=Path, default=ROOT / ".local/cache/contact_dynamics/s2_codes"
    )
    parser.add_argument(
        "--runtime-cache", type=Path, default=ROOT / ".local/cache/tactile_unit/s3_1"
    )
    parser.add_argument(
        "--state-action-stats",
        type=Path,
        default=ROOT / ".local/artifacts/tactile_unit/s3_1/state_action_normalization.json",
    )
    parser.add_argument("--timestamp-sample", type=int, default=256)
    return parser.parse_args()


def deterministic_indices(size: int, count: int, label: str) -> list[int]:
    ranked = sorted(
        range(size), key=lambda index: hashlib.sha256(f"{label}:{index}".encode()).digest()
    )
    return sorted(ranked[: min(size, count)])


def read_real_state_actions(
    dataset_root: Path, records: list[dict[str, Any]]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    import pyarrow.parquet as pq

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["state"]["relative_path"]].append(record)
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for relative_path, file_records in sorted(grouped.items()):
        table = pq.read_table(
            dataset_root / relative_path,
            columns=["index", "episode_index", "frame_index", "observation.state", "action"],
        )
        indices = np.asarray(table["index"])
        if len(indices) == 0 or not np.array_equal(indices, np.arange(indices[0], indices[0] + len(indices))):
            raise ValueError(f"non-contiguous global indices in {relative_path}")
        first_index = int(indices[0])
        episode_column = np.asarray(table["episode_index"])
        frame_column = np.asarray(table["frame_index"])
        for record in file_records:
            row = int(record["state"]["dataset_index"]) - first_index
            action_stop = row + ACTION_HORIZON
            if row < 0 or action_stop > len(table):
                raise ValueError(f"pair rows outside {relative_path}")
            episode = record["episode_id"]
            anchor = record["anchor"]["frame"]
            if int(episode_column[row]) != episode or int(frame_column[row]) != anchor:
                raise ValueError(f"state reference mismatch for {record['pair_id']}")
            if not np.all(episode_column[row:action_stop] == episode):
                raise ValueError(f"action crosses episode boundary for {record['pair_id']}")
            if not np.array_equal(frame_column[row:action_stop], np.arange(anchor, anchor + 16)):
                raise ValueError(f"action t:t+15 mismatch for {record['pair_id']}")
            state = np.asarray(table["observation.state"][row].as_py(), dtype=np.float32)
            action = np.asarray(table["action"].slice(row, ACTION_HORIZON).to_pylist(), dtype=np.float32)
            result[record["pair_id"]] = (state, action)
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    info = load_info(args.dataset_root)
    expected_state_names = (
        [f"left_arm_q_{index}" for index in range(7)]
        + [f"left_hand_q_{index}" for index in range(22)]
        + [f"right_arm_q_{index}" for index in range(7)]
        + [f"right_hand_q_{index}" for index in range(22)]
    )
    expected_action_names = (
        [f"left_arm_target_dof_{index}" for index in range(7)]
        + [f"left_hand_target_q_{index}" for index in range(22)]
        + [f"right_arm_target_dof_{index}" for index in range(7)]
        + [f"right_hand_target_q_{index}" for index in range(22)]
    )
    if info["features"]["observation.state"]["names"] != expected_state_names:
        raise ValueError("T-Rex observation.state ordering changed")
    if info["features"]["action"]["names"] != expected_action_names:
        raise ValueError("T-Rex action ordering changed")
    state_action_stats = json.loads(args.state_action_stats.read_text())
    if (
        state_action_stats.get("mode") != "mean_std"
        or state_action_stats.get("fit_split") != "frozen S1 train episodes only"
        or state_action_stats.get("s1_split_manifest_sha256")
        != "e4ef153904f013176ae9f5009751880dc6ee53bcd858d1d26c5793a6ba946e76"
    ):
        raise ValueError("T-Rex state/action normalization identity mismatch")
    pointers = load_episode_video_pointers(args.dataset_root)
    pointer_by_id = {item.episode_id: item for item in pointers}

    expected_hashes = {
        "s1_teacher": "54aedbfe0d72b18822624874ef3724512357c31ea03876513c6dea75d3aae8ac",
        "s2_encoder": "c36c0531bba461875384cebf6bd91c34d43d3f84d2083c15c47ae7dee4e64fa4",
        "transition_manifest": "2e9a14d13c80e24464e4e1bb47318ceb0aa8459f9e62cac3506d90c810667c72",
        "split_manifest": "e4ef153904f013176ae9f5009751880dc6ee53bcd858d1d26c5793a6ba946e76",
        "train_codes": "0c638f321336419a7b12d79858ea3db4396fdb5177fea98b3da736dfe31164c1",
        "test_codes": "d193bf9c623567d034f70451efbb62b28386f5cc00d808b9cba4f12012442402",
    }
    actual_hashes = {
        "s1_teacher": sha256_file(args.s1_checkpoint),
        "s2_encoder": sha256_file(args.s2_checkpoint),
        "transition_manifest": sha256_file(args.transition_cache / "manifest.json"),
        "split_manifest": sha256_file(args.s1_split),
        "train_codes": sha256_file(args.contact_code_root / "train.npy"),
        "test_codes": sha256_file(args.contact_code_root / "test.npy"),
    }
    if actual_hashes != expected_hashes:
        raise ValueError("frozen S1/S2 contact provenance mismatch")
    train_codes = np.load(args.contact_code_root / "train.npy", mmap_mode="r")
    test_codes = np.load(args.contact_code_root / "test.npy", mmap_mode="r")
    if train_codes.shape != (279680, 8, 32) or test_codes.shape != (17504, 8, 32):
        raise ValueError("frozen continuous z_c cache shape mismatch")
    if not np.isfinite(train_codes).all() or not np.isfinite(test_codes).all():
        raise ValueError("frozen continuous z_c cache contains non-finite values")
    contact_identity = cache_identity(
        teacher_sha256=actual_hashes["s1_teacher"],
        encoder_sha256=actual_hashes["s2_encoder"],
        transition_manifest_sha256=actual_hashes["transition_manifest"],
        split_sha256=actual_hashes["split_manifest"],
    )
    contact_identity["existing_code_cache_sha256"] = {
        "train": actual_hashes["train_codes"],
        "test": actual_hashes["test_codes"],
    }
    contact_identity["code_shapes"] = {
        "train": list(train_codes.shape),
        "test": list(test_codes.shape),
    }
    args.runtime_cache.mkdir(parents=True, exist_ok=True)
    (args.runtime_cache / "contact_identity.json").write_text(
        json.dumps(contact_identity, indent=2, sort_keys=True) + "\n"
    )

    # Audit all packed episode intervals before constructing paired records.
    duration_errors = np.asarray(
        [item.duration - item.length / CANONICAL_FPS for item in pointers], dtype=np.float64
    )
    negative_mapping = 0
    out_of_range = 0
    for pointer in pointers:
        try:
            validate_transition_anchor(pointer, 15)
            validate_transition_anchor(pointer, pointer.length - CANONICAL_HORIZON - 1)
        except ValueError:
            out_of_range += 1
        if pointer.from_timestamp < 0 or pointer.to_timestamp <= pointer.from_timestamp:
            negative_mapping += 1

    # Compare packed file duration against the latest metadata pointer endpoint.
    last_endpoint: dict[str, float] = defaultdict(float)
    for pointer in pointers:
        last_endpoint[pointer.relative_path] = max(last_endpoint[pointer.relative_path], pointer.to_timestamp)
    with ThreadPoolExecutor(max_workers=8) as executor:
        probes = list(
            executor.map(
                lambda path: (path, ffprobe_video(args.dataset_root / path)),
                sorted(last_endpoint),
            )
        )
    packed_file_tail_error = {
        path: probe["duration"] - last_endpoint[path] for path, probe in probes
    }

    split_manifest = json.loads(args.s1_split.read_text())
    split_sets = {
        name: set(map(int, split_manifest["episode_ids"][name])) for name in ("train", "val", "test")
    }
    leakage = {
        "train_val": len(split_sets["train"] & split_sets["val"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "val_test": len(split_sets["val"] & split_sets["test"]),
    }
    expected_pair_counts = {"train": 279680, "val": 17504, "test": 17504}
    split_summaries = {}
    all_pair_ids: set[str] = set()
    sample_records: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        arrays = load_transition_arrays(args.transition_cache, split)
        if len(arrays["episode_id"]) != expected_pair_counts[split]:
            raise ValueError(f"{split} pair count changed")
        missing_episodes = set(map(int, np.unique(arrays["episode_id"]))) - split_sets[split]
        if missing_episodes:
            raise ValueError(f"{split} transitions violate the frozen S1 split")
        indices = deterministic_indices(len(arrays["episode_id"]), min(512, len(arrays["episode_id"])), split)
        records = []
        for index in indices:
            episode = int(arrays["episode_id"][index])
            record = make_pair_record(
                split=split,
                source_index=index,
                pointer=pointer_by_id[episode],
                anchor_frame=int(arrays["anchor_frame"][index]),
                anchor_time=float(arrays["anchor_time"][index]),
                info=info,
                contact_transition=int(arrays["contact_transition"][index]),
                dynamic=bool(arrays["dynamic"][index]),
                force_trend_class=int(arrays["force_trend_class"][index]),
            )
            if record["pair_id"] in all_pair_ids:
                raise ValueError("duplicate pair identity")
            all_pair_ids.add(record["pair_id"])
            records.append(record)
        # Full coverage is checked without serializing 314k redundant records.
        invalid = 0
        identities = set()
        for episode, anchor, anchor_time in zip(
            arrays["episode_id"], arrays["anchor_frame"], arrays["anchor_time"]
        ):
            pointer = pointer_by_id[int(episode)]
            try:
                timing = validate_transition_anchor(pointer, int(anchor))
                if not math.isclose(
                    float(anchor_time), int(anchor) / CANONICAL_FPS, rel_tol=0, abs_tol=2e-5
                ):
                    invalid += 1
                identity = (int(episode), int(anchor))
                if identity in identities:
                    invalid += 1
                identities.add(identity)
            except (ValueError, IndexError):
                invalid += 1
        split_summaries[split] = {
            "episodes": len(split_sets[split]),
            "pairs": len(arrays["episode_id"]),
            "resolved_pairs": len(arrays["episode_id"]) - invalid,
            "invalid_pairs": invalid,
            "unique_pair_identities": len(identities),
        }
        sample_records[split] = records

    # Enrich exactly the immutable S3.0 960 contact pair IDs.
    s3_manifest_raw_sha = sha256_file(args.s3_0_manifest)
    if s3_manifest_raw_sha != "7c2d1be54536e0b6d2547c93202fd060211f9a46fb6dd71b83ac06b585e6626e":
        raise ValueError("S3.0 contact subset manifest identity mismatch")
    source_manifest = json.loads(args.s3_0_manifest.read_text())
    test_arrays = load_transition_arrays(args.transition_cache, "test")
    paired_rows = []
    for source in source_manifest["rows"]:
        index = int(source["source_index"])
        episode = int(test_arrays["episode_id"][index])
        anchor = int(test_arrays["anchor_frame"][index])
        if episode != int(source["episode_id"]) or anchor != int(source["anchor_frame"]):
            raise ValueError("S3.0 pair identity no longer matches S2")
        paired_rows.append(
            make_pair_record(
                split="test",
                source_index=index,
                pointer=pointer_by_id[episode],
                anchor_frame=anchor,
                anchor_time=float(test_arrays["anchor_time"][index]),
                info=info,
                contact_transition=int(test_arrays["contact_transition"][index]),
                dynamic=bool(test_arrays["dynamic"][index]),
                force_trend_class=int(test_arrays["force_trend_class"][index]),
            )
        )
    if len(paired_rows) != 960 or len({row["pair_id"] for row in paired_rows}) != 960:
        raise ValueError("canonical paired evaluation set is not exactly 960 unique pairs")

    # Real state/action and transform checks on the full 960 benchmark.
    raw_state_shapes = Counter()
    raw_action_shapes = Counter()
    transform_failures = []
    real_state_actions = read_real_state_actions(args.dataset_root, paired_rows)
    for row in paired_rows:
        state, action = real_state_actions[row["pair_id"]]
        raw_state_shapes[str(state.shape)] += 1
        raw_action_shapes[str(action.shape)] += 1
        try:
            raw_padded = pad_trex_state_action(state, action)
            if not np.array_equal(raw_padded["state"][:58], state):
                raise ValueError("raw state values changed during padding")
            if not np.array_equal(raw_padded["action"][:, :58], action):
                raise ValueError("raw action values changed during padding")
            transformed = normalize_and_pad_trex_state_action(state, action, state_action_stats)
            if not np.isfinite(transformed["state"]).all() or not np.isfinite(transformed["action"]).all():
                raise ValueError("normalized state/action is non-finite")
        except Exception as exc:
            transform_failures.append({"pair_id": row["pair_id"], "error": str(exc)})

    # Decode exact t/t+16 frames and check PTS error on a broad deterministic sample.
    timestamp_indices = set(deterministic_indices(960, args.timestamp_sample, "paired-timestamps"))
    first_per_primitive = {}
    for index, row in enumerate(paired_rows):
        first_per_primitive.setdefault(row["metadata"]["motor_primitive"], index)
    timestamp_indices.update(first_per_primitive.values())
    timestamp_errors = []
    processed_stats = []
    for index in sorted(timestamp_indices):
        row = paired_rows[index]
        path = args.dataset_root / row["vision"]["relative_path"]
        current, current_pts = decode_rgb_frame_nearest(
            path, row["vision"]["current"]["packed_timestamp"]
        )
        future, future_pts = decode_rgb_frame_nearest(
            path, row["vision"]["future"]["packed_timestamp"]
        )
        timestamp_errors.extend(
            [
                abs(current_pts - row["vision"]["current"]["packed_timestamp"]),
                abs(future_pts - row["vision"]["future"]["packed_timestamp"]),
            ]
        )
        for image in (current, future):
            processed = preprocess_trex_rgb(image)
            processed_stats.append(
                [float(processed.min()), float(processed.max()), float(processed.mean()), float(processed.std())]
            )

    transition_counts = Counter(str(row["contact"]["transition_class"]) for row in paired_rows)
    primitive_counts = Counter(row["metadata"]["motor_primitive"] for row in paired_rows)
    object_counts = Counter(row["metadata"]["object"] for row in paired_rows)
    manifest: dict[str, Any] = {
        "schema": "tactile3d-unit.s3-1-paired-eval.v1",
        "dataset": "T-Rex",
        "dataset_revision": discover_dataset_revision(args.dataset_root),
        "source_manifest": {
            "milestone": "S3.0",
            "sha256": s3_manifest_raw_sha,
            "count": 960,
        },
        "transition": {"horizon_frames": 16, "anchor_delta_sec": 16 / 30},
        "rows": paired_rows,
        "distribution": {
            "dynamic_fraction": float(np.mean([row["contact"]["dynamic"] for row in paired_rows])),
            "transition_class_counts": dict(sorted(transition_counts.items())),
            "primitive_counts": dict(sorted(primitive_counts.items())),
            "object_counts": dict(sorted(object_counts.items())),
            "primitive_coverage": len(primitive_counts),
            "object_coverage": len(object_counts),
        },
    }
    manifest["canonical_sha256"] = sha256_json(manifest)
    manifest_path = args.output_dir / "paired_eval_manifest.json"
    previous_manifest_sha = sha256_file(manifest_path) if manifest_path.is_file() else None
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    raw_manifest_sha = sha256_file(manifest_path)
    manifest_rebuild_matches_existing = (
        previous_manifest_sha is None or previous_manifest_sha == raw_manifest_sha
    )
    if not manifest_rebuild_matches_existing:
        raise ValueError("paired evaluation manifest changed across deterministic rebuild")
    (args.output_dir / "paired_eval_manifest.sha256").write_text(
        f"{raw_manifest_sha}  paired_eval_manifest.json\n"
    )

    summary = {
        "schema": "tactile3d-unit.s3-1-paired-contract-summary.v1",
        "dataset_revision": discover_dataset_revision(args.dataset_root),
        "video_pointer_mechanism": "LeRobot v3 per-episode chunk/file/from_timestamp/to_timestamp",
        "episode_duration_error_sec": {
            "max_abs": float(np.max(np.abs(duration_errors))),
            "mean_abs": float(np.mean(np.abs(duration_errors))),
        },
        "packed_file_tail_error_sec": {
            "min": float(min(packed_file_tail_error.values())),
            "max": float(max(packed_file_tail_error.values())),
        },
        "negative_or_invalid_episode_mappings": negative_mapping,
        "episode_boundary_audit_failures": out_of_range,
        "split_counts": {key: len(value) for key, value in split_sets.items()},
        "split_leakage": leakage,
        "pair_summaries": split_summaries,
        "pair_coverage_fraction": {
            key: value["resolved_pairs"] / value["pairs"] for key, value in split_summaries.items()
        },
        "raw_state_shapes_960": dict(raw_state_shapes),
        "raw_action_shapes_960": dict(raw_action_shapes),
        "state_action_feature_order": {
            "state": info["features"]["observation.state"]["names"],
            "action": info["features"]["action"]["names"],
            "status": "PASS",
            "semantic_layout": ["left arm 7", "left hand 22", "right arm 7", "right hand 22"],
            "trex_action_semantics": "absolute joint/hand target values",
            "gr1_raw_semantics_reused": False,
        },
        "state_action_normalization": {
            "mode": state_action_stats["mode"],
            "fit_split": state_action_stats["fit_split"],
            "canonical_sha256": state_action_stats["canonical_sha256"],
            "file_sha256": sha256_file(args.state_action_stats),
        },
        "action_transform_failure_count": len(transform_failures),
        "action_transform_failures": transform_failures,
        "embodiment": {
            "tag": TREX_EMBODIMENT_TAG,
            "id": TREX_EMBODIMENT_ID,
            "aliases_gr1": False,
            "released_tokenizer_max_num_embodiments": 30,
            "released_tokenizer_category_expansion_required": True,
            "learned_action_branch_parameters_required": True,
        },
        "contact": {
            "s1_teacher_identity": "PASS",
            "s2_encoder_identity": "PASS",
            "h_current_shape": [256],
            "h_future_shape": [256],
            "z_c_shape": [8, 32],
            "z_c_finite": True,
            "cache_identity_sha256": contact_identity["identity_sha256"],
            "adaptor_applied": False,
            "s3_0_decision": "ADAPTER_RECOMMENDED"
        },
        "timestamp_decode_sample_pairs": len(timestamp_indices),
        "timestamp_decode_max_abs_error_sec": float(max(timestamp_errors)),
        "timestamp_decode_tolerance_sec": 0.5 / CANONICAL_FPS + 1e-6,
        "processed_vision_shape": [3, 224, 224],
        "processed_vision_stats": {
            "min": float(np.min(np.asarray(processed_stats)[:, 0])),
            "max": float(np.max(np.asarray(processed_stats)[:, 1])),
            "mean": float(np.mean(np.asarray(processed_stats)[:, 2])),
            "std": float(np.mean(np.asarray(processed_stats)[:, 3])),
        },
        "paired_eval": {
            "expected": 960,
            "resolved": len(paired_rows),
            "manifest_canonical_sha256": manifest["canonical_sha256"],
            "manifest_file_sha256": raw_manifest_sha,
            "deterministic_rebuild_matches_existing": manifest_rebuild_matches_existing,
            **manifest["distribution"],
        },
        "off_by_one_contract": {
            "vision": ["t", "t+16"],
            "action_inclusive": ["t", "t+15"],
            "current_teacher_window_inclusive": ["t-15", "t"],
            "future_teacher_window_inclusive": ["t+1", "t+16"],
            "status": "PASS",
        },
    }
    required_pass = (
        not any(leakage.values())
        and not negative_mapping
        and not out_of_range
        and all(value["invalid_pairs"] == 0 for value in split_summaries.values())
        and len(paired_rows) == 960
        and not transform_failures
        and max(timestamp_errors) <= summary["timestamp_decode_tolerance_sec"]
    )
    summary["status"] = "PASS" if required_pass else "FAIL"
    summary["summary_sha256"] = sha256_json(summary)
    (args.output_dir / "paired_contract_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not required_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
