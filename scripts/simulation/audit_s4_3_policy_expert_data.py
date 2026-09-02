#!/usr/bin/env python3
"""Audit all frozen S4.3-PD formal attempts before dataset membership."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_3_pd import TASKS, sha256_file  # noqa: E402
from scripts.simulation.audit_s4_3_pd_pilot import (  # noqa: E402
    create_video,
    validate_attempt,
    write_json,
)
from scripts.simulation.generate_s4_3_policy_expert_data import (  # noqa: E402
    ARTIFACT_ROOT,
    FORMAL_ROOT,
    INFRASTRUCTURE_FAILURE_ROOT,
    formal_attempt_manifest,
    verify_frozen_protocol,
)


def validate_rgb_tree(item: dict[str, Any]) -> tuple[str, int, str | None]:
    import cv2

    digest = hashlib.sha256()
    count = 0
    try:
        references = item["arrays"]["rgb_reference"].tolist()
        for reference in references:
            path = item["directory"] / str(reference)
            payload = path.read_bytes()
            digest.update(path.name.encode("utf-8"))
            digest.update(hashlib.sha256(payload).digest())
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"unreadable RGB frame {reference}")
            count += 1
        return item["stats"]["attempt_id"], count, digest.hexdigest()
    except Exception as error:
        return item["stats"]["attempt_id"], count, f"{type(error).__name__}:{error}"


def classify_numeric_duplicates(
    summaries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        groups[row["steps_npz_sha256"]].append(row)
    expected = []
    unexpected = []
    for checksum, rows in groups.items():
        if len(rows) == 1:
            continue
        identities = {(row["task"], row["source_group_id"]) for row in rows}
        attempt_indices = {row["attempt_index"] for row in rows}
        rgb_hashes = {row["rgb_tree_sha256"] for row in rows}
        record = {
            "steps_npz_sha256": checksum,
            "attempt_ids": sorted(row["attempt_id"] for row in rows),
            "task_source_groups": sorted([list(value) for value in identities]),
            "attempt_indices": sorted(attempt_indices),
            "unique_rgb_tree_hashes": len(rgb_hashes),
        }
        if (
            len(rows) == 5
            and len(identities) == 1
            and attempt_indices == set(range(5))
            and len(rgb_hashes) == 5
        ):
            record["classification"] = "EXPECTED_WITHIN_SOURCE_VISUAL_PERTURBATIONS"
            expected.append(record)
        else:
            record["classification"] = "UNEXPECTED_NUMERIC_DUPLICATE"
            unexpected.append(record)
    return expected, unexpected


def subset_behavior(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {
            "attempts": 0,
            "episode_length": None,
            "tcp_variance_mean": None,
            "orientation_variance_mean": None,
            "hand_variance_mean": None,
            "hand_joint_std": None,
            "hand_joint_range": None,
            "meaningful_hand_motion_fraction": None,
            "active_hand_joints_min": None,
            "peak_contact_force": None,
            "mean_contact_force": None,
            "free_to_contact": 0,
            "contact_to_free": 0,
            "task_progress_max": None,
            "phase_counts": {},
        }
    stats = [item["stats"] for item in items]
    hand = np.concatenate(
        [item["arrays"]["policy_action"][:, 6:22].astype(np.float64) for item in items]
    )
    forces = np.concatenate(
        [item["arrays"]["normal_force"].astype(np.float64) for item in items]
    )
    lengths = np.asarray([row["length"] for row in stats], dtype=np.int64)
    phases = Counter()
    for row in stats:
        phases.update(row["phase_counts"])
    return {
        "attempts": len(items),
        "episode_length": {
            "min": int(lengths.min()),
            "p25": float(np.quantile(lengths, 0.25)),
            "median": float(np.median(lengths)),
            "mean": float(np.mean(lengths)),
            "p75": float(np.quantile(lengths, 0.75)),
            "max": int(lengths.max()),
        },
        "tcp_variance_mean": float(np.mean([row["tcp_variance"] for row in stats])),
        "orientation_variance_mean": float(
            np.mean([row["orientation_variance"] for row in stats])
        ),
        "hand_variance_mean": float(np.mean([row["hand_variance"] for row in stats])),
        "hand_joint_std": np.std(hand, axis=0).tolist(),
        "hand_joint_range": np.ptp(hand, axis=0).tolist(),
        "meaningful_hand_motion_fraction": float(
            np.mean([row["meaningful_hand_motion"] for row in stats])
        ),
        "active_hand_joints_min": int(min(row["active_hand_joints"] for row in stats)),
        "peak_contact_force": float(np.max(forces)),
        "mean_contact_force": float(np.mean(forces)),
        "free_to_contact": int(sum(row["free_to_contact"] for row in stats)),
        "contact_to_free": int(sum(row["contact_to_free"] for row in stats)),
        "task_progress_max": float(max(row["max_task_progress"] for row in stats)),
        "phase_counts": dict(sorted(phases.items())),
    }


def make_plots(items: list[dict[str, Any]], behavior: dict[str, Any], artifact_root: Path) -> None:
    import matplotlib.pyplot as plt

    plot_root = artifact_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for axis, task in zip(axes, TASKS):
        scoped = [item for item in items if item["stats"]["task"] == task]
        rates = []
        for group_index in range(25):
            group = [
                item for item in scoped if item["manifest"]["metadata"]["source_group_index"] == group_index
            ]
            rates.append(np.mean([item["stats"]["success"] for item in group]))
        colors = ["#1565c0"] * 20 + ["#ef6c00"] * 5
        axis.bar(range(25), rates, color=colors)
        axis.axhline(0.8, color="#c62828", linestyle="--")
        axis.set_ylim(0, 1.08)
        axis.set_ylabel("success rate")
        axis.set_title(task)
    axes[-1].set_xlabel("frozen formal source-group index (blue=train, orange=dev)")
    fig.tight_layout()
    fig.savefig(plot_root / "10_formal_success_rate_by_source_group.png", dpi=180)
    plt.close(fig)

    width = 0.35
    x = np.arange(len(TASKS))
    train = [behavior["tasks"][task]["success_counts"]["train"] for task in TASKS]
    dev = [behavior["tasks"][task]["success_counts"]["dev"] for task in TASKS]
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(x - width / 2, train, width, label="train / 100", color="#1565c0")
    axis.bar(x + width / 2, dev, width, label="dev / 25", color="#ef6c00")
    axis.scatter(x - width / 2, [80] * 3, marker="_", s=500, color="#c62828")
    axis.scatter(x + width / 2, [20] * 3, marker="_", s=500, color="#c62828")
    axis.set_xticks(x, TASKS)
    axis.set_ylabel("native successful attempts")
    axis.set_title("Frozen train/dev success counts and hard gates")
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_root / "11_train_dev_success_count.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    for task in TASKS:
        lengths = [item["stats"]["length"] for item in items if item["stats"]["task"] == task]
        axis.hist(lengths, bins=20, alpha=0.45, label=task)
    axis.set_xlabel("control steps")
    axis.set_ylabel("attempts")
    axis.set_title("Formal episode-length distribution")
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_root / "12_episode_length_distribution.png", dpi=180)
    plt.close(fig)

    matrix = np.asarray([behavior["tasks"][task]["all"]["hand_joint_std"] for task in TASKS])
    fig, axis = plt.subplots(figsize=(12, 4))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_yticks(range(len(TASKS)), TASKS)
    axis.set_xticks(range(16), [f"J{i}" for i in range(16)])
    axis.set_title("Formal hand-joint Action standard deviation")
    fig.colorbar(image, ax=axis, label="rad")
    fig.tight_layout()
    fig.savefig(plot_root / "13_hand_joint_variance_by_task.png", dpi=180)
    plt.close(fig)

    free_contact = [behavior["tasks"][task]["all"]["free_to_contact"] for task in TASKS]
    contact_free = [behavior["tasks"][task]["all"]["contact_to_free"] for task in TASKS]
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(x - width / 2, free_contact, width, label="free→contact")
    axis.bar(x + width / 2, contact_free, width, label="contact→free")
    axis.set_xticks(x, TASKS)
    axis.set_ylabel("transitions")
    axis.set_title("Formal Contact transition counts")
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_root / "14_contact_transition_counts.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, field, title in (
        (axes[0], "task_progress_max", "Max task progress"),
        (axes[1], "episode_length", "Mean episode length"),
    ):
        successful = []
        failed = []
        for task in TASKS:
            success_value = behavior["tasks"][task]["successful"][field]
            failure_value = behavior["tasks"][task]["failed"][field]
            if field == "episode_length":
                success_value = success_value["mean"] if success_value else np.nan
                failure_value = failure_value["mean"] if failure_value else np.nan
            successful.append(success_value)
            failed.append(np.nan if failure_value is None else failure_value)
        axis.bar(x - width / 2, successful, width, label="successful")
        axis.bar(x + width / 2, failed, width, label="failed")
        axis.set_xticks(x, TASKS, rotation=15)
        axis.set_title(title)
        axis.legend()
    fig.tight_layout()
    fig.savefig(plot_root / "15_successful_vs_failed_acquisition.png", dpi=180)
    plt.close(fig)


def create_review_media(items: list[dict[str, Any]], artifact_root: Path) -> None:
    video_root = artifact_root / "videos/formal"
    screenshot_root = artifact_root / "screenshots"
    for task in TASKS:
        successful = next(
            item for item in items if item["stats"]["task"] == task and item["stats"]["success"]
        )
        create_video(
            successful,
            video_root / task / "success" / f"{successful['stats']['attempt_id']}.mp4",
        )
        final_reference = str(successful["arrays"]["rgb_reference"][-1])
        shutil.copy2(
            successful["directory"] / final_reference,
            screenshot_root / f"{task}_formal_success_final.jpg",
        )
        failed = [
            item for item in items if item["stats"]["task"] == task and not item["stats"]["success"]
        ]
        if failed:
            create_video(
                failed[0],
                video_root / task / "failure" / f"{failed[0]['stats']['attempt_id']}.mp4",
            )


def audit(
    formal_root: Path, artifact_root: Path, infrastructure_failure_root: Path, rgb_workers: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = formal_attempt_manifest()
    freeze = verify_frozen_protocol(manifest, artifact_root)
    items = []
    integrity_failures = []
    summaries = []
    seen_attempt_ids = set()
    seen_seed_tuples = set()
    for expected in manifest["attempts"]:
        try:
            manifest_row, item = validate_attempt(
                formal_root / expected["attempt_id"], expected
            )
            metadata = manifest_row["metadata"]
            for name in (
                "split_role",
                "seed_namespace",
                "visual_randomization_seed",
                "controller_seed",
            ):
                if metadata.get(name) != expected[name]:
                    raise ValueError(f"frozen {name} mismatch")
            if metadata["expert_code_hashes"] != freeze["expert_code_hashes"]:
                raise ValueError("expert code hashes differ from freeze")
            if metadata["formal_attempt_manifest_sha256"] != freeze[
                "formal_attempt_manifest_sha256"
            ]:
                raise ValueError("formal manifest checksum differs from freeze")
            arrays = item["arrays"]
            length = item["stats"]["length"]
            if not np.array_equal(arrays["control_step"], np.arange(length)):
                raise ValueError("control-step sequence is not an episode-local range")
            if item["stats"]["success"]:
                if not bool(arrays["success"][-1]) or int(np.sum(arrays["success"])) != 1:
                    raise ValueError("native success is not the sole final transition")
                if metadata["termination_reason"] != "NATIVE_SUCCESS":
                    raise ValueError("successful attempt has non-success termination reason")
            elif np.any(arrays["success"]):
                raise ValueError("failed attempt contains a native-success transition")
            identity = expected["attempt_id"]
            seed_tuple = (
                expected["reset_seed"],
                expected["perturbation_seed"],
                expected["controller_seed"],
            )
            if identity in seen_attempt_ids or seed_tuple in seen_seed_tuples:
                raise ValueError("duplicate attempt identity or seed tuple")
            seen_attempt_ids.add(identity)
            seen_seed_tuples.add(seed_tuple)
            item["manifest"] = manifest_row
            items.append(item)
            summaries.append(
                {
                    "attempt_id": identity,
                    "task": expected["task"],
                    "split_role": expected["split_role"],
                    "source_group_id": expected["source_group_id"],
                    "source_group_index": expected["source_group_index"],
                    "attempt_index": expected["attempt_index"],
                    "reset_seed": expected["reset_seed"],
                    "environment_seed": expected["environment_seed"],
                    "perturbation_seed": expected["perturbation_seed"],
                    "controller_seed": expected["controller_seed"],
                    "success": item["stats"]["success"],
                    "termination_reason": item["stats"]["termination_reason"],
                    "length": item["stats"]["length"],
                    "steps_npz_sha256": manifest_row["checksums"]["steps_npz_sha256"],
                    "rgb_tree_sha256": manifest_row["checksums"]["rgb_tree_sha256"],
                }
            )
        except Exception as error:
            integrity_failures.append(f"{expected['attempt_id']}:{type(error).__name__}:{error}")

    rgb_results = []
    if not integrity_failures:
        with ThreadPoolExecutor(max_workers=max(1, rgb_workers)) as executor:
            rgb_results = list(executor.map(validate_rgb_tree, items))
        result_map = {attempt_id: (count, digest) for attempt_id, count, digest in rgb_results}
        for item in items:
            attempt_id = item["stats"]["attempt_id"]
            count, digest = result_map[attempt_id]
            expected_digest = item["manifest"]["checksums"]["rgb_tree_sha256"]
            if isinstance(digest, str) and digest.startswith(("OSError:", "ValueError:")):
                integrity_failures.append(f"{attempt_id}:RGB:{digest}")
            elif count != item["stats"]["length"] or digest != expected_digest:
                integrity_failures.append(f"{attempt_id}:RGB count/checksum mismatch")

    expected_duplicates, unexpected_duplicates = classify_numeric_duplicates(summaries)
    full_checksum_pairs = Counter(
        (row["steps_npz_sha256"], row["rgb_tree_sha256"]) for row in summaries
    )
    duplicate_full_pairs = [
        {"steps_npz_sha256": pair[0], "rgb_tree_sha256": pair[1], "count": count}
        for pair, count in full_checksum_pairs.items()
        if count > 1
    ]
    if duplicate_full_pairs:
        unexpected_duplicates.extend(duplicate_full_pairs)

    infrastructure_records = sorted(
        path.relative_to(infrastructure_failure_root).as_posix()
        for path in infrastructure_failure_root.rglob("infrastructure_failure.json")
    ) if infrastructure_failure_root.exists() else []

    acquisition_pass = (
        len(items) == 375
        and not integrity_failures
        and not unexpected_duplicates
        and len(seen_attempt_ids) == 375
        and len(seen_seed_tuples) == 375
    )
    acquisition = {
        "schema": "tactile3d-unit.s4-3-pd-formal-acquisition-results.v1",
        "stage": "PD6_PD7",
        "status": "PASS" if acquisition_pass else "FAIL",
        "expected_attempts": 375,
        "observed_schema_valid_attempts": len(items),
        "all_frozen_attempts_executed": len(items) == 375,
        "failed_attempts_retained": True,
        "replacement_success_seeking_seeds": False,
        "infrastructure_failure_records": infrastructure_records,
        "integrity_failures": integrity_failures,
        "rgb_frames_decoded": int(sum(result[1] for result in rgb_results)),
        "duplicate_audit": {
            "attempt_id_duplicates": len(summaries) - len(seen_attempt_ids),
            "seed_tuple_duplicates": len(summaries) - len(seen_seed_tuples),
            "expected_within_source_numeric_duplicate_groups": expected_duplicates,
            "unexpected_numeric_or_full_duplicates": unexpected_duplicates,
        },
        "attempts": summaries,
    }

    success_tasks = {}
    success_pass = acquisition_pass
    for task in TASKS:
        scoped = [row for row in summaries if row["task"] == task]
        train = [row for row in scoped if row["split_role"] == "POLICY_TRAIN"]
        dev = [row for row in scoped if row["split_role"] == "POLICY_DEV"]
        counts = {
            "train_attempts": len(train),
            "train_successes": sum(row["success"] for row in train),
            "train_success_rate": float(np.mean([row["success"] for row in train])),
            "dev_attempts": len(dev),
            "dev_successes": sum(row["success"] for row in dev),
            "dev_success_rate": float(np.mean([row["success"] for row in dev])),
            "total_attempts": len(scoped),
            "total_successes": sum(row["success"] for row in scoped),
            "total_success_rate": float(np.mean([row["success"] for row in scoped])),
        }
        gates = {
            "train_successes_at_least_80": counts["train_successes"] >= 80,
            "dev_successes_at_least_20": counts["dev_successes"] >= 20,
            "total_successes_at_least_100": counts["total_successes"] >= 100,
            "total_success_rate_at_least_80_percent": counts["total_success_rate"] >= 0.8,
        }
        task_pass = all(gates.values())
        success_pass = success_pass and task_pass
        success_tasks[task] = {**counts, "gates": gates, "status": "PASS" if task_pass else "FAIL"}
    success_audit = {
        "schema": "tactile3d-unit.s4-3-pd-formal-success-audit.v1",
        "stage": "PD7",
        "tasks": success_tasks,
        "all_task_hard_gates": "PASS" if success_pass else "FAIL",
        "status": "PASS" if success_pass else "FAIL",
    }

    behavior_tasks = {}
    exploit_behavior = False
    for task in TASKS:
        scoped = [item for item in items if item["stats"]["task"] == task]
        successful = [item for item in scoped if item["stats"]["success"]]
        failed = [item for item in scoped if not item["stats"]["success"]]
        all_behavior = subset_behavior(scoped)
        anti_degenerate = (
            all_behavior["meaningful_hand_motion_fraction"] == 1.0
            and all_behavior["active_hand_joints_min"] >= 3
        )
        behavior_tasks[task] = {
            "success_counts": {
                "train": success_tasks[task]["train_successes"],
                "dev": success_tasks[task]["dev_successes"],
                "total": success_tasks[task]["total_successes"],
            },
            "all": all_behavior,
            "successful": subset_behavior(successful),
            "failed": subset_behavior(failed),
            "anti_degeneracy": "PASS" if anti_degenerate else "FAIL",
            "exploit_behavior": False,
        }
        if not anti_degenerate:
            exploit_behavior = True
    behavior = {
        "schema": "tactile3d-unit.s4-3-pd-behavior-quality.v1",
        "stage": "PD7",
        "tasks": behavior_tasks,
        "exploit_behavior": exploit_behavior,
        "status": "PASS" if success_pass and not exploit_behavior else "FAIL",
    }

    warnings = {
        "schema": "tactile3d-unit.s4-3-pd-warnings.v1",
        "stage": "PD7",
        "warnings": [
            {
                "code": "EXPECTED_WITHIN_SOURCE_NUMERIC_REPLAY_DUPLICATION",
                "count": len(expected_duplicates),
                "meaning": "Five frozen attempts per source share physical replay arrays but have distinct visual-randomization seeds and RGB tree hashes.",
            },
            {
                "code": "FORMAL_EXPERT_FAILURES_RETAINED",
                "count": sum(not row["success"] for row in summaries),
                "meaning": "These attempts remain acquisition-audit-only and are excluded from BC membership.",
            },
        ],
        "unexpected_duplicates": unexpected_duplicates,
        "status": "PASS" if not unexpected_duplicates else "FAIL",
    }
    return acquisition, success_audit, behavior, warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, default=FORMAL_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument(
        "--infrastructure-failure-root",
        type=Path,
        default=INFRASTRUCTURE_FAILURE_ROOT,
    )
    parser.add_argument("--rgb-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    acquisition, success, behavior, warnings = audit(
        args.formal_root,
        args.artifact_root,
        args.infrastructure_failure_root,
        args.rgb_workers,
    )
    write_json(args.artifact_root / "formal_acquisition_results.json", acquisition)
    write_json(args.artifact_root / "formal_success_audit.json", success)
    write_json(args.artifact_root / "behavior_quality_audit.json", behavior)
    write_json(args.artifact_root / "warnings.json", warnings)
    if all(value["status"] == "PASS" for value in (acquisition, success, behavior, warnings)):
        items = []
        manifest = formal_attempt_manifest()
        for expected in manifest["attempts"]:
            manifest_row, item = validate_attempt(
                args.formal_root / expected["attempt_id"], expected
            )
            item["manifest"] = manifest_row
            items.append(item)
        make_plots(items, behavior, args.artifact_root)
        create_review_media(items, args.artifact_root)
        print(json.dumps({"stage": "PD7", "status": "PASS"}, sort_keys=True))
        return
    print(json.dumps({"stage": "PD7", "status": "FAIL"}, sort_keys=True))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
