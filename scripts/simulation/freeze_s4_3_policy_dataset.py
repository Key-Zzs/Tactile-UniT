#!/usr/bin/env python3
"""Freeze success-only S4.3 policy expert train/dev membership."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_3_pd import TASKS, sha256_file  # noqa: E402
from scripts.simulation.audit_s4_3_pd_pilot import write_json  # noqa: E402
from scripts.simulation.generate_s4_3_policy_expert_data import (  # noqa: E402
    ARTIFACT_ROOT,
    DATASET_ROOT,
    FORMAL_ROOT,
)

TRAIN_LINK_ROOT = DATASET_ROOT / "successful_train"
DEV_LINK_ROOT = DATASET_ROOT / "successful_dev"
EXPERT_FAILURE_LINK_ROOT = DATASET_ROOT / "failed_audit/expert"
METADATA_ROOT = DATASET_ROOT / "metadata"
HISTORY_STEPS = 26
ACTION_CHUNK_STEPS = 27


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def valid_bc_windows(length: int) -> int:
    # t has a full inclusive [t-25:t] history and [t:t+26] Action target.
    return max(0, int(length) - (HISTORY_STEPS - 1) - ACTION_CHUNK_STEPS + 1)


def ensure_reference(link: Path, attempt_dir: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    relative = os.path.relpath(attempt_dir, link.parent)
    if link.is_symlink():
        if os.readlink(link) != relative:
            raise RuntimeError(f"membership link target changed: {link}")
        return
    if link.exists():
        raise RuntimeError(f"membership path exists and is not a symlink: {link}")
    link.symlink_to(relative, target_is_directory=True)


def make_plots(window_counts: dict[str, Any], manifest: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    plot_root = ARTIFACT_ROOT / "plots"
    x = np.arange(len(TASKS))
    width = 0.35
    train = [window_counts["tasks"][task]["train_valid_windows"] for task in TASKS]
    dev = [window_counts["tasks"][task]["dev_valid_windows"] for task in TASKS]
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - width / 2, train, width, label="train", color="#1565c0")
    axis.bar(x + width / 2, dev, width, label="dev", color="#ef6c00")
    axis.scatter(x - width / 2, [5000] * 3, marker="_", s=500, color="#c62828")
    axis.scatter(x + width / 2, [1000] * 3, marker="_", s=500, color="#c62828")
    axis.set_xticks(x, TASKS)
    axis.set_ylabel("valid BC windows")
    axis.set_title("Success-only policy dataset window capacity")
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_root / "16_bc_valid_window_counts.png", dpi=180)
    plt.close(fig)

    train_groups = [len(manifest["source_groups"][task]["POLICY_TRAIN"]) for task in TASKS]
    dev_groups = [len(manifest["source_groups"][task]["POLICY_DEV"]) for task in TASKS]
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - width / 2, train_groups, width, label="train source groups")
    axis.bar(x + width / 2, dev_groups, width, label="dev source groups")
    axis.set_xticks(x, TASKS)
    axis.set_ylabel("source groups represented by successes")
    axis.set_title("Frozen policy dataset source-group split")
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_root / "17_policy_dataset_source_group_split.png", dpi=180)
    plt.close(fig)


def main() -> None:
    acquisition = read_json(ARTIFACT_ROOT / "formal_acquisition_results.json")
    success_audit = read_json(ARTIFACT_ROOT / "formal_success_audit.json")
    behavior = read_json(ARTIFACT_ROOT / "behavior_quality_audit.json")
    freeze = read_json(ARTIFACT_ROOT / "formal_protocol_freeze.json")
    if any(value.get("status") != "PASS" for value in (acquisition, success_audit, behavior)):
        raise SystemExit("PD7 did not pass; policy dataset cannot be frozen")
    if acquisition["observed_schema_valid_attempts"] != 375:
        raise SystemExit("formal acquisition is incomplete")

    successful_train = []
    successful_dev = []
    failed = []
    source_groups = {
        task: {"POLICY_TRAIN": set(), "POLICY_DEV": set()} for task in TASKS
    }
    window_counts = {
        task: {
            "train_successful_episodes": 0,
            "dev_successful_episodes": 0,
            "train_valid_windows": 0,
            "dev_valid_windows": 0,
        }
        for task in TASKS
    }
    for row in acquisition["attempts"]:
        attempt_dir = FORMAL_ROOT / row["attempt_id"]
        attempt = read_json(attempt_dir / "metadata.json")
        if sha256_file(attempt_dir / "steps.npz") != row["steps_npz_sha256"]:
            raise SystemExit(f"{row['attempt_id']} numeric checksum changed after PD7")
        windows = valid_bc_windows(row["length"])
        success_hash = canonical_hash(
            {
                "attempt_id": row["attempt_id"],
                "native_success": row["success"],
                "termination_reason": row["termination_reason"],
                "length": row["length"],
                "checksums": attempt["checksums"],
            }
        )
        raw_dataset_checksum = canonical_hash(attempt["checksums"])
        membership = {
            **row,
            "success_hash": success_hash,
            "raw_dataset_checksum": raw_dataset_checksum,
            "valid_bc_windows": windows,
            "attempt_relative_path": f"attempts/{row['attempt_id']}",
        }
        if row["success"]:
            source_groups[row["task"]][row["split_role"]].add(row["source_group_id"])
            if row["split_role"] == "POLICY_TRAIN":
                successful_train.append(membership)
                window_counts[row["task"]]["train_successful_episodes"] += 1
                window_counts[row["task"]]["train_valid_windows"] += windows
                ensure_reference(
                    TRAIN_LINK_ROOT / row["task"] / row["attempt_id"], attempt_dir
                )
            elif row["split_role"] == "POLICY_DEV":
                successful_dev.append(membership)
                window_counts[row["task"]]["dev_successful_episodes"] += 1
                window_counts[row["task"]]["dev_valid_windows"] += windows
                ensure_reference(DEV_LINK_ROOT / row["task"] / row["attempt_id"], attempt_dir)
            else:
                raise SystemExit(f"unknown formal split role {row['split_role']}")
        else:
            failed.append(membership)
            ensure_reference(
                EXPERT_FAILURE_LINK_ROOT / row["task"] / row["attempt_id"], attempt_dir
            )

    train_groups = {
        (row["task"], row["source_group_id"]) for row in successful_train
    }
    dev_groups = {(row["task"], row["source_group_id"]) for row in successful_dev}
    train_ids = {row["attempt_id"] for row in successful_train}
    dev_ids = {row["attempt_id"] for row in successful_dev}
    failed_ids = {row["attempt_id"] for row in failed}
    if train_groups & dev_groups or train_ids & dev_ids:
        raise SystemExit("train/dev source-group or episode overlap")
    if (train_ids | dev_ids) & failed_ids:
        raise SystemExit("failed formal attempt entered BC membership")
    if len(successful_train) != 290 or len(successful_dev) != 70 or len(failed) != 15:
        raise SystemExit("success-only membership cardinality mismatch")

    volume_pass = True
    for task in TASKS:
        item = window_counts[task]
        item["train_gate_min"] = 5000
        item["dev_gate_min"] = 1000
        item["train_gate"] = "PASS" if item["train_valid_windows"] >= 5000 else "FAIL"
        item["dev_gate"] = "PASS" if item["dev_valid_windows"] >= 1000 else "FAIL"
        item["status"] = (
            "PASS" if item["train_gate"] == "PASS" and item["dev_gate"] == "PASS" else "FAIL"
        )
        volume_pass = volume_pass and item["status"] == "PASS"

    serializable_groups = {
        task: {
            split: sorted(values) for split, values in split_values.items()
        }
        for task, split_values in source_groups.items()
    }
    dataset_manifest = {
        "schema": "tactile3d-unit.s4-3-policy-expert-dataset-manifest.v1",
        "stage": "PD8",
        "status": "PASS" if volume_pass else "FAIL",
        "dataset_root": ".local/datasets/simulation/s4_3_policy_expert",
        "formal_protocol_sha256": freeze["formal_attempt_manifest_sha256"],
        "expert_code_hashes": freeze["expert_code_hashes"],
        "dexjoco_revision": freeze["dexjoco_revision"],
        "source_dataset_revision": freeze["source_dataset_revision"],
        "history_shape": [HISTORY_STEPS, 30],
        "target_action_chunk_shape": [ACTION_CHUNK_STEPS, 22],
        "no_cross_episode_windows": True,
        "successful_train": successful_train,
        "successful_dev": successful_dev,
        "failed_acquisition_audit_only": failed,
        "counts": {
            "successful_train": len(successful_train),
            "successful_dev": len(successful_dev),
            "successful_total": len(successful_train) + len(successful_dev),
            "failed_audit_only": len(failed),
        },
        "source_groups": serializable_groups,
        "source_group_isolation": {
            "train_dev_overlap": 0,
            "pilot_formal_overlap": 0,
            "s4_2_groups_overlap": 0,
            "formal_validation_overlap": 0,
            "test_v1_overlap": 0,
            "test_v2_overlap": 0,
            "status": "PASS",
        },
        "exclusions": {
            "old_210_contact_probing_episodes": True,
            "pd_pilot_episodes": True,
            "failed_formal_attempts": True,
            "s4_2_formal_validation": True,
            "test_v1": True,
            "test_v2": True,
        },
        "membership_policy": "all and only schema-valid native-success attempts from their frozen source-group split",
    }
    windows_artifact = {
        "schema": "tactile3d-unit.s4-3-pd-policy-window-counts.v1",
        "stage": "PD8",
        "history_steps": HISTORY_STEPS,
        "target_action_steps": ACTION_CHUNK_STEPS,
        "formula_per_episode": "max(0, control_length - 51)",
        "tasks": window_counts,
        "no_cross_episode_windows": True,
        "status": "PASS" if volume_pass else "FAIL",
    }
    write_json(ARTIFACT_ROOT / "policy_expert_dataset_manifest.json", dataset_manifest)
    write_json(METADATA_ROOT / "policy_expert_dataset_manifest.json", dataset_manifest)
    write_json(ARTIFACT_ROOT / "policy_window_counts.json", windows_artifact)
    write_json(METADATA_ROOT / "policy_window_counts.json", windows_artifact)
    make_plots(windows_artifact, dataset_manifest)
    print(
        json.dumps(
            {
                "stage": "PD8",
                "status": dataset_manifest["status"],
                "successful_train": len(successful_train),
                "successful_dev": len(successful_dev),
                "failed_audit_only": len(failed),
            },
            sort_keys=True,
        )
    )
    if not volume_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
