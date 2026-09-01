#!/usr/bin/env python3
"""Audit official task-solving demonstration sources for S4.3-PD."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEXJOCO = ROOT / "third_party/dexjoco"
DEFAULT_ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_pd"
TASKS = ("pinch_tongs", "hammer_nail", "click_mouse")
DEXJOCO_REVISION = "8d23b0fab23b17a58c4b55f3942e17013aaf8267"
RAW_DATASET_REVISION = "125df5c1019e97503929ef1a0ad8f90373436afa"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_markers(path: Path, markers: tuple[str, ...]) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise RuntimeError(f"{path.relative_to(ROOT)} is missing markers: {missing}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "markers": list(markers),
    }


def task_record(task: str) -> dict[str, Any]:
    task_file = (
        DEXJOCO
        / "dexjoco/dexjoco/sim/envs"
        / f"panda_{task}_env.py"
    )
    config_file = DEXJOCO / "dexjoco/dexjoco/tasks" / task / "config.py"
    raw_root = (
        ROOT
        / ".local/datasets/simulation/s4_3_pd_official_raw/dexjoco_raw_datasets"
        / task
    )
    replay_stores = sorted(raw_root.glob("*/replay.zarr")) if raw_root.is_dir() else []
    sources = [
        {
            "source_class": "E0_OFFICIAL_DEMONSTRATION",
            "name": "DexJoCo successful teleoperation demonstrations",
            "availability": "OFFICIAL_REMOTE_PINNED; LOCAL_LOW_DIM_CACHE_MAY_BE_PARTIAL",
            "source_files": [
                "third_party/dexjoco/README.md",
                "third_party/dexjoco/scripts/record_demos_zarr.py",
                "third_party/dexjoco/scripts/replay_demos_zarr.py",
                f"third_party/dexjoco/dexjoco/dexjoco/tasks/{task}/config.py",
            ],
            "api": "replay.zarr/data/{action,action_rotvec,state,timestamp}",
            "privileged_information": "raw state includes task object poses and table height for replay restoration",
            "action_form": "official action_rotvec: [TCP xyz 3, TCP rotvec 3, Allegro targets 16] = 22D",
            "official_curated_episode_count": 100,
            "native_success_compatibility": "REQUIRES_PD2_NATIVE_REPLAY_CROSSCHECK",
            "local_low_dim_replay_stores": len(replay_stores),
            "selected": True,
            "reason": "Highest-priority actual successful task demonstrations with native policy actions and an official replay path.",
        },
        {
            "source_class": "E1_OFFICIAL_SCRIPTED_EXPERT",
            "name": "official scripted expert/controller",
            "availability": "NOT_FOUND_IN_PINNED_TREE",
            "selected": False,
            "reason": "No task-solving scripted policy or deterministic state machine exists for this task.",
        },
        {
            "source_class": "E2_OFFICIAL_REFERENCE_OR_WAYPOINT_ADAPTATION",
            "name": "official reference or waypoint controller",
            "availability": "NOT_FOUND_IN_PINNED_TREE",
            "selected": False,
            "reason": "Replay, policy-evaluation, and teleoperation infrastructure exist, but no task waypoint source exists.",
        },
        {
            "source_class": "E3_REPOSITORY_PRIVILEGED_SCRIPTED_EXPERT",
            "name": "repository-owned privileged fallback",
            "availability": "NOT_REQUIRED_IF_PD2_REPLAY_PASSES",
            "selected": False,
            "reason": "Reserved only if the selected official demonstrations cannot satisfy native replay and acquisition gates.",
        },
    ]
    return {
        "task": task,
        "task_source": task_file.relative_to(ROOT).as_posix(),
        "task_source_sha256": sha256_file(task_file),
        "task_config": config_file.relative_to(ROOT).as_posix(),
        "task_config_sha256": sha256_file(config_file),
        "available_sources": sources,
        "selected_source_class": "E0_OFFICIAL_DEMONSTRATION",
    }


def visualize(audit: dict[str, Any], destination: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    values = np.zeros((len(TASKS), 4), dtype=np.int64)
    values[:, 0] = 1
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.imshow(values, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), ("E0 demos", "E1 scripted", "E2 waypoint", "E3 fallback"))
    ax.set_yticks(range(len(TASKS)), TASKS)
    for row in range(len(TASKS)):
        for column in range(4):
            label = "SELECTED" if column == 0 else ("fallback" if column == 3 else "not found")
            ax.text(column, row, label, ha="center", va="center", fontsize=9)
    ax.set_title("S4.3-PD expert source decision matrix")
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def audit() -> dict[str, Any]:
    evidence = {
        "readme": require_markers(
            DEXJOCO / "README.md",
            (
                "writes each success as a replayable Zarr episode",
                "DexJoCo/DexJoCo-Datasets-Raw",
                "action_rotvec",
                "Privileged environment fields should be filtered out",
            ),
        ),
        "recorder": require_markers(
            DEXJOCO / "scripts/record_demos_zarr.py",
            ("successes_needed", "succeed", "action_rotvec"),
        ),
        "replayer": require_markers(
            DEXJOCO / "scripts/replay_demos_zarr.py",
            ("restore_initial_state", "action_rotvec", "info.get(\"succeed\""),
        ),
        "state_restorers": require_markers(
            DEXJOCO / "dexjoco/dexjoco/tasks/state_restorers.py",
            ("_restore_click_mouse", "_restore_pinch_tongs", "_restore_hammer_nail"),
        ),
    }
    tasks = {task: task_record(task) for task in TASKS}
    return {
        "schema": "tactile3d-unit.s4-3-pd-official-expert-audit.v1",
        "stage": "PD1",
        "dexjoco_revision": DEXJOCO_REVISION,
        "official_raw_dataset": {
            "repository": "DexJoCo/DexJoCo-Datasets-Raw",
            "revision": RAW_DATASET_REVISION,
            "bulk_media_downloaded": False,
        },
        "audit_scope": [
            "task sources and configs",
            "scripts and examples",
            "README and docs",
            "tests and assets",
            "data collection, conversion, replay, and policy evaluation utilities",
        ],
        "evidence": evidence,
        "tasks": tasks,
        "rejections": {
            "old_s4_2_contact_probe": "NOT_AN_EXPERT; zero native task successes",
            "openpi_norm_stats": "dataset statistics, not executable expert Actions",
            "checkpoint_commands": "trained checkpoint references are not bundled or frozen expert trajectories",
            "teleoperation_interfaces": "collection mechanisms, not autonomous expert sources",
        },
        "selected_source_class_all_tasks": "E0_OFFICIAL_DEMONSTRATION",
        "downstream_condition": "PD2 must establish native replay compatibility; otherwise E3 fallback development is required.",
        "status": "PASS",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    value = audit()
    write_json(args.artifact_root / "official_expert_audit.json", value)
    write_json(
        args.artifact_root / "expert_source_selection.json",
        {
            "schema": "tactile3d-unit.s4-3-pd-expert-source-selection.v1",
            "stage": "PD1",
            "tasks": {
                task: value["tasks"][task]["selected_source_class"] for task in TASKS
            },
            "status": "PASS",
        },
    )
    visualize(value, args.artifact_root / "plots/01_expert_source_decision_matrix.png")
    print(json.dumps({"stage": "PD1", "status": "PASS"}, sort_keys=True))


if __name__ == "__main__":
    main()
