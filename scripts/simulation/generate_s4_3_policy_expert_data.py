#!/usr/bin/env python3
"""Generate every attempt in the frozen S4.3-PD formal acquisition."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.dexjoco_adapter import DexJoCoRuntimeAdapter  # noqa: E402
from gr00t.simulation.s4_3_pd import (  # noqa: E402
    ATTEMPTS_PER_GROUP,
    FORMAL_GROUPS_PER_TASK,
    S43PDAttemptLogger,
    TASKS,
    OfficialReplaySource,
    expert_action_at,
    native_task_state,
    sha256_file,
    source_partition,
)

OFFICIAL_ROOT = (
    ROOT
    / ".local/datasets/simulation/s4_3_pd_official_raw/dexjoco_raw_datasets"
)
DATASET_ROOT = ROOT / ".local/datasets/simulation/s4_3_policy_expert"
FORMAL_ROOT = DATASET_ROOT / "attempts"
INFRASTRUCTURE_FAILURE_ROOT = DATASET_ROOT / "failed_audit/infrastructure"
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_pd"
RAW_DATASET_REVISION = "125df5c1019e97503929ef1a0ad8f90373436afa"
DEXJOCO_REVISION = "8d23b0fab23b17a58c4b55f3942e17013aaf8267"
MAX_HOLD_STEPS = 120
TRAIN_GROUPS_PER_TASK = 20
DEV_GROUPS_PER_TASK = 5
ENVIRONMENT_SEED_BASE = 4_500_000
PERTURBATION_SEED_BASE = 4_600_000


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def formal_attempt_manifest(official_root: Path = OFFICIAL_ROOT) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    attempts: list[dict[str, Any]] = []
    for task_index, task in enumerate(TASKS):
        task_groups = []
        for group_index, source_path in enumerate(source_partition(task, official_root, "FORMAL")):
            split_role = "POLICY_TRAIN" if group_index < TRAIN_GROUPS_PER_TASK else "POLICY_DEV"
            task_groups.append(
                {
                    "source_group_index": group_index,
                    "source_group_id": source_path.parent.name,
                    "split_role": split_role,
                }
            )
            for attempt_index in range(ATTEMPTS_PER_GROUP):
                environment_seed = (
                    ENVIRONMENT_SEED_BASE
                    + task_index * 10_000
                    + group_index * 100
                    + attempt_index
                )
                attempts.append(
                    {
                        "task": task,
                        "source_group_index": group_index,
                        "source_group_id": source_path.parent.name,
                        "attempt_index": attempt_index,
                        "attempt_id": (
                            f"formal-{task}-g{group_index:02d}-a{attempt_index:02d}"
                        ),
                        "split_role": split_role,
                        "seed_namespace": "PD_FORMAL",
                        "reset_seed": environment_seed,
                        "environment_seed": environment_seed,
                        "visual_randomization_seed": environment_seed,
                        "perturbation_seed": (
                            PERTURBATION_SEED_BASE
                            + task_index * 10_000
                            + group_index * 100
                            + attempt_index
                        ),
                        "controller_seed": None,
                    }
                )
        groups[task] = task_groups

    expected_attempts = len(TASKS) * FORMAL_GROUPS_PER_TASK * ATTEMPTS_PER_GROUP
    if len(attempts) != expected_attempts:
        raise RuntimeError("formal manifest attempt cardinality failure")
    if len({row["attempt_id"] for row in attempts}) != expected_attempts:
        raise RuntimeError("formal attempt IDs are not unique")
    seed_tuples = {
        (row["reset_seed"], row["perturbation_seed"], row["controller_seed"])
        for row in attempts
    }
    if len(seed_tuples) != expected_attempts:
        raise RuntimeError("formal seed tuples are not unique")
    for task in TASKS:
        task_rows = [row for row in attempts if row["task"] == task]
        if len(task_rows) != 125:
            raise RuntimeError(f"{task} formal attempt count is not 125")
        if sum(row["split_role"] == "POLICY_TRAIN" for row in task_rows) != 100:
            raise RuntimeError(f"{task} train attempt count is not 100")
        if sum(row["split_role"] == "POLICY_DEV" for row in task_rows) != 25:
            raise RuntimeError(f"{task} dev attempt count is not 25")
    return {
        "schema": "tactile3d-unit.s4-3-pd-formal-attempts.v1",
        "stage": "PD5_PD6",
        "status": "FROZEN_BEFORE_FORMAL_ACQUISITION",
        "namespace": "PD_FORMAL",
        "source_dataset": "DexJoCo/DexJoCo-Datasets-Raw",
        "source_dataset_revision": RAW_DATASET_REVISION,
        "groups_per_task": FORMAL_GROUPS_PER_TASK,
        "train_groups_per_task": TRAIN_GROUPS_PER_TASK,
        "dev_groups_per_task": DEV_GROUPS_PER_TASK,
        "attempts_per_group": ATTEMPTS_PER_GROUP,
        "attempts_per_task": 125,
        "total_attempts": expected_attempts,
        "groups": groups,
        "attempts": attempts,
    }


def code_hashes() -> dict[str, str]:
    paths = {
        "expert_adapter": ROOT / "gr00t/simulation/s4_3_pd.py",
        "central_action_adapter": ROOT / "gr00t/simulation/dexjoco_adapter.py",
        "formal_generator": ROOT / "scripts/simulation/generate_s4_3_policy_expert_data.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def verify_frozen_protocol(manifest: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    manifest_path = artifact_root / "formal_attempt_manifest.json"
    freeze_path = artifact_root / "formal_protocol_freeze.json"
    if not manifest_path.is_file() or not freeze_path.is_file():
        raise RuntimeError("formal protocol artifacts are not frozen")
    frozen_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if frozen_manifest != manifest:
        raise RuntimeError("formal attempt manifest differs from the frozen protocol")
    if freeze.get("status") != "FROZEN_BEFORE_FORMAL_ACQUISITION":
        raise RuntimeError("formal protocol does not have frozen status")
    if freeze.get("formal_attempt_manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("formal attempt manifest checksum changed after freeze")
    if freeze.get("expert_code_hashes") != code_hashes():
        raise RuntimeError("expert/generator code changed after formal freeze")
    return freeze


def _archive_infrastructure_failure(
    attempt_dir: Path,
    row: dict[str, Any],
    error: BaseException,
    failure_root: Path,
) -> Path:
    parent = failure_root / row["attempt_id"]
    parent.mkdir(parents=True, exist_ok=True)
    index = len(list(parent.glob("failure-*")))
    destination = parent / f"failure-{index:02d}"
    destination.mkdir()
    if attempt_dir.exists():
        shutil.move(str(attempt_dir), str(destination / "partial_attempt"))
    write_json(
        destination / "infrastructure_failure.json",
        {
            "schema": "tactile3d-unit.s4-3-pd-infrastructure-failure.v1",
            "classification": "INFRASTRUCTURE_FAILURE",
            "rerun_policy": "EXACT_SAME_FROZEN_ATTEMPT_ONLY",
            "attempt": row,
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )
    return destination


def run_attempt(
    row: dict[str, Any],
    *,
    official_root: Path,
    formal_root: Path,
    artifact_root: Path,
    failure_root: Path,
    freeze: dict[str, Any],
) -> dict[str, Any]:
    from dexjoco.tasks import CONFIG_MAPPING
    from dexjoco.tasks.state_restorers import restore_initial_state

    attempt_dir = formal_root / row["attempt_id"]
    metadata_path = attempt_dir / "metadata.json"
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing["metadata"].get("expert_code_hashes") != code_hashes():
            raise RuntimeError(f"{row['attempt_id']}: completed attempt code hash mismatch")
        return existing
    if attempt_dir.exists():
        _archive_infrastructure_failure(
            attempt_dir,
            row,
            RuntimeError("partial attempt found before rerun"),
            failure_root,
        )

    source_path = source_partition(row["task"], official_root, "FORMAL")[
        row["source_group_index"]
    ]
    if source_path.parent.name != row["source_group_id"]:
        raise RuntimeError("formal source identity changed after protocol freeze")
    source = OfficialReplaySource.load(row["task"], source_path)
    frozen_source = freeze["source_groups"][row["task"]][row["source_group_index"]]
    if (
        frozen_source["source_group_id"] != source.source_group_id
        or frozen_source["source_replay_tree_sha256"] != source.source_tree_sha256
        or frozen_source["split_role"] != row["split_role"]
    ):
        raise RuntimeError("formal source checksum or split changed after protocol freeze")
    metadata = {
        **row,
        "source_class": "E0_OFFICIAL_DEMONSTRATION",
        "source_dataset": "DexJoCo/DexJoCo-Datasets-Raw",
        "source_dataset_revision": RAW_DATASET_REVISION,
        "source_replay_tree_sha256": source.source_tree_sha256,
        "dexjoco_revision": DEXJOCO_REVISION,
        "expert_code_hashes": code_hashes(),
        "formal_attempt_manifest_sha256": freeze["formal_attempt_manifest_sha256"],
        "tracked_config_sha256": freeze["tracked_config_sha256"],
        "camera": "random_camera",
        "visual_randomization": True,
        "dynamics_randomization": False,
        "initial_state_restore": "official task state restorer",
        "control_dt_sec": 0.02,
        "control_hz": 50.0,
        "max_hold_steps": MAX_HOLD_STEPS,
        "pinch_tail_recovery": {
            "open_source_fraction": 0.88,
            "open_steps": 30,
            "close_steps": 30,
            "object_state_mutation": False,
        }
        if row["task"] == "pinch_tongs"
        else None,
        "student_privileged_state": False,
    }
    adapter = DexJoCoRuntimeAdapter(
        task_name=row["task"],
        seed=row["environment_seed"],
        episode_id=row["attempt_id"],
        camera_name="random_camera",
        randomize=True,
        randomize_dynamics=False,
    )
    logger: S43PDAttemptLogger | None = None
    state_names: tuple[str, ...] = ()
    termination_reason = "EXPERT_FAILURE_SOURCE_EXHAUSTED"
    try:
        adapter.start()
        adapter.reset()
        adapter.raw_env.hz = 1_000_000
        restored = restore_initial_state(
            adapter.env,
            row["task"],
            CONFIG_MAPPING[row["task"]](),
            source.initial_state,
        )
        adapter.last_raw_observation = restored
        current = adapter._observation(restored, False, False, False)
        logger = S43PDAttemptLogger(attempt_dir, metadata)
        for step in range(source.steps + MAX_HOLD_STEPS):
            action, phase, source_step = expert_action_at(source, step)
            current_diagnostics = dict(adapter.last_diagnostics)
            next_observation, reward, info, env_action = adapter.step(action)
            state, state_names, progress = native_task_state(
                row["task"], adapter.raw_env, info
            )
            logger.append(
                current,
                action,
                env_action,
                next_observation,
                reward,
                info,
                phase,
                source_step,
                current_diagnostics,
                state,
                progress,
            )
            current = next_observation
            if next_observation.success:
                termination_reason = "NATIVE_SUCCESS"
                break
            if next_observation.terminated or next_observation.truncated:
                termination_reason = "EXPERT_FAILURE_NATIVE_TERMINATION"
                break
        assert logger is not None
        return logger.finish(
            termination_reason=termination_reason, native_state_names=state_names
        )
    except BaseException as error:
        _archive_infrastructure_failure(attempt_dir, row, error, failure_root)
        raise
    finally:
        adapter.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--attempt-id")
    parser.add_argument("--official-root", type=Path, default=OFFICIAL_ROOT)
    parser.add_argument("--formal-root", type=Path, default=FORMAL_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument(
        "--failure-root", type=Path, default=INFRASTRUCTURE_FAILURE_ROOT
    )
    parser.add_argument("--freeze-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = formal_attempt_manifest(args.official_root)
    manifest_path = args.artifact_root / "formal_attempt_manifest.json"
    if args.freeze_only:
        write_json(manifest_path, manifest)
        print(json.dumps({"attempts": manifest["total_attempts"], "status": "FROZEN"}))
        return
    if os.environ.get("DISPLAY"):
        raise SystemExit("DISPLAY must be unset for headless formal generation")
    if os.environ.get("MUJOCO_GL") != "egl":
        raise SystemExit("MUJOCO_GL=egl is required for visual formal generation")
    freeze = verify_frozen_protocol(manifest, args.artifact_root)
    selected = manifest["attempts"]
    if args.task:
        selected = [row for row in selected if row["task"] == args.task]
    if args.attempt_id:
        selected = [row for row in selected if row["attempt_id"] == args.attempt_id]
    if not selected:
        raise SystemExit("no formal attempts selected")
    completed = []
    for row in selected:
        result = run_attempt(
            row,
            official_root=args.official_root,
            formal_root=args.formal_root,
            artifact_root=args.artifact_root,
            failure_root=args.failure_root,
            freeze=freeze,
        )
        completed.append(
            {"attempt_id": result["attempt_id"], "success": result["success"]}
        )
        print(json.dumps(completed[-1], sort_keys=True), flush=True)
    print(
        json.dumps(
            {
                "attempts": len(completed),
                "successes": sum(row["success"] for row in completed),
                "status": "COMPLETE",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
