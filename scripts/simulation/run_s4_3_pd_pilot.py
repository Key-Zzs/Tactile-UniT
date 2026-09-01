#!/usr/bin/env python3
"""Run the fixed 20-attempt/task S4.3-PD official-replay pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.dexjoco_adapter import DexJoCoRuntimeAdapter  # noqa: E402
from gr00t.simulation.s4_3_pd import (  # noqa: E402
    ATTEMPTS_PER_GROUP,
    PILOT_GROUPS_PER_TASK,
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
PILOT_ROOT = ROOT / ".local/datasets/simulation/s4_3_policy_expert/pilot/attempts"
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_pd"
RAW_DATASET_REVISION = "125df5c1019e97503929ef1a0ad8f90373436afa"
DEXJOCO_REVISION = "8d23b0fab23b17a58c4b55f3942e17013aaf8267"
MAX_HOLD_STEPS = 120


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def pilot_seed_manifest(official_root: Path = OFFICIAL_ROOT) -> dict[str, Any]:
    attempts = []
    for task_index, task in enumerate(TASKS):
        groups = source_partition(task, official_root, "PD_PILOT")
        for group_index, source_path in enumerate(groups):
            for attempt_index in range(ATTEMPTS_PER_GROUP):
                env_seed = 4_300_000 + task_index * 10_000 + group_index * 100 + attempt_index
                attempts.append(
                    {
                        "task": task,
                        "source_group_index": group_index,
                        "source_group_id": source_path.parent.name,
                        "attempt_index": attempt_index,
                        "attempt_id": f"pilot-{task}-g{group_index:02d}-a{attempt_index:02d}",
                        "seed_namespace": "PD_PILOT",
                        "reset_seed": env_seed,
                        "environment_seed": env_seed,
                        "visual_randomization_seed": env_seed,
                        "perturbation_seed": 4_400_000
                        + task_index * 10_000
                        + group_index * 100
                        + attempt_index,
                        "controller_seed": None,
                    }
                )
    expected = len(TASKS) * PILOT_GROUPS_PER_TASK * ATTEMPTS_PER_GROUP
    if len(attempts) != expected or len({row["attempt_id"] for row in attempts}) != expected:
        raise RuntimeError("pilot manifest cardinality or identity failure")
    seed_tuples = {
        (row["reset_seed"], row["perturbation_seed"], row["controller_seed"])
        for row in attempts
    }
    if len(seed_tuples) != expected:
        raise RuntimeError("pilot seed tuples are not unique")
    return {
        "schema": "tactile3d-unit.s4-3-pd-pilot-seeds.v1",
        "stage": "PD4",
        "namespace": "PD_PILOT",
        "groups_per_task": PILOT_GROUPS_PER_TASK,
        "attempts_per_group": ATTEMPTS_PER_GROUP,
        "attempts_per_task": PILOT_GROUPS_PER_TASK * ATTEMPTS_PER_GROUP,
        "total_attempts": expected,
        "formal_membership": False,
        "attempts": attempts,
        "status": "FROZEN_FOR_PILOT",
    }


def code_hashes() -> dict[str, str]:
    paths = {
        "expert_adapter": ROOT / "gr00t/simulation/s4_3_pd.py",
        "central_action_adapter": ROOT / "gr00t/simulation/dexjoco_adapter.py",
        "pilot_runner": ROOT / "scripts/simulation/run_s4_3_pd_pilot.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def run_attempt(row: dict[str, Any], *, official_root: Path, pilot_root: Path) -> dict[str, Any]:
    from dexjoco.tasks import CONFIG_MAPPING
    from dexjoco.tasks.state_restorers import restore_initial_state

    attempt_dir = pilot_root / row["attempt_id"]
    metadata_path = attempt_dir / "metadata.json"
    if metadata_path.is_file():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    if attempt_dir.exists():
        raise RuntimeError(f"refusing partial existing attempt: {attempt_dir}")

    source_path = source_partition(row["task"], official_root, "PD_PILOT")[
        row["source_group_index"]
    ]
    if source_path.parent.name != row["source_group_id"]:
        raise RuntimeError("pilot source identity changed after seed freeze")
    source = OfficialReplaySource.load(row["task"], source_path)
    metadata = {
        **row,
        "split_role": "PD_PILOT",
        "source_class": "E0_OFFICIAL_DEMONSTRATION",
        "source_dataset": "DexJoCo/DexJoCo-Datasets-Raw",
        "source_dataset_revision": RAW_DATASET_REVISION,
        "source_replay_tree_sha256": source.source_tree_sha256,
        "dexjoco_revision": DEXJOCO_REVISION,
        "expert_code_hashes": code_hashes(),
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
    finally:
        adapter.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--attempt-id")
    parser.add_argument("--official-root", type=Path, default=OFFICIAL_ROOT)
    parser.add_argument("--pilot-root", type=Path, default=PILOT_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--freeze-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pilot_seed_manifest(args.official_root)
    write_json(args.artifact_root / "pilot_seed_manifest.json", manifest)
    if args.freeze_only:
        print(json.dumps({"attempts": manifest["total_attempts"], "status": "FROZEN"}))
        return
    if os.environ.get("DISPLAY"):
        raise SystemExit("DISPLAY must be unset for headless pilot generation")
    if os.environ.get("MUJOCO_GL") != "egl":
        raise SystemExit("MUJOCO_GL=egl is required for the visual pilot")
    selected = manifest["attempts"]
    if args.task:
        selected = [row for row in selected if row["task"] == args.task]
    if args.attempt_id:
        selected = [row for row in selected if row["attempt_id"] == args.attempt_id]
    if not selected:
        raise SystemExit("no pilot attempts selected")
    completed = []
    for row in selected:
        result = run_attempt(
            row, official_root=args.official_root, pilot_root=args.pilot_root
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
