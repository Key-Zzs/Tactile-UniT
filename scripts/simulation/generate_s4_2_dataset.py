#!/usr/bin/env python3
"""Generate the frozen formal S4.2 DexJoCo Vision/Action/Contact dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.dexjoco_adapter import (  # noqa: E402
    DexJoCoRuntimeAdapter,
    SimPolicyAction,
    proprio_to_neutral_policy_action,
)
from gr00t.simulation.episode_logger import DexJoCoEpisodeLogger  # noqa: E402
from gr00t.simulation.s4_2_dataset import DEFAULT_DATASET_ROOT  # noqa: E402

CONTRACT_PATH = ROOT / "configs/simulation/s4_2_dataset_contract.json"
SPLIT_PATH = ROOT / "configs/simulation/s4_2_dataset_split_manifest.json"
REGION_PATH = ROOT / "configs/simulation/s4_2_dexjoco_contact_regions.json"


def split_lookup(manifest: dict[str, Any]) -> dict[tuple[str, str], str]:
    result = {}
    for task, splits in manifest["groups"].items():
        for split, groups in splits.items():
            for group in groups:
                key = (task, group)
                if key in result:
                    raise ValueError(f"duplicate source group {key}")
                result[key] = split
    return result


def group_parameters(group: int, contract: dict[str, Any]) -> dict[str, Any]:
    bounds = contract["script_family"]["bounded_group_variation"]
    return {
        "approach_steps": int(bounds["approach_steps"][0] + group % 9),
        "compression_m": float(
            np.linspace(*bounds["compression_m"], 20, dtype=np.float64)[group]
        ),
        "tangential_amplitude_m": float(
            np.linspace(*bounds["tangential_amplitude_m"], 20, dtype=np.float64)[group]
        ),
        "tangential_cycles": int(bounds["tangential_cycles"][0] + group % 3),
        "tangential_axis": "y" if group % 2 == 0 else "z",
    }


def action_for_step(
    initial: np.ndarray,
    contact_target: np.ndarray,
    parameters: dict[str, Any],
    step: int,
) -> SimPolicyAction:
    neutral = proprio_to_neutral_policy_action(initial).values
    approach_steps = parameters["approach_steps"]
    compressed = contact_target.copy()
    compressed[0] += parameters["compression_m"]
    release = compressed + np.asarray([0.0, 0.0, 0.30])
    if step < approach_steps:
        alpha = (step + 1) / approach_steps
        xyz = neutral[:3] * (1.0 - alpha) + contact_target * alpha
    elif step < 60:
        alpha = (step - approach_steps + 1) / max(1, 60 - approach_steps)
        xyz = contact_target * (1.0 - alpha) + compressed * alpha
    elif step < 100:
        phase = (step - 60) / 40.0
        xyz = compressed.copy()
        axis = 1 if parameters["tangential_axis"] == "y" else 2
        xyz[axis] += parameters["tangential_amplitude_m"] * np.sin(
            2.0 * np.pi * parameters["tangential_cycles"] * phase
        )
    elif step < 130:
        alpha = (step - 99) / 30.0
        xyz = compressed * (1.0 - alpha) + release * alpha
    else:
        xyz = release
    return SimPolicyAction(np.concatenate([xyz, neutral[3:]]))


def randomization_parameters(adapter: DexJoCoRuntimeAdapter, initial: np.ndarray) -> dict:
    raw = adapter.raw_env
    task = adapter.task_name
    body_names = sorted(adapter.region_map.object_body_names)  # type: ignore[union-attr]
    value: dict[str, Any] = {
        "table_delta_height_m": float(raw.delta_h),
        "primary_object_pose": initial[23:30].tolist(),
        "object_body_mass_kg": {
            name: float(raw.model.body_mass[raw.model.body(name).id]) for name in body_names
        },
    }
    if task == "pinch_tongs":
        value["object_joint_frictionloss"] = float(
            raw.model.dof_frictionloss[raw._tongs_dof_id]
        )
        value["object_joint_stiffness"] = float(
            raw.model.jnt_stiffness[raw._tongs_joint_id]
        )
    return value


def generate_episode(
    root: Path,
    task: str,
    group: int,
    perturbation: int,
    split: str,
    contract: dict[str, Any],
) -> dict[str, Any]:
    task_contract = contract["tasks"][task]
    source_id = f"{task}-script-{group:02d}"
    episode_id = f"{task}-g{group:02d}-p{perturbation:02d}"
    episode_dir = root / "episodes" / episode_id
    if (episode_dir / "metadata.json").is_file():
        return json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    if episode_dir.exists():
        raise RuntimeError(f"refusing partial existing episode directory: {episode_dir}")

    seed = int(task_contract["seed_base"] + 10 * group + perturbation)
    parameters = group_parameters(group, contract)
    offset = np.asarray(contract["script_family"]["base_contact_offsets_m"][task])
    offset += np.asarray(contract["script_family"]["perturbation_xyz_m"][perturbation])
    adapter = DexJoCoRuntimeAdapter(
        task_name=task,
        seed=seed,
        episode_id=episode_id,
        randomize=False,
        randomize_dynamics=True,
        region_config=REGION_PATH,
    )
    region_audit = adapter.start()
    observation = adapter.reset()
    adapter.raw_env.hz = 1_000_000
    initial = observation.proprio.copy()
    start, stop = task_contract["primary_object_state_slice"]
    contact_target = initial[start:stop][:3] + offset
    randomization = randomization_parameters(adapter, initial)
    metadata = {
        "episode_schema": "tactile3d-unit.dexjoco-s4-2-formal-episode.v1",
        "task": task,
        "split": split,
        "seed": seed,
        "source_type": "repository_owned_deterministic_script",
        "source_trajectory_id": source_id,
        "randomization_id": "native_seeded_id_dynamics",
        "randomization_parameters": randomization,
        "perturbation_id": perturbation,
        "perturbation_xyz_m": contract["script_family"]["perturbation_xyz_m"][perturbation],
        "script_parameters": parameters,
        "contact_target_m": contact_target.tolist(),
        "camera": "front",
        "visual_randomization": False,
        "dynamics_randomization": True,
        "wall_clock_pacing": "disabled_without_changing_physical_dt",
        "control_dt_sec": 0.02,
        "physics_dt_sec": 0.002,
        "region_audit": region_audit,
    }
    logger = DexJoCoEpisodeLogger(root / "episodes", episode_id, metadata)
    try:
        for step in range(int(contract["episode_plan"]["steps"])):
            action = action_for_step(initial, contact_target, parameters, step)
            observation, reward, _, env_action = adapter.step(action)
            logger.append(observation, action, env_action, reward, adapter.last_diagnostics)
        return logger.finish()
    finally:
        adapter.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--task", choices=("pinch_tongs", "hammer_nail", "click_mouse"))
    parser.add_argument("--group-start", type=int, default=0)
    parser.add_argument("--group-stop", type=int, default=20)
    parser.add_argument("--perturbation-stop", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    if os.environ.get("DISPLAY"):
        raise SystemExit("DISPLAY must be unset")
    args = parse_args()
    if not 0 <= args.group_start < args.group_stop <= 20:
        raise SystemExit("group range must lie within frozen [0,20)")
    if not 1 <= args.perturbation_stop <= 5:
        raise SystemExit("perturbation stop must lie within frozen [1,5]")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    split_manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    lookup = split_lookup(split_manifest)
    tasks = [args.task] if args.task else list(contract["tasks"])
    args.root.mkdir(parents=True, exist_ok=True)
    manifests = []
    for task in tasks:
        for group in range(args.group_start, args.group_stop):
            source_id = f"{task}-script-{group:02d}"
            for perturbation in range(args.perturbation_stop):
                manifest = generate_episode(
                    args.root,
                    task,
                    group,
                    perturbation,
                    lookup[(task, source_id)],
                    contract,
                )
                manifests.append(manifest["episode_id"])
                print(json.dumps({"episode": manifest["episode_id"], "status": "COMPLETE"}))
    print(json.dumps({"episodes": len(manifests), "root": str(args.root), "status": "COMPLETE"}))


if __name__ == "__main__":
    main()
