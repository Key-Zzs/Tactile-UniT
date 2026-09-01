#!/usr/bin/env python3
"""Replay representative frozen S4.2 trajectories and audit raw thumb contacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.dexjoco_adapter import DexJoCoRuntimeAdapter  # noqa: E402
from gr00t.simulation.s4_2_dataset import DEFAULT_DATASET_ROOT  # noqa: E402
from scripts.simulation.generate_s4_2_dataset import (  # noqa: E402
    CONTRACT_PATH,
    REGION_PATH,
    action_for_step,
    group_parameters,
)

ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_2r"
TASKS = ("pinch_tongs", "hammer_nail", "click_mouse")


def geom_descriptor(model: Any, mujoco: Any, geom_id: int) -> dict[str, Any]:
    body_id = int(model.geom_bodyid[geom_id])
    body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    geom = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    return {
        "geom_id": geom_id,
        "geom_name": geom,
        "body_id": body_id,
        "body_name": body,
        "descriptor": geom or f"body:{body}#collision-geom",
    }


def replay_task(
    task: str, contract: dict[str, Any], dataset_root: Path
) -> dict[str, Any]:
    import mujoco

    group = 0
    perturbation = 0
    task_contract = contract["tasks"][task]
    seed = int(task_contract["seed_base"])
    episode_id = f"{task}-g00-p00"
    adapter = DexJoCoRuntimeAdapter(
        task_name=task,
        seed=seed,
        episode_id=f"s4-2r-thumb-audit-{task}",
        randomize=False,
        randomize_dynamics=True,
        region_config=REGION_PATH,
    )
    region_audit = adapter.start()
    observation = adapter.reset()
    adapter.raw_env.hz = 1_000_000
    initial = observation.proprio.copy()
    start, stop = task_contract["primary_object_state_slice"]
    offset = np.asarray(contract["script_family"]["base_contact_offsets_m"][task])
    offset += np.asarray(contract["script_family"]["perturbation_xyz_m"][perturbation])
    contact_target = initial[start:stop][:3] + offset
    parameters = group_parameters(group, contract)
    thumb_geom_ids = sorted(
        geom_id
        for geom_id, region in adapter.region_map.geom_to_region.items()  # type: ignore[union-attr]
        if region == "right_thumb"
    )
    resolved_thumb_geoms = [
        geom_descriptor(adapter.raw_env.model, mujoco, geom_id) for geom_id in thumb_geom_ids
    ]
    object_bodies = adapter.region_map.object_body_names  # type: ignore[union-attr]
    raw_thumb_pairs: Counter[str] = Counter()
    raw_thumb_object_pairs: Counter[str] = Counter()
    raw_thumb_contact_steps = 0
    raw_thumb_object_contact_steps = 0
    extractor_thumb_pairs = 0
    tactile_thumb_active_steps = 0
    maximum_replay_tactile_difference = 0.0
    with np.load(
        dataset_root / "episodes" / episode_id / "steps.npz", allow_pickle=False
    ) as frozen:
        frozen_tactile = frozen["sim_tactile"].copy()
    try:
        for step in range(int(contract["episode_plan"]["steps"])):
            action = action_for_step(initial, contact_target, parameters, step)
            observation, _, _, _ = adapter.step(action)
            maximum_replay_tactile_difference = max(
                maximum_replay_tactile_difference,
                float(np.max(np.abs(observation.sim_tactile - frozen_tactile[step]))),
            )
            thumb_this_step = False
            thumb_object_this_step = False
            model = adapter.raw_env.model
            data = adapter.raw_env.data
            for contact_id in range(data.ncon):
                contact = data.contact[contact_id]
                geom1 = int(contact.geom1)
                geom2 = int(contact.geom2)
                region1 = adapter.region_map.region_for_geom(geom1)  # type: ignore[union-attr]
                region2 = adapter.region_map.region_for_geom(geom2)  # type: ignore[union-attr]
                if region1 != "right_thumb" and region2 != "right_thumb":
                    continue
                thumb_this_step = True
                first = geom_descriptor(model, mujoco, geom1)
                second = geom_descriptor(model, mujoco, geom2)
                pair = f"{first['descriptor']} <-> {second['descriptor']}"
                raw_thumb_pairs[pair] += 1
                body1 = first["body_name"]
                body2 = second["body_name"]
                matched_object = (region1 == "right_thumb" and body2 in object_bodies) or (
                    region2 == "right_thumb" and body1 in object_bodies
                )
                if matched_object:
                    thumb_object_this_step = True
                    raw_thumb_object_pairs[pair] += 1
            raw_thumb_contact_steps += int(thumb_this_step)
            raw_thumb_object_contact_steps += int(thumb_object_this_step)
            extractor_thumb_pairs += sum(
                row["region"] == "right_thumb"
                for row in adapter.last_diagnostics.get("matched_pairs", [])
            )
            tactile_thumb_active_steps += int(
                observation.sim_tactile.reshape(5, 6)[4, 0] > 0.5
            )
    finally:
        adapter.close()
    return {
        "task": task,
        "episode_id": episode_id,
        "seed": seed,
        "steps": int(contract["episode_plan"]["steps"]),
        "region_audit": region_audit,
        "resolved_thumb_geom_ids": thumb_geom_ids,
        "resolved_thumb_geoms": resolved_thumb_geoms,
        "raw_thumb_contact_steps_all_counterparts": raw_thumb_contact_steps,
        "raw_thumb_contact_pairs_all_counterparts": dict(sorted(raw_thumb_pairs.items())),
        "raw_thumb_object_contact_steps": raw_thumb_object_contact_steps,
        "raw_thumb_object_contact_pairs": dict(sorted(raw_thumb_object_pairs.items())),
        "extractor_thumb_matched_pairs": extractor_thumb_pairs,
        "tactile_thumb_active_steps": tactile_thumb_active_steps,
        "maximum_abs_tactile_difference_vs_frozen_episode": maximum_replay_tactile_difference,
        "deterministic_replay_matches_frozen": maximum_replay_tactile_difference == 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    return parser.parse_args()


def main() -> None:
    if os.environ.get("DISPLAY"):
        raise SystemExit("DISPLAY must be unset for deterministic headless replay")
    args = parse_args()
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    rows = [replay_task(task, contract, args.dataset) for task in TASKS]
    raw_object_contacts = sum(row["raw_thumb_object_contact_steps"] for row in rows)
    tactile_activity = sum(row["tactile_thumb_active_steps"] for row in rows)
    mapping_bug = raw_object_contacts > 0 and tactile_activity == 0
    result = {
        "schema": "tactile3d-unit.s4-2r-thumb-contact-audit.v1",
        "method": "deterministic replay of frozen group-00 perturbation-00 for each task",
        "tasks": rows,
        "raw_thumb_object_contact_steps": raw_object_contacts,
        "tactile_thumb_active_steps": tactile_activity,
        "mapping_bug": mapping_bug,
        "classification": (
            "CONTACT_REGION_MAPPING_BUG"
            if mapping_bug
            else "THUMB_INACTIVITY_IS_DATA_COVERAGE"
        ),
        "test_loaded": False,
        "selection_uses_test": False,
    }
    args.artifacts.mkdir(parents=True, exist_ok=True)
    destination = args.artifacts / "thumb_contact_audit.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if mapping_bug:
        raise SystemExit("S4_2R_CONTACT_MAPPING_FAIL")


if __name__ == "__main__":
    main()
