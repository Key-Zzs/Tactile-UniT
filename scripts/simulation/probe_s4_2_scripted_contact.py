#!/usr/bin/env python3
"""Run an unstored pre-generation contact probe for frozen S4.2 task offsets."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.dexjoco_adapter import (  # noqa: E402
    DexJoCoRuntimeAdapter,
    SimPolicyAction,
    proprio_to_neutral_policy_action,
)

CONTRACT = ROOT / "configs/simulation/s4_2_dataset_contract.json"
REGIONS = ROOT / "configs/simulation/s4_2_dexjoco_contact_regions.json"


def probe(task: str, config: dict) -> dict:
    adapter = DexJoCoRuntimeAdapter(
        task_name=task,
        seed=config["tasks"][task]["seed_base"],
        episode_id=f"probe-{task}",
        randomize=False,
        randomize_dynamics=False,
        region_config=REGIONS,
    )
    region_audit = adapter.start()
    observation = adapter.reset()
    initial = observation.proprio.copy()
    neutral = proprio_to_neutral_policy_action(initial).values
    start, stop = config["tasks"][task]["primary_object_state_slice"]
    object_xyz = initial[start:stop][:3]
    offset = np.asarray(config["script_family"]["base_contact_offsets_m"][task])
    contact_target = object_xyz + offset
    release_target = contact_target + np.asarray([0.0, 0.0, 0.30])
    tactile = []
    try:
        for step in range(130):
            if step < 35:
                alpha = (step + 1) / 35.0
                xyz = neutral[:3] * (1.0 - alpha) + contact_target * alpha
            elif step < 90:
                xyz = contact_target.copy()
                if step >= 60:
                    xyz[1] += 0.010 * np.sin((step - 60) * np.pi / 15.0)
            else:
                alpha = (step - 89) / 40.0
                xyz = contact_target * (1.0 - alpha) + release_target * alpha
            action = SimPolicyAction(np.concatenate([xyz, neutral[3:]]))
            observation, _, _, _ = adapter.step(action)
            tactile.append(observation.sim_tactile.copy())
    finally:
        adapter.close()
    values = np.stack(tactile).reshape(-1, 5, 6)
    normal = values[:, :, 1].sum(axis=1)
    tangent = values[:, :, 2].sum(axis=1)
    active = np.flatnonzero(normal > 1e-6)
    return {
        "task": task,
        "region_audit": region_audit["status"],
        "contact_target": contact_target.tolist(),
        "active_steps": int(len(active)),
        "first_contact_step": None if not len(active) else int(active[0]),
        "last_contact_step": None if not len(active) else int(active[-1]),
        "maximum_normal_force": float(normal.max()),
        "maximum_tangential_force": float(tangent.max()),
        "active_regions": np.flatnonzero(values[:, :, 0].sum(axis=0) > 0).tolist(),
        "final_occupancy": float(values[-1, :, 0].max()),
    }


def main() -> None:
    if os.environ.get("DISPLAY"):
        raise SystemExit("DISPLAY must be unset")
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    results = [probe(task, config) for task in config["tasks"]]
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
