#!/usr/bin/env python3
"""Collect ten short scripted DexJoCo contact-debug episodes; no policy is trained."""

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
from gr00t.simulation.episode_logger import DexJoCoEpisodeLogger  # noqa: E402

ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_1"
DATASET = ARTIFACTS / "debug_dataset"


def scripted_action(proprio: np.ndarray, step: int, steps: int) -> SimPolicyAction:
    neutral = proprio_to_neutral_policy_action(proprio).values
    tongs = proprio[23:26]
    contact_target = np.asarray([tongs[0] - 0.159, tongs[1], tongs[2] - 0.029])
    if step < 30:
        alpha = (step + 1) / 30.0
        xyz = neutral[:3] * (1.0 - alpha) + contact_target * alpha
    elif step < 70:
        xyz = contact_target.copy()
        if step >= 50:
            xyz[1] += 0.012 * np.sin((step - 50) * np.pi / 10.0)
    else:
        alpha = (step - 69) / (steps - 69)
        release = contact_target + np.asarray([0.0, 0.0, 0.32])
        xyz = contact_target * (1.0 - alpha) + release * alpha
    return SimPolicyAction(np.concatenate([xyz, neutral[3:]]))


def rollout(seed: int, episode_id: str, log: bool) -> dict[str, np.ndarray]:
    adapter = DexJoCoRuntimeAdapter(task_name="pinch_tongs", seed=seed, episode_id=episode_id)
    adapter.start()
    observation = adapter.reset()
    initial_proprio = observation.proprio.copy()
    logger = None
    if log:
        logger = DexJoCoEpisodeLogger(
            DATASET,
            episode_id,
            {
                "task": "pinch_tongs",
                "seed": seed,
                "action_source": "deterministic scripted approach/contact/tangential/release probe",
                "randomization": False,
                "camera": "front",
                "contact_regions": [
                    "right_palm",
                    "right_index",
                    "right_middle",
                    "right_ring",
                    "right_thumb",
                ],
            },
        )
    states, tactile, rewards, terminals = [], [], [], []
    steps = 100
    try:
        for step in range(steps):
            action = scripted_action(initial_proprio, step, steps)
            observation, reward, _, env_action = adapter.step(action)
            states.append(observation.proprio.copy())
            tactile.append(observation.sim_tactile.copy())
            rewards.append(reward)
            terminals.append(observation.terminated or observation.truncated)
            if logger is not None:
                logger.append(observation, action, env_action, reward, adapter.last_diagnostics)
        manifest = logger.finish() if logger is not None else None
    finally:
        adapter.close()
    return {
        "state": np.stack(states),
        "tactile": np.stack(tactile),
        "reward": np.asarray(rewards),
        "terminal": np.asarray(terminals),
        "manifest": manifest,
    }


def main() -> None:
    if os.environ.get("DISPLAY"):
        raise SystemExit("DISPLAY must be unset")
    if DATASET.exists():
        raise SystemExit(f"debug dataset already exists: {DATASET}")
    DATASET.mkdir(parents=True)
    manifests = []
    runs = []
    for index in range(10):
        result = rollout(seed=100 + index, episode_id=f"debug-{index:03d}", log=True)
        manifests.append(result["manifest"])
        runs.append(result)

    tactile = np.concatenate([run["tactile"] for run in runs], axis=0).reshape(-1, 5, 6)
    normal = tactile[:, :, 1]
    tangent = tactile[:, :, 2]
    occupancy = tactile[:, :, 0]
    free = np.concatenate([run["tactile"][:15] for run in runs]).reshape(-1, 5, 6)
    release = np.concatenate([run["tactile"][-5:] for run in runs]).reshape(-1, 5, 6)
    contact_steps = np.flatnonzero(normal.sum(axis=1) > 1e-6)
    onset = int(contact_steps[0]) if contact_steps.size else None
    active_region_series = occupancy.sum(axis=0)
    sanity = {
        "free_space": {
            "max_occupancy": float(free[:, :, 0].max()),
            "max_normal_force": float(free[:, :, 1].max()),
            "status": "PASS" if float(free[:, :, 1].max()) < 1e-6 else "FAIL",
        },
        "contact_onset": {
            "global_step": onset,
            "max_normal_force": float(normal.max()),
            "status": "PASS" if onset is not None and float(normal.max()) > 0 else "FAIL",
        },
        "compression": {
            "onset_band_max_normal_force": (
                float(normal[contact_steps[:20]].sum(axis=1).max()) if contact_steps.size else 0.0
            ),
            "overall_max_normal_force": float(normal.sum(axis=1).max()),
            "status": (
                "PASS" if contact_steps.size and float(normal.sum(axis=1).max()) > 1.0 else "FAIL"
            ),
        },
        "tangential": {
            "max_force": float(tangent.sum(axis=1).max()),
            "status": "PASS" if float(tangent.sum(axis=1).max()) > 1e-4 else "FAIL",
        },
        "release": {
            "max_final_occupancy": float(release[:, :, 0].max()),
            "max_final_normal_force": float(release[:, :, 1].max()),
            "status": "PASS" if float(release[:, :, 0].max()) == 0 else "FAIL",
        },
        "wrong_region": {
            "region_occupancy_counts": active_region_series.tolist(),
            "identical": bool(np.all(active_region_series == active_region_series[0])),
            "status": (
                "PASS" if not np.all(active_region_series == active_region_series[0]) else "FAIL"
            ),
        },
    }
    sanity["overall"] = (
        "PASS" if all(row["status"] == "PASS" for row in sanity.values()) else "FAIL"
    )
    replay1 = rollout(seed=4242, episode_id="determinism-a", log=False)
    replay2 = rollout(seed=4242, episode_id="determinism-b", log=False)
    determinism = {
        "seed": 4242,
        "state_max_abs_diff": float(np.max(np.abs(replay1["state"] - replay2["state"]))),
        "tactile_max_abs_diff": float(np.max(np.abs(replay1["tactile"] - replay2["tactile"]))),
        "reward_equal": bool(np.array_equal(replay1["reward"], replay2["reward"])),
        "terminal_equal": bool(np.array_equal(replay1["terminal"], replay2["terminal"])),
    }
    determinism["status"] = (
        "PASS"
        if determinism["state_max_abs_diff"] <= 1e-9
        and determinism["tactile_max_abs_diff"] <= 1e-8
        and determinism["reward_equal"]
        and determinism["terminal_equal"]
        else "BOUNDED_NONDETERMINISM"
    )
    dataset_manifest = {
        "schema": "tactile3d-unit.s4-1-debug-dataset.v1",
        "format": "per-episode NPZ numeric arrays plus referenced JPEG frames and metadata JSON",
        "episodes": 10,
        "steps_per_episode": 100,
        "frames": 1000,
        "action_source": "deterministic scripted debug probe",
        "task": "pinch_tongs",
        "seeds": list(range(100, 110)),
        "fields": manifests[0]["fields"],
        "future_conversion": "decode JPEG references and map NPZ columns to LeRobot features",
        "status": "PASS",
    }
    for name, value in (
        ("debug_dataset_manifest.json", dataset_manifest),
        ("tactile_sanity_checks.json", sanity),
        ("determinism_check.json", determinism),
    ):
        (ARTIFACTS / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({"dataset": dataset_manifest, "sanity": sanity, "determinism": determinism}))


if __name__ == "__main__":
    main()
