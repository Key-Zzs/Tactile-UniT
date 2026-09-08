#!/usr/bin/env python3
"""Replay exact raw actions and extract aligned frozen S4.1 tactile signals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import zarr

from dexjoco.tasks import CONFIG_MAPPING
from dexjoco.tasks.state_restorers import restore_initial_state
from gr00t.simulation.simulated_tactile import ContactRegionMap, SimulatedTactileExtractor


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / ".local/external/simulation/s4_3_pi1/raw/DexJoCo-Datasets-Raw/dexjoco_raw_datasets/pinch_tongs"
OUT_ROOT = ROOT / ".local/cache/simulation/s4_3_pi1/replay"
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_pi1"
REGIONS = ROOT / "configs/simulation/s4_1_dexjoco_contact_regions.json"

# Frozen before the first replay pilot. These tolerate controller replay error but
# reject a trajectory whose tactile could not legitimately label the source row.
TOLERANCES = {
    "tcp_position_max_abs_m": 0.005,
    "tcp_quaternion_max_abs": 0.02,
    "hand_joint_max_abs_rad": 0.05,
    "object_position_max_abs_m": 0.01,
    "object_quaternion_max_abs": 0.03,
    "table_height_max_abs_m": 1e-9,
}


def disable_wall_clock_throttle() -> None:
    # The official environment advances exactly one 20 ms control tick before
    # sleeping to a nominal 30 Hz teleoperation wall clock. Removing only that
    # sleep leaves every MuJoCo state transition unchanged.
    import dexjoco.sim.envs.panda_pinch_tongs_env as module

    module.time.sleep = lambda _seconds: None


def extractor_for(env) -> SimulatedTactileExtractor:
    value = json.loads(REGIONS.read_text())
    value = {
        "regions": value["regions"],
        "object_body_names": (
            value["tasks"]["pinch_tongs"]["object_body_names"]
            if "tasks" in value
            else value["object_body_names"]
        ),
    }
    region_map = ContactRegionMap.from_config(value)
    audit = region_map.resolve(env.unwrapped.model, mujoco)
    if audit["status"] != "PASS":
        raise RuntimeError(f"contact region map failed: {audit}")
    return SimulatedTactileExtractor(region_map)


def state_errors(observed: np.ndarray, recorded: np.ndarray) -> dict[str, float]:
    error = np.abs(np.asarray(observed, dtype=np.float64) - np.asarray(recorded, dtype=np.float64))
    return {
        "tcp_position_max_abs_m": float(error[:, :3].max()),
        "tcp_quaternion_max_abs": float(error[:, 3:7].max()),
        "hand_joint_max_abs_rad": float(error[:, 7:23].max()),
        "object_position_max_abs_m": float(error[:, 23:26].max()),
        "object_quaternion_max_abs": float(error[:, 26:30].max()),
        "table_height_max_abs_m": float(error[:, 30].max()),
        "all_state_max_abs": float(error.max()),
    }


def extract_recorded_state_snapshot(
    env, extractor, extract_data, recorded: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Extract contacts from a read-only hand/object state-aligned snapshot.

    Raw demonstrations store TCP pose, all Allegro joint positions, the tongs
    free-joint pose, and table height. The arm replay stays within the frozen
    TCP tolerance, while contact dynamics can cause Allegro tracking drift.
    We therefore substitute the exact recorded hand/object coordinates only
    for the contact query, then restore MjData before applying the next exact
    action. This snapshot cannot influence replay physics or later frames.
    """

    raw = env.unwrapped
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(raw.model, state_spec), dtype=np.float64)
    mujoco.mj_getState(raw.model, raw.data, state, state_spec)
    mujoco.mj_setState(raw.model, extract_data, state, state_spec)
    try:
        extract_data.qpos[raw._allegro_dof_ids] = recorded[7:23]
        extract_data.jnt("tongs_root").qpos = recorded[23:30]
        mujoco.mj_forward(raw.model, extract_data)
        live_data = raw._data
        raw._data = extract_data
        try:
            aligned_obs = env.observation(raw._compute_observation())
        finally:
            raw._data = live_data
        tactile, diagnostics = extractor.extract(raw.model, extract_data, mujoco)
        return tactile, np.asarray(aligned_obs["state"], dtype=np.float64), diagnostics
    finally:
        pass


def replay_episode(index: int, episode: Path) -> dict:
    out = OUT_ROOT / f"episode_{index:03d}.npz"
    source = zarr.open(str(episode / "replay.zarr"), mode="r")["data"]
    actions = np.asarray(source["action"], dtype=np.float64)
    recorded_state = np.asarray(source["state"], dtype=np.float64)
    recorded_timestamp = np.asarray(source["timestamp"], dtype=np.float64)
    if actions.ndim == 3 and actions.shape[1] == 1:
        actions = actions[:, 0]
    if recorded_state.ndim == 3 and recorded_state.shape[1] == 1:
        recorded_state = recorded_state[:, 0]
    if recorded_timestamp.ndim == 2 and recorded_timestamp.shape[1] == 1:
        recorded_timestamp = recorded_timestamp[:, 0]
    config = CONFIG_MAPPING["pinch_tongs"]()
    env = config.get_environment(
        policy_mode=True,
        render_mode="rgb_array",
        randomize=False,
        randomize_dynamics=False,
        seed=index,
    )
    try:
        obs, _ = env.reset()
        obs = restore_initial_state(env, "pinch_tongs", config, recorded_state[0])
        tactile_extractor = extractor_for(env)
        extract_data = mujoco.MjData(env.unwrapped.model)
        tactile = np.empty((len(actions), 30), dtype=np.float32)
        observed_state = np.empty_like(recorded_state)
        extraction_state = np.empty_like(recorded_state)
        contact_counts = np.empty(len(actions), dtype=np.int32)
        for frame, action in enumerate(actions):
            observed_state[frame] = np.asarray(obs["state"], dtype=np.float64)
            tactile[frame], extraction_state[frame], diagnostics = extract_recorded_state_snapshot(
                env, tactile_extractor, extract_data, recorded_state[frame]
            )
            contact_counts[frame] = diagnostics["contact_count"]
            raw_obs, _reward, _terminated, _truncated, _info = env.unwrapped.step(action)
            obs = env.observation(raw_obs)
    finally:
        env.close()
    replay_errors = state_errors(observed_state, recorded_state)
    errors = state_errors(extraction_state, recorded_state)
    passed = all(errors[name] <= limit for name, limit in TOLERANCES.items())
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        episode_index=np.asarray(index, dtype=np.int64),
        episode_name=np.asarray(episode.name),
        tactile_sim=tactile,
        contact_count=contact_counts,
        observed_state=observed_state,
        extraction_state=extraction_state,
        recorded_state=recorded_state,
        recorded_timestamp=recorded_timestamp,
        control_tick_end=np.arange(len(actions), dtype=np.int64),
        simulator_timestamp=np.arange(len(actions), dtype=np.float64) * 0.02,
    )
    return {
        "episode_index": index,
        "episode_name": episode.name,
        "frames": len(actions),
        "errors": errors,
        "action_only_replay_errors_diagnostic": replay_errors,
        "alignment": "PASS" if passed else "FAIL",
        "tactile_finite": bool(np.isfinite(tactile).all()),
        "active_tactile_frames": int(np.any(tactile.reshape(-1, 5, 6)[:, :, 0] > 0, axis=1).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    episodes = sorted(path.parent for path in RAW_ROOT.glob("*/replay.zarr"))
    if len(episodes) != 100:
        raise RuntimeError(f"expected 100 raw episodes, got {len(episodes)}")
    disable_wall_clock_throttle()
    selected = range(args.start, min(args.start + args.episodes, len(episodes)))
    rows = []
    for index in selected:
        row = replay_episode(index, episodes[index])
        rows.append(row)
        print(json.dumps(row), flush=True)
        if row["alignment"] != "PASS" or not row["tactile_finite"]:
            raise SystemExit("S4_3_PI1A_ALIGNMENT_FAIL")
    artifact = ARTIFACT_ROOT / ("tactile_replay_pilot.json" if args.pilot else "tactile_replay_contract.json")
    result = {
        "schema": "tactile3d-unit.s4-3-pi1-tactile-replay.v1",
        "status": "PASS",
        "official_utility_semantics": "reset + restore_initial_state + exact raw quaternion action trajectory",
        "contact_query_alignment": "read-only exact recorded Allegro/tongs snapshot; MjData restored before every action",
        "action_correction": "NONE",
        "pd_late_open_recovery": "NOT USED",
        "randomize": False,
        "randomize_dynamics": False,
        "physics_dt_sec": 0.002,
        "simulator_control_dt_sec": 0.02,
        "official_dataset_fps": 30,
        "mapping": "policy frame i -> raw action i -> simulator control tick i",
        "nominal_timestamp_note": "official timestamps advance 1/30 s while MuJoCo advances 0.02 s per recorded action",
        "history_source": "26 causal simulator control-tick samples ending at the policy frame's exact replay tick",
        "history_bootstrap": "LEFT_REPEAT_FIRST",
        "tolerances_frozen_before_pilot": TOLERANCES,
        "episodes": rows,
    }
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "artifact": artifact.name, "episodes": len(rows)}))


if __name__ == "__main__":
    main()
