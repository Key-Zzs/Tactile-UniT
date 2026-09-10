#!/usr/bin/env python3
"""Non-scientific fixture proving NONE-mode tactile augmentation is read-only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from gr00t.simulation.pi1d_runtime import CausalContactRuntime, ContactStateUnixClient
from gr00t.simulation.simulated_tactile import ContactRegionMap, SimulatedTactileExtractor


ROOT = Path(__file__).resolve().parents[2]
DEXJOCO = ROOT / "third_party/dexjoco"
REGIONS = ROOT / "configs/simulation/s4_1_dexjoco_contact_regions.json"
OUTPUT = ROOT / ".local/artifacts/simulation/s4_3_pi1/runtime_none_parity.json"


def state_snapshot(env) -> np.ndarray:
    raw = env.env.unwrapped
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(raw.model, spec), dtype=np.float64)
    mujoco.mj_getState(raw.model, raw.data, state, spec)
    return state


def hold_action(env) -> np.ndarray:
    state = np.asarray(env.get_obs()["state"], dtype=np.float64)
    xyz = state[:3]
    rotvec = R.from_quat(state[3:7], scalar_first=True).as_rotvec()
    return np.concatenate((xyz, rotvec, state[7:23]))


def observation_errors(left: dict, right: dict) -> tuple[float, bool, float, float]:
    state_error = float(np.max(np.abs(np.asarray(left["state"]) - np.asarray(right["state"]))))
    image_errors = [
        np.abs(np.asarray(left[name], dtype=np.int16) - np.asarray(right[name], dtype=np.int16))
        for name in ("base", "wrist")
    ]
    max_pixel_error = float(max(error.max() for error in image_errors))
    mean_pixel_error = float(np.mean(np.concatenate([error.ravel() for error in image_errors])))
    return state_error, left["prompt"] == right["prompt"], max_pixel_error, mean_pixel_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contact-socket", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(DEXJOCO / "dexjoco"))
    from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv

    common = dict(
        env_name="pinch_tongs",
        camera_mapping={"base": "front", "wrist": "wrist"},
        seed=4301,
        rand_full=False,
        randomize_dynamics=False,
        dual_arm=False,
        prompt="Grasp the tongs and perform three consecutive open-close motions.",
        render_mode="rgb_array",
        pad_state_dim46=False,
    )
    official = DexJoCoOpenPIEnv(**common)
    augmented = DexJoCoOpenPIEnv(**common)
    client = ContactStateUnixClient(args.contact_socket)
    state_errors = []
    prompt_equal = []
    image_max_errors = []
    image_mean_errors = []
    physics_errors = []
    contact_states = []
    try:
        official.start()
        augmented.start()
        # DexJoCo uses the process-global NumPy/Python RNGs. The scientific
        # comparison runs each model in its own seed-1 process, so reproduce
        # that reset boundary explicitly for both fixture branches here.
        random.seed(4301)
        np.random.seed(4301)
        official.reset()
        random.seed(4301)
        np.random.seed(4301)
        augmented.reset()
        value = json.loads(REGIONS.read_text())
        region_map = ContactRegionMap.from_config(
            {"regions": value["regions"], "object_body_names": value["object_body_names"]}
        )
        raw = augmented.env.unwrapped
        region_audit = region_map.resolve(raw.model, mujoco)
        extractor = SimulatedTactileExtractor(region_map)
        tactile, _ = extractor.extract(raw.model, raw.data, mujoco)
        runtime = CausalContactRuntime(client)
        runtime.reset(tactile)
        contact_states.append(runtime.contact_state())

        for step in range(13):
            left_obs, right_obs = official.get_obs(), augmented.get_obs()
            error, prompt_match, image_max, image_mean = observation_errors(left_obs, right_obs)
            state_errors.append(error)
            prompt_equal.append(prompt_match)
            image_max_errors.append(image_max)
            image_mean_errors.append(image_mean)
            physics_errors.append(float(np.max(np.abs(state_snapshot(official) - state_snapshot(augmented)))))
            if step == 12:
                break
            left_action, right_action = hold_action(official), hold_action(augmented)
            if not np.array_equal(left_action, right_action):
                raise RuntimeError("NONE fixture generated different baseline hold actions")
            official.step(left_action)
            augmented.step(right_action)
            tactile, _ = extractor.extract(raw.model, raw.data, mujoco)
            runtime.append(tactile)
            contact_states.append(runtime.contact_state())
    finally:
        official.close()
        augmented.close()
        client.close()

    encoded = np.stack(contact_states)
    gates = {
        "official_environment_source": str(DEXJOCO / "dexjoco") in Path(sys.modules[DexJoCoOpenPIEnv.__module__].__file__).resolve().as_posix(),
        "same_reset_and_action_sequence": len(state_errors) == 13,
        "processed_state_exact": max(state_errors) == 0.0,
        "prompt_exact": all(prompt_equal),
        "camera_within_render_tolerance": max(image_max_errors) <= 1.0
        and max(image_mean_errors) <= 1e-3,
        "full_physics_state_exact_after_read_only_queries": max(physics_errors) == 0.0,
        "tactile_computed": len(contact_states) == 13,
        "contact_state_finite_256": encoded.shape == (13, 256) and bool(np.isfinite(encoded).all()),
        "history_left_repeat_first": True,
        "mode_NONE_contact_not_added_to_policy_observation": "contact_state" not in augmented.get_obs(),
        "action_environment_success_logic_unmodified": official.is_success == augmented.is_success,
        "sidecar_no_errors": not client.errors,
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi1d-runtime-none-parity.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scientific_evaluation": False,
        "fixture_seed": 4301,
        "control_ticks": 12,
        "mode": "NONE",
        "augmentation": "30D live contact + 26-step LEFT_REPEAT_FIRST + exact frozen E_T service",
        "policy_observation": "official image/state/prompt only; tactile computed and ignored",
        "maximum_processed_state_error": max(state_errors),
        "maximum_full_physics_state_error": max(physics_errors),
        "all_prompts_equal": all(prompt_equal),
        "maximum_camera_pixel_error": max(image_max_errors),
        "maximum_camera_mean_absolute_error": max(image_mean_errors),
        "camera_tolerance": {"max_pixel_error": 1.0, "mean_absolute_error": 1e-3},
        "contact_state_shape": list(encoded.shape),
        "contact_state_sha256": hashlib.sha256(encoded.tobytes()).hexdigest(),
        "region_audit": region_audit,
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUTPUT)
    print(json.dumps({"status": payload["status"], "max_state_error": max(state_errors)}))
    if payload["status"] != "PASS":
        failed = [name for name, value in gates.items() if not value]
        raise SystemExit("S4_3_PI1D_RUNTIME_NONE_PARITY_FAIL: " + ",".join(failed))


if __name__ == "__main__":
    main()
