#!/usr/bin/env python3
"""Run the official DexJoCo evaluator through a read-only tactile augmentation layer."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import signal
import sys
from typing import Any

import numpy as np

from gr00t.simulation.pi1d_runtime import CausalContactRuntime, ContactStateUnixClient
from gr00t.simulation.s4_3_pi1 import TactileUnitMode
from gr00t.simulation.simulated_tactile import ContactRegionMap, SimulatedTactileExtractor


ROOT = Path(__file__).resolve().parents[2]
DEXJOCO = ROOT / "third_party/dexjoco"
REGIONS = ROOT / "configs/simulation/s4_1_dexjoco_contact_regions.json"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi1"
MODEL_ID = ""
MODE = TactileUnitMode.NONE
DIAGNOSTICS_JSONL = Path("/nonexistent")
OUTPUT_ROOT = Path("/nonexistent")
CONTACT_SOCKET = Path("/nonexistent")
EXPECTED_EPISODES = 50
EVALUATOR_SEED = 1


def sha256_array(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(payload, sort_keys=True) + "\n")
        output.flush()


def instrumented_inference_process(
    obs_queue,
    action_queue,
    stop_event,
    port: int,
    inferencing_event,
    seed: int,
    host: str,
):
    """Official inference worker with observation/action diagnostics only."""

    from dexjoco_openpi_client import eval_dexjoco_openpi as official
    from openpi_client import websocket_client_policy

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    official._set_seed(seed)
    client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    while not stop_event.is_set():
        observation = official.get_latest(obs_queue)
        if observation is None:
            stop_event.wait(0.01)
            continue
        payload = dict(observation.obs)
        episode_index = int(payload.pop("_pi1_episode_index"))
        reset_identity = str(payload.pop("_pi1_reset_identity"))
        forbidden = sorted(set(payload) & {"contact_shared_target", "physical_aux_valid"})
        try:
            result = client.infer(payload)
            action_chunk = np.asarray(result["actions"])
            record = {
                "type": "action_chunk",
                "model": MODEL_ID,
                "episode_index": episode_index,
                "reset_identity": reset_identity,
                "observation_timestamp": int(observation.timestamp),
                "shape": list(action_chunk.shape),
                "finite": bool(np.isfinite(action_chunk).all()),
                "minimum": float(action_chunk.min()),
                "maximum": float(action_chunk.max()),
                "mean_l2": float(np.linalg.norm(action_chunk, axis=-1).mean()),
                "contact_state_sent": "contact_state" in payload,
                "contact_state_sha256": (
                    sha256_array(np.asarray(payload["contact_state"], dtype=np.float32))
                    if "contact_state" in payload
                    else None
                ),
                "training_only_fields_sent": forbidden,
                "policy_timing": result.get("policy_timing"),
            }
            append_jsonl(DIAGNOSTICS_JSONL, record)
            action_queue.put(official.ActionChunk(action=action_chunk, timestamp=observation.timestamp))
            inferencing_event.clear()
        except Exception as error:
            append_jsonl(
                DIAGNOSTICS_JSONL,
                {
                    "type": "server_client_error",
                    "model": MODEL_ID,
                    "episode_index": episode_index,
                    "observation_timestamp": int(observation.timestamp),
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise


def build_augmented_environment(base_class):
    import mujoco

    class AugmentedDexJoCoOpenPIEnv(base_class):
        """Official client environment plus strictly read-only live tactile telemetry."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._contact_client: ContactStateUnixClient | None = None
            self._contact_runtime: CausalContactRuntime | None = None
            self._extractor: SimulatedTactileExtractor | None = None
            self._extractor_audit: dict[str, Any] | None = None
            self._episode_index = -1
            self._current: dict[str, Any] | None = None
            self._episodes: list[dict[str, Any]] = []
            self._last_info: dict[str, Any] = {}

        def start(self):
            super().start()
            value = json.loads(REGIONS.read_text())
            region_map = ContactRegionMap.from_config(
                {"regions": value["regions"], "object_body_names": value["object_body_names"]}
            )
            raw = self.env.unwrapped
            self._extractor_audit = region_map.resolve(raw.model, mujoco)
            if self._extractor_audit["status"] != "PASS" or region_map.tactile_dim != 30:
                raise RuntimeError(f"live tactile region resolution failed: {self._extractor_audit}")
            self._extractor = SimulatedTactileExtractor(region_map)
            self._contact_client = ContactStateUnixClient(CONTACT_SOCKET)
            self._contact_runtime = CausalContactRuntime(self._contact_client)
            original_step = self.env.step

            def captured_step(action):
                result = original_step(action)
                self._last_info = copy.deepcopy(result[4])
                return result

            self.env.step = captured_step

        def _reset_identity(self) -> str:
            raw = self.env.unwrapped
            state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
            state = np.empty(mujoco.mj_stateSize(raw.model, state_spec), dtype=np.float64)
            mujoco.mj_getState(raw.model, raw.data, state, state_spec)
            return sha256_array(state, np.asarray(self.obs["state"], dtype=np.float64))

        def _extract_tactile(self) -> tuple[np.ndarray, dict[str, Any]]:
            raw = self.env.unwrapped
            tactile, diagnostics = self._extractor.extract(raw.model, raw.data, mujoco)
            tactile = np.asarray(tactile, dtype=np.float32)
            if tactile.shape != (30,) or not np.isfinite(tactile).all():
                raise RuntimeError("live tactile query did not return finite [30]")
            return tactile, diagnostics

        def _record_tactile(self, tactile: np.ndarray, diagnostics: dict[str, Any]) -> None:
            matrix = tactile.reshape(5, 6)
            current = self._current
            current["tactile_samples"] += 1
            current["active_tactile_samples"] += int(np.any(matrix[:, 0] > 0))
            current["tactile_l2_sum"] += float(np.linalg.norm(tactile))
            current["tactile_l2_max"] = max(current["tactile_l2_max"], float(np.linalg.norm(tactile)))
            current["normal_force_max"] = max(current["normal_force_max"], float(matrix[:, 1].max()))
            current["matched_contact_count_sum"] += int(diagnostics["contact_count"])
            current["matched_contact_count_max"] = max(
                current["matched_contact_count_max"], int(diagnostics["contact_count"])
            )
            for name, count in diagnostics["contact_count_by_region"].items():
                current["matched_contacts_by_region"][name] += int(count)

        def _finalize_episode(self) -> None:
            if self._current is None:
                return
            current = self._current
            actions = np.asarray(current.pop("executed_actions"), dtype=np.float64)
            steps = int(len(actions))
            tactile_samples = int(current.pop("tactile_samples"))
            contact_state_norms = current.pop("contact_state_norms")
            current.update(
                success=bool(self.is_success),
                termination=("success" if self.is_success else "max_steps" if steps == 1000 else "terminated"),
                steps=steps,
                task_progress={
                    "final_pinch_count": int(self._last_info.get("pinch_count", 0)),
                    "max_pinch_count": int(current.pop("max_pinch_count")),
                },
                executed_action_stats={
                    "count": steps,
                    "shape": list(actions.shape),
                    "finite": bool(np.isfinite(actions).all()) if steps else False,
                    "minimum": float(actions.min()) if steps else None,
                    "maximum": float(actions.max()) if steps else None,
                    "mean_l2": float(np.linalg.norm(actions, axis=-1).mean()) if steps else None,
                },
                tactile_diagnostics={
                    "samples": tactile_samples,
                    "active_samples": current.pop("active_tactile_samples"),
                    "mean_l2": current.pop("tactile_l2_sum") / max(tactile_samples, 1),
                    "max_l2": current.pop("tactile_l2_max"),
                    "max_normal_force": current.pop("normal_force_max"),
                    "matched_contact_count_sum": current.pop("matched_contact_count_sum"),
                    "matched_contact_count_max": current.pop("matched_contact_count_max"),
                    "matched_contacts_by_region": current.pop("matched_contacts_by_region"),
                    "history_shape": [26, 30],
                    "history_bootstrap": "LEFT_REPEAT_FIRST",
                },
                contact_state_diagnostics={
                    "queries": len(contact_state_norms),
                    "finite": True,
                    "shape": [256],
                    "mean_l2": float(np.mean(contact_state_norms)) if contact_state_norms else None,
                    "max_l2": float(np.max(contact_state_norms)) if contact_state_norms else None,
                },
            )
            self._episodes.append(current)
            self._current = None

        def reset(self):
            self._finalize_episode()
            super().reset()
            self._episode_index += 1
            self._last_info = {}
            reset_identity = self._reset_identity()
            self._current = {
                "episode_index": self._episode_index,
                "reset_identity": reset_identity,
                "executed_actions": [],
                "max_pinch_count": 0,
                "tactile_samples": 0,
                "active_tactile_samples": 0,
                "tactile_l2_sum": 0.0,
                "tactile_l2_max": 0.0,
                "normal_force_max": 0.0,
                "matched_contact_count_sum": 0,
                "matched_contact_count_max": 0,
                "matched_contacts_by_region": {name: 0 for name in self._extractor.region_map.region_names},
                "contact_state_norms": [],
                "server_client_errors": [],
            }
            tactile, diagnostics = self._extract_tactile()
            self._contact_runtime.reset(tactile)
            self._record_tactile(tactile, diagnostics)

        def step(self, action: np.ndarray):
            self._current["executed_actions"].append(np.asarray(action, dtype=np.float64).copy())
            result = super().step(action)
            self._current["max_pinch_count"] = max(
                self._current["max_pinch_count"], int(self._last_info.get("pinch_count", 0))
            )
            tactile, diagnostics = self._extract_tactile()
            self._contact_runtime.append(tactile)
            self._record_tactile(tactile, diagnostics)
            return result

        def get_obs(self) -> dict[str, np.ndarray]:
            observation = super().get_obs()
            try:
                contact_state = self._contact_runtime.contact_state()
            except Exception as error:
                self._current["server_client_errors"].append(f"{type(error).__name__}: {error}")
                raise
            self._current["contact_state_norms"].append(float(np.linalg.norm(contact_state)))
            if MODE is not TactileUnitMode.NONE:
                observation["contact_state"] = contact_state
            observation["_pi1_episode_index"] = self._episode_index
            observation["_pi1_reset_identity"] = self._current["reset_identity"]
            return observation

        def close(self):
            self._finalize_episode()
            if self._contact_client is not None:
                self._contact_client.close()
            super().close()
            inference_rows = []
            if DIAGNOSTICS_JSONL.exists():
                inference_rows = [json.loads(line) for line in DIAGNOSTICS_JSONL.read_text().splitlines() if line]
            chunks = [row for row in inference_rows if row["type"] == "action_chunk"]
            errors = [row for row in inference_rows if row["type"] == "server_client_error"]
            for episode in self._episodes:
                episode["action_chunks"] = [
                    row for row in chunks if row["episode_index"] == episode["episode_index"]
                ]
                episode["server_client_errors"].extend(
                    row["error"] for row in errors if row["episode_index"] == episode["episode_index"]
                )
            episode_dirs = sorted(path for path in OUTPUT_ROOT.glob("episode_*_*") if path.is_dir())
            marker = list(OUTPUT_ROOT.glob("success_rate_*_*.txt"))
            correct_contact_delivery = (
                all(not row["contact_state_sent"] for row in chunks)
                if MODE is TactileUnitMode.NONE
                else all(row["contact_state_sent"] for row in chunks)
            )
            gates = {
                "official_evaluator_episode_count": len(self._episodes) == EXPECTED_EPISODES,
                "episode_indices_complete": [row["episode_index"] for row in self._episodes]
                == list(range(EXPECTED_EPISODES)),
                "one_output_directory_per_episode": len(episode_dirs) == EXPECTED_EPISODES,
                "one_success_marker": len(marker) == 1,
                "all_reset_identities_present": all(len(row["reset_identity"]) == 64 for row in self._episodes),
                "all_actions_finite": all(row["executed_action_stats"]["finite"] for row in self._episodes),
                "all_action_chunks_finite_30x22": bool(chunks)
                and all(row["finite"] and row["shape"] == [30, 22] for row in chunks),
                "all_tactile_finite_30d": all(
                    row["tactile_diagnostics"]["samples"] == row["steps"] + 1 for row in self._episodes
                ),
                "causal_history_26x30": all(
                    row["tactile_diagnostics"]["history_shape"] == [26, 30] for row in self._episodes
                ),
                "left_repeat_first": all(
                    row["tactile_diagnostics"]["history_bootstrap"] == "LEFT_REPEAT_FIRST"
                    for row in self._episodes
                ),
                "contact_state_computed_for_every_policy_query": sum(
                    row["contact_state_diagnostics"]["queries"] for row in self._episodes
                )
                == len(chunks),
                "contact_delivery_matches_mode": correct_contact_delivery,
                "training_only_targets_never_sent": all(not row["training_only_fields_sent"] for row in chunks),
                "B0_R0_tactile_computed_but_ignored": MODE is not TactileUnitMode.NONE
                or sum(row["contact_state_diagnostics"]["queries"] for row in self._episodes) == len(chunks),
                "no_server_client_errors": not errors
                and all(not row["server_client_errors"] for row in self._episodes),
                "no_episode_deleted": len(self._episodes) == EXPECTED_EPISODES,
            }
            payload = {
                "schema": "tactile3d-unit.s4-3-pi1d-model-eval.v1",
                "status": "PASS" if all(gates.values()) else "FAIL",
                "model": MODEL_ID,
                "mode": MODE.value,
                "task": "pinch_tongs",
                "regime": "rand_obj",
                "evaluator_seed": EVALUATOR_SEED,
                "episodes": EXPECTED_EPISODES,
                "successes": sum(int(row["success"]) for row in self._episodes),
                "success_rate": sum(int(row["success"]) for row in self._episodes) / EXPECTED_EPISODES,
                "official_evaluator_entrypoint": "dexjoco_openpi_client.eval_dexjoco_openpi.main",
                "official_environment_wrapper": "DexJoCoOpenPIEnv",
                "augmentation": "read-only live contact query + causal E_T sidecar",
                "contact_state_sent_to_policy": MODE is not TactileUnitMode.NONE,
                "physics_action_success_reset_camera_prompt_modified": False,
                "rand_full": False,
                "randomize_dynamics": False,
                "replan_ratio": 0.8,
                "history_bootstrap": "LEFT_REPEAT_FIRST",
                "extractor_audit": self._extractor_audit,
                "episode_results": self._episodes,
                "server_client_errors": errors,
                "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
            }
            artifact = ARTIFACTS / f"pi1d_{MODEL_ID.lower()}_eval.json"
            temporary = artifact.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            temporary.replace(artifact)
            print(json.dumps({"model": MODEL_ID, "status": payload["status"], "successes": payload["successes"]}), flush=True)

    return AugmentedDexJoCoOpenPIEnv


def main() -> None:
    global MODEL_ID, MODE, DIAGNOSTICS_JSONL, OUTPUT_ROOT, CONTACT_SOCKET, EXPECTED_EPISODES, EVALUATOR_SEED
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("R0", "B0", "B1", "B2"))
    parser.add_argument("--mode", required=True, choices=tuple(mode.value for mode in TactileUnitMode))
    parser.add_argument("--contact-socket", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics-jsonl", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    if args.seed != 1 or args.episodes != 50:
        raise SystemExit("PI1D requires exactly fresh evaluator seed 1 and 50 episodes")
    if args.output.exists() or args.diagnostics_jsonl.exists():
        raise SystemExit("Refusing to overwrite PI1D model output")
    expected_modes = {
        "R0": TactileUnitMode.NONE,
        "B0": TactileUnitMode.NONE,
        "B1": TactileUnitMode.CONTACT_STATE_TOKENS,
        "B2": TactileUnitMode.CONTACT_STATE_TOKENS_PHYSICAL_AUX,
    }
    MODEL_ID = args.model
    MODE = TactileUnitMode(args.mode)
    if MODE is not expected_modes[MODEL_ID]:
        raise SystemExit(f"mode/model mismatch: {MODEL_ID} {MODE.value}")
    DIAGNOSTICS_JSONL = args.diagnostics_jsonl
    OUTPUT_ROOT = args.output
    CONTACT_SOCKET = args.contact_socket
    EXPECTED_EPISODES = args.episodes
    EVALUATOR_SEED = args.seed
    DIAGNOSTICS_JSONL.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(DEXJOCO / "dexjoco"))
    from dexjoco_openpi_client import eval_dexjoco_openpi as official
    from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv

    official.DexJoCoOpenPIEnv = build_augmented_environment(DexJoCoOpenPIEnv)
    official.inference_process = instrumented_inference_process
    official.main(
        config=args.config,
        seed=args.seed,
        rand_full=False,
        randomize_dynamics=False,
        port=args.port,
        host="127.0.0.1",
        output=args.output,
        render_mode="rgb_array",
        replan_ratio=0.8,
        episodes=args.episodes,
        pad_state_dim46=False,
        record_pressed_digits=False,
    )


if __name__ == "__main__":
    main()
