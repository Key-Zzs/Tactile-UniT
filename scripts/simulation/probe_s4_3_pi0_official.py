#!/usr/bin/env python3
"""Probe the official DexJoCo/OpenPI server without altering either source tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv
from openpi_client import websocket_client_policy


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "third_party/dexjoco/configs/rand_obj/pinch_tongs.yaml"
DEFAULT_OUTPUT = ROOT / ".local/artifacts/simulation/s4_3_pi0/official_checkpoint_smoke.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8125)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--checkpoint-kind",
        choices=("official", "reproduced"),
        default="official",
        help="Select provenance labels for the server being probed.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    environment = DexJoCoOpenPIEnv(
        env_name=cfg["env_name"],
        camera_mapping=cfg["camera_mapping"],
        seed=0,
        rand_full=False,
        randomize_dynamics=False,
        dual_arm=False,
        prompt=cfg["prompt"],
        render_mode="rgb_array",
        pad_state_dim46=False,
    )
    try:
        environment.start()
        environment.reset()
        observation = environment.get_obs()
        client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
        result = client.infer(observation)
        # The official evaluator sends this chunk through a multiprocessing
        # queue, which materializes a writable copy before action conversion.
        # Mirror that boundary in this direct, single-process probe.
        actions = np.array(result["actions"], copy=True)
        environment_action = environment._process_action(actions[0])  # noqa: SLF001
        gates = {
            "task": cfg["env_name"] == "pinch_tongs",
            "prompt": observation["prompt"] == "Grasp the tongs and perform three consecutive open-close motions.",
            "state_23d": np.asarray(observation["state"]).shape == (23,),
            "base_image_224_uint8": observation["base"].shape == (224, 224, 3)
            and observation["base"].dtype == np.uint8,
            "wrist_image_224_uint8": observation["wrist"].shape == (224, 224, 3)
            and observation["wrist"].dtype == np.uint8,
            "action_chunk_30x22": actions.shape == (30, 22),
            "policy_actions_finite": bool(np.isfinite(actions).all()),
            "environment_action_23d": environment_action.shape == (23,),
            "environment_action_finite": bool(np.isfinite(environment_action).all()),
        }
        artifact = {
            "schema": (f"tactile3d-unit.s4-3-pi0-{args.checkpoint_kind}-checkpoint-server-smoke.v1"),
            "stage": "PI0-3" if args.checkpoint_kind == "official" else "PI0-10",
            "checkpoint_kind": args.checkpoint_kind,
            "task": "pinch_tongs",
            "regime": "rand_obj",
            "seed": 0,
            "rand_full": False,
            "randomize_dynamics": False,
            "server": {"host": args.host, "port": args.port},
            "state_shape": list(np.asarray(observation["state"]).shape),
            "image_shapes": {
                "base": list(observation["base"].shape),
                "wrist": list(observation["wrist"].shape),
            },
            "policy_action_shape": list(actions.shape),
            "environment_action_shape": list(environment_action.shape),
            "policy_action_min": float(actions.min()),
            "policy_action_max": float(actions.max()),
            "gates": {name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
            "status": "PASS" if all(gates.values()) else "FAIL",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(args.output)
        print(json.dumps(artifact, indent=2, sort_keys=True))
        if artifact["status"] != "PASS":
            raise SystemExit("S4_3_PI0_OFFICIAL_CHECKPOINT_SMOKE_FAIL")
    finally:
        environment.close()


if __name__ == "__main__":
    main()
