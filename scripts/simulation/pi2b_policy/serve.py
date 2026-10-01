#!/usr/bin/env python3
"""Serve one frozen PI2B Track-A checkpoint with training-only targets removed."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import socket
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.pi2b_policy.contract import MODEL_ORDER, Workspace
from gr00t.simulation.pi2b_policy.training import build_config, configure_imports


def checkpoint(workspace: Workspace, model: str, seed: int) -> Path:
    return workspace.checkpoint_seed42(model) if seed == 42 else workspace.final_checkpoint(model, seed)


def runtime_config(workspace: Workspace, model: str, seed: int):
    from gr00t.simulation.s4_3_pi1 import TactileUnitMode
    from openpi.training import config as training_config

    # The architecture and frozen lambda are identical across training seeds.
    # build_config deliberately restricts *new training* to 43/44, so seed42's
    # read-only runtime config is built from the same recipe and then relabelled.
    recipe_seed = seed if seed in {43, 44} else 43
    config = build_config(workspace, model, recipe_seed, fsdp_devices=1)
    config = dataclasses.replace(config, seed=seed)
    training_mode = config.model.tactile_unit_mode.value if model != "B0" else "NONE"
    lambda_phys = 0.0 if model == "B0" else float(config.model.lambda_phys)
    if model == "B0":
        return config, training_mode, lambda_phys
    if model == "B_VA27":
        config = dataclasses.replace(config, data=training_config.get_config("pinch_tongs").data)
    else:
        config = dataclasses.replace(
            config,
            data=dataclasses.replace(
                config.data,
                mode=TactileUnitMode.CONTACT_STATE_TOKENS,
                target_sidecar_path=None,
            ),
        )
    return config, training_mode, lambda_phys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_ORDER, required=True)
    parser.add_argument("--training-seed", choices=(42, 43, 44), type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    workspace = Workspace.load(ROOT)
    selected = checkpoint(workspace, args.model, args.training_seed)
    if not (selected / "params").is_dir():
        raise SystemExit(f"checkpoint missing: {selected}")
    configure_imports(workspace)
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server

    config, training_mode, lambda_phys = runtime_config(
        workspace, args.model, args.training_seed
    )
    logging.info(
        "PI2B_POLICY_IDENTITY model=%s training_seed=%d training_mode=%s "
        "runtime_mode=%s lambda_phys=%s checkpoint=%s",
        args.model,
        args.training_seed,
        training_mode,
        "NONE" if args.model in {"B0", "B_VA27"} else "CONTACT_STATE_TOKENS",
        lambda_phys,
        selected,
    )
    policy = policy_config.create_trained_policy(config, selected)
    logging.info(
        "PI2B_POLICY_SERVER_READY model=%s training_seed=%d port=%d host=%s",
        args.model,
        args.training_seed,
        args.port,
        socket.gethostname(),
    )
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
