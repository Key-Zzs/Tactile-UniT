#!/usr/bin/env python3
"""Serve one frozen R0/B0/B1/B2 policy for the paired PI1D evaluation."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path
import socket
import sys


ROOT = Path(__file__).resolve().parents[2]
OPENPI = ROOT / ".local/external/simulation/s4_3_pi1/openpi"
CHECKPOINTS = {
    "R0": ROOT / ".local/external/s4_3_pi0/models/DexJoCo-Pi05/pi05_dexjoco_ckpt/pinch_tongs",
    "B0": ROOT / ".local/experiments/simulation/s4_3_pi0/training/pinch_tongs/s43_pi0_official_seed42/29999",
    "B1": ROOT / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/s43_pi1b_contact_tokens_seed42/29999",
    "B2": ROOT / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/s43_pi1c_contact_tokens_physical_aux_seed42/29999",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=tuple(CHECKPOINTS))
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()
    checkpoint = CHECKPOINTS[args.model]
    if not (checkpoint / "params").is_dir():
        raise SystemExit(f"checkpoint is missing: {checkpoint}")

    os.chdir(OPENPI)
    sys.path[:0] = [str(ROOT), str(OPENPI / "src")]
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as training_config

    if args.model in {"R0", "B0"}:
        config = training_config.get_config("pinch_tongs")
        mode = "NONE"
        lambda_phys = 0.0
        inference_data_mode = "official SingleArmDataConfig"
    else:
        from gr00t.simulation.pi05_tactile_unit import install_openpi_runtime_hooks
        from gr00t.simulation.s4_3_pi1 import TactileUnitMode
        from scripts.simulation.train_s4_3_pi1 import build_config

        install_openpi_runtime_hooks()
        frozen = json.loads((ROOT / "configs/simulation/s4_3_pi1c_frozen.json").read_text())
        mode = (
            TactileUnitMode.CONTACT_STATE_TOKENS
            if args.model == "B1"
            else TactileUnitMode.CONTACT_STATE_TOKENS_PHYSICAL_AUX
        )
        lambda_phys = 0.0 if args.model == "B1" else float(frozen["lambda_phys"])
        config = build_config(mode.value, f"pi1d_{args.model.lower()}", lambda_phys)
        if args.model == "B2":
            # The auxiliary target exists only in training. Reuse B1's input transform
            # while retaining the frozen B2 model/checkpoint and action path.
            config = dataclasses.replace(
                config,
                data=dataclasses.replace(config.data, mode=TactileUnitMode.CONTACT_STATE_TOKENS),
            )
        inference_data_mode = "contact_state only; no training-only target"

    logging.info(
        "PI1D_POLICY_IDENTITY model=%s mode=%s lambda_phys=%s checkpoint=%s inference_data=%s",
        args.model,
        mode.value if hasattr(mode, "value") else mode,
        lambda_phys,
        checkpoint,
        inference_data_mode,
    )
    policy = policy_config.create_trained_policy(config, checkpoint)
    logging.info("PI1D_POLICY_COLD_LOAD_PASS model=%s", args.model)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    logging.info("PI1D_POLICY_SERVER_READY model=%s port=%d host=%s", args.model, args.port, socket.gethostname())
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
