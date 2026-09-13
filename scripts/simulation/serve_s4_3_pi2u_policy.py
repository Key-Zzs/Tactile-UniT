#!/usr/bin/env python3
"""Serve frozen B0/BVA/B1/B2 policies for the PI2U fresh ablation only."""

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
    "B0": ROOT / ".local/experiments/simulation/s4_3_pi0/training/pinch_tongs/s43_pi0_official_seed42/29999",
    "BVA": ROOT / ".local/experiments/simulation/s4_3_pi2u/bva/pinch_tongs/s43_pi2u_bva_seed42/29999",
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

    if args.model == "B0":
        config = training_config.get_config("pinch_tongs")
        mode, lambda_phys = "NONE", 0.0
        runtime_contract = "official B0 observations only"
    else:
        from gr00t.simulation.pi05_tactile_unit import install_openpi_runtime_hooks
        from gr00t.simulation.s4_3_pi1 import TactileUnitMode

        install_openpi_runtime_hooks()
        if args.model == "BVA":
            from scripts.simulation.train_s4_3_pi2u_bva import build_config

            protocol = json.loads((ROOT / ".local/artifacts/simulation/s4_3_pi2u/bva_training_protocol.json").read_text())
            config = build_config("s43_pi2u_bva_seed3_eval", float(protocol["lambda_phys"]), 1)
            # BVA inference must accept exactly B0's observation schema; the VA target
            # is a training-only sidecar field and is intentionally absent here.
            config = dataclasses.replace(config, data=training_config.get_config("pinch_tongs").data)
            mode, lambda_phys = TactileUnitMode.VA_PHYSICAL_AUX.value, float(protocol["lambda_phys"])
            runtime_contract = "B0-only observations; no tactile/contact/future target"
        else:
            from scripts.simulation.train_s4_3_pi1 import build_config

            frozen = json.loads((ROOT / "configs/simulation/s4_3_pi1c_frozen.json").read_text())
            mode = (
                TactileUnitMode.CONTACT_STATE_TOKENS
                if args.model == "B1"
                else TactileUnitMode.CONTACT_STATE_TOKENS_PHYSICAL_AUX
            )
            lambda_phys = 0.0 if args.model == "B1" else float(frozen["lambda_phys"])
            config = build_config(mode.value, f"s43_pi2u_{args.model.lower()}_seed3_eval", lambda_phys)
            if args.model == "B2":
                config = dataclasses.replace(
                    config, data=dataclasses.replace(config.data, mode=TactileUnitMode.CONTACT_STATE_TOKENS)
                )
            mode = mode.value
            runtime_contract = "frozen PI2A Contact-State runtime; no training-only target"

    logging.info(
        "PI2U_POLICY_IDENTITY model=%s mode=%s lambda_phys=%s checkpoint=%s runtime=%s",
        args.model, mode, lambda_phys, checkpoint, runtime_contract,
    )
    policy = policy_config.create_trained_policy(config, checkpoint)
    logging.info("PI2U_POLICY_COLD_LOAD_PASS model=%s", args.model)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata
    )
    logging.info("PI2U_POLICY_SERVER_READY model=%s port=%d host=%s", args.model, args.port, socket.gethostname())
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
