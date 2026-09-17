#!/usr/bin/env python3
"""Serve frozen B1/B_HVA/B2 policies through one matched-input runtime."""

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
    "B1": ROOT
    / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/"
    "s43_pi1b_contact_tokens_seed42/29999",
    "B_HVA": ROOT
    / ".local/experiments/simulation/s4_3_pi2m/bhva/pinch_tongs/"
    "s43_pi2m_bhva_seed42/29999",
    "B2": ROOT
    / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/"
    "s43_pi1c_contact_tokens_physical_aux_seed42/29999",
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
    from gr00t.simulation.pi05_tactile_unit import install_openpi_runtime_hooks
    from gr00t.simulation.s4_3_pi1 import TactileUnitMode
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server

    install_openpi_runtime_hooks()
    if args.model == "B_HVA":
        from scripts.simulation.train_s4_3_pi2m_bhva import build_config

        freeze = json.loads(
            (
                ROOT / ".local/artifacts/simulation/s4_3_pi2m/training_protocol_freeze.json"
            ).read_text()
        )
        lambda_phys = float(freeze["lambda_phys"])
        config = build_config("s43_pi2m_bhva_seed42", lambda_phys, 1)
        training_mode = TactileUnitMode.CONTACT_STATE_TOKENS_VA_PHYSICAL_AUX
    else:
        from scripts.simulation.train_s4_3_pi1 import build_config

        frozen = json.loads((ROOT / "configs/simulation/s4_3_pi1c_frozen.json").read_text())
        training_mode = (
            TactileUnitMode.CONTACT_STATE_TOKENS
            if args.model == "B1"
            else TactileUnitMode.CONTACT_STATE_TOKENS_PHYSICAL_AUX
        )
        lambda_phys = 0.0 if args.model == "B1" else float(frozen["lambda_phys"])
        config = build_config(training_mode.value, f"s43_pi2m_{args.model.lower()}_eval", lambda_phys)

    # Every formal policy receives the exact same online Contact-State schema.
    # Training-only target keys are absent from this transform and sample_actions
    # rejects them if a caller attempts to inject one.
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(
            config.data,
            mode=TactileUnitMode.CONTACT_STATE_TOKENS,
            target_sidecar_path=None,
        ),
    )
    logging.info(
        "PI2M_POLICY_IDENTITY model=%s training_mode=%s runtime_mode=%s lambda_phys=%s checkpoint=%s",
        args.model,
        training_mode.value,
        TactileUnitMode.CONTACT_STATE_TOKENS.value,
        lambda_phys,
        checkpoint,
    )
    policy = policy_config.create_trained_policy(config, checkpoint)
    logging.info("PI2M_POLICY_COLD_LOAD_PASS model=%s", args.model)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata
    )
    logging.info(
        "PI2M_POLICY_SERVER_READY model=%s port=%d host=%s",
        args.model,
        args.port,
        socket.gethostname(),
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
