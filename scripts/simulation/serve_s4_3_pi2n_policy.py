#!/usr/bin/env python3
"""Serve one frozen PI2N formal-cohort policy with training targets removed."""

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
PI2N_RUN_ROOT = ROOT / ".local/experiments/simulation/s4_3_pi2n"
CHECKPOINTS = {
    "B0": ROOT
    / ".local/experiments/simulation/s4_3_pi0/training/pinch_tongs/"
    "s43_pi0_official_seed42/29999",
    "B_VA27": PI2N_RUN_ROOT
    / "runs/B_VA27/pinch_tongs/s43_pi2n_b_va27_seed42/29999",
    "B1": ROOT
    / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/"
    "s43_pi1b_contact_tokens_seed42/29999",
    "B_HVA": ROOT
    / ".local/experiments/simulation/s4_3_pi2m/bhva/pinch_tongs/"
    "s43_pi2m_bhva_seed42/29999",
    "B2": ROOT
    / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/"
    "s43_pi1c_contact_tokens_physical_aux_seed42/29999",
    "B_VAC_V": PI2N_RUN_ROOT
    / "runs/B_VAC_V/pinch_tongs/s43_pi2n_b_vac_v_seed42/29999",
}
RUNTIME_MODES = {
    "B0": "NONE",
    "B_VA27": "NONE",
    "B1": "CONTACT_STATE_TOKENS",
    "B_HVA": "CONTACT_STATE_TOKENS",
    "B2": "CONTACT_STATE_TOKENS",
    "B_VAC_V": "CONTACT_STATE_TOKENS",
}


def _frozen_lambda(model: str) -> float:
    if model == "B_HVA":
        artifact = ROOT / ".local/artifacts/simulation/s4_3_pi2m/training_protocol_freeze.json"
    elif model == "B_VA27":
        artifact = ROOT / ".local/artifacts/simulation/s4_3_pi2n/b_va27_training_freeze.json"
    elif model == "B_VAC_V":
        artifact = ROOT / ".local/artifacts/simulation/s4_3_pi2n/b_vac_v_training_freeze.json"
    else:
        return 0.0
    payload = json.loads(artifact.read_text())
    return float(payload["lambda_phys"])


def _build_config(model: str):
    from gr00t.simulation.pi05_tactile_unit import install_openpi_runtime_hooks
    from gr00t.simulation.s4_3_pi1 import TactileUnitMode
    from openpi.training import config as training_config

    if model == "B0":
        return training_config.get_config("pinch_tongs"), "NONE", 0.0

    install_openpi_runtime_hooks()
    if model in {"B1", "B2"}:
        from scripts.simulation.train_s4_3_pi1 import build_config

        frozen = json.loads((ROOT / "configs/simulation/s4_3_pi1c_frozen.json").read_text())
        training_mode = (
            TactileUnitMode.CONTACT_STATE_TOKENS
            if model == "B1"
            else TactileUnitMode.CONTACT_STATE_TOKENS_PHYSICAL_AUX
        )
        lambda_phys = 0.0 if model == "B1" else float(frozen["lambda_phys"])
        config = build_config(training_mode.value, f"s43_pi2n_{model.lower()}_eval", lambda_phys)
    elif model == "B_HVA":
        from scripts.simulation.train_s4_3_pi2m_bhva import build_config

        training_mode = TactileUnitMode.CONTACT_STATE_TOKENS_VA_PHYSICAL_AUX
        lambda_phys = _frozen_lambda(model)
        config = build_config("s43_pi2m_bhva_seed42", lambda_phys, 1)
    else:
        from scripts.simulation.train_s4_3_pi2n import build_config

        training_mode = (
            TactileUnitMode.VA_PHYSICAL_AUX
            if model == "B_VA27"
            else TactileUnitMode.CONTACT_STATE_TOKENS_VAC_V_PHYSICAL_AUX
        )
        lambda_phys = _frozen_lambda(model)
        exp_name = (
            "s43_pi2n_b_va27_seed42"
            if model == "B_VA27"
            else "s43_pi2n_b_vac_v_seed42"
        )
        config = build_config(model, exp_name, lambda_phys, 1)

    if model == "B_VA27":
        # B_VA27 is the matched no-H control. Its training-only VA target must
        # not be requested by the production policy input transform.
        official = training_config.get_config("pinch_tongs")
        config = dataclasses.replace(config, data=official.data)
    else:
        # Every H-enabled model receives the same current Contact-State schema;
        # auxiliary targets and validity masks exist only in training.
        config = dataclasses.replace(
            config,
            data=dataclasses.replace(
                config.data,
                mode=TactileUnitMode.CONTACT_STATE_TOKENS,
                target_sidecar_path=None,
            ),
        )
    return config, training_mode.value, lambda_phys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=tuple(CHECKPOINTS))
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()
    checkpoint = CHECKPOINTS[args.model]
    if not (checkpoint / "params").is_dir():
        raise SystemExit(f"checkpoint is missing: {checkpoint}")
    resolved_pi2n = PI2N_RUN_ROOT.resolve()
    expected_pi2n = Path(
        "/mnt/ugreen_nas/storage/UniT_storage/experiments/simulation/s4_3_pi2n"
    )
    if args.model in {"B_VA27", "B_VAC_V"} and resolved_pi2n != expected_pi2n:
        raise SystemExit(f"PI2N run root is not the audited NAS directory: {resolved_pi2n}")

    os.chdir(OPENPI)
    sys.path[:0] = [str(ROOT), str(OPENPI / "src")]
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server

    config, training_mode, lambda_phys = _build_config(args.model)
    logging.info(
        "PI2N_POLICY_IDENTITY model=%s training_mode=%s runtime_mode=%s "
        "lambda_phys=%s checkpoint=%s",
        args.model,
        training_mode,
        RUNTIME_MODES[args.model],
        lambda_phys,
        checkpoint,
    )
    policy = policy_config.create_trained_policy(config, checkpoint)
    logging.info("PI2N_POLICY_COLD_LOAD_PASS model=%s", args.model)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    logging.info(
        "PI2N_POLICY_SERVER_READY model=%s port=%d host=%s",
        args.model,
        args.port,
        socket.gethostname(),
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
