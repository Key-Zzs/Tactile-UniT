#!/usr/bin/env python3
"""Launch one frozen S4.3-PI1 tactile pi0.5 training mode."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
OPENPI_ROOT = ROOT / ".local/external/simulation/s4_3_pi1/openpi"
OPENPI_SRC = OPENPI_ROOT / "src"
DATASET_ROOT = (
    ROOT
    / ".local/external/s4_3_pi0/datasets/DexJoCo-Datasets-LeRobot/"
    "dexjoco_lerobot_datasets/pinch_tongs"
)
SIDECAR = ROOT / ".local/datasets/simulation/s4_3_pi1/pinch_tongs_official_tactile/sidecar.npz"
BASE_PARAMS = ROOT / ".local/external/s4_3_pi0/models/DexJoCo-Pi05/pi05_base/params"
ASSETS = ROOT / ".local/cache/simulation/s4_3_pi0/assets"
CHECKPOINTS = ROOT / ".local/experiments/simulation/s4_3_pi1/training"


def configure_imports() -> None:
    if not OPENPI_ROOT.is_dir():
        raise FileNotFoundError(f"writable OpenPI clone is missing: {OPENPI_ROOT}")
    os.chdir(OPENPI_ROOT)
    sys.path.insert(0, str(OPENPI_SRC))
    sys.path.insert(0, str(ROOT))


def build_config(mode_name: str, exp_name: str, lambda_phys: float):
    from gr00t.simulation.pi05_tactile_unit import (
        TactileCheckpointWeightLoader,
        TactilePi0Config,
        TactileSingleArmDataConfig,
        install_openpi_runtime_hooks,
    )
    from gr00t.simulation.s4_3_pi1 import TactileUnitMode
    from openpi.training import config as training_config

    mode = TactileUnitMode(mode_name)
    if mode is TactileUnitMode.NONE:
        raise ValueError("PI1 training accepts only CONTACT_STATE_TOKENS or CONTACT_STATE_TOKENS_PHYSICAL_AUX")
    if mode is TactileUnitMode.CONTACT_STATE_TOKENS and lambda_phys != 0:
        raise ValueError("PI1B requires lambda_phys=0")
    if mode is TactileUnitMode.CONTACT_STATE_TOKENS_PHYSICAL_AUX and not 1e-3 <= lambda_phys <= 1e-1:
        raise ValueError("PI1C requires the frozen calibrated lambda_phys in [1e-3,1e-1]")

    install_openpi_runtime_hooks()
    official = training_config.get_config("pinch_tongs")
    model = TactilePi0Config(
        dtype="bfloat16",
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_dim=32,
        action_horizon=30,
        max_token_len=250,
        pi05=True,
        discrete_state_input=True,
        tactile_unit_mode=mode,
        lambda_phys=lambda_phys,
    )
    data = TactileSingleArmDataConfig(
        root=DATASET_ROOT,
        sidecar_path=SIDECAR,
        mode=mode,
        repo_id="local_repo",
        base_config=training_config.DataConfig(prompt_from_task=True),
    )
    return dataclasses.replace(
        official,
        name="pinch_tongs",
        exp_name=exp_name,
        model=model,
        data=data,
        weight_loader=TactileCheckpointWeightLoader(str(BASE_PARAMS)),
        freeze_filter=model.get_freeze_filter(),
        assets_base_dir=str(ASSETS),
        checkpoint_base_dir=str(CHECKPOINTS),
        seed=42,
        batch_size=32,
        num_workers=4,
        num_train_steps=30_000,
        log_interval=100,
        save_interval=10_000,
        keep_period=5_000,
        ema_decay=None,
        fsdp_devices=1,
        overwrite=False,
        resume=False,
    )


def load_official_train_module():
    path = OPENPI_ROOT / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("s4_3_pi1_official_train", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=("CONTACT_STATE_TOKENS", "CONTACT_STATE_TOKENS_PHYSICAL_AUX"),
    )
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--lambda-phys", type=float, default=0.0)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    configure_imports()
    config = build_config(args.mode, args.exp_name, args.lambda_phys)
    if args.print_config:
        print(dataclasses.asdict(config))
        return
    load_official_train_module().main(config)


if __name__ == "__main__":
    main()
