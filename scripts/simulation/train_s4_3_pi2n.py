#!/usr/bin/env python3
"""Launch one frozen PI2N candidate from an independent pi05_base initialization."""

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
CONTACT_SIDECAR = ROOT / ".local/datasets/simulation/s4_3_pi1/pinch_tongs_official_tactile/sidecar.npz"
VA27_TARGET = ROOT / ".local/datasets/simulation/s4_3_pi2m/pinch_tongs_va_t27/sidecar.npz"
BASE_PARAMS = ROOT / ".local/external/s4_3_pi0/models/DexJoCo-Pi05/pi05_base/params"
ASSETS = ROOT / ".local/cache/simulation/s4_3_pi0/assets"
MODEL_IDS = ("B_VA27", "B_VAC_V")


def pi2n_run_root() -> Path:
    configured = os.environ.get("PI2N_RUN_ROOT")
    path = Path(configured) if configured else ROOT / ".local/experiments/simulation/s4_3_pi2n"
    resolved = path.resolve()
    expected = Path("/mnt/ugreen_nas/storage/UniT_storage/experiments/simulation/s4_3_pi2n")
    if resolved != expected:
        raise RuntimeError(f"PI2N run root does not resolve to audited NAS path: {resolved}")
    return resolved


def vac_v_target() -> Path:
    return pi2n_run_root() / "caches/pinch_tongs_vac_v_t27/sidecar.npz"


def configure_imports() -> None:
    if not OPENPI_ROOT.is_dir():
        raise FileNotFoundError(f"writable OpenPI clone is missing: {OPENPI_ROOT}")
    os.chdir(OPENPI_ROOT)
    sys.path[:0] = [str(OPENPI_SRC), str(ROOT)]


def build_config(model_id: str, exp_name: str, lambda_phys: float, fsdp_devices: int):
    if model_id not in MODEL_IDS:
        raise ValueError(f"unsupported PI2N candidate: {model_id}")
    if fsdp_devices not in {1, 2, 4} or 32 % fsdp_devices:
        raise ValueError("PI2N fsdp_devices must be one, two, or four and divide global batch 32")
    if not 1e-3 <= lambda_phys <= 1e-1:
        raise ValueError("PI2N requires a frozen TRAIN-only calibrated lambda in [1e-3,1e-1]")
    from gr00t.simulation.pi05_tactile_unit import (
        TactileCheckpointWeightLoader,
        TactilePi0Config,
        TactileSingleArmDataConfig,
        install_openpi_runtime_hooks,
    )
    from gr00t.simulation.s4_3_pi1 import TactileUnitMode
    from openpi.training import config as training_config

    install_openpi_runtime_hooks()
    official = training_config.get_config("pinch_tongs")
    if model_id == "B_VA27":
        mode = TactileUnitMode.VA_PHYSICAL_AUX
        sidecar_path = VA27_TARGET
        target_sidecar_path = None
    else:
        mode = TactileUnitMode.CONTACT_STATE_TOKENS_VAC_V_PHYSICAL_AUX
        sidecar_path = CONTACT_SIDECAR
        target_sidecar_path = vac_v_target()
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
        sidecar_path=sidecar_path,
        target_sidecar_path=target_sidecar_path,
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
        checkpoint_base_dir=str(pi2n_run_root() / "runs" / model_id),
        seed=42,
        batch_size=32,
        num_workers=4,
        num_train_steps=30_000,
        log_interval=100,
        save_interval=10_000,
        keep_period=5_000,
        ema_decay=None,
        fsdp_devices=fsdp_devices,
        overwrite=False,
        resume=False,
    )


def load_official_train_module():
    path = OPENPI_ROOT / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("s4_3_pi2n_official_train", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", choices=MODEL_IDS, required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--lambda-phys", type=float, required=True)
    parser.add_argument("--fsdp-devices", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    configure_imports()
    config = build_config(args.model_id, args.exp_name, args.lambda_phys, args.fsdp_devices)
    if args.print_config:
        print(dataclasses.asdict(config))
        return
    load_official_train_module().main(config)


if __name__ == "__main__":
    main()
