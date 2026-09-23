"""Exact five-model training-config factory for PI2B Track A."""

from __future__ import annotations

import dataclasses
import importlib.util
import os
from pathlib import Path
import sys
from typing import Any

from .contract import RECIPES, Workspace


def configure_imports(workspace: Workspace) -> None:
    """Bind the pinned historical OpenPI source and this worktree explicitly."""

    openpi_src = workspace.openpi_root / "src"
    if not (workspace.openpi_root / "scripts/train.py").is_file() or not openpi_src.is_dir():
        raise FileNotFoundError("accepted OpenPI source copy is incomplete")
    root = str(workspace.root)
    source = str(openpi_src)
    sys.path[:] = [value for value in sys.path if value not in {root, source}]
    sys.path[:0] = [source, root]
    os.chdir(workspace.openpi_root)


def build_config(
    workspace: Workspace,
    model_id: str,
    seed: int,
    *,
    fsdp_devices: int,
    steps: int = 30_000,
    checkpoint_base_dir: Path | None = None,
    engineering_fixture: bool = False,
) -> Any:
    """Create the accepted recipe with only seed/output/device-count varied."""

    if model_id not in RECIPES:
        raise ValueError(f"unsupported model: {model_id}")
    if seed not in {43, 44}:
        raise ValueError("Track A new training seed must be 43 or 44")
    if fsdp_devices not in {1, 2, 4} or 32 % fsdp_devices:
        raise ValueError("device count must be 1, 2, or 4 and divide global batch 32")
    if engineering_fixture:
        if not 3 <= steps <= 5:
            raise ValueError("engineering fixture must be limited to 3-5 optimizer steps")
    elif steps != 30_000:
        raise ValueError("canonical PI2B runs require exactly 30000 optimizer steps")

    configure_imports(workspace)
    from gr00t.simulation.pi05_tactile_unit import (
        TactileCheckpointWeightLoader,
        TactilePi0Config,
        TactileSingleArmDataConfig,
        install_openpi_runtime_hooks,
    )
    from gr00t.simulation.s4_3_pi1 import TactileUnitMode
    from openpi.training import config as training_config
    from openpi.training import weight_loaders

    recipe = RECIPES[model_id]
    mode = TactileUnitMode(recipe.mode)
    official = training_config.get_config("pinch_tongs")
    if model_id == "B0":
        model = official.model
        data = dataclasses.replace(official.data, root=workspace.dataset_root)
        loader = weight_loaders.CheckpointWeightLoader(str(workspace.base_params))
        freeze_filter = official.freeze_filter
    else:
        install_openpi_runtime_hooks()
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
            lambda_phys=recipe.lambda_phys,
        )
        sidecar = workspace.va27_sidecar if model_id == "B_VA27" else workspace.contact_sidecar
        target_sidecar = workspace.va27_sidecar if model_id == "B_HVA" else None
        data = TactileSingleArmDataConfig(
            root=workspace.dataset_root,
            sidecar_path=sidecar,
            target_sidecar_path=target_sidecar,
            mode=mode,
            repo_id="local_repo",
            base_config=training_config.DataConfig(prompt_from_task=True),
        )
        loader = TactileCheckpointWeightLoader(str(workspace.base_params))
        freeze_filter = model.get_freeze_filter()

    base_dir = checkpoint_base_dir or workspace.run_root(model_id, seed)
    return dataclasses.replace(
        official,
        name="pinch_tongs",
        exp_name=workspace.experiment_name(model_id, seed),
        model=model,
        data=data,
        weight_loader=loader,
        freeze_filter=freeze_filter,
        assets_base_dir=str(workspace.assets),
        checkpoint_base_dir=str(base_dir),
        seed=seed,
        batch_size=32,
        num_workers=0 if engineering_fixture else 4,
        num_train_steps=steps,
        log_interval=1 if engineering_fixture else 100,
        save_interval=steps if engineering_fixture else 10_000,
        keep_period=5_000,
        ema_decay=None,
        fsdp_devices=fsdp_devices,
        overwrite=False,
        resume=False,
        wandb_enabled=not engineering_fixture,
    )


def load_official_train_module(workspace: Workspace):
    path = workspace.openpi_root / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("s4_3_pi2b_official_train", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config_summary(config: Any, model_id: str, fixture: bool) -> dict[str, Any]:
    recipe = RECIPES[model_id]
    return {
        "model_id": model_id,
        "mode": recipe.mode,
        "seed": config.seed,
        "lambda_phys": recipe.lambda_phys,
        "batch_size": config.batch_size,
        "num_train_steps": config.num_train_steps,
        "fsdp_devices": config.fsdp_devices,
        "checkpoint_dir": str(config.checkpoint_dir),
        "engineering_fixture": fixture,
        "online_contact_history": recipe.online_contact_history,
        "target": recipe.target,
    }
