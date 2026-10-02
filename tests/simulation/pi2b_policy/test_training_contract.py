from __future__ import annotations

import fcntl
import json
from pathlib import Path

import numpy as np
import pytest

from gr00t.simulation.pi2b_policy.contract import MODEL_ORDER, Workspace
from gr00t.simulation.pi2b_policy.coordination import open_lock
from gr00t.simulation.pi2b_policy.training import build_config


ROOT = Path(__file__).resolve().parents[3]


def test_five_model_configs_keep_batch_steps_seed_and_mode():
    workspace = Workspace.load(ROOT)
    summaries = {}
    for model_id in MODEL_ORDER:
        config = build_config(workspace, model_id, 43, fsdp_devices=1)
        summaries[model_id] = config
        assert config.seed == 43
        assert config.batch_size == 32
        assert config.num_train_steps == 30000
        assert config.fsdp_devices == 1
        assert config.overwrite is False and config.resume is False
    assert type(summaries["B0"].model).__name__ == "Pi0Config"
    assert type(summaries["B0"].data).__name__ == "SingleArmDataConfig"
    assert summaries["B_VA27"].model.tactile_unit_mode.value == "VA_PHYSICAL_AUX"
    assert summaries["B1"].model.tactile_unit_mode.value == "CONTACT_STATE_TOKENS"
    assert summaries["B_HVA"].model.tactile_unit_mode.value == "CONTACT_STATE_TOKENS_VA_PHYSICAL_AUX"
    assert summaries["B2"].model.tactile_unit_mode.value == "CONTACT_STATE_TOKENS_PHYSICAL_AUX"
    assert summaries["B_VA27"].data.sidecar_path == workspace.va27_sidecar
    assert summaries["B_HVA"].data.sidecar_path == workspace.contact_sidecar
    assert summaries["B_HVA"].data.target_sidecar_path == workspace.va27_sidecar


def test_canonical_config_rejects_extra_seed_or_changed_steps():
    workspace = Workspace.load(ROOT)
    with pytest.raises(ValueError):
        build_config(workspace, "B0", 45, fsdp_devices=1)
    with pytest.raises(ValueError):
        build_config(workspace, "B0", 43, fsdp_devices=1, steps=10)
    with pytest.raises(ValueError):
        build_config(workspace, "B0", 43, fsdp_devices=3)


def test_sidecar_indices_masks_and_no_h_va_fields():
    workspace = Workspace.load(ROOT)
    with np.load(workspace.contact_sidecar, allow_pickle=False) as contact, np.load(
        workspace.va27_sidecar, allow_pickle=False
    ) as va:
        assert set(va.files) == {"index", "va_shared_target", "va_aux_valid"}
        assert "contact_state" not in va.files
        np.testing.assert_array_equal(contact["index"], va["index"])
        np.testing.assert_array_equal(contact["physical_aux_valid"], va["va_aux_valid"])
        assert int(va["va_aux_valid"].sum()) == 37365
        assert int((~va["va_aux_valid"]).sum()) == 2700


def test_temporary_lock_proves_shared_and_exclusive_behavior(tmp_path):
    path = tmp_path / "barrier.lock"
    first = open_lock(path, shared=True)
    second = open_lock(path, shared=True)
    with pytest.raises(BlockingIOError):
        open_lock(path, shared=False)
    second.close()
    first.close()
    exclusive = open_lock(path, shared=False)
    with pytest.raises(BlockingIOError):
        open_lock(path, shared=True)
    exclusive.close()


def test_public_configs_do_not_embed_private_absolute_paths():
    for path in sorted((ROOT / "configs/simulation/pi2b_policy").glob("*.json")):
        text = path.read_text()
        assert "/home/" not in text
        assert "/mnt/" not in text
        json.loads(text)
