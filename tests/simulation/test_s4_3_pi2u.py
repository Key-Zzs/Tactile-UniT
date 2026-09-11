from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gr00t.simulation.s4_3_pi1 import TactileUnitMode
from gr00t.simulation.s4_3_pi2u_va import VAOnlyBridge, different_episode_info_nce


ROOT = Path(__file__).resolve().parents[2]


def test_bva_mode_is_explicit_and_contact_free() -> None:
    assert TactileUnitMode.VA_PHYSICAL_AUX.value == "VA_PHYSICAL_AUX"
    source = (ROOT / "gr00t/simulation/pi05_tactile_unit.py").read_text()
    branch = source[source.index("class TactilePi0") :]
    assert "TactileUnitMode.VA_PHYSICAL_AUX" in branch
    assert "if self.tactile_unit_mode is TactileUnitMode.VA_PHYSICAL_AUX" in branch


def test_va_bridge_shape_and_modality_rejection() -> None:
    model = VAOnlyBridge()
    value = torch.randn(3, 8, 32)
    assert model.encode("vision", value).shape == value.shape
    assert model.encode("action", value).shape == value.shape
    with pytest.raises(ValueError, match="rejects modality"):
        model.encode("contact", value)


def test_different_episode_nce_is_finite() -> None:
    query = torch.randn(4, 8, 32)
    candidate = torch.randn(4, 8, 32)
    episode = torch.tensor([0, 0, 1, 2])
    loss = different_episode_info_nce(query, candidate, episode, temperature=0.1)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_frozen_bva_protocol_contract() -> None:
    protocol = json.loads((ROOT / "configs/simulation/s4_3_pi2u_bva_protocol.json").read_text())
    assert protocol["status"] == "FROZEN_BEFORE_TRAINING"
    assert protocol["seed"] == 42
    assert protocol["steps"] == 30_000
    assert protocol["global_batch_size"] == 32
    assert protocol["auxiliary_target"]["sidecar_allowed_fields"] == [
        "index",
        "va_shared_target",
        "va_aux_valid",
    ]
    assert protocol["official_policy"]["tactile_or_contact_runtime_input"] is False


def test_materialized_sidecar_is_exact_and_tail_accounted() -> None:
    path = ROOT / ".local/datasets/simulation/s4_3_pi2u/pinch_tongs_va/sidecar.npz"
    if not path.exists():
        pytest.skip("local frozen BVA target sidecar is not materialized")
    with np.load(path, allow_pickle=False) as source:
        assert set(source.files) == {"index", "va_shared_target", "va_aux_valid"}
        assert source["index"].shape == (40_065,)
        assert source["va_shared_target"].shape == (40_065, 8, 32)
        valid = source["va_aux_valid"]
        assert valid.dtype == np.bool_
        assert int(valid.sum()) == 37_365
        assert int((~valid).sum()) == 2_700
        assert np.all(source["va_shared_target"][~valid] == 0)


def test_mode_and_none_parity_artifacts_pass() -> None:
    artifact_root = ROOT / ".local/artifacts/simulation/s4_3_pi2u"
    for name in ("bva_mode_contract.json", "bva_none_parity.json", "contact_leakage_audit.json"):
        path = artifact_root / name
        if not path.exists():
            pytest.skip(f"local validation artifact is not materialized: {name}")
        assert json.loads(path.read_text())["status"] == "PASS"
