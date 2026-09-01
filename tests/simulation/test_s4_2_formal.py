from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from gr00t.simulation.s4_2_formal import (
    ConditionalContactPredictor,
    FormalActionEncoder,
    FormalVACBridge,
    ScalarLogVarianceHead,
    SharedPrivateDecomposer,
)

ROOT = Path(__file__).resolve().parents[2]


def test_formal_protocol_preserves_frozen_scientific_boundary() -> None:
    value = json.loads((ROOT / "configs/simulation/s4_2_formal_downstream.json").read_text())
    boundary = value["scientific_boundary"]
    assert value["status"] == "FROZEN_BEFORE_FORMAL_S4_2_4_TRAINING"
    assert boundary["historical_10_percent_gate"] == "FAIL_UNCHANGED"
    assert boundary["canonical_contact"] == "C3"
    assert value["data"]["split_unit"] == ["task", "source_trajectory_id"]
    assert value["data"]["formal_test_loaded"] is False


def test_all_action_candidates_obey_exact_contract_and_budget() -> None:
    feature = torch.randn(2, 27, 66)
    state = torch.randn(2, 22)
    for candidate in ("A0", "A1", "A2"):
        model = FormalActionEncoder(candidate)
        output = model(feature, state)
        assert output["code"].shape == (2, 8, 32)
        assert output["action"].shape == (2, 27, 22)
        assert sum(parameter.numel() for parameter in model.parameters()) <= 25_000_000


def test_formal_bridge_is_independently_encodable_and_bounded() -> None:
    native = torch.randn(2, 8, 32)
    for adapter in ("affine", "mlp", "slot"):
        model = FormalVACBridge(adapter)
        outputs = {name: model.encode(name, native) for name in model.modalities}
        assert all(value.shape == (2, 8, 32) for value in outputs.values())
        assert list(inspect.signature(model.encode).parameters) == ["modality", "native"]
        counts = [
            sum(p.numel() for p in model.projectors[name].parameters()) for name in model.modalities
        ]
        assert len(set(counts)) == 1
        assert max(counts) <= 50_000
        assert sum(p.numel() for p in model.parameters()) <= 300_000


def test_shared_private_and_conditional_interfaces_are_explicit() -> None:
    native = torch.randn(3, 8, 32)
    shared = torch.randn(3, 8, 32)
    model = SharedPrivateDecomposer()
    private = model.encode_private("contact", native, shared)
    assert private.shape == (3, 8, 16)
    assert model.reconstruct("contact", shared, private).shape == native.shape
    assert model.cross_predict("vision", shared).shape == native.shape
    predictor = ConditionalContactPredictor(("vision", "action", "history"))
    values = {name: torch.randn(3, 8, 32) for name in predictor.modalities}
    assert predictor(values).shape == (3, 8, 32)
    uncertainty = ScalarLogVarianceHead(("vision", "action"))
    assert uncertainty(values).shape == (3,)


def test_formal_freeze_script_never_loads_a_test_array() -> None:
    source = (ROOT / "scripts/simulation/freeze_s4_2_formal_protocol.py").read_text()
    assert 'PAIR_ROOT / "test.npz"' not in source
    assert 'formal_test_loaded": False' in source


def test_s4_2_4_script_uses_exact_paired_train_and_validation_only() -> None:
    source = (ROOT / "scripts/simulation/run_s4_2_4_formal_representations.py").read_text()
    assert 'PAIR_ROOT / "train.npz"' in source
    assert 'PAIR_ROOT / "validation.npz"' in source
    assert 'PAIR_ROOT / "test.npz"' not in source
    assert "formal train/validation source-group leakage" in source


def test_s4_2_5_script_has_five_bounded_trials_and_no_test_access() -> None:
    source = (ROOT / "scripts/simulation/run_s4_2_5_formal_bridge.py").read_text()
    assert 'PAIR_ROOT / "paired_train.npz"' in source
    assert 'PAIR_ROOT / "paired_validation.npz"' in source
    assert 'PAIR_ROOT / "paired_test.npz"' not in source
    protocol = json.loads((ROOT / "configs/simulation/s4_2_formal_downstream.json").read_text())
    assert len(protocol["s4_2_5"]["trainable_candidates"]) == 5
