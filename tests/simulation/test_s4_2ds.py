from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import torch

from gr00t.contact_dynamics.models import (
    ContactDynamicsModel,
    DeltaMLPEncoder,
    LatentTransitionDecoder,
)
from gr00t.simulation.s4_2ds import ActionPilot, PilotVACBridge
from scripts.simulation.audit_s4_2ds_contact_representations import group_split


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/simulation/s4_2ds_representation_selection.json"


def test_c2_has_independent_explicit_decoder_bottleneck() -> None:
    model = ContactDynamicsModel(DeltaMLPEncoder(), LatentTransitionDecoder()).eval()
    current = torch.randn(4, 256)
    future = torch.randn(4, 256)
    with torch.inference_mode():
        code = model.encoder(current, future)
        decoded = model.decoder(code, current)
        zero = model.decoder(torch.zeros_like(code), current)
        repeated = model.encoder(current, future)
    assert code.shape == (4, 8, 32)
    assert decoded.shape == (4, 256)
    assert torch.equal(code, repeated)
    assert not torch.equal(decoded, zero)
    source = inspect.getsource(DeltaMLPEncoder.forward)
    assert "future - current" in source
    assert not any(name in source for name in ("reward", "success", "task", "label"))


def test_ds_split_is_source_group_isolated_and_train_internal() -> None:
    task = np.asarray(["a"] * 20 + ["b"] * 20)
    source = np.asarray([f"a-{index // 2}" for index in range(20)] + [f"b-{index // 2}" for index in range(20)])
    train, dev, manifest = group_split(task, source, 4242)
    train_groups = set(zip(task[train], source[train]))
    dev_groups = set(zip(task[dev], source[dev]))
    assert train_groups.isdisjoint(dev_groups)
    assert manifest["group_overlap"] == 0
    assert manifest["formal_validation_excluded"] is True
    assert manifest["formal_test_excluded"] is True


def test_action_pilot_contract_and_temporal_encoder() -> None:
    model = ActionPilot()
    output = model(torch.randn(3, 27, 66), torch.randn(3, 22))
    assert output["code"].shape == (3, 8, 32)
    assert output["action"].shape == (3, 27, 22)
    assert sum(parameter.numel() for parameter in model.parameters()) <= 5_000_000
    with torch.inference_mode():
        forward = model.encode(torch.arange(27 * 66, dtype=torch.float32).view(1, 27, 66))
        reversed_value = model.encode(torch.arange(27 * 66, dtype=torch.float32).view(1, 27, 66).flip(1))
    assert not torch.equal(forward, reversed_value)


def test_bridge_has_equal_independent_modality_interfaces() -> None:
    model = PilotVACBridge()
    value = torch.randn(2, 8, 32)
    outputs = {name: model.encode(name, value) for name in model.modalities}
    assert all(output.shape == (2, 8, 32) for output in outputs.values())
    counts = [sum(parameter.numel() for parameter in model.projectors[name].parameters()) for name in model.modalities]
    assert len(set(counts)) == 1
    assert sum(parameter.numel() for parameter in model.parameters()) <= 300_000
    assert list(inspect.signature(model.encode).parameters) == ["modality", "value"]


def test_selection_protocol_preserves_historical_failure_and_locked_test() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_BEFORE_ACTION_OR_BRIDGE_PILOT_TRAINING"
    assert config["scientific_boundary"]["historical_decision"] == "S4_2_3_CONTACT_DYNAMICS_FAIL"
    assert config["scientific_boundary"]["remediation_decision"] == "S4_2DR_DYNAMICS_REMEDIATION_FAIL"
    assert config["scientific_boundary"]["historical_gate_modified"] is False
    assert config["scientific_boundary"]["historical_10_percent_gate_used_for_ds_selection"] is False
    assert config["data"]["formal_test_model_metrics_loaded"] is False
    assert config["contact_candidates"]["C2"]["common_adapter"] == "identity"
    assert config["contact_candidates"]["C3"]["common_adapter"] == "identity"
