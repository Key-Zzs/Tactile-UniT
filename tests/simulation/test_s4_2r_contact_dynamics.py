from __future__ import annotations

import json
from pathlib import Path

import torch

from gr00t.contact_dynamics.models import (
    ContactDynamicsEncoder,
    ContactDynamicsModel,
    DeltaMLPEncoder,
    LatentTransitionDecoder,
)
from scripts.simulation import build_s4_2r_contact_dynamics_cache as cache_builder


ROOT = Path(__file__).resolve().parents[2]


def test_contact_dynamics_windows_are_exact_and_disjoint() -> None:
    protocol = json.loads(
        (ROOT / "configs/simulation/s4_2_protocol.json").read_text()
    )
    timing = protocol["timing"]
    current = set(range(timing["current_history_relative_steps"][0], 1))
    future = set(range(timing["future_history_relative_steps"][0], 28))
    assert timing["current_history_relative_steps"] == [-25, 0]
    assert timing["future_history_relative_steps"] == [2, 27]
    assert timing["transition_offset_steps"] == 27
    assert timing["transition_duration_sec"] == 0.54
    assert not current.intersection(future)


def test_dynamics_models_emit_exact_codes_under_parameter_budget() -> None:
    current = torch.randn(4, 256)
    future = torch.randn(4, 256)
    for encoder in (DeltaMLPEncoder(), ContactDynamicsEncoder()):
        model = ContactDynamicsModel(encoder, LatentTransitionDecoder()).eval()
        with torch.inference_mode():
            output = model(current, future)
        assert output["code"].shape == (4, 8, 32)
        assert output["future"].shape == (4, 256)
        assert torch.isfinite(output["code"]).all()
        assert sum(parameter.numel() for parameter in model.parameters()) <= 2_000_000


def test_dynamics_controls_change_code_or_decoding() -> None:
    model = ContactDynamicsModel(ContactDynamicsEncoder(), LatentTransitionDecoder()).eval()
    current = torch.randn(8, 256)
    future = torch.randn(8, 256)
    with torch.inference_mode():
        full = model(current, future)
        reversed_code = model.encoder(future, current)
        zero_prediction = model.decoder(torch.zeros_like(full["code"]), current)
    assert not torch.equal(full["code"], reversed_code)
    assert not torch.equal(full["future"], zero_prediction)


def test_dynamics_failure_blocks_all_downstream_stages_and_test_access() -> None:
    decision = json.loads(
        (ROOT / "configs/simulation/s4_2r_contact_dynamics_decision.json").read_text()
    )
    assert decision["decision"] == "S4_2_3_CONTACT_DYNAMICS_FAIL"
    assert decision["failed_gates"] == ["dynamic_improvement_at_least_10_percent"]
    assert decision["validation"]["dynamic_relative_improvement"] < decision[
        "validation"
    ]["required_dynamic_relative_improvement"]
    assert all(
        status == "NOT_RUN_DEPENDENCY_BLOCKED"
        for status in decision["downstream"].values()
    )
    assert decision["test_loaded"] is False
    assert cache_builder.SPLITS == ("train", "validation")
