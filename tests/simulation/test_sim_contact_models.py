from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from gr00t.simulation.sim_contact_models import (
    build_sim_contact_teacher,
    load_teacher_checkpoint,
    parameter_count,
    save_teacher_checkpoint,
)


ROOT = Path(__file__).resolve().parents[2]


def test_contact_state_collapse_gate_is_frozen() -> None:
    contract = json.loads(
        (ROOT / "configs/simulation/s4_2_representation_contract.json").read_text()
    )
    gates = contract["contact_teacher"]["gates"]
    assert contract["status"] == "FROZEN_BEFORE_REPRESENTATION_TRAINING"
    assert gates["effective_rank_min"] == 16.0
    assert gates["near_zero_variance_fraction_max"] == 0.5
    assert contract["test_loaded"] is False


@pytest.mark.parametrize("candidate", ["B0", "B1", "B2", "B3"])
def test_teacher_shapes_finite_and_parameter_budget(candidate: str) -> None:
    model = build_sim_contact_teacher(candidate).eval()
    history = torch.randn(3, 26, 30)
    with torch.inference_mode():
        output = model(history)
    assert output["latent"].shape == (3, 256)
    assert output["future"].shape == (3, 13, 30)
    assert torch.isfinite(output["latent"]).all()
    assert torch.isfinite(output["future"]).all()
    assert parameter_count(model) <= 5_000_000
    if candidate == "B3":
        assert output["reconstruction"].shape == (3, 26, 30)


def test_proposed_teacher_uses_order_and_first_differences() -> None:
    model = build_sim_contact_teacher("B3").eval()
    history = torch.arange(26, dtype=torch.float32).view(1, 26, 1).expand(-1, -1, 30)
    with torch.inference_mode():
        forward = model.encode(history)
        reversed_value = model.encode(history.flip(1))
    assert not torch.equal(forward, reversed_value)


def test_teacher_checkpoint_round_trip_is_exact(tmp_path: Path) -> None:
    model = build_sim_contact_teacher("B3", reconstruction_weight=0.5).eval()
    path = tmp_path / "teacher.pt"
    save_teacher_checkpoint(
        path,
        model,
        candidate="B3-N1-R0.50",
        normalization={"candidate": "N1"},
        metadata={"reconstruction_weight": 0.5, "test_loaded": False},
    )
    loaded, checkpoint = load_teacher_checkpoint(path)
    loaded.eval()
    value = torch.randn(2, 26, 30)
    with torch.inference_mode():
        torch.testing.assert_close(model(value)["future"], loaded(value)["future"], rtol=0, atol=0)
    assert checkpoint["metadata"]["test_loaded"] is False


def test_proposed_rejects_wrong_history_shape() -> None:
    model = build_sim_contact_teacher("B3")
    with pytest.raises(ValueError, match="expected history"):
        model(torch.randn(1, 25, 30))
