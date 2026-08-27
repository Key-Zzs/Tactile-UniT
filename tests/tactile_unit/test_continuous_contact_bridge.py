from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from gr00t.tactile_unit.continuous_contact_bridge import (
    CausalContactGate,
    TokenSetCrossAttentionBridge,
    TwoTowerContinuousProjector,
    bridge_objective,
    paired_info_nce,
    parameter_count,
)


def tokens(batch: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    return torch.randn(batch, 8, 32, generator=generator)


@pytest.mark.parametrize("kind", ["linear", "residual_mlp"])
def test_b1_projector_shape_finite_and_small(kind: str) -> None:
    model = TwoTowerContinuousProjector(kind)
    vision, contact = model(tokens(), tokens() + 0.2)
    assert vision.shape == contact.shape == (4, 8, 32)
    assert torch.isfinite(vision).all() and torch.isfinite(contact).all()
    assert parameter_count(model) < 20_000


def test_b2_cross_attention_shape_finite_and_no_position_assumption() -> None:
    model = TokenSetCrossAttentionBridge(heads=4).eval()
    vision, contact = tokens(), tokens() + 0.2
    projected_vision, projected_contact = model(vision, contact)
    assert projected_vision.shape == projected_contact.shape == (4, 8, 32)
    assert torch.isfinite(projected_vision).all()
    assert parameter_count(model) < 50_000
    permuted = contact[:, torch.tensor([7, 5, 3, 1, 6, 4, 2, 0])]
    permuted_vision, _ = model(vision, permuted)
    assert torch.allclose(projected_vision, permuted_vision, atol=1e-5, rtol=1e-5)


def test_missing_contact_is_exact_vision_fallback() -> None:
    model = CausalContactGate().eval()
    vision = tokens()
    assert torch.equal(model.residual_fuse(vision, None, None), vision)
    residual = tokens() * 0.1
    masked = torch.zeros(len(vision), dtype=torch.bool)
    fused = model.residual_fuse(vision, residual, torch.zeros(4, 256), masked)
    assert torch.equal(fused, vision)


def test_gate_available_missing_and_zero_current_are_finite_deterministic() -> None:
    torch.manual_seed(2)
    model = CausalContactGate().eval()
    vision = tokens()
    current = torch.zeros(4, 256)
    available = model(vision, current)
    missing = model(vision, None)
    assert available.shape == missing.shape == (4, 1, 1)
    assert torch.isfinite(available).all()
    assert torch.equal(missing, torch.zeros_like(missing))
    assert torch.equal(available, model(vision, current))


def test_gate_api_has_no_future_dependency() -> None:
    parameters = set(inspect.signature(CausalContactGate.forward).parameters)
    assert parameters == {"self", "current_vision", "current_contact", "contact_available"}


def test_bridge_objective_is_finite_and_weighted() -> None:
    vision, contact = tokens(), tokens() + 0.2
    losses = bridge_objective(
        vision,
        contact,
        vision,
        contact,
        weights=torch.tensor([1.0, 3.0, 1.0, 4.0]),
    )
    assert torch.isfinite(losses.total)
    assert losses.total.requires_grad is False
    assert paired_info_nce(vision, contact) < paired_info_nce(vision, contact.flip(0))


def test_deterministic_checkpoint_reload(tmp_path: Path) -> None:
    torch.manual_seed(11)
    before = TwoTowerContinuousProjector().eval()
    path = tmp_path / "bridge.pt"
    torch.save(before.state_dict(), path)
    after = TwoTowerContinuousProjector().eval()
    after.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    sample = tokens()
    for left, right in zip(before(sample, sample), after(sample, sample)):
        assert torch.equal(left, right)


def test_invalid_shapes_are_rejected() -> None:
    model = TwoTowerContinuousProjector()
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(2, 7, 32), torch.zeros(2, 8, 32))
