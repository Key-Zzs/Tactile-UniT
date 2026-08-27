from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from gr00t.tactile_unit.c2r_contact_preservation import (
    ACCEPTED_C2_CHECKPOINT_SHA256,
    C2RLossWeights,
    c2r_contact_loss,
    canonical_contact_probe,
    configure_contact_only_trainability,
    contact_relational_preservation,
    contact_sample_weight,
    frozen_state_digest,
    retention,
    verify_accepted_c2_checkpoint,
    weighted_sample_mse,
)
from gr00t.tactile_unit.continuous_vac_shared_space import (
    ContinuousVACSharedSpace,
    load_checkpoint,
    save_checkpoint,
)
from scripts.tactile_unit.train_c2r_contact_preservation import trial_grid


ROOT = Path(__file__).resolve().parents[2]


class FrozenDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mapping = nn.Linear(8 * 32, 256)

    def forward(self, z_value, current):
        return current + self.mapping(z_value.flatten(1))


def test_accepted_c2_checkpoint_sha_is_exact():
    path = ROOT / ".local/experiments/tactile_unit/vac_c2/selected.pt"
    assert verify_accepted_c2_checkpoint(path) == ACCEPTED_C2_CHECKPOINT_SHA256


def test_only_contact_projector_and_recovery_are_trainable():
    model = ContinuousVACSharedSpace("C2-slot")
    boundary = configure_contact_only_trainability(model)
    assert boundary["trainable_names"]
    assert all(name.startswith(("projectors.contact.", "recovery.contact.")) for name in boundary["trainable_names"])
    assert "shared_slots" in boundary["frozen_names"]
    assert all(not parameter.requires_grad for parameter in model.projectors["vision"].parameters())
    assert all(not parameter.requires_grad for parameter in model.projectors["action"].parameters())


def test_gradient_routes_through_frozen_decoder_to_contact_only():
    torch.manual_seed(3)
    model = ContinuousVACSharedSpace("C2-slot")
    configure_contact_only_trainability(model)
    decoder = FrozenDecoder().eval().requires_grad_(False)
    native = {name: torch.randn(8, 8, 32) for name in ("vision", "action", "contact")}
    current = torch.randn(8, 256)
    future = torch.randn(8, 256)
    loss, values = c2r_contact_loss(
        model, decoder, native, torch.arange(8), torch.arange(8) % 2 == 0,
        torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]), current, future,
        temperature=0.1, dynamic_weight=2.0, boundary_weight=2.0,
        weights=C2RLossWeights(),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert values["future"] == pytest.approx(values["delta"])
    assert any(parameter.grad is not None for parameter in model.projectors["contact"].parameters())
    assert any(parameter.grad is not None for parameter in model.recovery["contact"].parameters())
    assert all(parameter.grad is None for parameter in decoder.parameters())
    assert all(parameter.grad is None for parameter in model.projectors["vision"].parameters())
    assert all(parameter.grad is None for parameter in model.projectors["action"].parameters())
    assert model.shared_slots.grad is None
    assert all(value.grad is None for value in native.values())


def test_future_and_delta_formulas_use_frozen_target():
    torch.manual_seed(5)
    model = ContinuousVACSharedSpace("C2-slot")
    configure_contact_only_trainability(model)
    decoder = FrozenDecoder().eval().requires_grad_(False)
    native = {name: torch.randn(4, 8, 32) for name in ("vision", "action", "contact")}
    current = torch.randn(4, 256)
    future = torch.randn(4, 256)
    _, result = c2r_contact_loss(
        model, decoder, native, torch.arange(4), torch.zeros(4, dtype=torch.bool),
        torch.zeros(4, dtype=torch.long), current, future,
        temperature=0.1, dynamic_weight=2.0, boundary_weight=1.0,
        weights=C2RLossWeights(),
    )
    with torch.no_grad():
        shared = model.encode("contact", native["contact"])
        recovered = model.recover("contact", shared)
        predicted = decoder(recovered, current)
        expected_future = torch.square(predicted - future).mean()
        expected_delta = torch.square((predicted - current) - (future - current)).mean()
    assert result["future"] == pytest.approx(expected_future)
    assert result["delta"] == pytest.approx(expected_delta)


def test_train_derived_dynamic_and_boundary_weights():
    dynamic = torch.tensor([False, True, False, True])
    transition = torch.tensor([0, 0, 1, 2])
    weight = contact_sample_weight(dynamic, transition, dynamic_weight=2.0, boundary_weight=2.0)
    assert torch.equal(weight, torch.tensor([1.0, 2.0, 2.0, 4.0]))
    prediction = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    target = torch.zeros_like(prediction)
    expected = (1.0 + 2.0 * 4.0 + 2.0 * 9.0 + 4.0 * 16.0) / 9.0
    assert weighted_sample_mse(prediction, target, weight) == pytest.approx(expected)


def test_contact_relational_loss_covers_pairwise_neighborhood_and_ordering():
    torch.manual_seed(6)
    native = torch.randn(16, 8, 32)
    identical, identical_parts = contact_relational_preservation(native, native.clone())
    corrupted, corrupted_parts = contact_relational_preservation(native, native.roll(1, 0))
    assert identical == pytest.approx(0.0, abs=1e-7)
    assert set(identical_parts) == {"pairwise", "neighborhood", "ordering"}
    assert corrupted > identical
    assert corrupted_parts["pairwise"] > 0


def test_probe_protocol_is_identical_and_reports_rare_classes():
    rng = np.random.default_rng(7)
    train_x = rng.normal(size=(80, 8, 32)).astype(np.float32)
    test_x = rng.normal(size=(24, 8, 32)).astype(np.float32)
    train_y = np.arange(80) % 4
    test_y = np.arange(24) % 4
    native = canonical_contact_probe(train_x, test_x, train_y, test_y, 4)
    shared = canonical_contact_probe(train_x.copy(), test_x.copy(), train_y, test_y, 4)
    assert native == shared
    assert retention(shared, native) == pytest.approx(1.0)
    assert set(shared["per_class"]["1"]) == {"precision", "recall", "f1", "support"}


def test_bounded_grid_and_validation_only_selection():
    import json

    config = json.loads((ROOT / "configs/tactile_unit/c2r_contact_preservation_remediation.json").read_text())
    grid = trial_grid(config)
    assert len(grid) == 6
    assert {row["lambda_future"] for row in grid} == {0.5, 1.0, 2.0}
    assert {row["boundary_weight"] for row in grid} == {1.0, 2.0}
    assert all(row["lambda_delta"] == row["lambda_future"] for row in grid)
    trainer = (ROOT / "scripts/tactile_unit/train_c2r_contact_preservation.py").read_text()
    assert 'load_split(cache_root, "train"' in trainer
    assert 'load_split(cache_root, "validation"' in trainer
    assert 'load_split(cache_root, "test"' not in trainer
    assert '"test_loaded": False' in trainer


def test_selection_lock_is_verified_before_locked_test_load():
    source = (ROOT / "scripts/tactile_unit/evaluate_c2r_contact_preservation.py").read_text()
    lock_check = source.index("actual_selection_hash != expected_selection_hash")
    test_load = source.index('load_split(cache_root, "test"')
    assert lock_check < test_load
    assert "LOCKED RE-EVALUATION AFTER POST-C2 REMEDIATION" in source
    assert '"first_look_untouched_test": False' in source


def test_contact_update_preserves_va_outputs_and_frozen_digest():
    torch.manual_seed(11)
    model = ContinuousVACSharedSpace("C2-slot")
    configure_contact_only_trainability(model)
    vision = torch.randn(3, 8, 32)
    action = torch.randn(3, 8, 32)
    before_v = model.encode("vision", vision).detach().clone()
    before_a = model.encode("action", action).detach().clone()
    digest = frozen_state_digest(model)
    with torch.no_grad():
        next(parameter for parameter in model.projectors["contact"].parameters()).add_(0.1)
    assert torch.equal(before_v, model.encode("vision", vision))
    assert torch.equal(before_a, model.encode("action", action))
    assert frozen_state_digest(model) == digest


def test_checkpoint_reload_and_independent_contact_are_exact(tmp_path):
    torch.manual_seed(13)
    model = ContinuousVACSharedSpace("C2-slot").eval()
    value = torch.randn(3, 8, 32)
    before = model.encode("contact", value)
    checkpoint = tmp_path / "selected.pt"
    save_checkpoint(checkpoint, model, {"selection_split": "validation only", "test_loaded": False})
    loaded, metadata = load_checkpoint(checkpoint)
    after = loaded.eval().encode("contact", value)
    assert torch.equal(before, after)
    assert metadata["test_loaded"] is False
    unrelated = torch.randn(3, 8, 32) * 100
    _ = loaded.encode("vision", unrelated)
    assert torch.equal(after, loaded.encode("contact", value))
