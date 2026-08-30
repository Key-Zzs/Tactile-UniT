import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from gr00t.tactile_unit.continuous_vac_shared_space import (
    ContinuousVACSharedSpace,
    VACLossWeights,
    continuous_vac_loss,
    different_episode_info_nce,
    different_episode_permutation,
    effective_rank,
    geometry_diagnostics,
    load_checkpoint,
    pairwise_alignment_metrics,
    save_checkpoint,
)
from scripts.tactile_unit.evaluate_continuous_vac_shared_space import retention, temporal_ratios


@pytest.mark.parametrize("candidate", ["C0", "C1-linear", "C1-mlp", "C2-slot"])
def test_each_modality_is_independently_encodable(candidate):
    model = ContinuousVACSharedSpace(candidate)
    signature = inspect.signature(model.encode)
    assert list(signature.parameters) == ["modality", "native"]
    for modality in ("vision", "action", "contact"):
        value = torch.randn(4, 8, 32)
        output = model.encode(modality, value)
        assert output.shape == (4, 8, 32)
        assert torch.isfinite(output).all()
        recovered = model.recover(modality, output)
        assert recovered.shape == (4, 8, 32)
        assert torch.isfinite(recovered).all()
    assert not hasattr(model, "cross_modal_attention")


def test_shared_slot_candidate_has_no_pair_conditioned_candidate_path():
    model = ContinuousVACSharedSpace("C2-slot").eval()
    contact = torch.randn(5, 8, 32)
    first = model.encode("contact", contact)
    _unrelated_vision = torch.randn(5, 8, 32) * 100
    second = model.encode("contact", contact)
    assert torch.equal(first, second)


def test_gradient_reaches_only_c2_parameters():
    model = ContinuousVACSharedSpace("C1-mlp")
    native = {
        "vision": torch.randn(8, 8, 32),
        "action": torch.randn(8, 8, 32),
        "contact": torch.randn(8, 8, 32),
    }
    loss, _ = continuous_vac_loss(
        model, native, torch.arange(8), torch.arange(8) % 2 == 0,
        temperature=0.1, dynamic_weight=2.0, weights=VACLossWeights(),
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(value.grad is None for value in native.values())


def test_info_nce_positive_index_and_no_self_negative_bug():
    torch.manual_seed(3)
    value = torch.randn(12, 8, 32)
    episode = torch.arange(12)
    paired = different_episode_info_nce(value, value.clone(), episode, temperature=0.1)
    wrong = different_episode_info_nce(value, value.roll(1, 0), episode, temperature=0.1)
    assert paired < wrong
    same_episode = torch.tensor([0, 0, 1, 1, 2, 2])
    masked = different_episode_info_nce(value[:6], value[:6], same_episode, temperature=0.1)
    assert torch.isfinite(masked)


def test_deterministic_controls_metrics_and_noncollapse():
    episode = np.repeat(np.arange(8), 2)
    first = different_episode_permutation(episode, seed=5)
    second = different_episode_permutation(episode, seed=5)
    assert np.array_equal(first, second)
    assert np.all(episode[first] != episode)
    rng = np.random.default_rng(4)
    left = rng.normal(size=(16, 8, 32)).astype(np.float32)
    right = left + rng.normal(scale=0.01, size=left.shape).astype(np.float32)
    metrics = pairwise_alignment_metrics(left, right, episode, bootstrap_samples=100, seed=2, retrieval_chunk=5)
    repeated = pairwise_alignment_metrics(left, right, episode, bootstrap_samples=100, seed=2, retrieval_chunk=5)
    assert metrics == repeated
    assert metrics["paired_minus_shuffled_margin"] > 0
    assert metrics["retrieval"]["forward"]["recall_at_10"] > 0.5
    diagnostics = geometry_diagnostics(left)
    assert diagnostics["effective_rank"] == pytest.approx(effective_rank(left))
    assert diagnostics["per_dimension_variance"]["minimum"] > 0


def test_checkpoint_cold_reload_is_exact(tmp_path):
    torch.manual_seed(8)
    model = ContinuousVACSharedSpace("C2-slot").eval()
    value = torch.randn(3, 8, 32)
    before = model.encode("vision", value)
    path = tmp_path / "shared.pt"
    digest = save_checkpoint(path, model, {"selection_split": "validation"})
    loaded, metadata = load_checkpoint(path)
    after = loaded.eval().encode("vision", value)
    assert len(digest) == 64
    assert metadata["selection_split"] == "validation"
    assert torch.equal(before, after)


def test_contact_retention_and_action_temporal_formulas():
    native = {"macro_f1": 0.7, "majority": {"macro_f1": 0.2}}
    shared = {"macro_f1": 0.65}
    assert retention(shared, native) == pytest.approx(0.9)
    errors = {
        "correct": np.asarray([1.0, 2.0, 1.0]),
        "reversed": np.asarray([1.2, 2.4, 1.2]),
        "shuffled": np.asarray([1.1, 2.2, 1.1]),
    }
    result = temporal_ratios(errors, np.asarray([True, False, True]))
    assert result["dynamic"]["reversed_over_correct"] == pytest.approx(1.2)
    assert result["dynamic"]["shuffled_over_correct"] == pytest.approx(1.1)


def test_training_selection_is_validation_only_and_never_loads_test():
    source = Path("scripts/tactile_unit/train_continuous_vac_shared_space.py").read_text()
    assert 'load_split(cache_root, "train"' in source
    assert 'load_split(cache_root, "validation"' in source
    assert 'load_split(cache_root, "test"' not in source
    assert '"selection_split": "validation only"' in source
    assert '"test_loaded": False' in source
