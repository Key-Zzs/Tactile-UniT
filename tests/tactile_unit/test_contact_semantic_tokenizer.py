"""Structural, leakage, accounting, and determinism tests for S3.2-Q."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gr00t.tactile_unit.contact_semantic_tokenizer import (
    ContactSemanticTokenizer,
    WhiteningStatistics,
    assert_episode_disjoint,
    classify_shared_private,
    classify_single_stream,
    deterministic_different_episode_permutation,
    nominal_index_bitrate,
    private_stream_bypass,
    reconstruction_retention,
    same_episode_horizon_links,
    semantic_retention,
)


def synthetic_contact(seed: int = 42, samples: int = 96) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scale = np.linspace(0.05, 2.0, 32, dtype=np.float32)
    return (rng.normal(size=(samples, 8, 32)).astype(np.float32) * scale).astype(np.float32)


def test_public_spec_preregisters_equal_112_bit_budget_and_gates():
    spec = json.loads(Path("configs/tactile_unit/s3_2_q_semantic_tokenizer.json").read_text())
    assert spec["bitrate"]["baseline_bits"] == 112
    assert spec["bitrate"]["q2_total_bits"] == 112
    assert spec["gates"]["single"] == {"r_recon": 0.8, "r_contact": 0.9, "r_force": 0.9}
    assert spec["gates"]["rare_boundary"]["minimum_recall_gain_over_ordinary"] == 0.02
    assert spec["q2"]["anti_bypass"] == {
        "private_near_full_ratio": 1.05,
        "minimum_semantic_zero_relative_impact": 0.05,
    }
    assert spec["q0"]["variance_bands"] == {
        "rule": "fixed flattened-PC rank thirds before test access",
        "high": [0, 85],
        "mid": [85, 170],
        "low": [170, 256],
    }


def test_same_bitrate_accounting():
    assert nominal_index_bitrate(queries=8, stages=2, codes=128) == pytest.approx(112.0)
    assert nominal_index_bitrate(queries=8, stages=1, codes=128) == pytest.approx(56.0)
    semantic = nominal_index_bitrate(queries=8, stages=1, codes=128)
    private = nominal_index_bitrate(queries=8, stages=1, codes=128)
    assert semantic + private == pytest.approx(112.0)
    with pytest.raises(ValueError):
        nominal_index_bitrate(queries=8, stages=0, codes=128)


@pytest.mark.parametrize("kind", ["pca", "zca"])
def test_train_only_whitening_inverse_and_numerical_stability(kind: str):
    train = synthetic_contact()
    statistics = WhiteningStatistics.fit(train, kind=kind, regularization=1e-3)
    whitened = statistics.whiten_numpy(train)
    assert np.isfinite(whitened).all()
    assert statistics.inverse_consistency_error(train[:16]) < 2e-5
    assert np.max(np.abs(whitened.mean(axis=(0, 1)))) < 2e-4


def test_whitening_fit_is_deterministic_and_does_not_consume_test():
    train = synthetic_contact()
    first = WhiteningStatistics.fit(train, kind="zca", regularization=1e-3)
    second = WhiteningStatistics.fit(train.copy(), kind="zca", regularization=1e-3)
    assert np.array_equal(first.mean, second.mean)
    assert np.array_equal(first.transform, second.transform)
    perturbed_test = synthetic_contact(seed=99) * 100
    _ = first.whiten_numpy(perturbed_test)
    assert np.array_equal(first.transform, second.transform)


def test_split_leakage_guard():
    assert_episode_disjoint([1, 2], [3], [4, 5])
    with pytest.raises(ValueError, match="leakage"):
        assert_episode_disjoint([1, 2], [2, 3], [4])


def test_multi_horizon_links_are_same_episode_complete_and_deterministic():
    episode = np.repeat([10, 20], 4)
    anchor = np.tile([0, 8, 16, 24], 2)
    first = same_episode_horizon_links(episode, anchor, 8)
    second = same_episode_horizon_links(episode, anchor, 8)
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    source, target = first
    assert len(source) == 6
    assert np.all(episode[source] == episode[target])
    assert np.all(anchor[target] - anchor[source] == 8)


def test_different_episode_shuffle_and_reversed_transition_controls():
    episode = np.repeat(np.arange(8), 5)
    first = deterministic_different_episode_permutation(episode, seed=42)
    second = deterministic_different_episode_permutation(episode, seed=42)
    assert np.array_equal(first, second)
    assert np.all(episode[first] != episode)
    current = torch.randn(6, 256)
    future = torch.randn(6, 256)
    reversed_current, reversed_future = future, current
    assert torch.equal(reversed_current, future)
    assert torch.equal(reversed_future, current)


def test_single_semantic_shapes_indices_and_checkpoint_reload(tmp_path: Path):
    train = synthetic_contact()
    whitening = WhiteningStatistics.fit(train, kind="zca", regularization=1e-3)
    model = ContactSemanticTokenizer(semantic_stages=2, whitening=whitening).eval()
    value = torch.from_numpy(train[:5])
    with torch.no_grad():
        output = model(value)
    assert output["semantic"].shape == (5, 8, 32)
    assert output["semantic_indices"].shape == (5, 8, 2)
    assert output["full_native"].shape == value.shape
    assert model.semantic_bits == pytest.approx(112.0)
    path = tmp_path / "tokenizer.pt"
    torch.save(model.state_dict(), path)
    reloaded = ContactSemanticTokenizer(semantic_stages=2, whitening=whitening).eval()
    reloaded.load_state_dict(torch.load(path, weights_only=True), strict=True)
    with torch.no_grad():
        second = reloaded(value)
    assert torch.equal(output["semantic_indices"], second["semantic_indices"])
    assert torch.equal(output["full_native"], second["full_native"])


def test_semantic_private_shapes_zero_controls_and_no_private_side_inputs():
    model = ContactSemanticTokenizer(semantic_stages=1, private_stages=1).eval()
    value = torch.randn(7, 8, 32)
    with torch.no_grad():
        output = model(value)
    assert output["semantic_indices"].shape == (7, 8, 1)
    assert output["private_indices"].shape == (7, 8, 1)
    assert output["semantic_native"].shape == value.shape
    assert output["private"].shape == value.shape
    assert torch.equal(output["full_native"], output["semantic_native"] + output["private"])
    assert model.semantic_bits == pytest.approx(56.0)
    assert model.private_bits == pytest.approx(56.0)
    semantic_zero = output["private"]
    private_zero = output["semantic_native"]
    assert semantic_zero.shape == private_zero.shape == value.shape
    with pytest.raises(TypeError):
        model(value, torch.randn(7, 256))


def test_retention_formulas_are_raw_and_gate_boundaries_are_fixed():
    assert reconstruction_retention(0.2, 1.0, 0.0) == pytest.approx(0.8)
    assert reconstruction_retention(-0.2, 1.0, 0.0) == pytest.approx(1.2)
    assert semantic_retention(0.9, 1.0, 0.0) == pytest.approx(0.9)
    assert classify_single_stream(
        r_recon=0.8,
        r_contact=0.9,
        r_force=0.9,
        rare_boundary_pass=True,
        temporal_controls_pass=True,
        collapse=False,
    )
    assert not classify_single_stream(
        r_recon=0.8,
        r_contact=0.89,
        r_force=0.9,
        rare_boundary_pass=True,
        temporal_controls_pass=True,
        collapse=False,
    )
    assert classify_shared_private(
        semantic_r_contact=0.9,
        semantic_r_force=0.9,
        full_r_recon=0.8,
        rare_boundary_pass=True,
        temporal_controls_pass=True,
        bypass=False,
        collapse=False,
    )


def test_anti_bypass_rule_uses_private_only_and_semantic_zero_controls():
    assert private_stream_bypass(
        private_only_error=1.04,
        full_error=1.0,
        semantic_zero_error=1.04,
    )
    assert not private_stream_bypass(
        private_only_error=1.04,
        full_error=1.0,
        semantic_zero_error=1.20,
    )
    assert not private_stream_bypass(
        private_only_error=1.20,
        full_error=1.0,
        semantic_zero_error=1.01,
    )


def test_synthetic_hard_collapse_and_mild_noise_stability_sanity():
    model = ContactSemanticTokenizer(semantic_stages=1).eval()
    value = torch.zeros(32, 8, 32)
    with torch.no_grad():
        clean = model(value)["semantic_indices"]
        noisy = model(value + 1e-7 * torch.randn_like(value))["semantic_indices"]
    stability = (clean == noisy).float().mean().item()
    assert 0.0 <= stability <= 1.0
    assert torch.unique(clean).numel() == 1  # deliberate synthetic collapse sanity
