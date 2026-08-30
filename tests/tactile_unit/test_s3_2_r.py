"""Focused structural and metric tests for the S3.2-R decision tree."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gr00t.contact_dynamics.models import ContactDynamicsEncoder, LatentTransitionDecoder
from gr00t.tactile_unit.contact_adapter import ContactCodebookAdaptor
from gr00t.tactile_unit.s3_2_r import (
    ContactDecoderBridge,
    FrozenDigestGuard,
    assert_disjoint_splits,
    build_contact_rq,
    classify_sufficiency,
    collapse_diagnostics,
    deterministic_different_episode_shuffle,
    linear_cka,
    reconstruction_retention,
    select_gr1_rehearsal_episodes,
    semantic_retention,
)


def test_r0_private_rq_nominal_shape_capacity_and_indices():
    rq = build_contact_rq(stages=2, codes=128).eval()
    value = torch.randn(5, 8, 32)
    quantized, indices, _ = rq(value)
    assert quantized.shape == value.shape
    assert indices.shape == (5, 8, 2)
    assert len(rq.layers) == 2
    assert all(layer.n_e == 128 and layer.e_dim == 32 for layer in rq.layers)
    assert indices.min() >= 0 and indices.max() < 128


def test_public_spec_preregisters_nominal_and_capacity_sensitivity_geometry():
    spec = json.loads(Path("configs/tactile_unit/s3_2_r_diagnostics.json").read_text())
    assert spec["r0"]["architecture"] == {
        "queries": 8,
        "embedding_dim": 32,
        "stages": 2,
        "codes_per_stage": 128,
    }
    assert spec["r0"]["capacity_sensitivity_if_fail"] == {
        "stages": 3,
        "codes_per_stage": 128,
    }


def test_frozen_identity_guard_detects_mutation():
    module = torch.nn.Linear(3, 2)
    guard = FrozenDigestGuard.capture(component=module)
    assert guard.verify(component=module)["component"]["unchanged"]
    with torch.no_grad():
        module.weight.add_(1)
    with pytest.raises(RuntimeError, match="identity changed"):
        guard.verify(component=module)


def test_decoder_bridge_accepts_only_q_c_and_shared_per_query_weights():
    bridge = ContactDecoderBridge("residual_mlp")
    value = torch.randn(4, 8, 32)
    output = bridge(value)
    assert output.shape == value.shape
    assert torch.equal(output[:, 0] - value[:, 0], bridge.net(value[:, 0]))
    with pytest.raises(TypeError):
        bridge(value, torch.randn(4, 256))
    with pytest.raises(ValueError):
        bridge(torch.randn(4, 256))


def test_pc_rc_gradient_routing_and_frozen_shared_rq():
    pc = ContactCodebookAdaptor("affine")
    rc = ContactDecoderBridge("affine")
    rq = build_contact_rq(stages=2, codes=8).eval().requires_grad_(False)
    decoder = LatentTransitionDecoder().eval().requires_grad_(False)
    value = torch.randn(3, 8, 32)
    current = torch.randn(3, 256)
    target = torch.randn(3, 256)
    q, _, _ = rq(pc(value))
    loss = (decoder(rc(q), current) - target).square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in pc.parameters())
    assert all(parameter.grad is not None for parameter in rc.parameters())
    assert all(parameter.grad is None for parameter in rq.parameters())
    assert all(parameter.grad is None for parameter in decoder.parameters())


def test_r3_q_new_gets_gradients_while_q_old_never_changes():
    q_old = build_contact_rq(stages=2, codes=8).eval().requires_grad_(False)
    q_new = build_contact_rq(stages=2, codes=8)
    q_new.load_state_dict(q_old.state_dict())
    guard = FrozenDigestGuard.capture(q_old=q_old)
    value = torch.randn(4, 8, 32)
    old_value, _, _ = q_old(value)
    new_value, _, new_vq_loss = q_new(value)
    preserve = (new_value - old_value.detach()).square().mean()
    (new_vq_loss + preserve).backward()
    assert any(parameter.grad is not None for parameter in q_new.parameters())
    assert all(parameter.grad is None for parameter in q_old.parameters())
    assert guard.verify(q_old=q_old)["q_old"]["unchanged"]


def test_contact_split_leakage_rejection():
    assert_disjoint_splits([1, 2], [3], [4, 5])
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint_splits([1, 2], [2, 3], [4])


def test_gr1_rehearsal_excludes_last_ten_t4_episodes():
    train, held_out = select_gr1_rehearsal_episodes(range(30), held_out_count=10)
    assert train == list(range(20))
    assert held_out == list(range(20, 30))
    assert set(train).isdisjoint(held_out)


def test_retention_formulas_are_raw_and_unclipped():
    assert reconstruction_retention(0.046112, 0.073742, 0.070683, 0.000958) == pytest.approx(
        (0.070683 - 0.046112) / (0.070683 - 0.000958)
    )
    assert reconstruction_retention(-1.0, 2.0, 3.0, 0.0) == pytest.approx(1.5)
    assert semantic_retention(0.443, 0.6761, 0.17954) == pytest.approx(
        (0.443 - 0.17954) / (0.6761 - 0.17954)
    )
    assert semantic_retention(1.2, 1.0, 0.0) == pytest.approx(1.2)


def test_preregistered_gate_boundaries():
    assert classify_sufficiency(0.8, 0.9, 0.9, hard_code_collapse=False, query_collapse=False) == "STRONG_PASS"
    assert classify_sufficiency(0.6, 0.75, 0.75, hard_code_collapse=False, query_collapse=False) == "PARTIAL"
    assert classify_sufficiency(0.99, 0.99, 0.99, hard_code_collapse=True, query_collapse=False) == "FAIL"


def test_deterministic_different_episode_shuffle():
    episodes = np.repeat(np.arange(8), 5)
    first = deterministic_different_episode_shuffle(episodes, seed=42)
    second = deterministic_different_episode_shuffle(episodes, seed=42)
    assert np.array_equal(first, second)
    assert np.all(episodes[first] != episodes)


def test_synthetic_query_collapse_is_detected():
    values = np.ones((12, 8, 32), dtype=np.float32)
    indices = np.zeros((12, 8, 2), dtype=np.int64)
    result = collapse_diagnostics(indices, values, codebook_size=128)
    assert result["hard_code_collapse"]
    assert result["query_collapse"]


def test_linear_cka_identity_and_orthogonal_invariance():
    rng = np.random.default_rng(42)
    value = rng.normal(size=(50, 12))
    q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    assert linear_cka(value, value) == pytest.approx(1.0)
    assert linear_cka(value, value @ q) == pytest.approx(1.0)


def test_s1_s2_modules_can_be_guarded_together():
    encoder = ContactDynamicsEncoder()
    decoder = LatentTransitionDecoder()
    guard = FrozenDigestGuard.capture(s2_encoder=encoder, s2_decoder=decoder)
    result = guard.verify(s2_encoder=encoder, s2_decoder=decoder)
    assert all(row["unchanged"] for row in result.values())
