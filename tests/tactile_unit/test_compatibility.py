import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gr00t.model.tokenizer.vector_quantizer import (
    ResidualVectorQuantizer,
    ResidualVectorQuantizerConfig,
)
from gr00t.contact_dynamics.models import (
    ContactDynamicsEncoder,
    LatentTransitionDecoder,
)
from gr00t.tactile_unit.compatibility import (
    active_set_jaccard,
    codebook_usage,
    deterministic_contact_subset,
    jensen_shannon_divergence,
    parameter_digest,
    quantization_metrics,
    quantize_with_stage_diagnostics,
)


def tiny_rq() -> ResidualVectorQuantizer:
    config = ResidualVectorQuantizerConfig(
        stages=[
            {"n_e": 4, "e_dim": 2, "beta": 0.25, "code_restart": False},
            {"n_e": 4, "e_dim": 2, "beta": 0.25, "code_restart": False},
        ]
    )
    model = ResidualVectorQuantizer(config).eval().requires_grad_(False)
    with torch.no_grad():
        model.layers[0].embedding.weight.copy_(
            torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        )
        model.layers[1].embedding.weight.copy_(
            torch.tensor([[-0.25, -0.25], [-0.25, 0.25], [0.25, -0.25], [0.25, 0.25]])
        )
    return model


def test_contact_geometry_and_direct_deterministic_quantization():
    model = tiny_rq()
    value = torch.randn(5, 8, 2, generator=torch.Generator().manual_seed(42))
    first_q, first_i, rows = quantize_with_stage_diagnostics(model, value)
    second_q, second_i, _ = quantize_with_stage_diagnostics(model, value)
    reference_q, reference_i, _ = model(value)
    assert first_q.shape == value.shape
    assert first_i.shape == (5, 8, 2)
    assert torch.equal(first_q, second_q)
    assert torch.equal(first_i, second_i)
    assert torch.equal(first_q, reference_q)
    assert torch.equal(first_i, reference_i)
    assert first_i.min() >= 0 and first_i.max() < 4
    assert len(rows) == 2


def test_public_spec_freezes_canonical_original_unit_identity():
    spec_path = Path("configs/tactile_unit/s3_0_codebook_compatibility.json")
    spec = json.loads(spec_path.read_text())
    assert spec["original_unit"]["variant"] == "VLA-UniT-3B-fulldata"
    assert spec["original_unit"]["rq"] == {
        "frozen": True,
        "stages": 2,
        "codes_per_stage": 128,
        "embedding_dim": 32,
        "input": "Original UniT L2 or contact z_c directly",
        "forbidden_contact_path": "vq_down_resampler(z_c)",
    }
    t4_config = Path(spec["t4_reference"]["benchmark_config"])
    assert hashlib.sha256(t4_config.read_bytes()).hexdigest() == spec["t4_reference"][
        "benchmark_config_sha256"
    ]


def test_real_contact_geometry_and_quantized_decoder_path():
    encoder = ContactDynamicsEncoder().eval().requires_grad_(False)
    decoder = LatentTransitionDecoder().eval().requires_grad_(False)
    config = ResidualVectorQuantizerConfig(
        stages=[
            {"n_e": 8, "e_dim": 32, "beta": 0.25, "code_restart": False},
            {"n_e": 8, "e_dim": 32, "beta": 0.25, "code_restart": False},
        ]
    )
    rq = ResidualVectorQuantizer(config).eval().requires_grad_(False)
    current = torch.randn(3, 256)
    future = torch.randn(3, 256)
    continuous = encoder(current, future)
    quantized, indices, _ = quantize_with_stage_diagnostics(rq, continuous)
    prediction = decoder(quantized, current)
    assert continuous.shape == quantized.shape == (3, 8, 32)
    assert indices.shape == (3, 8, 2)
    assert prediction.shape == (3, 256)
    assert torch.isfinite(prediction).all()


def test_frozen_rq_parameter_digest_is_unchanged():
    model = tiny_rq()
    before = parameter_digest(model)
    quantize_with_stage_diagnostics(model, torch.randn(4, 8, 2))
    assert parameter_digest(model) == before


def test_quantization_metrics_are_energy_normalized():
    value = np.ones((2, 8, 2), dtype=np.float32) * 2
    quantized = value + 1
    metrics = quantization_metrics(value, quantized)
    assert metrics["absolute_mse"] == pytest.approx(1.0)
    assert metrics["input_energy"] == pytest.approx(4.0)
    assert metrics["relative_distortion"] == pytest.approx(0.25)


def test_codebook_usage_and_bounded_symmetric_js():
    codes = np.asarray([0, 0, 1, 2])
    usage = codebook_usage(codes, 4)
    assert usage["active_codes"] == 3
    assert usage["active_ratio"] == pytest.approx(0.75)
    left = np.asarray([0.5, 0.5, 0.0, 0.0])
    right = np.asarray([0.0, 0.5, 0.5, 0.0])
    assert active_set_jaccard(left, right) == pytest.approx(1 / 3)
    assert jensen_shannon_divergence(left, right) == pytest.approx(
        jensen_shannon_divergence(right, left)
    )
    assert 0 <= jensen_shannon_divergence(left, right) <= np.log(2)


def test_contact_subset_is_deterministic_unique_and_stratified():
    episode = np.repeat(np.arange(10), 10)
    anchor = np.tile(np.arange(10), 10)
    dynamic = np.asarray([False] * 70 + [True] * 30)
    transition = np.tile(np.arange(4), 25)
    first = deterministic_contact_subset(
        episode, anchor, dynamic, transition, count=40, seed=42
    )
    second = deterministic_contact_subset(
        episode, anchor, dynamic, transition, count=40, seed=42
    )
    assert np.array_equal(first, second)
    assert len(first) == len(np.unique(first)) == 40
    assert dynamic[first].mean() == pytest.approx(dynamic.mean(), abs=0.03)


def test_audit_source_has_no_double_down_resampler():
    source = Path("scripts/tactile_unit/audit_shared_rq_compatibility.py").read_text()
    forbidden = "vq" + "_down_resampler"
    assert forbidden not in source
