import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gr00t.contact_dynamics.evaluation import query_diversity
from gr00t.contact_dynamics.models import ContactDynamicsEncoder, LatentTransitionDecoder
from gr00t.model.tokenizer.vector_quantizer import (
    ResidualVectorQuantizer,
    ResidualVectorQuantizerConfig,
)
from gr00t.tactile_unit.compatibility import parameter_digest
from gr00t.tactile_unit.contact_adapter import ContactCodebookAdaptor


def tiny_rq() -> ResidualVectorQuantizer:
    rq = ResidualVectorQuantizer(
        ResidualVectorQuantizerConfig(
            stages=[
                {"n_e": 16, "e_dim": 32, "beta": 0.25, "code_restart": True},
                {"n_e": 16, "e_dim": 32, "beta": 0.25, "code_restart": True},
            ]
        )
    )
    return rq.eval().requires_grad_(False)


def test_candidate_shapes_parameter_counts_and_identity():
    value = torch.randn(5, 8, 32)
    identity = ContactCodebookAdaptor("identity")
    affine = ContactCodebookAdaptor("affine")
    mlp = ContactCodebookAdaptor("mlp")
    assert torch.equal(identity(value), value)
    assert affine(value).shape == mlp(value).shape == value.shape
    assert identity.parameter_count == 0
    assert affine.parameter_count == 1120
    assert mlp.parameter_count == 8416


def test_adaptor_shares_exact_weights_across_query_positions():
    adaptor = ContactCodebookAdaptor("mlp").eval()
    token = torch.randn(4, 1, 32)
    repeated = token.expand(-1, 8, -1)
    output = adaptor(repeated)
    assert all(torch.equal(output[:, 0], output[:, query]) for query in range(1, 8))
    assert not any("query" in name for name, _ in adaptor.named_parameters())


def test_frozen_rq_ste_gradient_integrity_and_no_mutation():
    torch.manual_seed(42)
    adaptor = ContactCodebookAdaptor("affine")
    rq = tiny_rq()
    encoder = ContactDynamicsEncoder().eval().requires_grad_(False)
    decoder = LatentTransitionDecoder().eval().requires_grad_(False)
    rq_before = parameter_digest(rq)
    current = torch.randn(6, 256)
    future = torch.randn(6, 256)
    with torch.no_grad():
        contact = encoder(current, future)
    adapted = adaptor(contact)
    quantized, indices, _ = rq(adapted)
    loss = F.mse_loss(decoder(quantized, current), future)
    loss.backward()
    gradients = [parameter.grad for parameter in adaptor.parameters()]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0
    assert all(parameter.grad is None for parameter in rq.parameters())
    assert all(parameter.grad is None for parameter in decoder.parameters())
    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert not rq.training and all(not layer.training for layer in rq.layers)
    assert all(int(layer.internal_step) == 0 for layer in rq.layers)
    assert parameter_digest(rq) == rq_before
    assert indices.min() >= 0 and indices.max() < 16


def test_checkpoint_round_trip_and_deterministic_evaluation(tmp_path: Path):
    torch.manual_seed(7)
    adaptor = ContactCodebookAdaptor("mlp").eval()
    value = torch.randn(3, 8, 32)
    expected = adaptor(value)
    checkpoint = tmp_path / "adaptor.pt"
    torch.save(
        {
            "schema": "tactile3d-unit.s3-2-contact-adaptor.v1",
            "architecture": "mlp",
            "state_dict": adaptor.state_dict(),
        },
        checkpoint,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored = ContactCodebookAdaptor(payload["architecture"]).eval()
    restored.load_state_dict(payload["state_dict"], strict=True)
    assert torch.equal(expected, restored(value))
    assert torch.equal(restored(value), restored(value))


def test_synthetic_adapted_queries_do_not_collapse():
    generator = torch.Generator().manual_seed(11)
    value = torch.randn(64, 8, 32, generator=generator)
    adapted = ContactCodebookAdaptor("affine").eval()(value).detach().numpy()
    metrics = query_diversity(adapted)
    assert metrics["collapsed_sample_fraction"] == 0.0
    assert np.isfinite(list(metrics.values())).all()


def test_public_protocol_prevents_split_leakage_and_test_selection():
    spec = json.loads(Path("configs/tactile_unit/s3_2_contact_adapter.json").read_text())
    assert spec["data"]["split_source"] == "accepted S1/S2 episode-disjoint manifest"
    assert spec["data"]["test_used_for_selection"] is False
    assert spec["selection"]["partition"] == "validation"
    assert "train-derived" in spec["data"]["dynamic_rule"]
    assert spec["interface"]["shared_weights_across_queries"] is True
    assert spec["frozen_components"]["rq_mode"] == "eval"
