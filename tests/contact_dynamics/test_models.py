import torch

from gr00t.contact_dynamics.models import (
    ContactDynamicsEncoder,
    ContactDynamicsModel,
    DeltaMLPEncoder,
    LatentTransitionDecoder,
)


def make_pair(batch=5):
    generator = torch.Generator().manual_seed(42)
    current = torch.randn(batch, 256, generator=generator)
    future = current + 0.1 * torch.randn(batch, 256, generator=generator)
    return current, future


def test_contact_encoder_and_decoder_shapes():
    current, future = make_pair()
    encoder = ContactDynamicsEncoder()
    decoder = LatentTransitionDecoder()
    code = encoder(current, future)
    prediction = decoder(code, current)
    assert code.shape == (5, 8, 32)
    assert prediction.shape == (5, 256)
    assert torch.isfinite(code).all()
    assert torch.isfinite(prediction).all()


def test_delta_baseline_uses_same_bottleneck_geometry():
    current, future = make_pair()
    assert DeltaMLPEncoder()(current, future).shape == (5, 8, 32)


def test_zero_shuffled_and_reversed_controls_are_executable():
    current, future = make_pair(6)
    model = ContactDynamicsModel(ContactDynamicsEncoder(), LatentTransitionDecoder()).eval()
    with torch.inference_mode():
        code = model.encoder(current, future)
        full = model.decoder(code, current)
        zero = model.decoder(torch.zeros_like(code), current)
        shuffled = model.decoder(code.roll(1, dims=0), current)
        reversed_code = model.encoder(future, current)
        reversed_prediction = model.decoder(reversed_code, current)
    for value in (full, zero, shuffled, reversed_prediction):
        assert value.shape == (6, 256)
        assert torch.isfinite(value).all()


def test_eval_is_deterministic():
    current, future = make_pair()
    model = ContactDynamicsModel(ContactDynamicsEncoder(), LatentTransitionDecoder()).eval()
    with torch.inference_mode():
        first = model(current, future)
        second = model(current, future)
    assert torch.equal(first["code"], second["code"])
    assert torch.equal(first["future"], second["future"])
