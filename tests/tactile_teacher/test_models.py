import pytest
import torch

from gr00t.tactile_teacher.models import (
    PredictiveContactTeacher,
    TactileVQBaseline,
    build_baseline,
)


@pytest.mark.parametrize("name", ["B0", "B1", "B2", "B3"])
def test_baseline_shapes(name):
    model = build_baseline(name, latent_dim=32)
    result = model(torch.randn(3, 16, 60))
    assert result["latent"].shape == (3, 32)
    assert result["future"].shape == (3, 8, 60)
    assert torch.isfinite(result["latent"]).all()


def test_vq_has_one_code_per_finger():
    model = TactileVQBaseline()
    result = model(torch.randn(2, 16, 60))
    assert result["indices"].shape == (2, 10)
    assert result["reconstruction"].shape == (2, 16, 60)
    assert result["indices"].max() < 64


def test_predictive_teacher_is_continuous_and_has_both_objectives():
    model = PredictiveContactTeacher(latent_dim=32, channels=32)
    result = model(torch.randn(2, 16, 60))
    assert result["latent"].shape == (2, 32)
    assert result["reconstruction"].shape == (2, 16, 60)
    assert result["future"].shape == (2, 8, 60)
    assert result["latent"].dtype.is_floating_point
