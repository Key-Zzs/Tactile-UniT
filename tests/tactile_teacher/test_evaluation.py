import numpy as np
import torch

from gr00t.tactile_teacher.evaluation import (
    classification_metrics,
    collapse_diagnostics,
    corrupt_history,
    regression_metrics,
    temporal_variant,
)


def test_temporal_variants_preserve_or_replace_expected_frames():
    value = torch.arange(2 * 4 * 3).reshape(2, 4, 3).float()
    assert torch.equal(temporal_variant(value, "full_history"), value)
    assert torch.equal(temporal_variant(value, "reversed_history"), value.flip(1))
    assert torch.equal(temporal_variant(value, "last_frame")[:, 0], value[:, -1])
    shuffled = temporal_variant(value, "shuffled_history", seed=7)
    assert torch.equal(shuffled.sort(dim=1).values, value.sort(dim=1).values)


def test_corruptions_are_deterministic_and_shape_safe():
    value = torch.randn(3, 16, 60)
    for name, severity in (
        ("gaussian_noise", 0.1),
        ("bias", 0.1),
        ("frame_dropout", 0.3),
        ("timestamp_jitter", 0.2),
    ):
        first = corrupt_history(value, name, severity, seed=9)
        second = corrupt_history(value, name, severity, seed=9)
        assert first.shape == value.shape
        assert torch.equal(first, second)


def test_metrics_and_collapse_diagnostics():
    target = np.array([0.0, 1.0, 2.0])
    assert regression_metrics(target, target)["r2"] == 1.0
    assert classification_metrics(target.astype(int), target.astype(int))["macro_f1"] == 1.0
    diagnostics = collapse_diagnostics(np.random.default_rng(0).normal(size=(100, 8)))
    assert diagnostics["effective_rank"] > 1
    assert diagnostics["per_dimension_variance"]["near_zero_fraction"] == 0
