import numpy as np
import pytest

from gr00t.contact_dynamics.evaluation import (
    different_episode_permutation,
    query_diversity,
    transition_metrics,
)


def test_transition_metric_correctness():
    current = np.zeros((2, 2), dtype=np.float32)
    target = np.array([[1, 0], [0, 1]], dtype=np.float32)
    prediction = target.copy()
    metrics = transition_metrics(current, target, prediction)
    assert metrics["future_mse"] == 0
    assert metrics["delta_mse"] == 0
    assert metrics["future_cosine"] == pytest.approx(1)


def test_negative_control_never_pairs_same_episode():
    episode = np.repeat(np.arange(8), 4)
    permutation = different_episode_permutation(episode, seed=42)
    assert sorted(permutation.tolist()) == list(range(len(episode)))
    assert np.all(episode[permutation] != episode)


def test_query_diversity_detects_identical_queries():
    code = np.ones((4, 8, 32), dtype=np.float32)
    result = query_diversity(code)
    assert result["collapsed_sample_fraction"] == 1.0
    assert result["mean_off_diagonal_cosine"] == pytest.approx(1.0)
