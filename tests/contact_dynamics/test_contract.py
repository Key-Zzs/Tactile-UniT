import numpy as np
import pytest

from gr00t.contact_dynamics.contract import (
    TransitionPairContract,
    evenly_spaced_anchors,
    validate_episode_splits,
)


def test_canonical_k16_windows_do_not_overlap():
    contract = TransitionPairContract(horizon_frames=16)
    current = contract.current_indices(100)
    future = contract.future_indices(100)
    assert current.tolist() == list(range(85, 101))
    assert future.tolist() == list(range(101, 117))
    assert np.intersect1d(current, future).size == 0
    assert contract.overlap_samples == 0
    assert contract.history_physical_span_sec == pytest.approx(0.5)
    assert contract.anchor_delta_sec == pytest.approx(16 / 30)


def test_k8_overlap_is_detected():
    contract = TransitionPairContract(horizon_frames=8)
    assert contract.future_indices(100).tolist() == list(range(93, 109))
    assert contract.overlap_samples == 8
    assert contract.overlap_fraction == pytest.approx(0.5)


def test_episode_boundary_rejection():
    contract = TransitionPairContract(horizon_frames=16)
    with pytest.raises(ValueError):
        contract.validate_anchor(14, 100)
    with pytest.raises(ValueError):
        contract.validate_anchor(90, 100)
    contract.validate_anchor(15, 32)


def test_anchor_selection_is_unique_and_valid_for_longest_horizon():
    anchors = evenly_spaced_anchors(1000, 64, maximum_horizon_frames=24)
    assert len(anchors) == len(np.unique(anchors)) == 64
    assert anchors[0] == 15
    assert anchors[-1] == 975
    for anchor in anchors:
        TransitionPairContract(horizon_frames=24).validate_anchor(int(anchor), 1000)


def test_split_episode_sets_cannot_leak():
    splits = {"train": [1, 2], "val": [3], "test": [4, 5]}
    assert validate_episode_splits(splits) == {
        "train_val": 0,
        "train_test": 0,
        "val_test": 0,
    }
    with pytest.raises(ValueError, match="episode leakage"):
        validate_episode_splits({"train": [1, 2], "val": [2, 3], "test": [4]})
