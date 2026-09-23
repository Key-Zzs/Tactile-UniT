from __future__ import annotations

import numpy as np

from gr00t.simulation.pi2b_policy.statistics import (
    conditional_reset_bootstrap,
    discordant_table,
    exact_mcnemar,
    holm_adjust,
    two_way_bootstrap,
    wilson_interval,
)


def test_wilson_boundaries_are_finite_and_bounded():
    assert wilson_interval(0, 200)[0] == 0.0
    assert wilson_interval(200, 200)[1] == 1.0
    low, high = wilson_interval(100, 200)
    assert 0.0 < low < 0.5 < high < 1.0


def test_mcnemar_zero_discordance_and_known_table():
    assert exact_mcnemar([False, True], [False, True]) == 1.0
    table = discordant_table([True, True, False, False], [False, False, True, False])
    assert table == {"both_fail": 1, "first_only": 2, "second_only": 1, "both_success": 0}
    assert exact_mcnemar([True, True, False, False], [False, False, True, False]) == 1.0


def test_holm_preserves_original_order_and_monotonicity():
    adjusted = holm_adjust([0.01, 0.04, 0.02])
    assert adjusted == [0.03, 0.04, 0.04]


def test_bootstraps_cover_all_success_all_failure_and_identical():
    success = np.ones((3, 200), dtype=np.bool_)
    failure = np.zeros((3, 200), dtype=np.bool_)
    for function in (conditional_reset_bootstrap, two_way_bootstrap):
        positive = function(success, failure, repetitions=200, seed=1)
        identical = function(success, success, repetitions=200, seed=1)
        negative = function(failure, success, repetitions=200, seed=1)
        assert positive["estimate"] == 1.0 and positive["interval"] == (1.0, 1.0)
        assert identical["estimate"] == 0.0 and identical["interval"] == (0.0, 0.0)
        assert negative["estimate"] == -1.0 and negative["interval"] == (-1.0, -1.0)


def test_two_way_uses_crossed_shared_indices():
    first = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.bool_)
    second = np.zeros_like(first)
    result = two_way_bootstrap(first, second, repetitions=500, seed=7, chunk_size=100)
    assert result["estimate"] == 4 / 6
    assert result["unit"] == "shared training-seed index crossed with shared reset index"
