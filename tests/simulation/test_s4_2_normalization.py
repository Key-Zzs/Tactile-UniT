from __future__ import annotations

import numpy as np
import pytest

from gr00t.simulation.s4_2_normalization import (
    TactileNormalization,
    fit_tactile_normalization,
)


def raw_rows() -> np.ndarray:
    values = np.zeros((4, 5, 6), dtype=np.float32)
    values[1:, 1, 0] = 1
    values[1:, 1, 1] = [1, 3, 8]
    values[1:, 1, 2] = [0.5, 1, 2]
    values[1:, 1, 3:6] = [[1, 2, 3], [2, 4, 6], [3, 6, 9]]
    return values.reshape(4, 30)


@pytest.mark.parametrize("candidate", ["N0", "N1"])
def test_train_normalization_preserves_binary_occupancy_and_empty_cop(candidate: str) -> None:
    raw = raw_rows()
    fitted = fit_tactile_normalization(raw, candidate)
    transformed = fitted.transform(raw).reshape(4, 5, 6)
    np.testing.assert_array_equal(transformed[..., 0], raw.reshape(4, 5, 6)[..., 0])
    assert np.count_nonzero(transformed[0, :, 3:6]) == 0
    assert np.count_nonzero(transformed[:, 0, 3:6]) == 0
    assert np.isfinite(transformed).all()


def test_n1_is_log_force_and_json_round_trip_is_exact() -> None:
    raw = raw_rows()
    n0 = fit_tactile_normalization(raw, "N0")
    n1 = fit_tactile_normalization(raw, "N1")
    assert not np.allclose(n0.force_mean, n1.force_mean)
    restored = TactileNormalization.from_json(n1.to_json())
    np.testing.assert_array_equal(restored.transform(raw), n1.transform(raw))


def test_fit_rejects_unknown_candidate_and_nonfinite_data() -> None:
    with pytest.raises(ValueError, match="unknown"):
        fit_tactile_normalization(raw_rows(), "N2")
    damaged = raw_rows()
    damaged[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_tactile_normalization(damaged, "N0")
