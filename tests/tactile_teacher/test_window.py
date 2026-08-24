import numpy as np
import pytest

from gr00t.tactile_teacher.window import TemporalWindow, resample_window, resample_windows


def test_resample_window_uses_physical_timestamps():
    timestamps = np.asarray([0.0, 0.1, 0.22, 0.31, 0.45, 0.6, 0.75])
    values = np.stack([timestamps, 2.0 * timestamps], axis=-1).astype(np.float32)
    window = TemporalWindow(history_sec=0.3, future_sec=0.2, history_steps=4, future_steps=2)

    history, future, history_query, future_query = resample_window(
        timestamps, values, anchor_time=0.4, window=window
    )

    np.testing.assert_allclose(history_query, [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(future_query, [0.5, 0.6])
    np.testing.assert_allclose(history[:, 0], history_query, atol=1e-6)
    np.testing.assert_allclose(future[:, 1], 2 * future_query, atol=1e-6)


def test_resample_window_rejects_extrapolation_and_bad_timestamps():
    timestamps = np.arange(5, dtype=np.float64) / 10
    values = np.zeros((5, 60), dtype=np.float32)
    window = TemporalWindow(history_sec=0.3, future_sec=0.2, history_steps=4, future_steps=2)

    with pytest.raises(ValueError, match="does not cover"):
        resample_window(timestamps, values, anchor_time=0.1, window=window)
    timestamps[2] = timestamps[1]
    with pytest.raises(ValueError, match="strictly increasing"):
        resample_window(timestamps, values, anchor_time=0.2, window=window)


def test_vectorized_resampling_matches_single_window():
    timestamps = np.linspace(0, 2, 61)
    values = np.stack([np.sin(timestamps), np.cos(timestamps)], axis=-1).astype(np.float32)
    window = TemporalWindow(history_sec=0.5, future_sec=0.25, history_steps=16, future_steps=8)
    anchors = np.asarray([0.75, 1.0, 1.5])
    histories, futures = resample_windows(timestamps, values, anchors, window)
    for index, anchor in enumerate(anchors):
        history, future, _, _ = resample_window(timestamps, values, anchor, window)
        np.testing.assert_allclose(histories[index], history, atol=1e-6)
        np.testing.assert_allclose(futures[index], future, atol=1e-6)
