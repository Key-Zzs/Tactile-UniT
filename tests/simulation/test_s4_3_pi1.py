from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gr00t.simulation.s4_3_pi1 import HISTORY_STEPS, OnlineTactileHistory, left_repeat_history


ROOT = Path(__file__).resolve().parents[2]


def test_left_repeat_history_never_reads_future() -> None:
    tactile = np.arange(40 * 30, dtype=np.float32).reshape(40, 30)
    first, repeated = left_repeat_history(tactile, 0)
    assert repeated == 25
    np.testing.assert_array_equal(first, np.repeat(tactile[:1], HISTORY_STEPS, axis=0))

    middle, repeated = left_repeat_history(tactile, 12)
    assert repeated == 13
    np.testing.assert_array_equal(middle[-1], tactile[12])
    assert not np.isin(tactile[13:, 0], middle[:, 0]).any()

    full, repeated = left_repeat_history(tactile, 39)
    assert repeated == 0
    np.testing.assert_array_equal(full, tactile[14:40])


def test_online_history_matches_offline_contract() -> None:
    tactile = np.arange(33 * 30, dtype=np.float32).reshape(33, 30)
    online = OnlineTactileHistory()
    np.testing.assert_array_equal(online.reset(tactile[0]), left_repeat_history(tactile, 0)[0])
    for index in range(1, len(tactile)):
        np.testing.assert_array_equal(online.append(tactile[index]), left_repeat_history(tactile, index)[0])


def test_tactile_shape_and_finiteness_guards() -> None:
    with pytest.raises(ValueError):
        left_repeat_history(np.zeros((2, 29), dtype=np.float32), 0)
    with pytest.raises(IndexError):
        left_repeat_history(np.zeros((2, 30), dtype=np.float32), 2)
    online = OnlineTactileHistory()
    invalid = np.zeros(30, dtype=np.float32)
    invalid[0] = np.nan
    with pytest.raises(ValueError):
        online.reset(invalid)


def test_frozen_pi1_protocols_are_consistent() -> None:
    modes = json.loads((ROOT / "configs/simulation/s4_3_pi1_modes.json").read_text())
    training = json.loads((ROOT / "configs/simulation/s4_3_pi1_training_protocol.json").read_text())
    evaluation = json.loads((ROOT / "configs/simulation/s4_3_pi1d_eval_protocol.json").read_text())
    assert modes["contact_token_adapter"]["parameter_count"] == 199_680
    assert modes["physical_auxiliary"]["parameter_count"] == 658_176 < 1_000_000
    assert training["common"]["seed"] == 42
    assert training["common"]["steps"] == 30_000
    assert training["common"]["global_batch_size"] == 32
    assert training["B1"]["lambda_phys"] == 0.0
    assert evaluation["episodes"] == 50
    assert evaluation["evaluator_seed"] == 1
    assert evaluation["evaluation_performance_seen"] is False
