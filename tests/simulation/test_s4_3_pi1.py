from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gr00t.simulation.s4_3_pi1 import HISTORY_STEPS, OnlineTactileHistory, left_repeat_history
from gr00t.simulation.pi1d_runtime import CausalContactRuntime
from scripts.simulation.summarize_s4_3_pi1d import exact_mcnemar, wilson


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


def test_pi1c_lambda_is_frozen_from_the_preregistered_formula() -> None:
    frozen = json.loads((ROOT / "configs/simulation/s4_3_pi1c_frozen.json").read_text())
    calibration = frozen["calibration"]
    expected = 0.1 * calibration["mean_official_pi05_loss"] / calibration["mean_physical_loss"]
    expected = min(frozen["lambda_clamp"][1], max(frozen["lambda_clamp"][0], expected))
    assert frozen["status"] == "FROZEN_BEFORE_TRAINING"
    assert frozen["mode"] == "CONTACT_STATE_TOKENS_PHYSICAL_AUX"
    assert frozen["seed"] == 42
    assert frozen["steps"] == 30_000
    assert frozen["global_batch_size"] == 32
    assert frozen["lambda_phys"] == pytest.approx(expected, abs=0.0, rel=1e-15)
    assert calibration["batches"] == 4
    assert calibration["split"] == "TRAIN only"
    assert calibration["rollout_performance_used"] is False
    assert calibration["dev_or_pi1d_data_used"] is False
    assert calibration["recalculate_during_training"] is False


def test_pi1d_causal_runtime_is_cached_and_never_reads_future() -> None:
    calls: list[np.ndarray] = []

    def encoder(history: np.ndarray) -> np.ndarray:
        calls.append(history.copy())
        return np.full(256, history[-1, 0], dtype=np.float32)

    runtime = CausalContactRuntime(encoder)
    first = np.arange(30, dtype=np.float32)
    runtime.reset(first)
    np.testing.assert_array_equal(runtime.current_history(), np.repeat(first[None], 26, axis=0))
    np.testing.assert_array_equal(runtime.contact_state(), np.zeros(256, dtype=np.float32))
    runtime.contact_state()
    assert len(calls) == 1

    second = first + 100
    runtime.append(second)
    np.testing.assert_array_equal(runtime.current_history()[-2:], np.stack((first, second)))
    np.testing.assert_array_equal(runtime.contact_state(), np.full(256, 100, dtype=np.float32))
    assert len(calls) == 2
    assert not np.any(calls[-1] == first + 200)


def test_pi1d_statistics_primitives_are_exact() -> None:
    assert wilson(0, 50)[0] == pytest.approx(0.0, abs=1e-16)
    assert wilson(50, 50)[1] == pytest.approx(1.0, abs=1e-16)
    assert exact_mcnemar(0, 0) == 1.0
    assert exact_mcnemar(0, 5) == pytest.approx(0.0625)


def test_pi1d_runtime_sources_freeze_exact_evaluation_contract() -> None:
    evaluation = (ROOT / "scripts/simulation/evaluate_s4_3_pi1d_augmented.py").read_text()
    serving = (ROOT / "scripts/simulation/serve_s4_3_pi1_policy.py").read_text()
    runner = (ROOT / "scripts/simulation/run_s4_3_pi1d_evaluation.sh").read_text()
    assert "args.seed != 1 or args.episodes != 50" in evaluation
    assert "replan_ratio=0.8" in evaluation
    assert "rand_full=False" in evaluation
    assert "randomize_dynamics=False" in evaluation
    assert 'for model in R0 B0 B1 B2' in runner
    assert "mode=TactileUnitMode.CONTACT_STATE_TOKENS" in serving
    assert "The auxiliary target exists only in training" in serving
