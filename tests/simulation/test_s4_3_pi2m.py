from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/simulation/s4_3_pi2m_bhva_target_protocol.json"
BUILDER = ROOT / "scripts/simulation/build_s4_3_pi2m_bhva_targets.py"
BHVA_PROTOCOL = ROOT / "configs/simulation/s4_3_pi2m_bhva_protocol.json"
MODE_SOURCE = ROOT / "gr00t/simulation/pi05_tactile_unit.py"
MODE_ENUM = ROOT / "gr00t/simulation/s4_3_pi1.py"
TRAINER = ROOT / "scripts/simulation/train_s4_3_pi2m_bhva.py"
ANALYZER = ROOT / "scripts/simulation/analyze_s4_3_pi2m.py"
EVAL_PROTOCOL = ROOT / "configs/simulation/s4_3_pi2m_evaluation_protocol.json"
STAT_PROTOCOL = ROOT / "configs/simulation/s4_3_pi2m_statistical_protocol.json"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_s4_3_pi2m_bhva_targets", BUILDER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_analyzer():
    spec = importlib.util.spec_from_file_location("analyze_s4_3_pi2m", ANALYZER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_corrected_target_protocol_is_distinct_and_frozen() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    target = protocol["target"]
    historical = protocol["historical_BVA"]
    assert protocol["status"] == "FROZEN_BEFORE_TARGET_BUILD"
    assert protocol["model_id"] == "B_HVA"
    assert target["future_row_offset"] == 27
    assert target["control_dt_seconds"] == 0.02
    assert target["physical_horizon_seconds"] == 0.54
    assert target["future_selection"] == "same episode exact row i+27; no interpolation"
    assert target["output"].startswith(".local/datasets/simulation/s4_3_pi2m/")
    assert historical["target_row_offset"] == 16
    assert historical["actual_physical_horizon_seconds"] == 0.32
    assert historical["modified_by_PI2M"] is False
    assert target["output"] != historical["target_sidecar"]


def test_expected_mask_is_episode_local_plus_27() -> None:
    builder = load_builder()
    lengths = [31, 35, 40]
    mask = builder.expected_valid_mask(lengths)
    cursor = 0
    for length in lengths:
        local = mask[cursor : cursor + length]
        assert np.all(local[: length - 27])
        assert not np.any(local[length - 27 :])
        cursor += length
    assert cursor == len(mask)


def test_builder_uses_only_clean_vision_teacher_path() -> None:
    source = BUILDER.read_text()
    assert "bridge.encode(\"vision\"" in source
    assert "future_row_i_plus_27" in source
    assert "B3_VAC_projector_used\": False" in source
    assert "contact_shared_target" not in source
    assert "FrozenS42PolicyStack" not in source
    assert "refusing to overwrite frozen PI2M target output" in source


def test_bhva_mode_is_contact_conditioned_but_uses_only_va_target() -> None:
    protocol = json.loads(BHVA_PROTOCOL.read_text())
    assert protocol["model_id"] == "B_HVA"
    assert protocol["mode"] == "CONTACT_STATE_TOKENS_VA_PHYSICAL_AUX"
    assert protocol["online_inputs"]["identical_to_B2"] is True
    assert protocol["training_target"]["field"] == "va_shared_target"
    assert protocol["training_target"]["valid_field"] == "va_aux_valid"
    assert protocol["training_target"]["mask_equal_B2"] is True
    assert protocol["contact_input_sidecar"]["field_read"] == "contact_state"
    assert protocol["contact_input_sidecar"]["contact_target_fields_read"] is False
    assert protocol["historical_BVA"]["actual_physical_horizon_seconds"] == 0.32
    assert protocol["historical_BVA"]["matched_horizon_control_for_B2"] is False


def test_bhva_training_wiring_preserves_b2_input_and_separate_target_cache() -> None:
    enum_source = MODE_ENUM.read_text()
    mode_source = MODE_SOURCE.read_text()
    trainer = TRAINER.read_text()
    assert 'CONTACT_STATE_TOKENS_VA_PHYSICAL_AUX = "CONTACT_STATE_TOKENS_VA_PHYSICAL_AUX"' in enum_source
    assert "target_sidecar_path=VA_TARGET_SIDECAR" in trainer
    assert "sidecar_path=CONTACT_SIDECAR" in trainer
    assert "TactileUnitMode.CONTACT_STATE_TOKENS_VA_PHYSICAL_AUX" in trainer
    assert '"contact_state": self._contact_state[position]' in mode_source
    assert '"va_shared_target": self._va_shared_target[position]' in mode_source
    assert '"va_aux_valid": self._va_aux_valid[position]' in mode_source
    combined_branch = mode_source.split(
        "elif mode is TactileUnitMode.CONTACT_STATE_TOKENS_VA_PHYSICAL_AUX:", 1
    )[1].split("else:", 1)[0]
    assert 'source["contact_shared_target"]' not in combined_branch
    assert 'source["physical_aux_valid"]' not in combined_branch


def test_bhva_training_recipe_is_unique_seed42_30k_global_batch32() -> None:
    protocol = json.loads(BHVA_PROTOCOL.read_text())
    trainer = TRAINER.read_text()
    assert protocol["seed"] == 42
    assert protocol["steps"] == 30_000
    assert protocol["global_batch_size"] == 32
    assert protocol["allowed_device_counts"] == [1, 2, 4]
    assert "seed=42" in trainer
    assert "batch_size=32" in trainer
    assert "num_train_steps=30_000" in trainer
    assert "resume=False" in trainer
    assert "overwrite=False" in trainer


def test_evaluation_and_statistics_are_frozen_for_fresh_seed7() -> None:
    evaluation = json.loads(EVAL_PROTOCOL.read_text())
    statistics = json.loads(STAT_PROTOCOL.read_text())
    assert evaluation["models"] == ["B1", "B_HVA", "B2"]
    assert evaluation["episodes_per_model"] == 200
    assert evaluation["total_rollouts"] == 600
    assert evaluation["evaluator_seed"] == 7
    assert evaluation["runtime_mode_all_models"] == "CONTACT_STATE_TOKENS"
    assert statistics["bootstrap_resamples"] == 100_000
    assert statistics["bootstrap_seed"] == 4317
    assert [row["name"] for row in statistics["comparisons"]] == [
        "B2-B_HVA",
        "B_HVA-B1",
        "B2-B1",
    ]
    assert statistics["comparisons"][0]["role"] == "PRIMARY_TARGET_EFFECT"
    assert statistics["historical_BVA"].startswith("context only")


def test_exact_paired_statistics_helpers() -> None:
    analyzer = load_analyzer()
    assert analyzer.exact_mcnemar(0, 0) == 1.0
    assert analyzer.exact_mcnemar(10, 0) == pytest.approx(2 / 2**10)
    assert analyzer.holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2}) == {
        "a": pytest.approx(0.03),
        "b": pytest.approx(0.06),
        "c": pytest.approx(0.2),
    }
    left = np.ones(200, dtype=np.bool_)
    right = np.zeros(200, dtype=np.bool_)
    assert analyzer.paired_bootstrap(left, right, resamples=1_000, seed=4317) == (1.0, 1.0)
