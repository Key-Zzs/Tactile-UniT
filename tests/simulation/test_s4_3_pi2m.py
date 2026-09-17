from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/simulation/s4_3_pi2m_bhva_target_protocol.json"
BUILDER = ROOT / "scripts/simulation/build_s4_3_pi2m_bhva_targets.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_s4_3_pi2m_bhva_targets", BUILDER)
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
