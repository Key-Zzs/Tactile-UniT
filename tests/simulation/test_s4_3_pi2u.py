from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from queue import SimpleQueue
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gr00t.simulation.s4_3_pi1 import TactileUnitMode
from gr00t.simulation.s4_3_pi2u_va import VAOnlyBridge, different_episode_info_nce
from scripts.simulation.run_s4_3_pi2u_eval import install_cross_episode_action_quarantine


ROOT = Path(__file__).resolve().parents[2]


def test_formal_seed6_is_consistent_across_protocol_and_scripts() -> None:
    protocol = json.loads((ROOT / "configs/simulation/s4_3_pi2u_ablation_retry_seed6.json").read_text())
    assert protocol["evaluator_seed"] == 6
    assert protocol["temporal_remediation"]["canonical_control_steps"] == 27
    assert protocol["temporal_remediation"]["canonical_horizon_seconds"] == pytest.approx(0.54)
    assert protocol["temporal_remediation"]["source_dataset_offset_frames"] == 16
    assert protocol["temporal_remediation"]["bva_checkpoint_tree_sha256"] == "04b609d7cc89e5fffdfab8219bf34da362117a15d0c7e3d5cd4ee20a9ee4770d"
    sources = {
        name: (ROOT / f"scripts/simulation/{name}").read_text()
        for name in (
            "run_s4_3_pi2u_eval.py",
            "analyze_s4_3_pi2u_ablation.py",
            "plot_s4_3_pi2u_results.py",
            "audit_s4_3_pi2u_final.py",
        )
    }
    assert "EVALUATOR_SEED = 6" in sources["run_s4_3_pi2u_eval.py"]
    assert "EVALUATOR_SEED = 6" in sources["analyze_s4_3_pi2u_ablation.py"]
    assert "SEED = 6" in sources["plot_s4_3_pi2u_results.py"]
    assert "EVALUATOR_SEED = 6" in sources["audit_s4_3_pi2u_final.py"]
    assert ".local/tmp/s43u6" in sources["audit_s4_3_pi2u_final.py"]
    assert 'event.get("type") == "stale_cross_episode_action_discarded"' in sources["audit_s4_3_pi2u_final.py"]
    assert '"all_cross_episode_action_quarantines_valid_and_discarded"' in sources["audit_s4_3_pi2u_final.py"]


def test_cross_episode_future_action_chunk_is_discarded(tmp_path: Path) -> None:
    diagnostics = tmp_path / "inference.jsonl"
    official = SimpleNamespace(
        _interp_single_arm_action=lambda left, right, ratio: (1 - ratio) * left + ratio * right,
        _interp_dual_arm_action=lambda left, right, ratio: (1 - ratio) * left + ratio * right,
        Action=lambda action, timestamp: SimpleNamespace(action=action, timestamp=timestamp),
    )
    install_cross_episode_action_quarantine(official, diagnostics, "B0")
    action_queue = SimpleQueue()
    action_queue.put(SimpleNamespace(timestamp=602, action=np.full((2, 1), 99.0)))
    action_queue.put(SimpleNamespace(timestamp=0, action=np.asarray([[1.0], [2.0]])))
    actions_buffer = deque()

    official.receive_actions(action_queue, actions_buffer, now_timestamp=0, dual_arm=False)

    assert [row.timestamp for row in actions_buffer] == [0, 1]
    assert [float(row.action[0]) for row in actions_buffer] == [1.0, 2.0]
    events = [json.loads(line) for line in diagnostics.read_text().splitlines()]
    assert events == [{
        "action_timestamp": 602,
        "current_timestamp": 0,
        "model": "B0",
        "reason": "future timestamp is impossible within one causal episode",
        "type": "stale_cross_episode_action_discarded",
    }]


def test_bva_mode_is_explicit_and_contact_free() -> None:
    assert TactileUnitMode.VA_PHYSICAL_AUX.value == "VA_PHYSICAL_AUX"
    source = (ROOT / "gr00t/simulation/pi05_tactile_unit.py").read_text()
    branch = source[source.index("class TactilePi0") :]
    assert "TactileUnitMode.VA_PHYSICAL_AUX" in branch
    assert "if self.tactile_unit_mode is TactileUnitMode.VA_PHYSICAL_AUX" in branch


def test_va_bridge_shape_and_modality_rejection() -> None:
    model = VAOnlyBridge()
    value = torch.randn(3, 8, 32)
    assert model.encode("vision", value).shape == value.shape
    assert model.encode("action", value).shape == value.shape
    with pytest.raises(ValueError, match="rejects modality"):
        model.encode("contact", value)


def test_different_episode_nce_is_finite() -> None:
    query = torch.randn(4, 8, 32)
    candidate = torch.randn(4, 8, 32)
    episode = torch.tensor([0, 0, 1, 2])
    loss = different_episode_info_nce(query, candidate, episode, temperature=0.1)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_frozen_bva_protocol_contract() -> None:
    protocol = json.loads((ROOT / "configs/simulation/s4_3_pi2u_bva_protocol.json").read_text())
    assert protocol["status"] == "FROZEN_BEFORE_TRAINING"
    assert protocol["seed"] == 42
    assert protocol["steps"] == 30_000
    assert protocol["global_batch_size"] == 32
    assert protocol["auxiliary_target"]["sidecar_allowed_fields"] == [
        "index",
        "va_shared_target",
        "va_aux_valid",
    ]
    assert protocol["official_policy"]["tactile_or_contact_runtime_input"] is False
    temporal = protocol["auxiliary_target"]["temporal_alignment"]
    assert temporal["canonical_offset_steps"] == 27
    assert temporal["canonical_horizon_seconds"] == pytest.approx(0.54)
    assert temporal["source_dataset_fps"] == pytest.approx(30.0)
    assert temporal["source_offset_frames"] == 16
    assert temporal["source_horizon_seconds"] == pytest.approx(16 / 30)
    assert temporal["absolute_timing_error_seconds"] == pytest.approx(0.54 - 16 / 30)
    assert temporal["selection_rule"] == "nearest native source frame; no RGB interpolation"


def test_materialized_sidecar_is_exact_and_tail_accounted() -> None:
    path = ROOT / ".local/datasets/simulation/s4_3_pi2u/pinch_tongs_va/sidecar.npz"
    if not path.exists():
        pytest.skip("local frozen BVA target sidecar is not materialized")
    with np.load(path, allow_pickle=False) as source:
        assert set(source.files) == {"index", "va_shared_target", "va_aux_valid"}
        assert source["index"].shape == (40_065,)
        assert source["va_shared_target"].shape == (40_065, 8, 32)
        valid = source["va_aux_valid"]
        assert valid.dtype == np.bool_
        assert int(valid.sum()) == 38_465
        assert int((~valid).sum()) == 1_600
        assert np.all(source["va_shared_target"][~valid] == 0)


def test_mode_and_none_parity_artifacts_pass() -> None:
    artifact_root = ROOT / ".local/artifacts/simulation/s4_3_pi2u"
    for name in ("bva_mode_contract.json", "bva_none_parity.json", "contact_leakage_audit.json"):
        path = artifact_root / name
        if not path.exists():
            pytest.skip(f"local validation artifact is not materialized: {name}")
        assert json.loads(path.read_text())["status"] == "PASS"
