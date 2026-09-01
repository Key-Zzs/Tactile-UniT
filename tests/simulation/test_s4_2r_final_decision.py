from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_final_decision_preserves_failure_and_hard_stop() -> None:
    decision = json.loads(
        (ROOT / "configs/simulation/s4_2r_final_decision.json").read_text()
    )
    assert decision["decision"] == "S4_2R_CONTACT_DYNAMICS_FAIL"
    assert decision["original_s4_2"] == {
        "decision": "S4_2_CONTACT_STATE_FAIL",
        "effective_rank": 11.255626031431431,
        "threshold": 16.0,
        "modified": False,
    }
    failure = decision["contact_dynamics_failure"]
    assert failure["relative_improvement"] < failure["required_relative_improvement"]
    assert decision["stages"]["S4.2-3"] == "S4_2_3_CONTACT_DYNAMICS_FAIL"
    assert all(
        decision["stages"][stage] == "NOT_RUN_DEPENDENCY_BLOCKED"
        for stage in ("S4.2-4A", "S4.2-4B", "S4.2-4C", "S4.2-5")
    )
    assert decision["test_loaded"] is False
    assert decision["selection_uses_test"] is False


def test_unexecuted_stage_artifacts_are_not_fabricated() -> None:
    artifact_root = ROOT / ".local/artifacts/simulation/s4_2r"
    forbidden = (
        "action_selection.json",
        "action_validation.json",
        "vision_identity.json",
        "paired_vac_manifest.json",
        "bridge_selection.json",
        "bridge_validation.json",
    )
    assert not any((artifact_root / name).exists() for name in forbidden)
