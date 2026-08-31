from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_final_decision_honestly_records_contact_state_failure() -> None:
    decision = json.loads(
        (ROOT / "configs/simulation/s4_2_final_decision.json").read_text()
    )
    assert decision["decision"] == "S4_2_CONTACT_STATE_FAIL"
    assert decision["milestone_status"] == "FAILED"
    assert decision["stages"]["S4.2-2"]["status"] == "FAIL"
    assert decision["contact_state"]["effective_rank"] < decision["contact_state"]["effective_rank_min"]
    assert decision["contact_state"]["failed_gates"] == ["no_collapse"]
    assert decision["failure"]["threshold_changed_after_result"] is False
    assert decision["failure"]["retraining_after_failure"] is False


def test_dependent_stages_and_locked_test_are_not_claimed() -> None:
    decision = json.loads(
        (ROOT / "configs/simulation/s4_2_final_decision.json").read_text()
    )
    for stage in ("S4.2-3", "S4.2-4", "S4.2-5", "S4.2-6", "S4.2-7"):
        assert decision["stages"][stage]["status"] == "NOT_RUN_DEPENDENCY_BLOCKED"
    test = decision["test_protocol"]
    assert test["training_uses_test"] is False
    assert test["selection_uses_test"] is False
    assert test["locked_test_evaluation"] == "NOT_RUN_DEPENDENCY_BLOCKED"
    assert decision["s4_3_readiness"] == "NOT_READY"
    assert decision["policy_training"] == "NOT_STARTED"
