from __future__ import annotations

import json
from pathlib import Path

from gr00t.simulation.pi2b_policy.contract import MODEL_ORDER, NEW_SEEDS, RECIPES


ROOT = Path(__file__).resolve().parents[3]


def test_exact_scope_and_lambdas():
    assert MODEL_ORDER == ("B0", "B_VA27", "B1", "B_HVA", "B2")
    assert NEW_SEEDS == (43, 44)
    assert RECIPES["B_VA27"].lambda_phys == 0.03077957631925596
    assert RECIPES["B_HVA"].lambda_phys == 0.026468189597253295
    assert RECIPES["B2"].lambda_phys == 0.020927851827513076
    assert RECIPES["B0"].lambda_phys == RECIPES["B1"].lambda_phys == 0.0


def test_no_h_and_h_modes_are_exact():
    assert not RECIPES["B0"].online_contact_history
    assert not RECIPES["B_VA27"].online_contact_history
    assert all(RECIPES[key].online_contact_history for key in ("B1", "B_HVA", "B2"))
    assert RECIPES["B_HVA"].target == RECIPES["B_VA27"].target == "VA27"
    assert RECIPES["B2"].target == "VAC_CONTACT27"


def test_public_protocol_authorizes_exact_counts():
    protocol = json.loads((ROOT / "configs/simulation/pi2b_policy/protocol.json").read_text())
    assert protocol["new_policy_runs"] == 10
    assert protocol["teacher_runs"] == 0
    assert protocol["formal_checkpoints"] == 15
    assert protocol["formal_rollouts"] == 3000
    assert len(protocol["static_run_order"]) == 10
    assert protocol["isolation"]["read_new_teacher_weights"] is False
