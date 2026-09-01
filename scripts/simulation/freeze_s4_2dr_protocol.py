#!/usr/bin/env python3
"""Freeze the S4.2-DR follow-up protocol after DR1-DR3 and before DR5."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.simulation.s4_2dr_common import (  # noqa: E402
    ARTIFACT_ROOT,
    CACHE_ROOT,
    CONTACT_STATE_PATH,
    HISTORICAL_PATH,
    ROOT,
    SELECTION_PATH,
    atomic_json,
    canonical_hash,
    load_json,
    sha256_file,
)

CONFIG = ROOT / "configs/simulation/s4_2dr_contact_dynamics_remediation.json"


def main() -> None:
    fairness = load_json(ARTIFACT_ROOT / "baseline_fairness.json")
    reproduction = load_json(ARTIFACT_ROOT / "baseline_reproduction.json")
    reference = load_json(ARTIFACT_ROOT / "empirical_predictability_reference.json")
    regimes = load_json(ARTIFACT_ROOT / "regime_decomposition.json")
    thresholds = load_json(ARTIFACT_ROOT / "regime_thresholds.json")
    historical = load_json(HISTORICAL_PATH)
    selection = load_json(SELECTION_PATH)
    contact_state = load_json(CONTACT_STATE_PATH)
    if fairness["gate"] != "PASS" or reproduction["gate"] != "PASS":
        raise RuntimeError("DR1 is not complete")
    if historical["decision"] != "S4_2_3_CONTACT_DYNAMICS_FAIL":
        raise RuntimeError("historical S4.2-3 failure was not preserved")
    if any((ROOT / ".local/experiments/simulation/s4_2dr").glob("R*/best.pt")):
        raise RuntimeError("new Dynamics training started before protocol freeze")
    config = load_json(CONFIG)
    freeze = {
        "schema": "tactile3d-unit.s4-2dr-protocol-freeze.v1",
        "status": "FROZEN_BEFORE_DR5_AND_BEFORE_NEW_DYNAMICS_TRAINING",
        "protocol_path": "configs/simulation/s4_2dr_contact_dynamics_remediation.json",
        "protocol_sha256": sha256_file(CONFIG),
        "dataset": {
            "train_sha256": sha256_file(CACHE_ROOT / "train.npz"),
            "validation_sha256": sha256_file(CACHE_ROOT / "validation.npz"),
        },
        "contact_state_checkpoint": contact_state["accepted_teacher"]["checkpoint"],
        "contact_state_checkpoint_sha256": contact_state["accepted_teacher"]["checkpoint_sha256"],
        "C2_checkpoint": next(row["checkpoint"] for row in selection["trials"] if row["model"] == "C2"),
        "C2_checkpoint_sha256": next(row["checkpoint_sha256"] for row in selection["trials"] if row["model"] == "C2"),
        "C3_checkpoint": selection["selected_checkpoint"],
        "C3_checkpoint_sha256": selection["selected_checkpoint_sha256"],
        "DR1_fairness_result_hash": fairness["fairness_result_hash"],
        "DR2_result_hash": reference["result_hash"],
        "E_ref": reference["selected_reference"],
        "critical_regime_thresholds": thresholds,
        "DR3_result_hash": regimes["result_hash"],
        "route_A_gates": config["route_A"]["gates"],
        "route_B_gates": config["route_B"]["gates"],
        "historical_result": config["historical_result"],
        "test_loaded": False,
        "new_training_started": False,
    }
    freeze["protocol_freeze_hash"] = canonical_hash(freeze)
    atomic_json(ARTIFACT_ROOT / "protocol_freeze.json", freeze)
    print(json.dumps({"status": freeze["status"], "protocol_freeze_hash": freeze["protocol_freeze_hash"], "test_loaded": False}, indent=2))


if __name__ == "__main__":
    main()
