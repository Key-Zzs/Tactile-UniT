#!/usr/bin/env python3
"""Freeze the preregistered S4.2-DS selection protocol before pilot training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/simulation/s4_2ds_representation_selection.json"
ARTIFACT = ROOT / ".local/artifacts/simulation/s4_2ds/protocol_freeze.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_ACTION_OR_BRIDGE_PILOT_TRAINING":
        raise RuntimeError("S4.2-DS protocol is not in a freezable state")
    if protocol["data"]["formal_test_model_metrics_loaded"] is not False:
        raise RuntimeError("formal test integrity violated")
    utility_path = ROOT / ".local/artifacts/simulation/s4_2ds/contact_representation_utility.json"
    utility = json.loads(utility_path.read_text(encoding="utf-8"))
    result = {
        "schema": "tactile3d-unit.s4-2ds-protocol-freeze.v1",
        "stage": "DS3",
        "historical_decisions": {
            "original": "S4_2_3_CONTACT_DYNAMICS_FAIL",
            "remediation": "S4_2DR_DYNAMICS_REMEDIATION_FAIL",
            "modified": False,
        },
        "c2_eligibility": utility["candidates"]["C2"]["contract"]["eligibility"],
        "contact_only_eligibility_gates": protocol["contact_only_hard_gates"],
        "bridge_gates": protocol["bridge_pilot"]["hard_gates"],
        "material_advantage_thresholds": protocol["material_advantage"],
        "selection_rule": protocol["canonical_selection_rule"],
        "config": "configs/simulation/s4_2ds_representation_selection.json",
        "config_sha256": sha256_file(CONFIG),
        "contact_utility_sha256": sha256_file(utility_path),
        "formal_test_loaded": False,
        "formal_validation_confirmation_loaded": False,
        "ds_training_started": False,
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    temporary = ARTIFACT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(ARTIFACT)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
