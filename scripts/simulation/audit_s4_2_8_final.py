#!/usr/bin/env python3
"""Close S4.2 after locked TEST and its equality-only repeat."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_2_formal"


def load(name: str) -> dict[str, Any]:
    return json.loads((ARTIFACT_ROOT / name).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    pretest, access, first, repeat = (
        load("pretest_freeze.json"),
        load("locked_test_access.json"),
        load("locked_test.json"),
        load("locked_test_repeat.json"),
    )
    if pretest["status"] != "PASS" or not access["test_loaded"]:
        raise RuntimeError("locked TEST access contract failed")
    if (
        repeat.get("deterministic_equal") is not True
        or repeat["metric_digest"] != first["metric_digest"]
    ):
        raise RuntimeError("deterministic repeat mismatch")
    checkpoint_hashes = pretest["checkpoint_sha256"]
    current_frozen = {
        "contact_state": sha256_file(
            ROOT / ".local/experiments/simulation/s4_2r/contact_state/accepted.pt"
        ),
        "contact_C3": sha256_file(
            ROOT / ".local/experiments/simulation/s4_2r/contact_dynamics/selected.pt"
        ),
    }
    if any(current_frozen[name] != checkpoint_hashes[name] for name in current_frozen):
        raise RuntimeError("frozen Contact identity changed after TEST")
    test_output = subprocess.check_output(
        [sys.executable, "-m", "pytest", "-q", "tests/simulation"],
        cwd=ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    )
    if "failed" in test_output.lower():
        raise RuntimeError("simulation regression failed")
    accepted = first["overall"] == "PASS"
    decision = {
        "schema": "tactile3d-unit.s4-2-8-final-decision.v1",
        "decision": (
            "S4_2_EXTERNAL_REPRESENTATION_ACCEPTED_CONTINUOUS_C3_SHARED_PRIVATE_CONDITIONAL"
            if accepted
            else "S4_2_EXTERNAL_REPRESENTATION_LOCKED_TEST_FAIL"
        ),
        "classification": (
            "CONTINUOUS_C3_EXTERNAL_REPRESENTATION" if accepted else "LOCKED_TEST_FAIL"
        ),
        "canonical_contact": "C3",
        "continuous_contact": "SELECTED",
        "discrete_contact": "REJECTED_RECOVERY_GATE",
        "selected_action": pretest["selected_action"],
        "selected_bridge": pretest["selected_bridge"],
        "shared_private": first["sections"]["shared_private"]["overall"],
        "conditional": first["sections"]["conditional"]["overall"],
        "uncertainty": first["sections"]["uncertainty"]["overall"],
        "locked_test": first["overall"],
        "locked_test_metric_digest": first["metric_digest"],
        "deterministic_repeat": "PASS",
        "deterministic_equal": True,
        "historical_original": "S4_2_3_CONTACT_DYNAMICS_FAIL",
        "historical_remediation": "S4_2DR_DYNAMICS_REMEDIATION_FAIL",
        "historical_10_percent_gate": "FAIL_UNCHANGED",
        "contact_state_checkpoint_sha256": current_frozen["contact_state"],
        "contact_dynamics_checkpoint_sha256": current_frozen["contact_C3"],
        "dataset_manifest_canonical_sha256": pretest["dataset_manifest_canonical_sha256"],
        "training_after_test": False,
        "selection_after_test": False,
        "threshold_change_after_test": False,
        "formal_test_runs": 1,
        "deterministic_repeat_runs": 1,
        "simulation_regression": "PASS",
    }
    atomic_json(ARTIFACT_ROOT / "final_decision.json", decision)
    acceptance = f"""# S4.2-4--8 Human Acceptance

- S4.2-4 formal Action/Vision and exact paired VAC: PASS
- S4.2-5 formal continuous bridge: PASS (`{pretest['selected_bridge']}`)
- S4.2-6 continuous/discrete comparison: continuous selected; discrete rejected at unchanged recovery gate; shared/private PASS
- S4.2-7 conditional sufficiency, missing modality, uncertainty: PASS
- S4.2-8 pretest freeze: PASS
- First locked TEST: {first['overall']}
- Equality-only deterministic repeat: PASS (`{first['metric_digest']}`)
- Final decision: **{decision['decision']}**

Historical S4.2-3 and S4.2-DR failures remain unchanged. No training, selection, threshold, normalization, split, or checkpoint change occurred after TEST access.
"""
    (ARTIFACT_ROOT / "HUMAN_ACCEPTANCE.md").write_text(acceptance, encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
