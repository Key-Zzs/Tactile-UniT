#!/usr/bin/env python3
"""Write ignored evidence that the S4.2-R protocol preceded R4/training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/simulation/s4_2r_contact_state_rank_remediation.json"
DOC = ROOT / "docs/research/s4_2r_contact_state_rank_remediation.md"
ORIGINAL = ROOT / "configs/simulation/s4_2_final_decision.json"
SOURCE = (
    ROOT
    / ".local/artifacts/simulation/s4_2r/intrinsic_dimension/source_intrinsic_dimension.json"
)
DESTINATION = ROOT / ".local/artifacts/simulation/s4_2r/protocol_freeze.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    if original["decision"] != "S4_2_CONTACT_STATE_FAIL":
        raise RuntimeError("original S4.2 decision was modified")
    reference = source["source_dimension_reference"]
    frozen_reference = protocol["r1_source_dimension_reference"]
    if reference["d_src"] != frozen_reference["d_src"]:
        raise RuntimeError("source dimension value mismatch")
    if reference["canonical_sha256"] != frozen_reference["terms_canonical_sha256"]:
        raise RuntimeError("source dimension hash mismatch")
    value = {
        "schema": "tactile3d-unit.s4-2r-protocol-freeze.v1",
        "new_gate_definition": protocol["new_structural_collapse_contract"],
        "source_dimension_reference": frozen_reference,
        "dataset_manifest_hash": protocol["frozen_inputs"][
            "dataset_manifest_canonical_sha256"
        ],
        "teacher_checkpoint_hash": protocol["frozen_inputs"][
            "teacher_checkpoint_sha256"
        ],
        "protocol_files": {
            str(CONFIG.relative_to(ROOT)): sha256(CONFIG),
            str(DOC.relative_to(ROOT)): sha256(DOC),
        },
        "original_s4_2_result": original["decision"],
        "test_loaded": False,
        "training_started": False,
        "r4_started": False,
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(DESTINATION.relative_to(ROOT)), "sha256": sha256(DESTINATION)}, indent=2))


if __name__ == "__main__":
    main()
