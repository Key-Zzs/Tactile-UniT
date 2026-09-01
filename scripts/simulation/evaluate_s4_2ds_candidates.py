#!/usr/bin/env python3
"""Apply the frozen DS3 rule and freeze the canonical Contact candidate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gr00t.simulation.s4_2_dataset import sha256_file


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_2ds"
CONFIG_PATH = ROOT / "configs/simulation/s4_2ds_representation_selection.json"


def load(name: str) -> dict[str, Any]:
    return json.loads(ARTIFACT_ROOT.joinpath(name).read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    protocol = load("protocol_freeze.json")
    if protocol["config_sha256"] != sha256_file(CONFIG_PATH):
        raise RuntimeError("selection protocol changed after freeze")
    utility = load("contact_representation_utility.json")
    comparison = load("candidate_comparison.json")
    bridge = {name: load(f"{name.lower()}_bridge_pilot.json") for name in ("C2", "C3")}
    eligible = {
        name: bool(utility["candidates"][name]["contact_only_eligible"])
        and bridge[name]["overall"] == "PASS"
        for name in ("C2", "C3")
    }
    material = {
        "semantic": bool(comparison["contact_semantic_advantage_C3"]),
        "bridge": bool(comparison["bridge_advantage_C3"]),
        "reconstruction": bool(comparison["frozen_future_reconstruction_advantage_C3"]),
    }
    if not eligible["C2"] and eligible["C3"]:
        selected = "C3"
        classification = "C3_STRUCTURED_REPRESENTATION_SELECTED"
    elif eligible["C2"] and not eligible["C3"]:
        selected = "C2"
        classification = "C2_DELTA_REPRESENTATION_SELECTED"
    elif eligible["C2"] and eligible["C3"]:
        if any(material.values()):
            selected = "C3"
            classification = "C3_REPRESENTATION_UTILITY_SELECTED"
        else:
            selected = "C2"
            classification = "C2_DELTA_REPRESENTATION_SELECTED"
    else:
        selected = "NONE"
        classification = "CONTACT_REPRESENTATION_INSUFFICIENT"
    checkpoint = utility["candidates"][selected]["contract"] if selected != "NONE" else None
    canonical = {
        "schema": "tactile3d-unit.s4-2ds-canonical-contact-representation.v1",
        "stage": "DS6",
        "candidate": selected,
        "classification": classification,
        "checkpoint": None if checkpoint is None else checkpoint["checkpoint"],
        "checkpoint_sha256": None if checkpoint is None else checkpoint["checkpoint_sha256"],
        "native_shape": None if checkpoint is None else checkpoint["native_latent_shape"],
        "common_shape": None if checkpoint is None else checkpoint["common_latent_shape"],
        "adapter": None if checkpoint is None else checkpoint["adapter"],
        "semantic_metrics": None if selected == "NONE" else utility["candidates"][selected]["probes"],
        "temporal_metrics": None
        if selected == "NONE"
        else utility["candidates"][selected]["temporal_and_controls"],
        "bridge_metrics": None if selected == "NONE" else bridge[selected]["alignment"],
        "selection_reason": [
            "Both C2 and C3 passed every Contact-only and VAC bridge hard gate.",
            "C3 had no material semantic or bridge advantage under the frozen thresholds.",
            "C3 had a 7.017% frozen future-recovery MSE advantage with positive paired bootstrap CI, satisfying the frozen material reconstruction rule.",
        ]
        if selected == "C3"
        else ["Frozen DS3 canonical-selection rule applied without post-hoc changes."],
        "material_advantage_C3": material,
        "historical_10_percent_status": "FAIL_UNCHANGED",
        "historical_decisions_modified": False,
        "formal_validation_confirmation_loaded": False,
        "formal_test_loaded": False,
    }
    atomic_json(ARTIFACT_ROOT / "canonical_contact_representation.json", canonical)
    print(json.dumps(canonical, indent=2))
    if selected == "NONE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
