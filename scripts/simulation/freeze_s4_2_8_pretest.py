#!/usr/bin/env python3
"""Freeze every selected identity before the first formal TEST access."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/simulation/s4_2_formal_downstream.json"
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_2_formal"
OUTPUT = ARTIFACT_ROOT / "pretest_freeze.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    config = load_json(CONFIG)
    decisions = {
        stage: load_json(ARTIFACT_ROOT / f"s4_2_{stage}_final.json") for stage in (4, 5, 6, 7)
    }
    expected = {
        4: "S4_2_4_FORMAL_ACTION_VISION_ACCEPTED",
        5: "S4_2_5_FORMAL_CONTINUOUS_BRIDGE_ACCEPTED",
        6: "S4_2_6_CONTINUOUS_SELECTED_DISCRETE_REJECTED_SHARED_PRIVATE_ACCEPTED",
        7: "S4_2_7_CONDITIONAL_MISSING_MODALITY_UNCERTAINTY_ACCEPTED",
    }
    for stage, decision in decisions.items():
        if decision["decision"] != expected[stage] or decision["formal_test_loaded"]:
            raise RuntimeError(f"S4.2-{stage} is not pretest-freezable")
    checkpoints = {
        "contact_state": ROOT / ".local/experiments/simulation/s4_2r/contact_state/accepted.pt",
        "contact_C3": ROOT / ".local/experiments/simulation/s4_2r/contact_dynamics/selected.pt",
        "action": ROOT / ".local/experiments/simulation/s4_2_formal/action/selected.pt",
        "bridge": ROOT / ".local/experiments/simulation/s4_2_formal/bridge/selected.pt",
        "shared_private": ROOT
        / ".local/experiments/simulation/s4_2_formal/s4_2_6/shared_private.pt",
        "rejected_contact_rq": ROOT
        / ".local/experiments/simulation/s4_2_formal/s4_2_6/contact_rq_selected.pt",
        "conditional_A_plus_H": ROOT
        / ".local/experiments/simulation/s4_2_formal/s4_2_7/A_plus_H.pt",
        "conditional_V_plus_A_plus_H": ROOT
        / ".local/experiments/simulation/s4_2_formal/s4_2_7/V_plus_A_plus_H.pt",
        "conditional_V_plus_A_missing_H": ROOT
        / ".local/experiments/simulation/s4_2_formal/s4_2_7/V_plus_A_missing_H.pt",
        "conditional_A_only_missing_H": ROOT
        / ".local/experiments/simulation/s4_2_formal/s4_2_7/A_only_missing_H.pt",
        "uncertainty_full": ROOT
        / ".local/experiments/simulation/s4_2_formal/s4_2_7/uncertainty_full.pt",
        "uncertainty_missing_H": ROOT
        / ".local/experiments/simulation/s4_2_formal/s4_2_7/uncertainty_missing_H.pt",
    }
    checkpoint_hashes = {name: sha256_file(path) for name, path in checkpoints.items()}
    if (
        checkpoint_hashes["contact_state"]
        != config["scientific_boundary"]["contact_state_checkpoint_sha256"]
    ):
        raise RuntimeError("accepted Contact-State changed before pretest freeze")
    if (
        checkpoint_hashes["contact_C3"]
        != config["scientific_boundary"]["canonical_contact_checkpoint_sha256"]
    ):
        raise RuntimeError("canonical C3 changed before pretest freeze")
    implementations = {
        path: sha256_file(ROOT / path)
        for path in (
            "configs/simulation/s4_2_formal_downstream.json",
            "gr00t/simulation/s4_2_formal.py",
            "scripts/simulation/run_s4_2_4_formal_representations.py",
            "scripts/simulation/run_s4_2_5_formal_bridge.py",
            "scripts/simulation/run_s4_2_6_bottleneck_shared_private.py",
            "scripts/simulation/run_s4_2_7_conditional_uncertainty.py",
            "scripts/simulation/run_s4_2_8_locked_test.py",
            "scripts/simulation/audit_s4_2_8_final.py",
        )
    }
    caches = {
        name: sha256_file(ROOT / path)
        for name, path in {
            "paired_train": ".local/cache/simulation/s4_2_formal/paired_train.npz",
            "paired_validation": ".local/cache/simulation/s4_2_formal/paired_validation.npz",
            "shared_train": ".local/cache/simulation/s4_2_formal/shared_train.npz",
            "shared_validation": ".local/cache/simulation/s4_2_formal/shared_validation.npz",
        }.items()
    }
    forbidden = (
        ROOT / ".local/cache/simulation/s4_2/pairs/test.npz",
        ROOT / ".local/cache/simulation/s4_2_formal/paired_test.npz",
        ARTIFACT_ROOT / "locked_test.json",
    )
    if any(path.exists() for path in forbidden):
        raise RuntimeError("TEST material exists before the authorized pretest freeze")
    output = {
        "schema": "tactile3d-unit.s4-2-8-pretest-freeze.v1",
        "stage": "S4.2-8_PRETEST_FREEZE",
        "status": "PASS",
        "training_complete": True,
        "selection_complete": True,
        "test_loaded": False,
        "formal_test_model_metrics_loaded": False,
        "dataset_manifest_canonical_sha256": config["data"]["manifest_canonical_sha256"],
        "historical_10_percent_gate": "FAIL_UNCHANGED",
        "canonical_contact": "C3",
        "selected_action": load_json(ARTIFACT_ROOT / "action_selection.json")["selected"],
        "selected_bridge": decisions[5]["selected"],
        "continuous_contact": "PRIMARY",
        "discrete_contact": "REJECTED_RECOVERY_GATE",
        "checkpoint_sha256": checkpoint_hashes,
        "implementation_sha256": implementations,
        "selection_cache_sha256": caches,
        "vision_checkpoint_file_sha256": load_json(ARTIFACT_ROOT / "vision_identity.json")[
            "checkpoint_file_sha256"
        ],
        "locked_test_runs_allowed": int(config["s4_2_8"]["locked_test_runs"]),
        "deterministic_repeat_runs_allowed": int(config["s4_2_8"]["deterministic_repeat_runs"]),
        "post_test_changes_forbidden": config["s4_2_8"]["post_test_changes_forbidden"],
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    atomic_json(OUTPUT, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
