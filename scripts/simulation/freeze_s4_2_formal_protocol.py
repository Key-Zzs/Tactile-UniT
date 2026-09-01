#!/usr/bin/env python3
"""Audit and freeze the S4.2-4--8 protocol before formal downstream training."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/simulation/s4_2_formal_downstream.json"
DATASET = ROOT / ".local/artifacts/simulation/s4_2/dataset_manifest.json"
DS_FINAL = ROOT / ".local/artifacts/simulation/s4_2ds/final_decision.json"
CONTACT_STATE = ROOT / ".local/experiments/simulation/s4_2r/contact_state/accepted.pt"
CONTACT_C3 = ROOT / ".local/experiments/simulation/s4_2r/contact_dynamics/selected.pt"
PAIR_ROOT = ROOT / ".local/cache/simulation/s4_2/pairs"
OUTPUT = ROOT / ".local/artifacts/simulation/s4_2_formal/protocol_freeze.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_fields(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {
            name: source[name]
            for name in (
                "pair_id",
                "task",
                "source_trajectory_id",
                "anchor_step",
                "future_step",
            )
        }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    config = load_json(CONFIG)
    dataset = load_json(DATASET)
    ds_final = load_json(DS_FINAL)
    if config["status"] != "FROZEN_BEFORE_FORMAL_S4_2_4_TRAINING":
        raise RuntimeError("formal downstream config is not freezable")
    if dataset["canonical_sha256"] != config["data"]["manifest_canonical_sha256"]:
        raise RuntimeError("formal dataset identity mismatch")
    if ds_final["decision"] != "S4_2DS_C3_REPRESENTATION_UTILITY_SELECTED":
        raise RuntimeError("canonical Contact representation was not selected")
    if ds_final["formal_test_model_metrics_loaded"]:
        raise RuntimeError("formal test was already accessed")
    identities = {
        "contact_state": sha256_file(CONTACT_STATE),
        "canonical_C3": sha256_file(CONTACT_C3),
    }
    expected = {
        "contact_state": config["scientific_boundary"]["contact_state_checkpoint_sha256"],
        "canonical_C3": config["scientific_boundary"]["canonical_contact_checkpoint_sha256"],
    }
    if identities != expected:
        raise RuntimeError("frozen Contact identity mismatch")
    train = load_fields(PAIR_ROOT / "train.npz")
    validation = load_fields(PAIR_ROOT / "validation.npz")
    if len(train["pair_id"]) != 22680 or len(validation["pair_id"]) != 4860:
        raise RuntimeError("formal pair count mismatch")
    groups = {
        split: set(zip(value["task"].tolist(), value["source_trajectory_id"].tolist()))
        for split, value in (("train", train), ("validation", validation))
    }
    if groups["train"] & groups["validation"]:
        raise RuntimeError("train/validation source-group leakage")
    if not all(
        np.all(value["future_step"] - value["anchor_step"] == 27) for value in (train, validation)
    ):
        raise RuntimeError("transition offset mismatch")
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    if branch != "develop/sim-benchmark":
        raise RuntimeError("wrong branch")
    result = {
        "schema": "tactile3d-unit.s4-2-formal-protocol-freeze.v1",
        "stage": "S4.2-4_PRETRAIN_FREEZE",
        "status": "PASS",
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": sha256_file(CONFIG),
        "dataset_manifest_canonical_sha256": dataset["canonical_sha256"],
        "frozen_checkpoint_sha256": identities,
        "historical_10_percent_gate": "FAIL_UNCHANGED",
        "canonical_contact": "C3",
        "pair_counts": {"train": len(train["pair_id"]), "validation": len(validation["pair_id"])},
        "source_group_counts": {name: len(value) for name, value in groups.items()},
        "source_group_overlap": 0,
        "transition_offset_exact": True,
        "formal_test_loaded": False,
        "training_uses_test": False,
        "selection_uses_test": False,
        "starting_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "branch": branch,
    }
    atomic_json(OUTPUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
