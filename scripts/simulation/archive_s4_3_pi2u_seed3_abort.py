#!/usr/bin/env python3
"""Preserve the invalidated PI2U seed-3 attempt without reusing its outcomes."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2u"
TARGET = ARTIFACTS / "aborted_seed3"


def move(source: Path, destination: Path) -> None:
    if source.exists():
        if destination.exists():
            raise SystemExit(f"archive destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))


def main() -> None:
    if TARGET.exists():
        raise SystemExit("seed3 abort archive already exists")
    TARGET.mkdir(parents=True)
    for model in ("b0", "bva", "b1", "b2"):
        move(ARTIFACTS / f"{model}_raw_rollouts.json", TARGET / f"{model}_raw_rollouts.json")
        move(ARTIFACTS / f"{model}_contact_state_service.json", TARGET / f"{model}_contact_state_service.json")
    move(ROOT / ".local/logs/simulation/s4_3_pi2u/evaluation", TARGET / "logs")
    move(ROOT / ".local/cache/simulation/s4_3_pi2u/evaluation/seed3", TARGET / "cache")
    move(ROOT / ".local/tmp/simulation/s4_3_pi2u/evaluation", TARGET / "tmp")
    manifest = {
        "schema": "tactile3d-unit.s4-3-pi2u-aborted-seed3.v1",
        "status": "ABORTED_INFRASTRUCTURE_FAILURE",
        "evaluator_seed": 3,
        "B0_completed_episodes": 161,
        "B1_completed_episodes": 200,
        "BVA_completed_episodes": 200,
        "B2_started": False,
        "cause": "late prior-episode ActionChunk crossed timestamp reset and triggered official receive_actions future-timestamp assertion",
        "formal_outcomes_reusable": False,
        "replacement_seed": 4,
    }
    (TARGET / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
