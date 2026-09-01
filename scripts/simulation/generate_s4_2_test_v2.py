#!/usr/bin/env python3
"""Generate the preregistered independent FORMAL_TEST_V2 episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.simulation.generate_s4_2_dataset import generate_episode  # noqa: E402

CONFIG = ROOT / "configs/simulation/s4_2_tf_locked_test_v2.json"
DATASET_CONTRACT = ROOT / "configs/simulation/s4_2_dataset_contract.json"
DEFAULT_ROOT = ROOT / ".local/datasets/simulation/s4_2_test_v2"
PREREGISTRATION = ROOT / ".local/artifacts/simulation/s4_2_tf/formal_test_v2_preregistration.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if os.environ.get("DISPLAY"):
        raise SystemExit("DISPLAY must be unset")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    protocol = config["formal_test_v2"]
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if preregistration["status"] != "FROZEN_BEFORE_FORMAL_TEST_V2_GENERATION":
        raise RuntimeError("FORMAL_TEST_V2 preregistration is not frozen")
    if preregistration["generation_implementation_sha256"] != sha256_file(Path(__file__)):
        raise RuntimeError("FORMAL_TEST_V2 generator changed after preregistration")
    contract = json.loads(DATASET_CONTRACT.read_text(encoding="utf-8"))
    completed = []
    for task in protocol["tasks"]:
        for group in protocol["source_group_indices"]:
            source_id = protocol["source_trajectory_id"].format(task=task, group=group)
            parameters = dict(protocol["group_parameters"][str(group)])
            for perturbation in range(protocol["perturbations_per_group"]):
                episode_id = protocol["episode_id"].format(
                    task=task, group=group, perturbation=perturbation
                )
                seed = int(protocol["seed_bases"][task] + 10 * group + perturbation)
                manifest = generate_episode(
                    args.root,
                    task,
                    group,
                    perturbation,
                    "test",
                    contract,
                    source_id=source_id,
                    episode_id=episode_id,
                    seed=seed,
                    parameters=parameters,
                    metadata_updates={
                        "formal_test_name": "FORMAL_TEST_V2",
                        "generation_preregistration_sha256": sha256_file(PREREGISTRATION),
                    },
                )
                completed.append(manifest["episode_id"])
                print(json.dumps({"episode": manifest["episode_id"], "status": "COMPLETE"}))
    if len(completed) != protocol["episode_count"] or len(set(completed)) != len(completed):
        raise RuntimeError("FORMAL_TEST_V2 episode count/identity mismatch")
    print(json.dumps({"episodes": len(completed), "root": str(args.root), "status": "COMPLETE"}))


if __name__ == "__main__":
    main()
