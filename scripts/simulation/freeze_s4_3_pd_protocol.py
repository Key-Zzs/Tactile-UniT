#!/usr/bin/env python3
"""Validate and freeze the S4.3-PD formal acquisition before any rollout."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_3_pd import (  # noqa: E402
    TASKS,
    replay_tree_sha256,
    sha256_file,
    source_partition,
)
from scripts.simulation.generate_s4_3_policy_expert_data import (  # noqa: E402
    ARTIFACT_ROOT,
    FORMAL_ROOT,
    OFFICIAL_ROOT,
    code_hashes,
    formal_attempt_manifest,
    write_json,
)

CONFIG_ROOT = ROOT / "configs/simulation"
CONFIG_PATHS = {
    "protocol": CONFIG_ROOT / "s4_3_pd_protocol.json",
    "expert_contract": CONFIG_ROOT / "s4_3_pd_expert_contract.json",
    "formal_acquisition": CONFIG_ROOT / "s4_3_pd_formal_acquisition.json",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_seed_integers(value: Any, key_hint: str = "") -> set[int]:
    result: set[int] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            result.update(collect_seed_integers(child, str(key).lower()))
    elif isinstance(value, list):
        for child in value:
            result.update(collect_seed_integers(child, key_hint))
    elif "seed" in key_hint and isinstance(value, int) and not isinstance(value, bool):
        result.add(value)
    return result


def prior_recorded_seeds() -> set[int]:
    result: set[int] = set()
    for path in sorted(CONFIG_ROOT.glob("*.json")):
        if path.name.startswith("s4_3_pd_"):
            continue
        result.update(collect_seed_integers(read_json(path)))
    for path in sorted(ARTIFACT_ROOT.glob("*.json")):
        if path.name.startswith("formal_") or path.name == "pilot_seed_manifest.json":
            continue
        result.update(collect_seed_integers(read_json(path)))
    return result


def main() -> None:
    manifest_path = ARTIFACT_ROOT / "formal_attempt_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("run formal generator --freeze-only before protocol freeze")
    manifest = formal_attempt_manifest(OFFICIAL_ROOT)
    if read_json(manifest_path) != manifest:
        raise SystemExit("expanded formal manifest differs from current source partition")
    if FORMAL_ROOT.exists() and any(FORMAL_ROOT.iterdir()):
        raise SystemExit("formal attempt data exists before protocol freeze")

    configs = {name: read_json(path) for name, path in CONFIG_PATHS.items()}
    pilot = read_json(ARTIFACT_ROOT / "pilot_results.json")
    starting = read_json(ARTIFACT_ROOT / "starting_integrity.json")
    task_contract = read_json(ARTIFACT_ROOT / "task_success_contract.json")
    pilot_manifest = read_json(ARTIFACT_ROOT / "pilot_seed_manifest.json")
    if pilot.get("status") != "PASS" or not pilot.get("all_tasks_at_least_80_percent"):
        raise SystemExit("PD4 did not pass; formal protocol cannot be frozen")
    if configs["protocol"].get("status") != "FROZEN_BEFORE_FORMAL_ACQUISITION":
        raise SystemExit("tracked protocol status is not frozen")

    expected_group_ids = {
        task: [row["source_group_id"] for row in manifest["groups"][task]]
        for task in TASKS
    }
    if configs["formal_acquisition"].get("formal_group_ids") != expected_group_ids:
        raise SystemExit("tracked formal group IDs differ from the expanded manifest")

    current_code_hashes = code_hashes()
    expert_contract = configs["expert_contract"]
    expected_code_hashes = {
        "expert_adapter": expert_contract["expert_code"]["sha256"],
        "central_action_adapter": expert_contract["central_action_adapter"]["sha256"],
        "formal_generator": expert_contract["formal_generator"]["sha256"],
    }
    if current_code_hashes != expected_code_hashes:
        raise SystemExit("tracked expert code hashes do not match the source tree")

    task_success_hashes = {}
    for task in TASKS:
        item = expert_contract["native_success_sources"][task]
        actual = sha256_file(ROOT / item["path"])
        if actual != item["sha256"]:
            raise SystemExit(f"{task} native success source changed before freeze")
        if task_contract["tasks"][task]["source_sha256"] != actual:
            raise SystemExit(f"{task} PD2 success audit hash mismatch")
        task_success_hashes[task] = {"path": item["path"], "sha256": actual}

    pilot_groups = {
        (row["task"], row["source_group_id"]) for row in pilot_manifest["attempts"]
    }
    formal_groups = {
        (row["task"], row["source_group_id"]) for row in manifest["attempts"]
    }
    group_overlap = sorted(pilot_groups & formal_groups)
    if group_overlap:
        raise SystemExit(f"pilot/formal source-group overlap: {group_overlap}")
    pilot_seeds = {
        (row["reset_seed"], row["perturbation_seed"], row["controller_seed"])
        for row in pilot_manifest["attempts"]
    }
    formal_seeds = {
        (row["reset_seed"], row["perturbation_seed"], row["controller_seed"])
        for row in manifest["attempts"]
    }
    if pilot_seeds & formal_seeds:
        raise SystemExit("pilot/formal seed-tuple overlap")
    formal_integer_seeds = {
        seed
        for row in manifest["attempts"]
        for seed in (
            row["reset_seed"],
            row["environment_seed"],
            row["visual_randomization_seed"],
            row["perturbation_seed"],
        )
    }
    prior_collision = sorted(formal_integer_seeds & prior_recorded_seeds())
    if prior_collision:
        raise SystemExit(f"formal seeds collide with prior recorded seeds: {prior_collision}")

    source_groups: dict[str, list[dict[str, Any]]] = {}
    for task in TASKS:
        rows = []
        paths = source_partition(task, OFFICIAL_ROOT, "FORMAL")
        for group, path in zip(manifest["groups"][task], paths):
            rows.append(
                {
                    **group,
                    "source_replay_tree_sha256": replay_tree_sha256(path),
                }
            )
        source_groups[task] = rows

    tracked_hashes = {name: sha256_file(path) for name, path in CONFIG_PATHS.items()}
    freeze = {
        "schema": "tactile3d-unit.s4-3-pd-formal-protocol-freeze.v1",
        "stage": "PD5",
        "status": "FROZEN_BEFORE_FORMAL_ACQUISITION",
        "pilot_gate": pilot,
        "expert_source_class": {
            task: expert_contract["source_class_per_task"][task] for task in TASKS
        },
        "expert_code_hashes": current_code_hashes,
        "task_success_hashes": task_success_hashes,
        "tracked_config_sha256": tracked_hashes,
        "formal_attempt_manifest_sha256": sha256_file(manifest_path),
        "source_dataset_revision": expert_contract["source_dataset_revision"],
        "dexjoco_revision": expert_contract["dexjoco_revision"],
        "source_groups": source_groups,
        "attempts": manifest["attempts"],
        "counts": {
            "source_groups_per_task": 25,
            "attempts_per_group": 5,
            "attempts_per_task": 125,
            "total_attempts": 375,
            "train_groups_per_task": 20,
            "train_attempts_per_task": 100,
            "dev_groups_per_task": 5,
            "dev_attempts_per_task": 25,
        },
        "success_thresholds_per_task": {
            "train_successes_min": 80,
            "dev_successes_min": 20,
            "total_successes_min": 100,
            "total_success_rate_min": 0.8,
        },
        "seed_freshness": {
            "pilot_formal_group_overlap": group_overlap,
            "pilot_formal_seed_tuple_overlap": 0,
            "prior_recorded_integer_seed_collisions": prior_collision,
            "environment_seed_min": min(row["environment_seed"] for row in manifest["attempts"]),
            "environment_seed_max": max(row["environment_seed"] for row in manifest["attempts"]),
            "perturbation_seed_min": min(row["perturbation_seed"] for row in manifest["attempts"]),
            "perturbation_seed_max": max(row["perturbation_seed"] for row in manifest["attempts"]),
            "status": "PASS",
        },
        "randomization": configs["formal_acquisition"]["randomization"],
        "failure_policy": configs["formal_acquisition"]["failure_policy"],
        "action_contract": configs["protocol"]["policy_action_contract"],
        "student_observation_contract": configs["protocol"]["policy_observation_contract"],
        "s4_2_immutable_checkpoint_hashes": starting[
            "immutable_checkpoint_hashes_before"
        ],
        "expert_controller_changes_after_freeze": "FORBIDDEN",
        "formal_attempts_present_at_freeze": 0,
    }
    write_json(ARTIFACT_ROOT / "formal_protocol_freeze.json", freeze)
    print(
        json.dumps(
            {
                "stage": "PD5",
                "status": freeze["status"],
                "attempts": len(freeze["attempts"]),
                "source_groups": sum(len(rows) for rows in source_groups.values()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
