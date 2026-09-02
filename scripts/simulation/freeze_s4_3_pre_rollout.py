#!/usr/bin/env python3
"""Freeze checkpoint, reset, runtime, and statistical identity before R12."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_3_training import (  # noqa: E402
    atomic_json,
    canonical_sha256,
    read_json,
    sha256_file,
)
from scripts.simulation.freeze_s4_3_restart_protocol import (  # noqa: E402
    CHECKPOINTS,
    tracked_s4_2_hashes,
)

ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_restart"
TRAINING = ARTIFACT_ROOT / "training_jobs.json"
OFFLINE = ARTIFACT_ROOT / "offline_dev_evaluation.json"
POLICY = ROOT / "configs/simulation/s4_3_restart_policy_protocol.json"
ACT = ROOT / "configs/simulation/s4_3_restart_act_protocol.json"
EVALUATION = ROOT / "configs/simulation/s4_3_policy_eval_v1.json"
SUCCESS = ARTIFACT_ROOT / "task_success_contract.json"
STATISTICS_CODE = ROOT / "scripts/simulation/analyze_s4_3_closed_loop.py"
ROLLOUT_CODE = (
    ROOT / "scripts/simulation/run_s4_3_policy_rollouts_dex.py",
    ROOT / "scripts/simulation/serve_s4_3_act_policy.py",
    ROOT / "scripts/simulation/run_s4_3_rollout_job.py",
    ROOT / "scripts/simulation/run_s4_3_rollout_queue.py",
)


def parameter_counts(jobs: list[dict[str, Any]]) -> dict[str, int]:
    counts = {}
    for row in jobs:
        checkpoint = __import__("torch").load(
            ROOT / row["checkpoint"], map_location="cpu", weights_only=False
        )
        counts[row["variant"]] = int(checkpoint["trainable_parameter_count"])
    return counts


def main() -> None:
    if (ROOT / ".local/logs/simulation/s4_3_restart/closed_loop").exists():
        if any((ROOT / ".local/logs/simulation/s4_3_restart/closed_loop").rglob("metadata.json")):
            raise RuntimeError("scientific rollout metadata exists before pre-rollout freeze")
    training = read_json(TRAINING)
    offline = read_json(OFFLINE)
    policy = read_json(POLICY)
    evaluation = read_json(EVALUATION)
    if training.get("status") != "PASS" or training.get("canonical_checkpoints") != 36:
        raise RuntimeError("R9 complete training gate has not passed")
    if offline.get("status") != "PASS" or offline["hard_sanity"]["status"] != "PASS":
        raise RuntimeError("R10 offline hard sanity has not passed")
    checkpoint_hashes = {}
    identities = set()
    for row in training["jobs"]:
        identity = (row["task"], row["variant"], row["training_seed"])
        if identity in identities:
            raise RuntimeError("duplicate canonical checkpoint identity")
        identities.add(identity)
        path = ROOT / row["checkpoint"]
        actual = sha256_file(path)
        if actual != row["checkpoint_sha256"]:
            raise RuntimeError("canonical checkpoint changed before rollout")
        checkpoint_hashes["/".join(map(str, identity))] = actual
    if len(checkpoint_hashes) != 36:
        raise RuntimeError("pre-rollout checkpoint count is not 36")
    s4_2 = {name: sha256_file(ROOT / path) for name, path in CHECKPOINTS.items()}
    expected = {name: policy["s4_2_immutable"][name] for name in CHECKPOINTS}
    if s4_2 != expected:
        raise RuntimeError("S4.2 checkpoint mutation before rollout")
    starting = read_json(ARTIFACT_ROOT / "starting_integrity.json")
    configs = tracked_s4_2_hashes()
    if configs != starting["s4_2_tracked_config_sha256_before"]:
        raise RuntimeError("S4.2 tracked config mutation before rollout")
    adapter = ROOT / "gr00t/simulation/dexjoco_adapter.py"
    manifest = {
        "schema": "tactile3d-unit.s4-3-pre-rollout-freeze.v1",
        "stage": "R11",
        "checkpoint_sha256": dict(sorted(checkpoint_hashes.items())),
        "checkpoint_set_sha256": canonical_sha256(checkpoint_hashes),
        "checkpoint_count": len(checkpoint_hashes),
        "checkpoint_parameter_counts": parameter_counts(training["jobs"]),
        "evaluation_reset_config": "configs/simulation/s4_3_policy_eval_v1.json",
        "evaluation_reset_config_sha256": sha256_file(EVALUATION),
        "evaluation_resets": evaluation["resets"],
        "evaluation_reset_count": len(evaluation["resets"]),
        "success_predicates": read_json(SUCCESS)["tasks"],
        "success_contract_sha256": sha256_file(SUCCESS),
        "timeouts": policy["timeouts"],
        "replan_stride": policy["action_contract"]["replan_stride"],
        "action_adapter": "gr00t.simulation.dexjoco_adapter.policy_action_to_env_action",
        "action_adapter_source_sha256": sha256_file(adapter),
        "warmup": policy["warmup"],
        "primary_metrics": [
            policy["primary_endpoint"],
            policy["preregistered_secondary"],
            *policy["secondary_metrics"],
        ],
        "statistical_code": str(STATISTICS_CODE.relative_to(ROOT)),
        "statistical_code_sha256": sha256_file(STATISTICS_CODE),
        "rollout_code_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in ROLLOUT_CODE
        },
        "policy_protocol_sha256": sha256_file(POLICY),
        "act_protocol_sha256": sha256_file(ACT),
        "s4_2_checkpoint_sha256": s4_2,
        "s4_2_tracked_config_sha256": configs,
        "s4_2_vision_identity_sha256": sha256_file(
            ROOT / ".local/artifacts/simulation/s4_2_formal/vision_identity.json"
        ),
        "rollout_performance_seen": False,
        "checkpoint_selection_complete": True,
        "training_complete": True,
        "retuning_after_rollout_start_allowed": False,
        "status": "PASS",
    }
    atomic_json(ARTIFACT_ROOT / "pre_rollout_freeze.json", manifest)
    training_manifest = {
        "schema": "tactile3d-unit.s4-3-act-training-manifest.v1",
        "stage": "R9",
        "jobs": training["jobs"],
        "expected_jobs": 36,
        "completed_jobs": 36,
        "hyperparameters": read_json(ACT)["optimizer"],
        "budget": read_json(ACT)["budget"],
        "status": "PASS",
    }
    checkpoint_manifest = {
        "schema": "tactile3d-unit.s4-3-act-checkpoint-manifest.v1",
        "stage": "R9.6",
        "checkpoints": checkpoint_hashes,
        "checkpoint_set_sha256": canonical_sha256(checkpoint_hashes),
        "count": 36,
        "selection": "lowest POLICY_DEV normalized full-chunk Action L1",
        "closed_loop_selection_used": False,
        "status": "PASS",
    }
    atomic_json(ARTIFACT_ROOT / "act_training_manifest.json", training_manifest)
    atomic_json(ARTIFACT_ROOT / "act_checkpoint_manifest.json", checkpoint_manifest)
    print(
        json.dumps(
            {
                "checkpoint_count": 36,
                "rollout_performance_seen": False,
                "status": "PASS",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
