#!/usr/bin/env python3
"""Audit the canonical S4.3-RR rollout matrix before statistical analysis."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_3_training import (  # noqa: E402
    TASKS,
    VARIANTS,
    atomic_json,
    read_json,
    sha256_file,
)

ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_rr"
ROLLOUTS = ARTIFACT_ROOT / "closed_loop_rollouts.json"
RETRIES = ARTIFACT_ROOT / "infrastructure_retry_log.json"
FREEZE = ARTIFACT_ROOT / "pre_rollout_freeze_v2.json"
RESET_CONFIG = ROOT / "configs/simulation/s4_3_policy_eval_v1.json"
SUCCESS_CONTRACT = ROOT / ".local/artifacts/simulation/s4_3_restart/task_success_contract.json"
OUTPUT = ARTIFACT_ROOT / "rollout_completeness.json"


def identity(row: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row["task"]),
        str(row["variant"]),
        int(row["training_seed"]),
        str(row["evaluation_reset_id"]),
    )


def main() -> None:
    source = read_json(ROLLOUTS)
    retries = read_json(RETRIES)
    freeze = read_json(FREEZE)
    reset_config = read_json(RESET_CONFIG)
    rows = source.get("rollouts", [])
    jobs = source.get("jobs", [])
    reset_lookup = {
        (row["task"], row["evaluation_reset_id"]): row
        for row in reset_config["resets"]
    }
    expected = {
        (task, variant, seed, reset["evaluation_reset_id"])
        for task in TASKS
        for variant in VARIANTS
        for seed in range(3)
        for reset in reset_config["resets"]
        if reset["task"] == task
    }
    identities = [identity(row) for row in rows]
    observed = set(identities)
    duplicates = sorted(key for key, count in Counter(identities).items() if count != 1)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)

    reset_fields = (
        "reset_index",
        "seed_namespace",
        "reset_seed",
        "environment_seed",
        "visual_randomization_seed",
        "dynamics_randomization",
        "perturbation_seed",
        "overlap_with_prior_sources",
    )
    reset_mismatches: list[dict[str, Any]] = []
    for row in rows:
        expected_reset = reset_lookup.get((row["task"], row["evaluation_reset_id"]))
        if expected_reset is None:
            continue
        mismatched = {
            field: {"expected": expected_reset[field], "actual": row.get(field)}
            for field in reset_fields
            if row.get(field) != expected_reset[field]
        }
        if mismatched:
            reset_mismatches.append({"identity": list(identity(row)), "fields": mismatched})

    checkpoint_mismatches = []
    for row in rows:
        checkpoint_key = f"{row['task']}/{row['variant']}/{row['training_seed']}"
        expected_hash = freeze["checkpoint_sha256"].get(checkpoint_key)
        if row.get("checkpoint_sha256") != expected_hash:
            checkpoint_mismatches.append(
                {
                    "identity": list(identity(row)),
                    "expected": expected_hash,
                    "actual": row.get("checkpoint_sha256"),
                }
            )

    expected_job_identities = {
        (task, variant, seed) for task in TASKS for variant in VARIANTS for seed in range(3)
    }
    job_identities = [
        (str(row["task"]), str(row["variant"]), int(row["training_seed"])) for row in jobs
    ]
    job_counts = Counter(job_identities)
    job_duplicates = sorted(key for key, count in job_counts.items() if count != 1)
    missing_jobs = sorted(expected_job_identities - set(job_identities))
    unexpected_jobs = sorted(set(job_identities) - expected_job_identities)
    job_contract_failures = []
    for job in jobs:
        job_key = (job["task"], job["variant"], int(job["training_seed"]))
        metadata = job.get("metadata", [])
        expected_hash = freeze["checkpoint_sha256"].get("/".join(map(str, job_key)))
        valid = (
            job.get("status") == "PASS"
            and job.get("run_kind") == "scientific"
            and job.get("scientific_result") is True
            and job.get("rollouts") == 30
            and len(metadata) == 30
            and job.get("checkpoint_sha256") == expected_hash
            and job.get("logical_device") == "cuda:0"
            and job.get("endpoint", {}).get("encoded_length", 81) <= 80
            and job.get("endpoint", {}).get("cleanup_pass") is True
        )
        if not valid:
            job_contract_failures.append(list(job_key))

    warmup = freeze["warmup"]
    timeouts = freeze["timeouts"]["tasks"]
    scientific_outcomes = {"SUCCESS", "TIMEOUT", "ENV_TERMINATION_FAILURE", "INVALID_ACTION", "NUMERIC_FAILURE", "SIMULATION_EXCEPTION"}
    row_contract_failures: list[dict[str, Any]] = []
    artifact_failures: list[dict[str, Any]] = []
    for row in rows:
        adapter = row.get("action_adapter", {})
        vision = row.get("vision_transport", {})
        valid = (
            row.get("run_kind") == "scientific"
            and row.get("scientific_result") is True
            and row.get("logging_status") == "PASS"
            and row.get("termination_reason") in scientific_outcomes
            and bool(row.get("success")) == (row.get("termination_reason") == "SUCCESS")
            and row.get("timeout_steps") == timeouts[row["task"]]["timeout_steps"]
            and row.get("warmup_samples") == warmup["control_steps"]
            # The frozen runtime executes 25 x 20 ms hold actions and records
            # the initial frame as well, yielding the registered 26 samples.
            and row.get("warmup_action_steps") == warmup["control_steps"] - 1
            and row.get("warmup_duration_sec") == warmup["duration_sec"]
            and row.get("warmup_counted_in_timeout") is warmup["counted_in_timeout"]
            and row.get("replan_stride") == freeze["replan_stride"] == 5
            and row.get("action_chunk_shape") == [27, 22]
            and adapter.get("source") == freeze["action_adapter"]
            and adapter.get("policy_shape") == [22]
            and adapter.get("environment_shape") == [23]
            and adapter.get("status") == "PASS"
            and vision.get("source") == "current RGB I_t only"
            and vision.get("matches_policy_expert_storage") is True
            and row.get("actual_future_contact_read") is False
            and row.get("future_observation_read") is False
            and row.get("expert_action_read") is False
            and row.get("uncertainty", {}).get("invoked") is False
            and row.get("uncertainty", {}).get("intervention") is False
            and row.get("runtime_exceptions") == []
            and row.get("physical_gpu") in (0, 1, 2, 3)
            and row.get("logical_device") == "cuda:0"
            and bool(row.get("socket_endpoint_short_hash"))
        )
        if not valid:
            row_contract_failures.append(
                {"identity": list(identity(row)), "rollout_id": row.get("rollout_id")}
            )
        for path_field, hash_field in (("trace", "trace_sha256"), ("raw_video", "raw_video_sha256")):
            path = ROOT / row[path_field]
            actual = sha256_file(path) if path.is_file() else None
            if actual != row.get(hash_field):
                artifact_failures.append(
                    {
                        "rollout_id": row.get("rollout_id"),
                        "field": path_field,
                        "expected": row.get(hash_field),
                        "actual": actual,
                    }
                )

    reset_sets = {}
    for task in TASKS:
        expected_ids = {
            reset["evaluation_reset_id"] for reset in reset_config["resets"] if reset["task"] == task
        }
        reset_sets[task] = all(
            {
                row["evaluation_reset_id"]
                for row in rows
                if row["task"] == task
                and row["variant"] == variant
                and int(row["training_seed"]) == seed
            }
            == expected_ids
            for variant in VARIANTS
            for seed in range(3)
        )

    termination_counts = dict(Counter(row.get("termination_reason") for row in rows))
    failed_scientific_rollouts = sum(not bool(row.get("success")) for row in rows)
    controller_restarts = retries.get("controller_restarts", [])
    controller_restarts_safe = all(
        event.get("partial_scientific_rollout_metadata") == 0
        and (
            event.get("classification") != "PRE_RESET_INFRASTRUCTURE_RELAUNCH"
            or (
                event.get("same_checkpoint")
                and event.get("same_code")
                and event.get("same_ordered_reset_stream")
                and event.get("same_scientific_configuration")
            )
        )
        for event in controller_restarts
    )
    gates = {
        "source_pass": source.get("status") == "PASS",
        "canonical_rollout_count": len(rows) == source.get("completed_rollouts") == 1080,
        "canonical_job_count": len(jobs) == source.get("completed_jobs") == 36,
        "unique_canonical_identities": not duplicates and len(observed) == 1080,
        "identity_set_exact": not missing and not unexpected and observed == expected,
        "job_identity_set_exact": not job_duplicates and not missing_jobs and not unexpected_jobs,
        "frozen_reset_fields_exact": not reset_mismatches,
        "same_ordered_reset_specs": all(reset_sets.values()),
        "checkpoint_hashes_exact": not checkpoint_mismatches,
        "job_contracts": not job_contract_failures,
        "rollout_contracts": not row_contract_failures,
        "raw_artifacts_present_and_hashed": not artifact_failures,
        "evaluation_config_hash": sha256_file(RESET_CONFIG) == freeze["evaluation_reset_config_sha256"],
        "success_contract_hash": sha256_file(SUCCESS_CONTRACT) == freeze["success_contract_sha256"],
        "statistics_hash": sha256_file(ROOT / freeze["statistical_code"]) == freeze["statistical_code_sha256"],
        "runtime_code_hashes": all(
            sha256_file(ROOT / path) == expected
            for path, expected in freeze["rollout_code_sha256"].items()
        ),
        "policy_dependency_hashes": all(
            sha256_file(ROOT / path) == expected
            for path, expected in freeze["policy_dependency_code_sha256"].items()
        ),
        "s4_2_precheck": read_json(ARTIFACT_ROOT / "s4_2_immutability.json").get("status") == "PASS",
        "infrastructure_retry_log": retries.get("status") == "PASS" and not retries.get("retries"),
        "controller_restarts_scientifically_safe": controller_restarts_safe,
        "scientific_failures_retained": failed_scientific_rollouts + termination_counts.get("SUCCESS", 0) == 1080,
        "reset_replacements_zero": not reset_mismatches and not missing and not unexpected,
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    result = {
        "schema": "tactile3d-unit.s4-3-rr-rollout-completeness.v1",
        "stage": "RR7",
        "expected_rollouts": 1080,
        "canonical_rollouts": len(rows),
        "expected_jobs": 36,
        "canonical_jobs": len(jobs),
        "expected_unique_identities": len(expected),
        "canonical_unique_identities": len(observed),
        "duplicate_canonical_identities": [list(key) for key in duplicates],
        "missing_identities": [list(key) for key in missing],
        "unexpected_identities": [list(key) for key in unexpected],
        "missing_jobs": [list(key) for key in missing_jobs],
        "unexpected_jobs": [list(key) for key in unexpected_jobs],
        "duplicate_jobs": [list(key) for key in job_duplicates],
        "reset_field_mismatches": reset_mismatches,
        "checkpoint_mismatches": checkpoint_mismatches,
        "job_contract_failures": job_contract_failures,
        "rollout_contract_failures": row_contract_failures,
        "raw_artifact_failures": artifact_failures,
        "same_reset_specs_by_task": reset_sets,
        "termination_counts": termination_counts,
        "scientific_successes": termination_counts.get("SUCCESS", 0),
        "scientific_failures_retained": failed_scientific_rollouts,
        "canonical_scientific_retries": len(retries.get("retries", [])),
        "controller_restarts": controller_restarts,
        "pre_reset_infrastructure_relaunches": retries.get("exact_pre_reset_relaunches", 0),
        "reset_replacements": 0 if gates["reset_replacements_zero"] else None,
        "gates": gates,
        "decision": "PASS" if status == "PASS" else "S4_3_RR_ROLLOUT_COMPLETENESS_FAIL",
        "status": status,
    }
    atomic_json(OUTPUT, result)
    print(json.dumps({"gates": gates, "status": status}, sort_keys=True))
    if status != "PASS":
        raise SystemExit("S4_3_RR_ROLLOUT_COMPLETENESS_FAIL")


if __name__ == "__main__":
    main()
