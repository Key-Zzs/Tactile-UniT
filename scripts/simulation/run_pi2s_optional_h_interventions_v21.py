#!/usr/bin/env python3
"""Correct the v2 H-audit hash domain without changing its scientific runtime.

Three completed v2 episodes are re-audited byte-for-byte and never rerun.  This
runner executes only the 39 protocol tuples that have no completed episode.
The original v2 FAIL records remain immutable evidence.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import string
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.simulation import run_pi2s_diagnostic_rollouts as base  # noqa: E402
from scripts.simulation import run_pi2s_optional_h_interventions as v1  # noqa: E402
from scripts.simulation import run_pi2s_optional_h_interventions_v2 as v2  # noqa: E402

PI2S_ROOT = base.PI2S_ROOT
DECISION = ROOT / "configs/simulation/pi2s/optional_h_intervention_audit_recovery_v21.json"
PERSISTED_DECISION = PI2S_ROOT / "artifacts/optional_h_intervention_audit_recovery_v21.json"
V2_MANIFEST = PI2S_ROOT / "artifacts/diagnostic_rollout_optional_recovery_v2_manifest.json"
V2_PROGRESS = PI2S_ROOT / "diagnostics/rollouts_optional_h_v2/progress.json"
V2_SUPERVISOR_LOG = PI2S_ROOT / "logs/rollouts_optional_h_v2/supervisor.log"
V2_FAILED_RESUME_SNAPSHOT = (
    PI2S_ROOT / "artifacts/diagnostic_rollout_optional_v2_failed_resume_state.json"
)
MANIFEST = PI2S_ROOT / "artifacts/diagnostic_rollout_optional_recovery_v21_manifest.json"
RESULTS = PI2S_ROOT / "artifacts/diagnostic_rollout_optional_results_v21.json"
RUN_ROOT = PI2S_ROOT / "diagnostics/rollouts_optional_h_v21"
LOG_ROOT = PI2S_ROOT / "logs/rollouts_optional_h_v21"
CACHE_ROOT = PI2S_ROOT / "cache/rollouts_optional_h_v21"
PROGRESS = RUN_ROOT / "progress.json"
RESUME = PI2S_ROOT / "resume_state.json"

AUDIT_VERSION = "OPTIONAL_H_AUDIT_V21_HASH_DOMAINS_SEPARATED"
SCHEMA_MANIFEST = "tactile3d-unit.s4-3-pi2s-optional-h-recovery-manifest.v2.1"
SCHEMA_AUDIT = "tactile3d-unit.s4-3-pi2s-h-delivery-audit.v2.1"
SCHEMA_RESULTS = "tactile3d-unit.s4-3-pi2s-optional-h-rollout-results.v2.1"
SCHEMA_TUPLE = v2.SCHEMA_TUPLE
FOCUS_KEYS = v2.FOCUS_KEYS
FOCUS_IDS = v2.FOCUS_IDS
CONDITIONS = v2.CONDITIONS
REUSED_V1_TUPLES = 6
REAUDITED_V2_TUPLES = 3
EXPECTED_RECOVERY_TUPLES = 39
EXPECTED_FINAL_TUPLES = 48
TUPLES_PER_BLOCK = 13

BASE_SCRIPT = v2.BASE_SCRIPT
V1_SCRIPT = v2.V1_SCRIPT
V2_SCRIPT = Path(v2.__file__).resolve()
BASE_VALIDATE_TUPLE = v2.BASE_VALIDATE_TUPLE
BASE_PROGRESS_PAYLOAD = v2.BASE_PROGRESS_PAYLOAD
BASE_AGGREGATE_RESULTS = v2.BASE_AGGREGATE_RESULTS
BASE_RUN_TUPLE_ATTEMPT = v2.BASE_RUN_TUPLE_ATTEMPT


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in string.hexdigits for character in value)
    )


def _resolve_pi2s_symbolic(value: str) -> Path:
    prefix = "$PI2S_ROOT/"
    if not value.startswith(prefix):
        raise base.ContractError(f"not a PI2S symbolic path: {value}")
    path = PI2S_ROOT / value.removeprefix(prefix)
    root = PI2S_ROOT.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise base.ContractError(f"PI2S symbolic path escaped the root: {value}")
    return path


def _resolve_new_pi2s_symbolic(value: str) -> Path:
    prefix = "$PI2S_ROOT/"
    if not value.startswith(prefix):
        raise base.ContractError(f"not a PI2S symbolic output path: {value}")
    path = PI2S_ROOT / value.removeprefix(prefix)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    root = PI2S_ROOT.resolve(strict=True)
    parent = path.parent.resolve(strict=True)
    if not parent.is_relative_to(root):
        raise base.ContractError(f"PI2S symbolic output escaped the root: {value}")
    return path


def _load_contract() -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    protocol, protocol_sha, v2_decision = v2._load_contract()
    decision = base.read_json(DECISION)
    if (
        decision.get("status") != "AUTHORIZED_AUDIT_ONLY_CORRECTION_AFTER_V2_FALSE_POSITIVE_STOP"
        or decision.get("decision")
        != "PRESERVE_V2_FAILURES_REAUDIT_THREE_COMPLETED_EPISODES_WITH_HASH_DOMAINS_SEPARATED_AND_RUN_ONLY_39_UNEXECUTED_TUPLES"
    ):
        raise base.ContractError("v2.1 audit correction is not authorized")
    cause = decision.get("root_cause", {})
    if (
        cause.get("classification") != "AUDIT_HASH_DOMAIN_MISMATCH_CONFIRMED"
        or cause.get("scientific_runtime_changed") is not False
        or cause.get("post_result_tolerance_relaxation") is not False
    ):
        raise base.ContractError("v2.1 root-cause contract drifted")
    budget = decision.get("recovery_budget", {})
    if (
        budget.get("valid_v1_protocol_tuples_reused") != REUSED_V1_TUPLES
        or budget.get("completed_v2_episodes_reaudited_without_rerun") != REAUDITED_V2_TUPLES
        or budget.get("v21_unexecuted_protocol_tuples_to_run") != EXPECTED_RECOVERY_TUPLES
        or budget.get("final_unique_optional_protocol_tuples") != EXPECTED_FINAL_TUPLES
        or budget.get("final_total_s5_protocol_tuples_including_base") != 120
        or budget.get("further_expansion_authorized") is not False
    ):
        raise base.ContractError("v2.1 recovery budget drifted")
    guardrails = decision.get("guardrails", {})
    if (
        guardrails.get("completed_v2_episodes_may_be_rerun") is not False
        or guardrails.get("v2_failure_files_may_be_modified_or_removed") is not False
        or guardrails.get("v2_attempt_2_may_be_used") is not False
        or guardrails.get("scientific_runtime_may_change") is not False
        or guardrails.get("performance_selection") is not False
        or guardrails.get("formal_track_a_score_replacement") is not False
        or guardrails.get("best_intervention_deployment_selection") is not False
        or guardrails.get("training_or_optimizer_updates") != 0
        or guardrails.get("checkpoint_writes") != 0
        or guardrails.get("real_robot_allowed") is not False
    ):
        raise base.ContractError("v2.1 guardrails drifted")
    return protocol, protocol_sha, v2_decision, decision


def _completed_v2_keys(decision: Mapping[str, Any]) -> set[str]:
    rows = decision["completed_v2_episodes_for_reaudit"]
    keys = {str(row["v2_tuple_key"]) for row in rows}
    if len(rows) != REAUDITED_V2_TUPLES or len(keys) != REAUDITED_V2_TUPLES:
        raise base.ContractError("v2 completed tuple list is not exact")
    return keys


def build_recovery_tuples(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    frozen, _protocol_sha, _v2_decision, decision = _load_contract()
    if dict(protocol) != frozen:
        raise base.ContractError("v2.1 tuple builder received a non-frozen protocol")
    v2_manifest = base.read_json(V2_MANIFEST)
    completed = _completed_v2_keys(decision)
    rows = [dict(row) for row in v2_manifest["tuples"] if row["tuple_key"] not in completed]
    if (
        len(rows) != EXPECTED_RECOVERY_TUPLES
        or len({row["tuple_key"] for row in rows}) != EXPECTED_RECOVERY_TUPLES
        or completed & {row["tuple_key"] for row in rows}
        or len(completed | {row["tuple_key"] for row in rows}) != 42
    ):
        raise base.ContractError("v2.1 recovery partition failed")
    return rows


def _verify_v2_evidence(v2_decision: Mapping[str, Any], decision: Mapping[str, Any]) -> None:
    v2._verify_v1_evidence(v2_decision)
    v2_info = decision["v2"]
    if (
        base.sha256_file(V2_SCRIPT) != v2_info["runner_sha256"]
        or base.sha256_file(V2_MANIFEST) != v2_info["manifest_sha256"]
        or base.sha256_file(V2_PROGRESS) != v2_info["terminal_progress_sha256"]
        or base.sha256_file(V2_SUPERVISOR_LOG) != v2_info["supervisor_log_sha256"]
        or v2.RESULTS.exists()
    ):
        raise base.ContractError("v2 stopped-wave evidence drifted")
    failed_resume = V2_FAILED_RESUME_SNAPSHOT if V2_FAILED_RESUME_SNAPSHOT.is_file() else RESUME
    if base.sha256_file(failed_resume) != v2_info["failed_resume_state_sha256"]:
        raise base.ContractError("v2 failed resume-state evidence drifted")
    v2_manifest = base.read_json(V2_MANIFEST)
    observed_sources = {
        relative: base.sha256_file(ROOT / relative) for relative in v2_manifest["source_sha256"]
    }
    if observed_sources != v2_manifest["source_sha256"]:
        raise base.ContractError("v2 source evidence drifted")
    manifest_index = {row["tuple_key"]: row for row in v2_manifest["tuples"]}
    for row in decision["completed_v2_episodes_for_reaudit"]:
        artifact = _resolve_pi2s_symbolic(row["tuple_artifact"])
        failure = _resolve_pi2s_symbolic(row["original_failure"])
        if (
            base.sha256_file(artifact) != row["tuple_artifact_sha256"]
            or base.sha256_file(failure) != row["original_failure_sha256"]
            or row.get("attempt") != 1
            or row.get("canonical_only_if_corrected_audit_passes") is not True
        ):
            raise base.ContractError(f"completed v2 evidence drifted: {row['v2_tuple_key']}")
        payload = base.read_json(artifact)
        failure_payload = base.read_json(failure)
        if (
            payload.get("status") != "PASS"
            or payload.get("tuple") != manifest_index.get(row["v2_tuple_key"])
            or payload.get("steps") != row["steps"]
            or payload.get("success") is not row["success"]
            or failure_payload.get("status") != "FAIL"
            or failure_payload.get("type") != "ContractError"
            or "contradictory correct H at physical tick" not in failure_payload.get("message", "")
        ):
            raise base.ContractError(f"completed v2 payload drifted: {row['v2_tuple_key']}")
        if (artifact.parents[1] / "attempt_2").exists():
            raise base.ContractError(f"prohibited v2 attempt_2 exists: {row['v2_tuple_key']}")


def _audit_path(tuple_artifact: Path) -> Path:
    return tuple_artifact.with_name("h_intervention_audit_v21.json")


def _audit_failure_path(tuple_artifact: Path) -> Path:
    return tuple_artifact.with_name("h_intervention_audit_v21_failure.json")


def build_corrected_audit(tuple_artifact: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    payload = base.read_json(tuple_artifact)
    events = payload.get("observation_events", [])
    chunks = payload.get("action_chunks", [])
    steps = payload.get("step_telemetry", [])
    if not events or not chunks or not steps:
        raise base.ContractError("v2.1 tuple lacks observation/query/step evidence")
    condition = str(expected["h_condition"])
    event_ticks = [int(row["control_step"]) for row in events]
    query_ticks = [int(row["observation_timestamp"]) for row in chunks]
    post_by_tick: dict[int, str] = {}
    post_sequence_exact = True
    for row in steps:
        record = row.get("post_step_correct_h")
        if condition == "train_mean":
            post_sequence_exact &= record is None
            continue
        if record is None:
            post_sequence_exact = False
            continue
        tick = int(record["control_tick"])
        sha = record.get("sha256")
        post_sequence_exact &= (
            tick == int(row["control_step"]) + 1
            and _is_sha256(sha)
            and record.get("shape") == [256]
            and record.get("dtype") == "float32"
            and record.get("sent_to_policy_at_this_step") is False
            and np.isfinite(float(record["l2"]))
            and tick not in post_by_tick
        )
        post_by_tick[tick] = str(sha)
    if condition == "same_episode_lag5":
        tick_zero = next((row for row in events if int(row["control_step"]) == 0), None)
        if tick_zero is None:
            raise base.ContractError("v2.1 lag audit lacks tick-zero observation")
        post_by_tick[0] = str(tick_zero["delivered_h_sha256"])
    mean_sha = v2._sha256_array(v1.train_mean_h())
    event_gates = []
    bootstrap_count = 0
    same_tick_runtime_checks = 0
    for row in events:
        current = int(row["control_step"])
        delivered = row.get("delivered_h_sha256")
        correct_structured = row.get("correct_h_sha256")
        common = (
            row.get("diagnostic_runtime_version") == v2.RUNTIME_VERSION
            and row.get("h_condition") == condition
            and row.get("h_sha256") == delivered
            and row.get("lag_clock") == "raw_50hz_physical_control_tick"
            and _is_sha256(delivered)
            and _is_sha256(correct_structured)
            and np.isfinite(float(row["correct_h_l2"]))
            and np.isfinite(float(row["delivered_h_l2"]))
        )
        if condition == "train_mean":
            valid = (
                row.get("requested_control_tick") is None
                and row.get("actual_control_tick") is None
                and row.get("bootstrap_affected") is False
                and row.get("lag_source_correct_h_sha256") is None
                and delivered == mean_sha
            )
        else:
            requested = current - 5
            actual = max(0, requested)
            bootstrap = requested < 0
            bootstrap_count += int(bootstrap)
            source_raw_sha = post_by_tick.get(actual)
            valid = (
                row.get("requested_control_tick") == requested
                and row.get("actual_control_tick") == actual
                and row.get("bootstrap_affected") is bootstrap
                and row.get("lag_source_correct_h_sha256") == source_raw_sha
                and delivered == source_raw_sha
            )
            if current > 0 and current in post_by_tick:
                same_tick_runtime_checks += 1
        event_gates.append(common and valid)
    delivered_by_tick = {int(row["control_step"]): row["delivered_h_sha256"] for row in events}
    query_gates = []
    for row in chunks:
        control = row.get("sampling_control", {})
        tick = int(row["observation_timestamp"])
        query_gates.append(
            control.get("h_condition") == condition
            and control.get("delivered_h_sha256") == delivered_by_tick.get(tick)
            and np.isfinite(float(control.get("delivered_h_l2")))
        )
    raw_tick_gate = post_sequence_exact and (
        condition == "train_mean"
        or (
            sorted(post_by_tick) == list(range(len(steps) + 1))
            and same_tick_runtime_checks == sum(tick > 0 for tick in event_ticks)
        )
    )
    gates = {
        "tuple_identity_exact": payload.get("tuple") == dict(expected),
        "base_tuple_integrity_pass": payload.get("status") == "PASS"
        and all(value == "PASS" for value in payload.get("gates", {}).values()),
        "hash_domains_separated_not_compared_cross_domain": True,
        "frozen_runtime_in_memory_array_equality_gate_completed": condition == "train_mean"
        or same_tick_runtime_checks == sum(tick > 0 for tick in event_ticks),
        "all_observation_events_conditioned_exactly": bool(event_gates) and all(event_gates),
        "all_policy_queries_received_audited_h": bool(query_gates) and all(query_gates),
        "policy_queries_match_observation_events_one_to_one": query_ticks == event_ticks,
        "raw_tick_buffer_complete_without_train_mean_extra_queries": raw_tick_gate,
        "training_only_targets_absent": all(
            not row.get("training_only_fields_sent") for row in chunks
        ),
    }
    return {
        "schema": SCHEMA_AUDIT,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "tuple_key": expected["tuple_key"],
        "protocol_tuple_key": expected["protocol_tuple_key"],
        "tuple_identity_sha256": expected["tuple_identity_sha256"],
        "diagnostic_runtime_version": v2.RUNTIME_VERSION,
        "audit_version": AUDIT_VERSION,
        "h_condition": condition,
        "observation_event_count": len(events),
        "policy_query_count": len(chunks),
        "physical_step_count": len(steps),
        "post_step_h_query_count": len(post_by_tick) - int(condition == "same_episode_lag5"),
        "same_tick_in_memory_array_equality_checks": same_tick_runtime_checks,
        "bootstrap_affected_observation_count": bootstrap_count,
        "hash_domains": {
            "observation_correct_h_sha256": "BASE_DTYPE_SHAPE_BYTES",
            "post_step_and_delivered_h_sha256": "V2_RAW_BYTES",
            "cross_domain_hash_equality_required": False,
        },
        "original_v2_false_positive_failure_preserved": tuple_artifact.with_name(
            "h_intervention_audit_v2_failure.json"
        ).is_file(),
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "optimizer_updates": 0,
        "checkpoint_writes": 0,
    }


def evaluate_tuple(
    tuple_spec_path: Path,
    contact_socket: Path,
    output: Path,
    diagnostics: Path,
    artifact: Path,
    port: int,
) -> None:
    audit_path = _audit_path(artifact)
    failure_path = _audit_failure_path(artifact)
    if audit_path.exists() or failure_path.exists():
        raise FileExistsError("v2.1 audit output already exists")
    base.evaluate_tuple(tuple_spec_path, contact_socket, output, diagnostics, artifact, port)
    spec = base.read_json(tuple_spec_path)
    try:
        audit = build_corrected_audit(artifact, spec)
    except base.ContractError as error:
        base.publish_json_no_clobber(
            failure_path,
            {
                "schema": "tactile3d-unit.s4-3-pi2s-h-delivery-integrity-failure.v2.1",
                "status": "FAIL",
                "tuple_key": spec["tuple_key"],
                "type": type(error).__name__,
                "message": str(error),
                "retry_classification": "SCIENTIFIC_INTEGRITY_FAILURE_NO_RETRY",
                "canonical_outcome_produced": False,
            },
        )
        raise
    base.publish_json_no_clobber(audit_path, audit)
    if audit["status"] != "PASS":
        raise base.ContractError(f"v2.1 H delivery audit failed: {spec['tuple_key']}")


def validate_tuple_artifact(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    payload = BASE_VALIDATE_TUPLE(path, expected)
    audit_path = _audit_path(path)
    if not audit_path.is_file():
        raise base.ContractError(f"v2.1 H audit is absent: {audit_path}")
    audit = base.read_json(audit_path)
    if (
        audit.get("schema") != SCHEMA_AUDIT
        or audit.get("status") != "PASS"
        or audit.get("tuple_key") != expected["tuple_key"]
        or audit.get("tuple_identity_sha256") != expected["tuple_identity_sha256"]
        or audit.get("audit_version") != AUDIT_VERSION
        or any(value != "PASS" for value in audit.get("gates", {}).values())
    ):
        raise base.ContractError(f"v2.1 H audit drifted: {audit_path}")
    return payload


def run_tuple_attempt(
    tuple_row: Mapping[str, Any],
    attempt: int,
    port: int,
    contact_socket: Path,
    physical_gpu: int,
) -> dict[str, Any]:
    try:
        return BASE_RUN_TUPLE_ATTEMPT(tuple_row, attempt, port, contact_socket, physical_gpu)
    except RuntimeError as error:
        artifact = base.attempt_paths(tuple_row, attempt).artifact
        audit = _audit_path(artifact)
        failure = _audit_failure_path(artifact)
        if (audit.is_file() and base.read_json(audit).get("status") == "FAIL") or failure.is_file():
            raise base.ContractError(
                f"v2.1 H scientific integrity gate failed without retry: "
                f"{tuple_row['tuple_key']} attempt={attempt}"
            ) from error
        raise


def block_paths(checkpoint_name: str) -> SimpleNamespace:
    lower = checkpoint_name.lower()
    return SimpleNamespace(
        run=RUN_ROOT / "blocks" / lower,
        log=LOG_ROOT / lower,
        cache=CACHE_ROOT / lower,
        socket=Path(f"/tmp/pi2s_s5_optional_v21_{lower}.sock"),
    )


def build_manifest(source_commit: str) -> dict[str, Any]:
    protocol, protocol_sha, v2_decision, decision = _load_contract()
    _verify_v2_evidence(v2_decision, decision)
    if base.git_output("branch", "--show-current") != "develop/sim-benchmark":
        raise base.ContractError("v2.1 may only be frozen on develop/sim-benchmark")
    runner = Path(__file__).resolve()
    source_files = (BASE_SCRIPT, V1_SCRIPT, V2_SCRIPT, runner, DECISION)
    for path in source_files:
        relative = path.relative_to(ROOT).as_posix()
        if base.committed_blob_sha(source_commit, relative) != base.sha256_file(path):
            raise base.ContractError(f"committed source binding failed: {relative}")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"], cwd=ROOT, check=False
    ).returncode:
        raise base.ContractError("v2.1 source commit is not an ancestor of HEAD")
    check = subprocess.run(
        [str(base.UNIT_PYTHON), str(BASE_SCRIPT), "check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if check.returncode or "CHECK_PASS" not in check.stdout:
        raise base.ContractError("base 72 no longer passes its frozen check")
    pending = build_recovery_tuples(protocol)
    for row in pending[::TUPLES_PER_BLOCK]:
        base.validate_checkpoint_presence(row)
    corrected = []
    v2_index = {row["tuple_key"]: row for row in base.read_json(V2_MANIFEST)["tuples"]}
    for row in decision["completed_v2_episodes_for_reaudit"]:
        artifact = _resolve_pi2s_symbolic(row["tuple_artifact"])
        audit = build_corrected_audit(artifact, v2_index[row["v2_tuple_key"]])
        if audit["status"] != "PASS":
            raise base.ContractError(f"corrected audit did not pass: {row['v2_tuple_key']}")
        corrected.append(
            {
                **row,
                "corrected_audit_semantic_sha256": base.canonical_sha(audit),
            }
        )
    source_paths = tuple(base.SOURCE_PATHS) + tuple(
        path.relative_to(ROOT).as_posix() for path in source_files
    )
    return {
        "schema": SCHEMA_MANIFEST,
        "status": "FROZEN_PRE_EXECUTION",
        "created_at_utc": base.now_utc(),
        "source_commit": source_commit,
        "source_branch": base.git_output("branch", "--show-current"),
        "runner": "$REPO_ROOT/" + runner.relative_to(ROOT).as_posix(),
        "runner_sha256": base.sha256_file(runner),
        "v2_runner_sha256": base.sha256_file(V2_SCRIPT),
        "protocol_sha256": protocol_sha,
        "audit_recovery_decision_sha256": base.sha256_file(DECISION),
        "v2_manifest_sha256": base.sha256_file(V2_MANIFEST),
        "v2_terminal_progress_sha256": base.sha256_file(V2_PROGRESS),
        "v2_failed_resume_state_sha256": decision["v2"]["failed_resume_state_sha256"],
        "audit_version": AUDIT_VERSION,
        "diagnostic_runtime_version": v2.RUNTIME_VERSION,
        "source_sha256": {relative: base.sha256_file(ROOT / relative) for relative in source_paths},
        "reused_valid_v1_tuples": base.read_json(V2_MANIFEST)["reused_valid_v1_tuples"],
        "preserved_invalid_v1_attempts": base.read_json(V2_MANIFEST)[
            "preserved_invalid_v1_attempts"
        ],
        "reaudited_completed_v2_tuples": corrected,
        "tuples": pending,
        "tuple_sequence_sha256": base.canonical_sha(pending),
        "budget": {
            "reused_v1": REUSED_V1_TUPLES,
            "reaudited_v2_without_rerun": REAUDITED_V2_TUPLES,
            "v21_recovery": EXPECTED_RECOVERY_TUPLES,
            "final_optional": EXPECTED_FINAL_TUPLES,
            "base_plus_optional_total": 120,
            "further_expansion_authorized": False,
        },
        "performance_selection_used": False,
        "formal_track_a_scores_replaced": False,
        "optimizer_updates": 0,
        "checkpoint_writes": 0,
        "real_robot_used": False,
    }


def freeze_manifest(source_commit: str) -> None:
    protocol, _protocol_sha, _v2_decision, decision = _load_contract()
    del protocol
    targets = [MANIFEST, PERSISTED_DECISION, V2_FAILED_RESUME_SNAPSHOT]
    targets.extend(
        _resolve_new_pi2s_symbolic(row["corrected_audit"])
        for row in decision["completed_v2_episodes_for_reaudit"]
    )
    if any(path.exists() for path in targets):
        raise FileExistsError("v2.1 freeze target already exists")
    payload = build_manifest(source_commit)
    base.copy_file_no_clobber(DECISION, PERSISTED_DECISION)
    base.copy_file_no_clobber(RESUME, V2_FAILED_RESUME_SNAPSHOT)
    v2_index = {row["tuple_key"]: row for row in base.read_json(V2_MANIFEST)["tuples"]}
    for row in decision["completed_v2_episodes_for_reaudit"]:
        artifact = _resolve_pi2s_symbolic(row["tuple_artifact"])
        audit_path = _resolve_new_pi2s_symbolic(row["corrected_audit"])
        audit = build_corrected_audit(artifact, v2_index[row["v2_tuple_key"]])
        base.publish_json_no_clobber(audit_path, audit)
    base.publish_json_no_clobber(MANIFEST, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "reused_v1": REUSED_V1_TUPLES,
                "reaudited_v2": REAUDITED_V2_TUPLES,
                "recovery_tuples": len(payload["tuples"]),
                "sha256": base.sha256_file(MANIFEST),
            },
            sort_keys=True,
        )
    )


def validate_manifest() -> dict[str, Any]:
    protocol, protocol_sha, v2_decision, decision = _load_contract()
    _verify_v2_evidence(v2_decision, decision)
    manifest = base.read_json(MANIFEST)
    if (
        manifest.get("schema") != SCHEMA_MANIFEST
        or manifest.get("status") != "FROZEN_PRE_EXECUTION"
    ):
        raise base.ContractError("v2.1 manifest is not eligible")
    if (
        manifest.get("protocol_sha256") != protocol_sha
        or manifest.get("audit_recovery_decision_sha256") != base.sha256_file(DECISION)
        or manifest.get("v2_manifest_sha256") != base.sha256_file(V2_MANIFEST)
        or manifest.get("audit_version") != AUDIT_VERSION
        or manifest.get("diagnostic_runtime_version") != v2.RUNTIME_VERSION
    ):
        raise base.ContractError("v2.1 manifest binding drifted")
    if (
        base.sha256_file(PERSISTED_DECISION) != base.sha256_file(DECISION)
        or base.sha256_file(V2_FAILED_RESUME_SNAPSHOT)
        != decision["v2"]["failed_resume_state_sha256"]
    ):
        raise base.ContractError("v2.1 persisted recovery evidence drifted")
    source_commit = str(manifest["source_commit"])
    runner_relative = Path(__file__).resolve().relative_to(ROOT).as_posix()
    if (
        manifest.get("runner_sha256") != base.sha256_file(Path(__file__).resolve())
        or base.committed_blob_sha(source_commit, runner_relative) != manifest["runner_sha256"]
        or manifest.get("v2_runner_sha256") != base.sha256_file(V2_SCRIPT)
    ):
        raise base.ContractError("v2.1 runner binding drifted")
    observed_sources = {
        relative: base.sha256_file(ROOT / relative) for relative in manifest["source_sha256"]
    }
    if observed_sources != manifest["source_sha256"]:
        raise base.ContractError("v2.1 execution source drifted")
    expected = build_recovery_tuples(protocol)
    if manifest.get("tuples") != expected or manifest.get(
        "tuple_sequence_sha256"
    ) != base.canonical_sha(expected):
        raise base.ContractError("v2.1 tuple manifest drifted")
    v2_index = {row["tuple_key"]: row for row in base.read_json(V2_MANIFEST)["tuples"]}
    for row in manifest["reaudited_completed_v2_tuples"]:
        artifact = _resolve_pi2s_symbolic(row["tuple_artifact"])
        audit_path = _resolve_pi2s_symbolic(row["corrected_audit"])
        audit = base.read_json(audit_path)
        expected_audit = build_corrected_audit(artifact, v2_index[row["v2_tuple_key"]])
        if (
            audit != expected_audit
            or base.canonical_sha(audit) != row["corrected_audit_semantic_sha256"]
            or audit.get("status") != "PASS"
        ):
            raise base.ContractError(f"corrected v2 audit drifted: {row['v2_tuple_key']}")
    for row in expected[::TUPLES_PER_BLOCK]:
        base.validate_checkpoint_presence(row)
    return manifest


def progress_payload(
    manifest: Mapping[str, Any],
    assignments: Sequence[Mapping[str, Any]],
    started: float,
    status: str,
) -> dict[str, Any]:
    payload = BASE_PROGRESS_PAYLOAD(manifest, assignments, started, status)
    payload.update(
        {
            "schema": "tactile3d-unit.s4-3-pi2s-optional-h-recovery-progress.v2.1",
            "completed_blocks": sum(
                1
                for identifier in FOCUS_IDS
                if sum(
                    row["tuple_key"].startswith(identifier.lower() + "__") for row in assignments
                )
                == TUPLES_PER_BLOCK
            ),
            "planned_blocks": len(FOCUS_IDS),
            "reused_valid_v1_protocol_tuples": REUSED_V1_TUPLES,
            "reaudited_completed_v2_protocol_tuples": REAUDITED_V2_TUPLES,
            "completed_recovery_tuples": len(assignments),
            "completed_optional_protocol_tuples": REUSED_V1_TUPLES
            + REAUDITED_V2_TUPLES
            + len(assignments),
            "planned_optional_protocol_tuples": EXPECTED_FINAL_TUPLES,
            "preserved_v1_invalid_attempts": 3,
            "preserved_v2_false_positive_audit_failures": 3,
            "audit_version": AUDIT_VERSION,
            "diagnostic_runtime_version": v2.RUNTIME_VERSION,
            "optional_48_started": True,
        }
    )
    return payload


def update_progress(
    manifest: Mapping[str, Any],
    assignments: Sequence[Mapping[str, Any]],
    started: float,
    status: str,
) -> None:
    payload = progress_payload(manifest, assignments, started, status)
    base.atomic_mutable_json(PROGRESS, payload)
    base.atomic_mutable_json(
        RESUME,
        {
            "schema": "tactile3d-unit.pi2s-resume.v2",
            "status": "PI2S_LONG_JOB_RUNNING" if status == "RUNNING" else status,
            "phase": "S5_OPTIONAL_H_V21_AUDIT_CORRECTION_RECOVERY_39",
            "updated_at_utc": payload["updated_at_utc"],
            "supervisor_pid": os.getpid(),
            "source_commit": manifest["source_commit"],
            "protocol_sha256": manifest["protocol_sha256"],
            "manifest_sha256": payload["manifest_sha256"],
            "progress_path": str(PROGRESS),
            "supervisor_log": str(LOG_ROOT / "supervisor.log"),
            "reused_valid_v1_protocol_tuples": REUSED_V1_TUPLES,
            "reaudited_completed_v2_protocol_tuples": REAUDITED_V2_TUPLES,
            "completed_recovery_tuples": len(assignments),
            "planned_recovery_tuples": EXPECTED_RECOVERY_TUPLES,
            "completed_optional_protocol_tuples": REUSED_V1_TUPLES
            + REAUDITED_V2_TUPLES
            + len(assignments),
            "planned_optional_protocol_tuples": EXPECTED_FINAL_TUPLES,
            "base_canonical_tuples_completed": 72,
            "total_s5_protocol_tuples_if_complete": 120,
            "long_job_running": status == "RUNNING",
            "training_performed": False,
            "push_performed": False,
            "worktree_removal_performed": False,
            "real_robot_used": False,
        },
    )


def _episode_summary(
    tuple_row: Mapping[str, Any],
    payload: Mapping[str, Any],
    artifact: Path,
    audit_path: Path,
    provenance: str,
    attempt: int,
) -> dict[str, Any]:
    return {
        "protocol_tuple_key": tuple_row["protocol_tuple_key"],
        "tuple": dict(tuple_row),
        "diagnostic_runtime_version": v2.RUNTIME_VERSION,
        "audit_version": AUDIT_VERSION,
        "provenance": provenance,
        "attempt": attempt,
        "artifact": str(artifact),
        "artifact_sha256": base.sha256_file(artifact),
        "h_audit": str(audit_path),
        "h_audit_sha256": base.sha256_file(audit_path),
        "success": bool(payload["success"]),
        "termination": payload["termination"],
        "steps": int(payload["steps"]),
        "ever_object_contact": bool(payload["ever_object_contact"]),
        "ever_lifted": bool(payload["ever_lifted"]),
        "max_pinch_count": int(payload["max_pinch_count"]),
        "ever_native_trigger": bool(payload["ever_native_trigger"]),
        "max_success_counter": int(payload["max_success_counter"]),
        "max_consecutive_object_contact_run": int(payload["max_consecutive_object_contact_run"]),
        "exact_first_failure_stage": payload["exact_first_failure_stage"],
    }


def aggregate_results(
    manifest: Mapping[str, Any],
    assignments: Sequence[Mapping[str, Any]],
    snapshots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    recovery = BASE_AGGREGATE_RESULTS(manifest, assignments, snapshots)
    by_protocol_key: dict[str, dict[str, Any]] = {}
    attempt_ledger = []
    for episode in recovery["episodes"]:
        tuple_row = episode["tuple"]
        artifact = Path(episode["artifact"])
        payload = validate_tuple_artifact(artifact, tuple_row)
        by_protocol_key[tuple_row["protocol_tuple_key"]] = _episode_summary(
            tuple_row,
            payload,
            artifact,
            _audit_path(artifact),
            "V21_RECOVERY",
            int(episode["attempt"]),
        )
    attempt_ledger.extend(
        dict(row) | {"diagnostic_runtime_version": v2.RUNTIME_VERSION}
        for row in recovery["attempt_ledger"]
    )
    for row in manifest["reaudited_completed_v2_tuples"]:
        artifact = _resolve_pi2s_symbolic(row["tuple_artifact"])
        audit_path = _resolve_pi2s_symbolic(row["corrected_audit"])
        payload = base.read_json(artifact)
        tuple_row = payload["tuple"]
        by_protocol_key[row["protocol_tuple_key"]] = _episode_summary(
            tuple_row, payload, artifact, audit_path, "V2_REAUDITED_WITH_V21", 1
        )
        attempt_ledger.extend(
            [
                {
                    "protocol_tuple_key": row["protocol_tuple_key"],
                    "attempt": 1,
                    "kind": "SCIENTIFIC_RESULT_REAUDITED_WITH_V21",
                    "path": str(artifact),
                    "sha256": row["tuple_artifact_sha256"],
                    "status": "PASS",
                    "canonical_outcome": True,
                },
                {
                    "protocol_tuple_key": row["protocol_tuple_key"],
                    "attempt": 1,
                    "kind": "FALSE_POSITIVE_V2_AUDIT_FAILURE_PRESERVED",
                    "path": str(_resolve_pi2s_symbolic(row["original_failure"])),
                    "sha256": row["original_failure_sha256"],
                    "status": "FAIL",
                    "canonical_outcome": False,
                },
            ]
        )
    for row in manifest["reused_valid_v1_tuples"]:
        artifact = _resolve_pi2s_symbolic(row["tuple_artifact"])
        audit_path = _resolve_pi2s_symbolic(row["h_audit"])
        payload = base.read_json(artifact)
        tuple_row = payload["tuple"]
        key = row["protocol_tuple_key"]
        by_protocol_key[key] = {
            **v2._episode_summary(
                tuple_row,
                payload,
                artifact,
                audit_path,
                "OPTIONAL_H_V1",
                "REUSED_VALID_V1",
                1,
            ),
            "audit_version": "V1",
        }
        attempt_ledger.append(
            {
                "protocol_tuple_key": key,
                "attempt": 1,
                "kind": "SCIENTIFIC_RESULT_REUSED_VALID_V1",
                "path": str(artifact),
                "sha256": row["tuple_artifact_sha256"],
                "status": "PASS",
                "canonical_outcome": True,
            }
        )
    for row in manifest["preserved_invalid_v1_attempts"]:
        attempt_ledger.append(
            {
                "protocol_tuple_key": row["protocol_tuple_key"],
                "attempt": row["attempt"],
                "kind": "SCIENTIFIC_INTEGRITY_FAILURE_PRESERVED_V1",
                "path": str(_resolve_pi2s_symbolic(row["artifact"])),
                "sha256": row["sha256"],
                "status": "FAIL",
                "canonical_outcome": False,
            }
        )
    protocol, _protocol_sha, _v2_decision, _decision = _load_contract()
    ordered = [row["tuple_key"] for row in v2._full_protocol_tuples(protocol)]
    if set(by_protocol_key) != set(ordered) or len(by_protocol_key) != EXPECTED_FINAL_TUPLES:
        raise base.ContractError("combined v1+v2+v2.1 optional coverage is not exact")
    episodes = [by_protocol_key[key] for key in ordered]
    return {
        "schema": SCHEMA_RESULTS,
        "status": "PASS",
        "created_at_utc": base.now_utc(),
        "source_commit": manifest["source_commit"],
        "protocol_sha256": manifest["protocol_sha256"],
        "manifest_sha256": base.sha256_file(MANIFEST),
        "audit_recovery_decision_sha256": manifest["audit_recovery_decision_sha256"],
        "v2_manifest_sha256": manifest["v2_manifest_sha256"],
        "canonical_tuples_planned": EXPECTED_FINAL_TUPLES,
        "canonical_tuples_completed": len(episodes),
        "canonical_tuples_invalid": 0,
        "reused_valid_v1_protocol_tuples": REUSED_V1_TUPLES,
        "reaudited_v2_protocol_tuples_without_rerun": REAUDITED_V2_TUPLES,
        "v21_recovery_tuples_completed": len(recovery["episodes"]),
        "optional_48_executed": True,
        "episodes": episodes,
        "attempt_ledger": attempt_ledger,
        "runtime_logs": recovery["runtime_logs"],
        "successes": sum(int(row["success"]) for row in episodes),
        "failures": sum(int(not row["success"]) for row in episodes),
        "stage_definition": base.STAGE_DEFINITION,
        "exact_first_failure_available": False,
        "coarse_native_milestone_localization_available": True,
        "gpu_snapshots": list(snapshots),
        "scientific_role": "POST_EXPOSURE_MECHANISM_DIAGNOSTIC_NOT_FORMAL_SCORE",
        "formal_track_a_scores_replaced": False,
        "performance_selection_used": False,
        "further_rollout_expansion_authorized": False,
        "optimizer_updates": 0,
        "checkpoint_writes": 0,
        "real_robot_used": False,
    }


def status_payload() -> dict[str, Any]:
    manifest = validate_manifest()
    completed = base.collect_completed(manifest)
    progress = base.read_json(PROGRESS) if PROGRESS.is_file() else None
    result = base.read_json(RESULTS) if RESULTS.is_file() else None
    return {
        "status": (
            result.get("status")
            if result
            else progress.get("status") if progress else "NOT_STARTED"
        ),
        "manifest_sha256": base.sha256_file(MANIFEST),
        "source_commit": manifest["source_commit"],
        "reused_valid_v1_protocol_tuples": REUSED_V1_TUPLES,
        "reaudited_completed_v2_protocol_tuples": REAUDITED_V2_TUPLES,
        "completed_recovery_tuples": len(completed),
        "remaining_recovery_tuples": EXPECTED_RECOVERY_TUPLES - len(completed),
        "completed_optional_protocol_tuples": REUSED_V1_TUPLES
        + REAUDITED_V2_TUPLES
        + len(completed),
        "planned_optional_protocol_tuples": EXPECTED_FINAL_TUPLES,
        "progress": progress,
        "results_present": result is not None,
        "results_sha256": base.sha256_file(RESULTS) if RESULTS.is_file() else None,
    }


def serve_policy(checkpoint_name: str, port: int) -> None:
    v2.serve_policy(checkpoint_name, port)


def activate_base_runtime() -> None:
    base.__file__ = str(Path(__file__).resolve())
    base.MANIFEST = MANIFEST
    base.RESULTS = RESULTS
    base.RUN_ROOT = RUN_ROOT
    base.LOG_ROOT = LOG_ROOT
    base.CACHE_ROOT = CACHE_ROOT
    base.PROGRESS = PROGRESS
    base.RESUME = RESUME
    base.FOCUS_KEYS = FOCUS_KEYS
    base.SAMPLING_SEEDS = v1.SAMPLING_SEEDS
    base.EXPECTED_TUPLES = EXPECTED_RECOVERY_TUPLES
    base.SCHEMA_MANIFEST = SCHEMA_MANIFEST
    base.SCHEMA_TUPLE = SCHEMA_TUPLE
    base.SCHEMA_RESULTS = SCHEMA_RESULTS
    base.build_base_tuples = build_recovery_tuples
    base.validate_manifest = validate_manifest
    base.build_diagnostic_environment = v2.build_v2_environment
    base.validate_tuple_artifact = validate_tuple_artifact
    base.run_tuple_attempt = run_tuple_attempt
    base.block_paths = block_paths
    base.progress_payload = progress_payload
    base.update_progress = update_progress
    base.aggregate_results = aggregate_results
    v2.validate_manifest = validate_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--source-commit", required=True)
    sub.add_parser("validate")
    contact = sub.add_parser("contact-service")
    contact.add_argument("--artifact", type=Path, required=True)
    contact.add_argument("--socket", type=Path, required=True)
    serve = sub.add_parser("serve-policy")
    serve.add_argument("--checkpoint-id", required=True)
    serve.add_argument("--port", type=int, required=True)
    evaluate = sub.add_parser("evaluate-tuple")
    evaluate.add_argument("--tuple-spec", type=Path, required=True)
    evaluate.add_argument("--contact-socket", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--diagnostics", type=Path, required=True)
    evaluate.add_argument("--artifact", type=Path, required=True)
    evaluate.add_argument("--port", type=int, required=True)
    launch = sub.add_parser("orchestrate")
    launch.add_argument("--gpus", required=True)
    launch.add_argument("--resume", action="store_true")
    sub.add_parser("status")
    sub.add_parser("check")
    return parser.parse_args()


def main() -> None:
    activate_base_runtime()
    args = parse_args()
    if args.command == "freeze":
        freeze_manifest(args.source_commit)
    elif args.command == "validate":
        manifest = validate_manifest()
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "reused_v1": REUSED_V1_TUPLES,
                    "reaudited_v2": REAUDITED_V2_TUPLES,
                    "recovery_tuples": len(manifest["tuples"]),
                    "sha256": base.sha256_file(MANIFEST),
                },
                sort_keys=True,
            )
        )
    elif args.command == "contact-service":
        base.contact_service(args.artifact, args.socket)
    elif args.command == "serve-policy":
        serve_policy(args.checkpoint_id, args.port)
    elif args.command == "evaluate-tuple":
        evaluate_tuple(
            args.tuple_spec,
            args.contact_socket,
            args.output,
            args.diagnostics,
            args.artifact,
            args.port,
        )
    elif args.command == "orchestrate":
        gpus = [int(value) for value in args.gpus.split(",") if value]
        base.orchestrate(gpus, resume=args.resume)
    elif args.command == "status":
        print(json.dumps(status_payload(), indent=2, sort_keys=True))
    elif args.command == "check":
        base.check_completed()
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
