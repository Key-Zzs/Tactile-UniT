#!/usr/bin/env python3
"""Recover the frozen PI2S optional H wave after the v1 lag-clock stop.

The v1 runner and every v1 output remain immutable.  Six valid TRAIN-mean
protocol tuples are reused byte-for-byte.  This v2 runtime executes only the
remaining 42 unique protocol tuples.  For ``same_episode_lag5`` it computes
the correct contact state after every raw 50 Hz physical control step and
delivers the value from five ticks earlier at the next policy observation.
TRAIN-mean tuples do not perform the additional per-step encoder queries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
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

PI2S_ROOT = base.PI2S_ROOT
DECISION = ROOT / "configs/simulation/pi2s/optional_h_intervention_recovery_v2.json"
PERSISTED_DECISION = PI2S_ROOT / "artifacts/optional_h_intervention_recovery_v2.json"
V1_MANIFEST = PI2S_ROOT / "artifacts/diagnostic_rollout_optional_manifest.json"
V1_RESULT = PI2S_ROOT / "artifacts/diagnostic_rollout_optional_results.json"
V1_PROGRESS = PI2S_ROOT / "diagnostics/rollouts_optional_h_v1/progress.json"
V1_FAILED_RESUME_SNAPSHOT = (
    PI2S_ROOT / "artifacts/diagnostic_rollout_optional_v1_failed_resume_state.json"
)
MANIFEST = PI2S_ROOT / "artifacts/diagnostic_rollout_optional_recovery_v2_manifest.json"
RESULTS = PI2S_ROOT / "artifacts/diagnostic_rollout_optional_results_v2.json"
RUN_ROOT = PI2S_ROOT / "diagnostics/rollouts_optional_h_v2"
LOG_ROOT = PI2S_ROOT / "logs/rollouts_optional_h_v2"
CACHE_ROOT = PI2S_ROOT / "cache/rollouts_optional_h_v2"
PROGRESS = RUN_ROOT / "progress.json"
RESUME = PI2S_ROOT / "resume_state.json"

RUNTIME_VERSION = "OPTIONAL_H_V2_RAW50HZ_TICK_BUFFER"
SCHEMA_MANIFEST = "tactile3d-unit.s4-3-pi2s-optional-h-recovery-manifest.v2"
SCHEMA_TUPLE = "tactile3d-unit.s4-3-pi2s-optional-h-rollout-tuple.v2"
SCHEMA_AUDIT = "tactile3d-unit.s4-3-pi2s-h-delivery-audit.v2"
SCHEMA_RESULTS = "tactile3d-unit.s4-3-pi2s-optional-h-rollout-results.v2"
FOCUS_KEYS = v1.FOCUS_KEYS
FOCUS_IDS = v1.FOCUS_IDS
CONDITIONS = v1.CONDITIONS
EXPECTED_RECOVERY_TUPLES = 42
REUSED_V1_TUPLES = 6
TUPLES_PER_BLOCK = 14
EXPECTED_FINAL_TUPLES = 48

BASE_SCRIPT = v1.BASE_SCRIPT
V1_SCRIPT = Path(v1.__file__).resolve()
BASE_BUILD_ENVIRONMENT = v1.BASE_BUILD_ENVIRONMENT
BASE_VALIDATE_TUPLE = v1.BASE_VALIDATE_TUPLE
BASE_PROGRESS_PAYLOAD = v1.BASE_PROGRESS_PAYLOAD
BASE_AGGREGATE_RESULTS = v1.BASE_AGGREGATE_RESULTS
BASE_RUN_TUPLE_ATTEMPT = base.run_tuple_attempt


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _resolve_pi2s_symbolic(value: str) -> Path:
    prefix = "$PI2S_ROOT/"
    if not value.startswith(prefix):
        raise base.ContractError(f"not a PI2S symbolic path: {value}")
    path = PI2S_ROOT / value.removeprefix(prefix)
    resolved_root = PI2S_ROOT.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root):
        raise base.ContractError(f"PI2S symbolic path escaped the root: {value}")
    return path


def _load_contract() -> tuple[dict[str, Any], str, dict[str, Any]]:
    protocol, protocol_sha = base.load_protocol()
    decision = base.read_json(DECISION)
    if (
        decision.get("status") != "AUTHORIZED_AFTER_V1_SCIENTIFIC_CONTRACT_STOP"
        or decision.get("decision")
        != "PRESERVE_V1_AND_RUN_ONLY_THE_42_UNIQUE_UNCOMPLETED_PROTOCOL_TUPLES_WITH_V2_RUNTIME"
    ):
        raise base.ContractError("v2 recovery decision is not authorized")
    budget = decision.get("recovery_budget", {})
    if (
        budget.get("valid_v1_protocol_tuples_reused") != REUSED_V1_TUPLES
        or budget.get("v2_protocol_tuples_to_run") != EXPECTED_RECOVERY_TUPLES
        or budget.get("final_unique_optional_protocol_tuples") != EXPECTED_FINAL_TUPLES
        or budget.get("final_total_s5_protocol_tuples_including_base") != 120
        or budget.get("invalid_v1_attempts_excluded_from_canonical_count") != 3
        or budget.get("further_expansion_authorized") is not False
    ):
        raise base.ContractError("v2 recovery budget drifted")
    runtime = decision.get("v2_runtime", {})
    if (
        runtime.get("version") != RUNTIME_VERSION
        or runtime.get("new_identity_required") is not True
        or runtime.get("train_mean_path_change")
        != "NONE: per-physical-step H queries are disabled for TRAIN-mean tuples."
    ):
        raise base.ContractError("v2 runtime contract drifted")
    guardrails = decision.get("guardrails", {})
    if (
        guardrails.get("v1_files_may_be_modified_or_removed") is not False
        or guardrails.get("v1_attempt_2_may_be_used") is not False
        or guardrails.get("valid_v1_protocol_tuples_may_be_rerun") is not False
        or guardrails.get("performance_selection") is not False
        or guardrails.get("formal_track_a_score_replacement") is not False
        or guardrails.get("best_intervention_deployment_selection") is not False
        or guardrails.get("training_or_optimizer_updates") != 0
        or guardrails.get("checkpoint_writes") != 0
        or guardrails.get("real_robot_allowed") is not False
    ):
        raise base.ContractError("v2 recovery guardrails drifted")
    return protocol, protocol_sha, decision


def _full_protocol_tuples(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    return v1.build_optional_tuples(protocol)


def _accepted_protocol_keys(decision: Mapping[str, Any]) -> set[str]:
    rows = decision["preserved_valid_v1_tuples"]
    keys = {str(row["protocol_tuple_key"]) for row in rows}
    if len(rows) != REUSED_V1_TUPLES or len(keys) != REUSED_V1_TUPLES:
        raise base.ContractError("v1 accepted tuple list is not exact")
    return keys


def build_recovery_tuples(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    frozen_protocol, _protocol_sha, decision = _load_contract()
    if dict(protocol) != frozen_protocol:
        raise base.ContractError("v2 recovery tuple builder received a non-frozen protocol")
    accepted = _accepted_protocol_keys(decision)
    recovery: list[dict[str, Any]] = []
    for original in _full_protocol_tuples(protocol):
        protocol_key = str(original["tuple_key"])
        if protocol_key in accepted:
            continue
        row = dict(original)
        row["protocol_tuple_key"] = protocol_key
        row["diagnostic_runtime_version"] = RUNTIME_VERSION
        row["tuple_key"] = protocol_key + "__v2"
        row["tuple_identity_sha256"] = base.canonical_sha(row)
        recovery.append(row)
    if (
        len(recovery) != EXPECTED_RECOVERY_TUPLES
        or len({row["tuple_key"] for row in recovery}) != EXPECTED_RECOVERY_TUPLES
        or {row["protocol_tuple_key"] for row in recovery} & accepted
    ):
        raise base.ContractError("v2 recovery tuple partition failed")
    if len(accepted | {str(row["protocol_tuple_key"]) for row in recovery}) != 48:
        raise base.ContractError("v1+v2 protocol tuple coverage is not exactly 48")
    return recovery


def _verify_v1_evidence(decision: Mapping[str, Any]) -> None:
    if base.sha256_file(V1_SCRIPT) != decision["v1"]["runner_sha256"]:
        raise base.ContractError("v1 runner bytes drifted")
    if base.sha256_file(V1_MANIFEST) != decision["v1"]["manifest_sha256"]:
        raise base.ContractError("v1 manifest bytes drifted")
    if base.sha256_file(V1_PROGRESS) != decision["v1"]["terminal_progress_sha256"]:
        raise base.ContractError("v1 terminal progress bytes drifted")
    v1_manifest = base.read_json(V1_MANIFEST)
    if base.sha256_file(v1.TRAIN_SIDECAR) != v1_manifest.get(
        "train_sidecar_sha256"
    ) or base.sha256_file(v1.BASE_RESULTS) != v1_manifest.get("base_results_sha256"):
        raise base.ContractError("v1 scientific input evidence drifted")
    failed_resume = V1_FAILED_RESUME_SNAPSHOT if V1_FAILED_RESUME_SNAPSHOT.is_file() else RESUME
    if base.sha256_file(failed_resume) != decision["v1"]["failed_resume_state_sha256"]:
        raise base.ContractError("v1 failed resume-state evidence drifted")
    if V1_RESULT.exists():
        raise base.ContractError("v1 unexpectedly published a formal result")
    for row in decision["preserved_valid_v1_tuples"]:
        artifact = _resolve_pi2s_symbolic(row["tuple_artifact"])
        audit_path = _resolve_pi2s_symbolic(row["h_audit"])
        if (
            base.sha256_file(artifact) != row["tuple_artifact_sha256"]
            or base.sha256_file(audit_path) != row["h_audit_sha256"]
        ):
            raise base.ContractError(
                f"preserved valid v1 tuple drifted: {row['protocol_tuple_key']}"
            )
        payload = base.read_json(artifact)
        audit = base.read_json(audit_path)
        if (
            payload.get("status") != "PASS"
            or payload.get("tuple", {}).get("tuple_key") != row["protocol_tuple_key"]
            or payload.get("tuple", {}).get("h_condition") != "train_mean"
            or audit.get("status") != "PASS"
            or audit.get("tuple_key") != row["protocol_tuple_key"]
            or any(value != "PASS" for value in audit.get("gates", {}).values())
        ):
            raise base.ContractError(
                f"preserved valid v1 tuple is not eligible: {row['protocol_tuple_key']}"
            )
    for row in decision["preserved_invalid_v1_attempts"]:
        artifact = _resolve_pi2s_symbolic(row["artifact"])
        if base.sha256_file(artifact) != row["sha256"]:
            raise base.ContractError(f"invalid v1 attempt drifted: {row['protocol_tuple_key']}")
        payload = base.read_json(artifact)
        if (
            payload.get("status") != "FAIL"
            or payload.get("steps") != 7
            or payload.get("tuple", {}).get("tuple_key") != row["protocol_tuple_key"]
            or payload.get("tuple", {}).get("h_condition") != "same_episode_lag5"
            or payload.get("gates", {}).get("native_terminal_matches_steps") != "FAIL"
            or [name for name, value in payload.get("gates", {}).items() if value != "PASS"]
            != ["native_terminal_matches_steps"]
            or row.get("attempt") != 1
            or row.get("canonical_outcome") is not False
        ):
            raise base.ContractError(
                f"invalid v1 evidence contract drifted: {row['protocol_tuple_key']}"
            )
        attempt_2 = artifact.parents[1] / "attempt_2"
        if attempt_2.exists():
            raise base.ContractError(f"prohibited v1 attempt_2 exists: {attempt_2}")


def build_v2_environment(base_class: Any) -> Any:
    parent = BASE_BUILD_ENVIRONMENT(base_class)

    class RawTickBufferedHEnvironment(parent):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self._pi2s_correct_h_by_tick: dict[int, np.ndarray] = {}

        def step(self, action: np.ndarray) -> Any:
            result = super().step(action)
            if base.TUPLE_SPEC["h_condition"] == "same_episode_lag5":
                correct = np.asarray(self._contact_runtime.contact_state(), dtype=np.float32)
                if correct.shape != (256,) or not np.isfinite(correct).all():
                    raise base.ContractError("post-step correct H is not finite [256]")
                tick = int(self._control_step)
                existing = self._pi2s_correct_h_by_tick.get(tick)
                if existing is not None and not np.array_equal(existing, correct):
                    raise base.ContractError(f"correct H changed within raw tick {tick}")
                self._pi2s_correct_h_by_tick[tick] = correct.copy()
                self._steps[-1]["post_step_correct_h"] = {
                    "control_tick": tick,
                    "sha256": _sha256_array(correct),
                    "l2": float(np.linalg.norm(correct)),
                    "shape": [256],
                    "dtype": "float32",
                    "source": "same accepted ContactStateUnixClient after physical-step tactile append",
                    "sent_to_policy_at_this_step": False,
                }
            return result

        def get_obs(self) -> dict[str, np.ndarray]:
            observation = super().get_obs()
            condition = str(base.TUPLE_SPEC["h_condition"])
            if condition not in CONDITIONS or base.TUPLE_SPEC["model"] not in base.CONTACT_MODELS:
                raise base.ContractError("v2 optional H condition/model contract failed")
            correct = np.asarray(observation.get("contact_state"), dtype=np.float32)
            if correct.shape != (256,) or not np.isfinite(correct).all():
                raise base.ContractError("correct live H is not finite [256]")
            current_tick = int(self._control_step)
            existing = self._pi2s_correct_h_by_tick.get(current_tick)
            if existing is not None and not np.array_equal(existing, correct):
                raise base.ContractError(
                    f"get_obs H disagrees with buffered raw tick {current_tick}"
                )
            self._pi2s_correct_h_by_tick[current_tick] = correct.copy()
            requested_tick: int | None = None
            actual_tick: int | None = None
            bootstrap_affected = False
            lag_source_sha: str | None = None
            if condition == "train_mean":
                delivered = v1.train_mean_h().copy()
            else:
                requested_tick = current_tick - 5
                actual_tick = max(0, requested_tick)
                bootstrap_affected = requested_tick < 0
                if actual_tick not in self._pi2s_correct_h_by_tick:
                    raise base.ContractError(
                        f"v2 raw lag buffer missing tick: current={current_tick} actual={actual_tick}"
                    )
                delivered = self._pi2s_correct_h_by_tick[actual_tick].copy()
                lag_source_sha = _sha256_array(delivered)
            observation["contact_state"] = delivered
            observation["_pi2s_h_condition"] = condition
            row = self._observation_events[-1]
            row.update(
                {
                    "diagnostic_runtime_version": RUNTIME_VERSION,
                    "h_condition": condition,
                    "correct_h_sha256": row["h_sha256"],
                    "correct_h_l2": row["h_l2"],
                    "delivered_h_sha256": _sha256_array(delivered),
                    "delivered_h_l2": float(np.linalg.norm(delivered)),
                    "requested_control_tick": requested_tick,
                    "actual_control_tick": actual_tick,
                    "bootstrap_affected": bootstrap_affected,
                    "lag_source_correct_h_sha256": lag_source_sha,
                    "lag_clock": "raw_50hz_physical_control_tick",
                }
            )
            row["h_sha256"] = row["delivered_h_sha256"]
            row["h_l2"] = row["delivered_h_l2"]
            self._last_h = row
            return observation

    return RawTickBufferedHEnvironment


def _audit_path(tuple_artifact: Path) -> Path:
    return tuple_artifact.with_name("h_intervention_audit_v2.json")


def _audit_failure_path(tuple_artifact: Path) -> Path:
    return tuple_artifact.with_name("h_intervention_audit_v2_failure.json")


def build_v2_audit(tuple_artifact: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    payload = base.read_json(tuple_artifact)
    events = payload.get("observation_events", [])
    chunks = payload.get("action_chunks", [])
    steps = payload.get("step_telemetry", [])
    if not events or not chunks or not steps:
        raise base.ContractError("v2 tuple lacks observation/query/physical-step evidence")
    condition = str(expected["h_condition"])
    correct_by_tick: dict[int, str] = {}
    for row in events:
        tick = int(row["control_step"])
        sha = str(row["correct_h_sha256"])
        if tick in correct_by_tick and correct_by_tick[tick] != sha:
            raise base.ContractError(f"contradictory correct H at observation tick {tick}")
        correct_by_tick[tick] = sha
    post_step_records = []
    post_step_sequence_exact = True
    for row in steps:
        record = row.get("post_step_correct_h")
        if record is None:
            continue
        tick = int(record["control_tick"])
        sha = str(record["sha256"])
        post_step_sequence_exact &= (
            tick == int(row["control_step"]) + 1
            and record.get("shape") == [256]
            and record.get("dtype") == "float32"
            and record.get("sent_to_policy_at_this_step") is False
            and np.isfinite(float(record["l2"]))
        )
        if tick in correct_by_tick and correct_by_tick[tick] != sha:
            raise base.ContractError(f"contradictory correct H at physical tick {tick}")
        correct_by_tick[tick] = sha
        post_step_records.append(record)
    mean_sha = _sha256_array(v1.train_mean_h())
    event_gates = []
    bootstrap_count = 0
    for row in events:
        current = int(row["control_step"])
        delivered = str(row["delivered_h_sha256"])
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
            source_sha = correct_by_tick.get(actual)
            valid = (
                row.get("requested_control_tick") == requested
                and row.get("actual_control_tick") == actual
                and row.get("bootstrap_affected") is bootstrap
                and row.get("lag_source_correct_h_sha256") == source_sha
                and delivered == source_sha
            )
        event_gates.append(
            valid
            and row.get("diagnostic_runtime_version") == RUNTIME_VERSION
            and row.get("h_condition") == condition
            and row.get("h_sha256") == delivered
            and row.get("lag_clock") == "raw_50hz_physical_control_tick"
            and np.isfinite(float(row["delivered_h_l2"]))
        )
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
    event_ticks = [int(row["control_step"]) for row in events]
    query_ticks = [int(row["observation_timestamp"]) for row in chunks]
    max_observation_tick = max(int(row["control_step"]) for row in events)
    expected_ticks = set(range(max_observation_tick + 1))
    lag_buffer_complete = (condition == "train_mean" and not post_step_records) or (
        condition == "same_episode_lag5"
        and post_step_sequence_exact
        and [int(row["control_tick"]) for row in post_step_records]
        == list(range(1, len(steps) + 1))
        and expected_ticks.issubset(correct_by_tick)
        and len(post_step_records) == len(steps)
    )
    gates = {
        "tuple_identity_exact": payload.get("tuple") == dict(expected),
        "runtime_version_exact": expected.get("diagnostic_runtime_version") == RUNTIME_VERSION,
        "all_observation_events_conditioned_exactly": bool(event_gates) and all(event_gates),
        "all_policy_queries_received_audited_h": bool(query_gates) and all(query_gates),
        "policy_queries_match_observation_events_one_to_one": query_ticks == event_ticks,
        "raw_tick_buffer_complete_without_train_mean_extra_queries": lag_buffer_complete,
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
        "diagnostic_runtime_version": RUNTIME_VERSION,
        "h_condition": condition,
        "observation_event_count": len(events),
        "policy_query_count": len(chunks),
        "physical_step_count": len(steps),
        "post_step_h_query_count": len(post_step_records),
        "buffered_correct_tick_count_through_last_observation": len(
            set(correct_by_tick) & expected_ticks
        ),
        "last_observation_tick": max_observation_tick,
        "bootstrap_affected_observation_count": bootstrap_count,
        "train_mean_h_sha256": mean_sha,
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
        raise FileExistsError("v2 H audit output already exists")
    base.evaluate_tuple(tuple_spec_path, contact_socket, output, diagnostics, artifact, port)
    spec = base.read_json(tuple_spec_path)
    try:
        audit = build_v2_audit(artifact, spec)
    except base.ContractError as error:
        base.publish_json_no_clobber(
            failure_path,
            {
                "schema": "tactile3d-unit.s4-3-pi2s-h-delivery-integrity-failure.v2",
                "status": "FAIL",
                "tuple_key": spec["tuple_key"],
                "protocol_tuple_key": spec["protocol_tuple_key"],
                "tuple_identity_sha256": spec["tuple_identity_sha256"],
                "diagnostic_runtime_version": RUNTIME_VERSION,
                "type": type(error).__name__,
                "message": str(error),
                "retry_classification": "SCIENTIFIC_INTEGRITY_FAILURE_NO_RETRY",
                "canonical_outcome_produced": False,
            },
        )
        raise
    base.publish_json_no_clobber(audit_path, audit)
    if audit["status"] != "PASS":
        raise base.ContractError(f"v2 H delivery audit failed: {spec['tuple_key']}")


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
        audit_path = _audit_path(artifact)
        failure_path = _audit_failure_path(artifact)
        audit_failed = audit_path.is_file() and base.read_json(audit_path).get("status") == "FAIL"
        if audit_failed or failure_path.is_file():
            raise base.ContractError(
                f"v2 H scientific integrity gate failed without retry: "
                f"{tuple_row['tuple_key']} attempt={attempt}"
            ) from error
        raise


def validate_tuple_artifact(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    payload = BASE_VALIDATE_TUPLE(path, expected)
    audit_path = _audit_path(path)
    if not audit_path.is_file():
        raise base.ContractError(f"v2 H audit is absent: {audit_path}")
    audit = base.read_json(audit_path)
    if (
        audit.get("schema") != SCHEMA_AUDIT
        or audit.get("status") != "PASS"
        or audit.get("tuple_key") != expected["tuple_key"]
        or audit.get("protocol_tuple_key") != expected["protocol_tuple_key"]
        or audit.get("tuple_identity_sha256") != expected["tuple_identity_sha256"]
        or any(value != "PASS" for value in audit.get("gates", {}).values())
    ):
        raise base.ContractError(f"v2 H audit drifted: {audit_path}")
    return payload


def block_paths(checkpoint_name: str) -> SimpleNamespace:
    lower = checkpoint_name.lower()
    return SimpleNamespace(
        run=RUN_ROOT / "blocks" / lower,
        log=LOG_ROOT / lower,
        cache=CACHE_ROOT / lower,
        socket=Path(f"/tmp/pi2s_s5_optional_v2_{lower}.sock"),
    )


def build_manifest(source_commit: str) -> dict[str, Any]:
    protocol, protocol_sha, decision = _load_contract()
    _verify_v1_evidence(decision)
    if base.git_output("branch", "--show-current") != "develop/sim-benchmark":
        raise base.ContractError("v2 manifest may only be frozen on develop/sim-benchmark")
    runner = Path(__file__).resolve()
    source_files = (BASE_SCRIPT, V1_SCRIPT, runner, DECISION)
    for path in source_files:
        relative = path.relative_to(ROOT).as_posix()
        if base.committed_blob_sha(source_commit, relative) != base.sha256_file(path):
            raise base.ContractError(f"committed source binding failed: {relative}")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"], cwd=ROOT, check=False
    ).returncode:
        raise base.ContractError("v2 source commit is not an ancestor of HEAD")
    base_check = subprocess.run(
        [str(base.UNIT_PYTHON), str(BASE_SCRIPT), "check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if base_check.returncode or "CHECK_PASS" not in base_check.stdout:
        raise base.ContractError("base 72 no longer passes its frozen check")
    tuples = build_recovery_tuples(protocol)
    for row in tuples[::TUPLES_PER_BLOCK]:
        base.validate_checkpoint_presence(row)
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
        "base_runner_sha256": base.sha256_file(BASE_SCRIPT),
        "v1_runner_sha256": base.sha256_file(V1_SCRIPT),
        "protocol_sha256": protocol_sha,
        "recovery_decision": "$REPO_ROOT/" + DECISION.relative_to(ROOT).as_posix(),
        "recovery_decision_sha256": base.sha256_file(DECISION),
        "v1_manifest_sha256": base.sha256_file(V1_MANIFEST),
        "v1_terminal_progress_sha256": base.sha256_file(V1_PROGRESS),
        "v1_failed_resume_state_sha256": decision["v1"]["failed_resume_state_sha256"],
        "train_sidecar_sha256": base.sha256_file(v1.TRAIN_SIDECAR),
        "base_72_results_sha256": base.sha256_file(v1.BASE_RESULTS),
        "source_sha256": {relative: base.sha256_file(ROOT / relative) for relative in source_paths},
        "diagnostic_runtime_version": RUNTIME_VERSION,
        "reused_valid_v1_tuples": decision["preserved_valid_v1_tuples"],
        "preserved_invalid_v1_attempts": decision["preserved_invalid_v1_attempts"],
        "budget": {
            "reused_valid_v1_protocol_tuples": REUSED_V1_TUPLES,
            "v2_recovery_tuples": EXPECTED_RECOVERY_TUPLES,
            "final_unique_optional_protocol_tuples": EXPECTED_FINAL_TUPLES,
            "base_plus_optional_total": 120,
            "further_expansion_authorized": False,
        },
        "runtime_change": decision["v2_runtime"],
        "tuples": tuples,
        "tuple_sequence_sha256": base.canonical_sha(tuples),
        "performance_selection_used": False,
        "formal_track_a_scores_replaced": False,
        "optimizer_updates": 0,
        "checkpoint_writes": 0,
        "real_robot_used": False,
    }


def freeze_manifest(source_commit: str) -> None:
    if MANIFEST.exists() or PERSISTED_DECISION.exists() or V1_FAILED_RESUME_SNAPSHOT.exists():
        raise FileExistsError("v2 freeze target already exists")
    payload = build_manifest(source_commit)
    base.copy_file_no_clobber(DECISION, PERSISTED_DECISION)
    base.copy_file_no_clobber(RESUME, V1_FAILED_RESUME_SNAPSHOT)
    base.publish_json_no_clobber(MANIFEST, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "reused_v1": REUSED_V1_TUPLES,
                "recovery_tuples": len(payload["tuples"]),
                "sha256": base.sha256_file(MANIFEST),
            },
            sort_keys=True,
        )
    )


def validate_manifest() -> dict[str, Any]:
    protocol, protocol_sha, decision = _load_contract()
    _verify_v1_evidence(decision)
    manifest = base.read_json(MANIFEST)
    if (
        manifest.get("schema") != SCHEMA_MANIFEST
        or manifest.get("status") != "FROZEN_PRE_EXECUTION"
    ):
        raise base.ContractError("v2 recovery manifest is not eligible")
    if (
        manifest.get("protocol_sha256") != protocol_sha
        or manifest.get("recovery_decision_sha256") != base.sha256_file(DECISION)
        or manifest.get("v1_manifest_sha256") != base.sha256_file(V1_MANIFEST)
        or manifest.get("v1_terminal_progress_sha256") != base.sha256_file(V1_PROGRESS)
        or manifest.get("train_sidecar_sha256") != base.sha256_file(v1.TRAIN_SIDECAR)
        or manifest.get("base_72_results_sha256") != base.sha256_file(v1.BASE_RESULTS)
        or manifest.get("diagnostic_runtime_version") != RUNTIME_VERSION
    ):
        raise base.ContractError("v2 manifest binding drifted")
    if not PERSISTED_DECISION.is_file() or base.sha256_file(PERSISTED_DECISION) != base.sha256_file(
        DECISION
    ):
        raise base.ContractError("persisted v2 recovery decision drifted")
    if base.sha256_file(V1_FAILED_RESUME_SNAPSHOT) != decision["v1"]["failed_resume_state_sha256"]:
        raise base.ContractError("persisted v1 failed resume-state snapshot drifted")
    source_commit = str(manifest.get("source_commit"))
    runner_relative = Path(__file__).resolve().relative_to(ROOT).as_posix()
    if (
        manifest.get("runner_sha256") != base.sha256_file(Path(__file__).resolve())
        or base.committed_blob_sha(source_commit, runner_relative) != manifest["runner_sha256"]
        or manifest.get("base_runner_sha256") != base.sha256_file(BASE_SCRIPT)
        or manifest.get("v1_runner_sha256") != base.sha256_file(V1_SCRIPT)
    ):
        raise base.ContractError("v2 runner binding drifted")
    observed_sources = {
        relative: base.sha256_file(ROOT / relative) for relative in manifest["source_sha256"]
    }
    if observed_sources != manifest["source_sha256"]:
        raise base.ContractError("v2 execution source drifted")
    expected = build_recovery_tuples(protocol)
    if manifest.get("tuples") != expected or manifest.get(
        "tuple_sequence_sha256"
    ) != base.canonical_sha(expected):
        raise base.ContractError("v2 recovery tuple manifest drifted")
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
            "schema": "tactile3d-unit.s4-3-pi2s-optional-h-recovery-progress.v2",
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
            "completed_recovery_tuples": len(assignments),
            "completed_optional_protocol_tuples": REUSED_V1_TUPLES + len(assignments),
            "planned_optional_protocol_tuples": EXPECTED_FINAL_TUPLES,
            "preserved_invalid_v1_attempts": 3,
            "optional_48_started": True,
            "diagnostic_runtime_version": RUNTIME_VERSION,
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
            "phase": "S5_OPTIONAL_H_V2_RECOVERY_42",
            "updated_at_utc": payload["updated_at_utc"],
            "supervisor_pid": os.getpid(),
            "source_commit": manifest["source_commit"],
            "protocol_sha256": manifest["protocol_sha256"],
            "manifest_sha256": payload["manifest_sha256"],
            "progress_path": str(PROGRESS),
            "supervisor_log": str(LOG_ROOT / "supervisor.log"),
            "reused_valid_v1_protocol_tuples": REUSED_V1_TUPLES,
            "completed_recovery_tuples": len(assignments),
            "planned_recovery_tuples": EXPECTED_RECOVERY_TUPLES,
            "completed_optional_protocol_tuples": REUSED_V1_TUPLES + len(assignments),
            "planned_optional_protocol_tuples": EXPECTED_FINAL_TUPLES,
            "base_canonical_tuples_completed": 72,
            "total_s5_protocol_tuples_if_complete": 120,
            "preserved_invalid_v1_attempts": 3,
            "long_job_running": status == "RUNNING",
            "optional_48_started": True,
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
    runtime_version: str,
    provenance: str,
    attempt: int,
) -> dict[str, Any]:
    return {
        "protocol_tuple_key": tuple_row.get("protocol_tuple_key", tuple_row["tuple_key"]),
        "tuple": dict(tuple_row),
        "diagnostic_runtime_version": runtime_version,
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
    v2_attempts = []
    for episode in recovery["episodes"]:
        tuple_row = episode["tuple"]
        artifact = Path(episode["artifact"])
        payload = validate_tuple_artifact(artifact, tuple_row)
        audit_path = _audit_path(artifact)
        by_protocol_key[str(tuple_row["protocol_tuple_key"])] = _episode_summary(
            tuple_row,
            payload,
            artifact,
            audit_path,
            RUNTIME_VERSION,
            "V2_RECOVERY",
            int(episode["attempt"]),
        )
    for row in recovery["attempt_ledger"]:
        v2_attempts.append(dict(row) | {"diagnostic_runtime_version": RUNTIME_VERSION})
    v1_attempts = []
    for row in manifest["reused_valid_v1_tuples"]:
        artifact = _resolve_pi2s_symbolic(row["tuple_artifact"])
        audit_path = _resolve_pi2s_symbolic(row["h_audit"])
        payload = base.read_json(artifact)
        tuple_row = payload["tuple"]
        key = str(row["protocol_tuple_key"])
        by_protocol_key[key] = _episode_summary(
            tuple_row,
            payload,
            artifact,
            audit_path,
            "OPTIONAL_H_V1",
            "REUSED_VALID_V1",
            1,
        )
        v1_attempts.append(
            {
                "protocol_tuple_key": key,
                "attempt": 1,
                "kind": "SCIENTIFIC_RESULT_REUSED_VALID_V1",
                "path": str(artifact),
                "sha256": row["tuple_artifact_sha256"],
                "status": "PASS",
                "canonical_outcome": True,
                "diagnostic_runtime_version": "OPTIONAL_H_V1",
            }
        )
    invalid_attempts = []
    for row in manifest["preserved_invalid_v1_attempts"]:
        invalid_attempts.append(
            {
                "protocol_tuple_key": row["protocol_tuple_key"],
                "attempt": row["attempt"],
                "kind": "SCIENTIFIC_INTEGRITY_FAILURE_PRESERVED_V1",
                "path": str(_resolve_pi2s_symbolic(row["artifact"])),
                "sha256": row["sha256"],
                "status": "FAIL",
                "canonical_outcome": False,
                "diagnostic_runtime_version": "OPTIONAL_H_V1",
            }
        )
    protocol, _protocol_sha, _decision = _load_contract()
    ordered_keys = [row["tuple_key"] for row in _full_protocol_tuples(protocol)]
    if set(by_protocol_key) != set(ordered_keys) or len(by_protocol_key) != EXPECTED_FINAL_TUPLES:
        raise base.ContractError("combined v1+v2 optional tuple coverage is not exact")
    episodes = [by_protocol_key[key] for key in ordered_keys]
    return {
        "schema": SCHEMA_RESULTS,
        "status": "PASS",
        "created_at_utc": base.now_utc(),
        "source_commit": manifest["source_commit"],
        "protocol_sha256": manifest["protocol_sha256"],
        "manifest_sha256": base.sha256_file(MANIFEST),
        "recovery_decision_sha256": manifest["recovery_decision_sha256"],
        "v1_manifest_sha256": manifest["v1_manifest_sha256"],
        "canonical_tuples_planned": EXPECTED_FINAL_TUPLES,
        "canonical_tuples_completed": len(episodes),
        "canonical_tuples_invalid": 0,
        "reused_valid_v1_protocol_tuples": REUSED_V1_TUPLES,
        "v2_recovery_tuples_completed": len(recovery["episodes"]),
        "preserved_invalid_v1_attempts": len(invalid_attempts),
        "invalid_v1_attempts_counted_as_canonical_outcomes": False,
        "optional_48_executed": True,
        "episodes": episodes,
        "attempt_ledger": v1_attempts + invalid_attempts + v2_attempts,
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
        "protocol_sha256": manifest["protocol_sha256"],
        "source_commit": manifest["source_commit"],
        "reused_valid_v1_protocol_tuples": REUSED_V1_TUPLES,
        "completed_recovery_tuples": len(completed),
        "remaining_recovery_tuples": EXPECTED_RECOVERY_TUPLES - len(completed),
        "completed_optional_protocol_tuples": REUSED_V1_TUPLES + len(completed),
        "planned_optional_protocol_tuples": EXPECTED_FINAL_TUPLES,
        "preserved_invalid_v1_attempts": 3,
        "progress": progress,
        "results_present": result is not None,
        "results_sha256": base.sha256_file(RESULTS) if RESULTS.is_file() else None,
        "optional_48_started": True,
    }


def serve_policy(checkpoint_name: str, port: int) -> None:
    manifest = validate_manifest()
    row = next(
        (value for value in manifest["tuples"] if value["checkpoint_id"] == checkpoint_name), None
    )
    if row is None:
        raise base.ContractError(f"unknown v2 recovery checkpoint {checkpoint_name}")
    from gr00t.simulation.pi2b_policy.contract import Workspace
    from gr00t.simulation.pi2b_policy.training import configure_imports
    from scripts.simulation.pi2b_policy.serve import runtime_config

    workspace = Workspace.load_readonly(ROOT)
    checkpoint = base.symbolic_checkpoint(row["checkpoint_path"])
    configure_imports(workspace)
    import jax
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server

    runtime_evidence = base.verify_runtime_environment("OPENPI_POLICY_SERVER")
    config, training_mode, lambda_phys = runtime_config(
        workspace, str(row["model"]), int(row["training_seed"])
    )
    policy = policy_config.create_trained_policy(config, checkpoint)
    wrapped = v1.OptionalSeedResetPolicy(policy, jax)
    print(
        "PI2S_POLICY_SERVER_READY "
        f"checkpoint_id={checkpoint_name} training_mode={training_mode} "
        f"runtime_mode={row['runtime_mode']} lambda_phys={lambda_phys} port={port} "
        f"wave={RUNTIME_VERSION}",
        flush=True,
    )
    print("PI2S_POLICY_RUNTIME " + json.dumps(runtime_evidence, sort_keys=True), flush=True)
    websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped, host="127.0.0.1", port=port, metadata=wrapped.metadata
    ).serve_forever()


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
    base.build_diagnostic_environment = build_v2_environment
    base.validate_tuple_artifact = validate_tuple_artifact
    base.run_tuple_attempt = run_tuple_attempt
    base.block_paths = block_paths
    base.progress_payload = progress_payload
    base.update_progress = update_progress
    base.aggregate_results = aggregate_results
    v1.validate_manifest = validate_manifest


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
