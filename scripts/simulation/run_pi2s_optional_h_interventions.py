#!/usr/bin/env python3
"""Run the frozen S4.3-PI2S optional closed-loop H interventions.

This is a versioned companion to ``run_pi2s_diagnostic_rollouts.py``.  It
reuses that already-validated evaluator and orchestration implementation
without changing its bytes or its 72 canonical base results.  The only
scientific change is the preregistered H delivered to the three contact-token
policies: the frozen TRAIN mean or the causally available value from five raw
control ticks earlier.  Correct-H outcomes remain in the completed base wave.

The optional wave is exactly 3 checkpoints x 4 reset identities x 2 H
conditions x 2 sampling seeds = 48 canonical tuples.  It performs no optimizer
update, checkpoint write, model selection, formal-score replacement, or robot
execution.
"""

from __future__ import annotations

import argparse
import functools
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

PI2S_ROOT = base.PI2S_ROOT
PROTOCOL = base.PROTOCOL
DECISION = ROOT / "configs/simulation/pi2s/optional_h_intervention_decision.json"
PERSISTED_DECISION = PI2S_ROOT / "artifacts/optional_h_intervention_decision.json"
BASE_RESULTS = PI2S_ROOT / "artifacts/diagnostic_rollout_results.json"
MANIFEST = PI2S_ROOT / "artifacts/diagnostic_rollout_optional_manifest.json"
RESULTS = PI2S_ROOT / "artifacts/diagnostic_rollout_optional_results.json"
RUN_ROOT = PI2S_ROOT / "diagnostics/rollouts_optional_h_v1"
LOG_ROOT = PI2S_ROOT / "logs/rollouts_optional_h_v1"
CACHE_ROOT = PI2S_ROOT / "cache/rollouts_optional_h_v1"
PROGRESS = RUN_ROOT / "progress.json"
RESUME = PI2S_ROOT / "resume_state.json"
TRAIN_SIDECAR = PI2S_ROOT / "source_snapshots/data/pi1_contact_sidecar.npz"

SCHEMA_MANIFEST = "tactile3d-unit.s4-3-pi2s-optional-h-rollout-manifest.v1"
SCHEMA_TUPLE = "tactile3d-unit.s4-3-pi2s-optional-h-rollout-tuple.v1"
SCHEMA_AUDIT = "tactile3d-unit.s4-3-pi2s-h-delivery-audit.v1"
SCHEMA_RESULTS = "tactile3d-unit.s4-3-pi2s-optional-h-rollout-results.v1"
FOCUS_IDS = ("B_HVA_42", "B_HVA_43", "B1_43")
FOCUS_KEYS = (("B_HVA", 42), ("B_HVA", 43), ("B1", 43))
CONDITIONS = ("train_mean", "same_episode_lag5")
SAMPLING_SEEDS = (4317, 4318)
EXPECTED_TUPLES = 48
TUPLES_PER_BLOCK = 16

BASE_BUILD_ENVIRONMENT = base.build_diagnostic_environment
BASE_VALIDATE_TUPLE = base.validate_tuple_artifact
BASE_PROGRESS_PAYLOAD = base.progress_payload
BASE_AGGREGATE_RESULTS = base.aggregate_results
BASE_SCRIPT = Path(base.__file__).resolve()


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _load_contract() -> tuple[dict[str, Any], str, dict[str, Any]]:
    protocol, protocol_sha = base.load_protocol()
    decision = base.read_json(DECISION)
    if decision.get("status") != "AUTHORIZED_BY_PREREGISTERED_INFORMATION_NEED":
        raise base.ContractError("optional H decision is not authorized")
    if decision.get("decision") != "RUN_EXACTLY_THE_FROZEN_OPTIONAL_48_AND_DO_NOT_EXPAND_FURTHER":
        raise base.ContractError("optional H decision drifted")
    if decision.get("protocol", {}).get("sha256") != protocol_sha:
        raise base.ContractError("optional H decision protocol binding drifted")
    budget = decision.get("frozen_budget", {})
    if (
        budget.get("canonical_tuples") != EXPECTED_TUPLES
        or tuple(budget.get("checkpoints", ())) != FOCUS_IDS
        or tuple(budget.get("h_conditions", ())) != CONDITIONS
        or tuple(budget.get("sampling_seeds", ())) != SAMPLING_SEEDS
        or budget.get("further_rollout_expansion_authorized") is not False
    ):
        raise base.ContractError("optional H budget is not exact")
    if (
        decision.get("guardrails", {}).get("purpose")
        != "MECHANISM_DIAGNOSIS_NOT_POSITIVE_RESULT_SEARCH"
    ):
        raise base.ContractError("optional H scientific purpose drifted")
    return protocol, protocol_sha, decision


def _optional_tuple_key(row: Mapping[str, Any]) -> str:
    return (
        f"{str(row['checkpoint_id']).lower()}__r{int(row['reset_seed'])}_"
        f"i{int(row['reset_block_index']):02d}__p{int(row['sampling_seed'])}__"
        f"h{row['h_condition']}"
    )


def build_optional_tuples(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    optional = protocol["development_rollouts"]["optional_h_intervention"]
    if (
        int(optional.get("canonical_tuples", -1)) != EXPECTED_TUPLES
        or tuple(optional.get("models", ())) != FOCUS_IDS
        or tuple(optional.get("conditions", ())) != CONDITIONS
        or tuple(int(value) for value in optional.get("sampling_seeds", ())) != SAMPLING_SEEDS
        or optional.get("enabled_only_after_written_s2_s3_information_need") is not True
    ):
        raise base.ContractError("protocol optional H contract drifted")
    focus_by_id = {base.checkpoint_id(row): row for row in protocol["focus_checkpoints"]}
    if any(identifier not in focus_by_id for identifier in FOCUS_IDS):
        raise base.ContractError("optional H focus checkpoint is absent")
    resets = optional["reset_specs"]
    if len(resets) != 4:
        raise base.ContractError("optional H reset count drifted")
    tuples: list[dict[str, Any]] = []
    for identifier in FOCUS_IDS:
        checkpoint = focus_by_id[identifier]
        for reset in resets:
            for condition in CONDITIONS:
                for sampling_seed in SAMPLING_SEEDS:
                    row = {
                        "checkpoint_id": identifier,
                        "model": checkpoint["model"],
                        "training_seed": int(checkpoint["training_seed"]),
                        "checkpoint_path": checkpoint["path"],
                        "checkpoint_tree_sha256": checkpoint["tree_sha256"],
                        "reset_identity": reset["reset_identity"],
                        "reset_seed": int(reset["seed"]),
                        "reset_block_index": int(reset["block_index"]),
                        "reset_global_index": int(reset["global_index"]),
                        "mjstate_sha256": reset["mjstate_sha256"],
                        "processed_state_sha256": reset["processed_state_sha256"],
                        "sampling_seed": sampling_seed,
                        "h_condition": condition,
                        "runtime_mode": "CONTACT_STATE_TOKENS",
                    }
                    row["tuple_key"] = _optional_tuple_key(row)
                    row["tuple_identity_sha256"] = base.canonical_sha(row)
                    tuples.append(row)
    if (
        len(tuples) != EXPECTED_TUPLES
        or len({row["tuple_key"] for row in tuples}) != EXPECTED_TUPLES
    ):
        raise base.ContractError("optional H tuple count/uniqueness failed")
    return tuples


@functools.lru_cache(maxsize=1)
def train_mean_h() -> np.ndarray:
    protocol, _protocol_sha, _decision = _load_contract()
    if not TRAIN_SIDECAR.is_file():
        raise base.ContractError("persistent TRAIN sidecar is missing")
    with np.load(TRAIN_SIDECAR, allow_pickle=False) as data:
        contact = np.asarray(data["contact_state"], dtype=np.float32)
    if contact.shape != (40065, 256) or not np.isfinite(contact).all():
        raise base.ContractError("persistent TRAIN contact_state contract failed")
    mean = np.mean(contact, axis=0, dtype=np.float64).astype(np.float32)
    frozen = protocol["h_conditions"]
    if _sha256_array(mean) != frozen["train_mean_h_float32_sha256"] or not np.isclose(
        np.linalg.norm(mean.astype(np.float64)),
        float(frozen["train_mean_h_l2"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise base.ContractError("TRAIN-mean H identity drifted")
    mean.setflags(write=False)
    return mean


def build_optional_environment(base_class: Any) -> Any:
    parent = BASE_BUILD_ENVIRONMENT(base_class)

    class OptionalHEnvironment(parent):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self._pi2s_correct_h_by_step: dict[int, np.ndarray] = {}

        def get_obs(self) -> dict[str, np.ndarray]:
            observation = super().get_obs()
            condition = str(base.TUPLE_SPEC["h_condition"])
            if condition not in CONDITIONS or base.TUPLE_SPEC["model"] not in base.CONTACT_MODELS:
                raise base.ContractError("optional H condition/model contract failed")
            correct = np.asarray(observation.get("contact_state"), dtype=np.float32)
            if correct.shape != (256,) or not np.isfinite(correct).all():
                raise base.ContractError("correct live H is not finite [256]")
            control_step = int(self._control_step)
            self._pi2s_correct_h_by_step[control_step] = correct.copy()
            requested_step: int | None = None
            actual_step: int | None = None
            bootstrap_affected = False
            if condition == "train_mean":
                delivered = train_mean_h().copy()
            else:
                requested_step = control_step - 5
                actual_step = max(0, requested_step)
                bootstrap_affected = requested_step < 0
                if actual_step not in self._pi2s_correct_h_by_step:
                    raise base.ContractError(
                        f"lag5 requires an unavailable control tick: current={control_step} actual={actual_step}"
                    )
                delivered = self._pi2s_correct_h_by_step[actual_step].copy()
            observation["contact_state"] = delivered
            observation["_pi2s_h_condition"] = condition
            row = self._observation_events[-1]
            row.update(
                {
                    "h_condition": condition,
                    "correct_h_sha256": row["h_sha256"],
                    "correct_h_l2": row["h_l2"],
                    "delivered_h_sha256": _sha256_array(delivered),
                    "delivered_h_l2": float(np.linalg.norm(delivered)),
                    "requested_control_step": requested_step,
                    "actual_control_step": actual_step,
                    "bootstrap_affected": bootstrap_affected,
                }
            )
            row["h_sha256"] = row["delivered_h_sha256"]
            row["h_l2"] = row["delivered_h_l2"]
            self._last_h = row
            return observation

    return OptionalHEnvironment


class OptionalSeedResetPolicy(base.SeedResetPolicy):
    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(observation)
        try:
            condition = str(payload.pop("_pi2s_h_condition"))
        except KeyError as error:
            raise base.ContractError("optional H condition metadata is missing") from error
        if condition not in CONDITIONS:
            raise base.ContractError(f"unknown optional H condition: {condition}")
        contact = np.asarray(payload.get("contact_state"), dtype=np.float32)
        if contact.shape != (256,) or not np.isfinite(contact).all():
            raise base.ContractError("optional policy server did not receive finite H [256]")
        result = super().infer(payload)
        control = dict(result["pi2s_sampling_control"])
        control.update(
            {
                "h_condition": condition,
                "delivered_h_sha256": _sha256_array(contact),
                "delivered_h_l2": float(np.linalg.norm(contact)),
            }
        )
        result["pi2s_sampling_control"] = control
        return result


def serve_policy(checkpoint_name: str, port: int) -> None:
    manifest = validate_manifest()
    row = next(
        (value for value in manifest["tuples"] if value["checkpoint_id"] == checkpoint_name), None
    )
    if row is None:
        raise base.ContractError(f"unknown optional checkpoint {checkpoint_name}")
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
    wrapped = OptionalSeedResetPolicy(policy, jax)
    print(
        "PI2S_POLICY_SERVER_READY "
        f"checkpoint_id={checkpoint_name} training_mode={training_mode} "
        f"runtime_mode={row['runtime_mode']} lambda_phys={lambda_phys} port={port} "
        "wave=OPTIONAL_H_V1",
        flush=True,
    )
    print("PI2S_POLICY_RUNTIME " + json.dumps(runtime_evidence, sort_keys=True), flush=True)
    websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped, host="127.0.0.1", port=port, metadata=wrapped.metadata
    ).serve_forever()


def _intervention_audit_path(tuple_artifact: Path) -> Path:
    return tuple_artifact.with_name("h_intervention_audit.json")


def build_intervention_audit(tuple_artifact: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    payload = base.read_json(tuple_artifact)
    events = payload.get("observation_events", [])
    chunks = payload.get("action_chunks", [])
    if not events or not chunks:
        raise base.ContractError("optional tuple lacks H observation/query evidence")
    condition = str(expected["h_condition"])
    correct_by_step: dict[int, str] = {}
    for row in events:
        step = int(row["control_step"])
        correct_by_step.setdefault(step, str(row["correct_h_sha256"]))
    frozen_mean_sha = _sha256_array(train_mean_h())
    event_gates = []
    bootstrap_count = 0
    for row in events:
        step = int(row["control_step"])
        delivered = str(row["delivered_h_sha256"])
        if condition == "train_mean":
            valid = (
                row.get("requested_control_step") is None
                and row.get("actual_control_step") is None
                and row.get("bootstrap_affected") is False
                and delivered == frozen_mean_sha
            )
        else:
            requested = step - 5
            actual = max(0, requested)
            bootstrap = requested < 0
            bootstrap_count += int(bootstrap)
            valid = (
                row.get("requested_control_step") == requested
                and row.get("actual_control_step") == actual
                and row.get("bootstrap_affected") is bootstrap
                and delivered == correct_by_step.get(actual)
            )
        event_gates.append(
            valid
            and row.get("h_condition") == condition
            and row.get("h_sha256") == delivered
            and np.isfinite(float(row["delivered_h_l2"]))
        )
    delivered_by_step = {int(row["control_step"]): row["delivered_h_sha256"] for row in events}
    query_gates = []
    for row in chunks:
        control = row.get("sampling_control", {})
        step = int(row["observation_timestamp"])
        query_gates.append(
            control.get("h_condition") == condition
            and control.get("delivered_h_sha256") == delivered_by_step.get(step)
            and np.isfinite(float(control.get("delivered_h_l2")))
        )
    gates = {
        "tuple_identity_exact": payload.get("tuple") == dict(expected),
        "all_observation_events_conditioned_exactly": bool(event_gates) and all(event_gates),
        "all_policy_queries_received_audited_h": bool(query_gates) and all(query_gates),
        "correct_h_retained_for_counterfactual_audit": all(
            isinstance(row.get("correct_h_sha256"), str) for row in events
        ),
        "training_only_targets_absent": all(
            not row.get("training_only_fields_sent") for row in chunks
        ),
    }
    return {
        "schema": SCHEMA_AUDIT,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "tuple_key": expected["tuple_key"],
        "tuple_identity_sha256": expected["tuple_identity_sha256"],
        "h_condition": condition,
        "observation_event_count": len(events),
        "policy_query_count": len(chunks),
        "bootstrap_affected_observation_count": bootstrap_count,
        "train_mean_h_sha256": frozen_mean_sha,
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
    audit = _intervention_audit_path(artifact)
    if audit.exists():
        raise FileExistsError(audit)
    base.evaluate_tuple(tuple_spec_path, contact_socket, output, diagnostics, artifact, port)
    spec = base.read_json(tuple_spec_path)
    payload = build_intervention_audit(artifact, spec)
    if payload["status"] != "PASS":
        raise base.ContractError(f"H delivery audit failed: {spec['tuple_key']}")
    base.publish_json_no_clobber(audit, payload)


def validate_tuple_artifact(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    payload = BASE_VALIDATE_TUPLE(path, expected)
    audit_path = _intervention_audit_path(path)
    if not audit_path.is_file():
        raise base.ContractError(f"H delivery audit is absent: {audit_path}")
    audit = base.read_json(audit_path)
    if (
        audit.get("schema") != SCHEMA_AUDIT
        or audit.get("status") != "PASS"
        or audit.get("tuple_key") != expected["tuple_key"]
        or audit.get("tuple_identity_sha256") != expected["tuple_identity_sha256"]
        or any(value != "PASS" for value in audit.get("gates", {}).values())
    ):
        raise base.ContractError(f"H delivery audit drifted: {audit_path}")
    return payload


def block_paths(checkpoint_name: str) -> SimpleNamespace:
    lower = checkpoint_name.lower()
    return SimpleNamespace(
        run=RUN_ROOT / "blocks" / lower,
        log=LOG_ROOT / lower,
        cache=CACHE_ROOT / lower,
        socket=Path(f"/tmp/pi2s_s5_optional_{lower}.sock"),
    )


def build_manifest(source_commit: str) -> dict[str, Any]:
    protocol, protocol_sha, decision = _load_contract()
    if base.git_output("branch", "--show-current") != "develop/sim-benchmark":
        raise base.ContractError("optional manifest may only be frozen on develop/sim-benchmark")
    runner_relative = Path(__file__).resolve().relative_to(ROOT).as_posix()
    base_relative = BASE_SCRIPT.relative_to(ROOT).as_posix()
    for relative, path in (
        (runner_relative, Path(__file__).resolve()),
        (base_relative, BASE_SCRIPT),
    ):
        if base.committed_blob_sha(source_commit, relative) != base.sha256_file(path):
            raise base.ContractError(f"committed runner binding failed: {relative}")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"], cwd=ROOT, check=False
    ).returncode:
        raise base.ContractError("optional source commit is not an ancestor of HEAD")
    if base.sha256_file(BASE_RESULTS) != decision["evidence"]["base_72"]["sha256"]:
        raise base.ContractError("base 72 result drifted")
    base_check = subprocess.run(
        [str(base.UNIT_PYTHON), str(BASE_SCRIPT), "check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if base_check.returncode or "CHECK_PASS" not in base_check.stdout:
        raise base.ContractError("base 72 no longer passes its frozen check")
    tuples = build_optional_tuples(protocol)
    for row in tuples[::TUPLES_PER_BLOCK]:
        base.validate_checkpoint_presence(row)
    mean = train_mean_h()
    source_paths = tuple(base.SOURCE_PATHS) + (
        base_relative,
        runner_relative,
        DECISION.relative_to(ROOT).as_posix(),
    )
    return {
        "schema": SCHEMA_MANIFEST,
        "status": "FROZEN_PRE_EXECUTION",
        "created_at_utc": base.now_utc(),
        "source_commit": source_commit,
        "source_branch": base.git_output("branch", "--show-current"),
        "runner": "$REPO_ROOT/" + runner_relative,
        "runner_sha256": base.sha256_file(Path(__file__).resolve()),
        "base_runner": "$REPO_ROOT/" + base_relative,
        "base_runner_sha256": base.sha256_file(BASE_SCRIPT),
        "protocol_sha256": protocol_sha,
        "post_base_decision": "$REPO_ROOT/" + DECISION.relative_to(ROOT).as_posix(),
        "post_base_decision_sha256": base.sha256_file(DECISION),
        "base_results": "$PI2S_ROOT/artifacts/diagnostic_rollout_results.json",
        "base_results_sha256": base.sha256_file(BASE_RESULTS),
        "source_sha256": {relative: base.sha256_file(ROOT / relative) for relative in source_paths},
        "train_sidecar": "$PI2S_ROOT/source_snapshots/data/pi1_contact_sidecar.npz",
        "train_sidecar_sha256": base.sha256_file(TRAIN_SIDECAR),
        "train_mean_h_sha256": _sha256_array(mean),
        "train_mean_h_l2": float(np.linalg.norm(mean.astype(np.float64))),
        "lag_contract": {
            "clock": "raw physical control_step at 50 Hz",
            "requested_step": "current_control_step - 5",
            "bootstrap": "clamp to control_step 0 and mark affected",
            "causal": True,
        },
        "budget": {
            "canonical_tuples": EXPECTED_TUPLES,
            "checkpoints": len(FOCUS_IDS),
            "resets": 4,
            "conditions": list(CONDITIONS),
            "sampling_seeds": list(SAMPLING_SEEDS),
            "base_plus_optional_total": 120,
            "further_expansion_authorized": False,
        },
        "tuples": tuples,
        "tuple_sequence_sha256": base.canonical_sha(tuples),
        "performance_selection_used": False,
        "formal_track_a_scores_replaced": False,
        "optimizer_updates": 0,
        "checkpoint_writes": 0,
        "real_robot_used": False,
    }


def freeze_manifest(source_commit: str) -> None:
    if MANIFEST.exists() or PERSISTED_DECISION.exists():
        raise FileExistsError("optional manifest or persisted decision already exists")
    payload = build_manifest(source_commit)
    base.copy_file_no_clobber(DECISION, PERSISTED_DECISION)
    base.publish_json_no_clobber(MANIFEST, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "tuples": len(payload["tuples"]),
                "sha256": base.sha256_file(MANIFEST),
            },
            sort_keys=True,
        )
    )


def validate_manifest() -> dict[str, Any]:
    protocol, protocol_sha, decision = _load_contract()
    manifest = base.read_json(MANIFEST)
    if (
        manifest.get("schema") != SCHEMA_MANIFEST
        or manifest.get("status") != "FROZEN_PRE_EXECUTION"
    ):
        raise base.ContractError("optional manifest is not eligible")
    if manifest.get("protocol_sha256") != protocol_sha:
        raise base.ContractError("optional protocol binding drifted")
    if manifest.get("post_base_decision_sha256") != base.sha256_file(DECISION):
        raise base.ContractError("optional decision binding drifted")
    if not PERSISTED_DECISION.is_file() or base.sha256_file(PERSISTED_DECISION) != base.sha256_file(
        DECISION
    ):
        raise base.ContractError("persisted optional decision drifted")
    if base.sha256_file(BASE_RESULTS) != decision["evidence"]["base_72"]["sha256"]:
        raise base.ContractError("base result drifted after optional freeze")
    source_commit = str(manifest.get("source_commit"))
    runner_relative = Path(__file__).resolve().relative_to(ROOT).as_posix()
    base_relative = BASE_SCRIPT.relative_to(ROOT).as_posix()
    if (
        manifest.get("runner_sha256") != base.sha256_file(Path(__file__).resolve())
        or base.committed_blob_sha(source_commit, runner_relative) != manifest["runner_sha256"]
        or manifest.get("base_runner_sha256") != base.sha256_file(BASE_SCRIPT)
        or base.committed_blob_sha(source_commit, base_relative) != manifest["base_runner_sha256"]
    ):
        raise base.ContractError("optional runner binding drifted")
    observed_sources = {
        relative: base.sha256_file(ROOT / relative) for relative in manifest["source_sha256"]
    }
    if observed_sources != manifest["source_sha256"]:
        raise base.ContractError("optional execution source drifted")
    if base.sha256_file(TRAIN_SIDECAR) != manifest["train_sidecar_sha256"]:
        raise base.ContractError("optional TRAIN sidecar drifted")
    expected = build_optional_tuples(protocol)
    if manifest.get("tuples") != expected or manifest.get(
        "tuple_sequence_sha256"
    ) != base.canonical_sha(expected):
        raise base.ContractError("optional tuple manifest drifted")
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
            "schema": "tactile3d-unit.s4-3-pi2s-optional-h-progress.v1",
            "completed_blocks": sum(
                1
                for identifier in FOCUS_IDS
                if sum(
                    row["tuple_key"].startswith(identifier.lower() + "__") for row in assignments
                )
                == TUPLES_PER_BLOCK
            ),
            "planned_blocks": len(FOCUS_IDS),
            "optional_48_started": True,
            "base_results_sha256": manifest["base_results_sha256"],
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
            "phase": "S5_OPTIONAL_48_H_INTERVENTIONS",
            "updated_at_utc": payload["updated_at_utc"],
            "supervisor_pid": os.getpid(),
            "source_commit": manifest["source_commit"],
            "protocol_sha256": manifest["protocol_sha256"],
            "manifest_sha256": payload["manifest_sha256"],
            "progress_path": str(PROGRESS),
            "supervisor_log": str(LOG_ROOT / "supervisor.log"),
            "completed_canonical_tuples": payload["completed_canonical_tuples"],
            "planned_canonical_tuples": EXPECTED_TUPLES,
            "base_canonical_tuples_completed": 72,
            "total_s5_canonical_tuples_if_complete": 120,
            "long_job_running": status == "RUNNING",
            "optional_48_started": True,
            "training_performed": False,
            "push_performed": False,
            "worktree_removal_performed": False,
            "real_robot_used": False,
        },
    )


def aggregate_results(
    manifest: Mapping[str, Any],
    assignments: Sequence[Mapping[str, Any]],
    snapshots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result = BASE_AGGREGATE_RESULTS(manifest, assignments, snapshots)
    intervention_audits = []
    for episode in result["episodes"]:
        audit_path = _intervention_audit_path(Path(episode["artifact"]))
        audit = base.read_json(audit_path)
        intervention_audits.append(
            {
                "tuple_key": episode["tuple"]["tuple_key"],
                "path": str(audit_path),
                "sha256": base.sha256_file(audit_path),
                "h_condition": audit["h_condition"],
                "observation_event_count": audit["observation_event_count"],
                "policy_query_count": audit["policy_query_count"],
                "bootstrap_affected_observation_count": audit[
                    "bootstrap_affected_observation_count"
                ],
            }
        )
    result.update(
        {
            "schema": SCHEMA_RESULTS,
            "optional_48_executed": True,
            "base_72_results_sha256": manifest["base_results_sha256"],
            "post_base_decision_sha256": manifest["post_base_decision_sha256"],
            "intervention_audits": intervention_audits,
            "scientific_role": "POST_EXPOSURE_MECHANISM_DIAGNOSTIC_NOT_FORMAL_SCORE",
            "further_rollout_expansion_authorized": False,
        }
    )
    return result


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
        "completed_canonical_tuples": len(completed),
        "planned_canonical_tuples": EXPECTED_TUPLES,
        "remaining_canonical_tuples": EXPECTED_TUPLES - len(completed),
        "progress": progress,
        "results_present": result is not None,
        "results_sha256": base.sha256_file(RESULTS) if RESULTS.is_file() else None,
        "optional_48_started": progress is not None or result is not None,
        "base_72_results_sha256": manifest["base_results_sha256"],
    }


def activate_base_runtime() -> None:
    # Base functions resolve these names in their own module globals.  Pointing
    # them at the optional immutable roots lets us reuse the validated runner
    # while keeping every base artifact and code byte untouched.
    base.__file__ = str(Path(__file__).resolve())
    base.MANIFEST = MANIFEST
    base.RESULTS = RESULTS
    base.RUN_ROOT = RUN_ROOT
    base.LOG_ROOT = LOG_ROOT
    base.CACHE_ROOT = CACHE_ROOT
    base.PROGRESS = PROGRESS
    base.RESUME = RESUME
    base.FOCUS_KEYS = FOCUS_KEYS
    base.SAMPLING_SEEDS = SAMPLING_SEEDS
    base.EXPECTED_TUPLES = EXPECTED_TUPLES
    base.SCHEMA_MANIFEST = SCHEMA_MANIFEST
    base.SCHEMA_TUPLE = SCHEMA_TUPLE
    base.SCHEMA_RESULTS = SCHEMA_RESULTS
    base.build_base_tuples = build_optional_tuples
    base.validate_manifest = validate_manifest
    base.build_diagnostic_environment = build_optional_environment
    base.validate_tuple_artifact = validate_tuple_artifact
    base.block_paths = block_paths
    base.progress_payload = progress_payload
    base.update_progress = update_progress
    base.aggregate_results = aggregate_results
    base.serve_policy = serve_policy


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
                    "tuples": len(manifest["tuples"]),
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
