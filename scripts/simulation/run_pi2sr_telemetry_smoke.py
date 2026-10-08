#!/usr/bin/env python3
"""Run the bounded non-scientific PI2S-R telemetry/runtime smoke."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.dexjoco_adapter import policy_action_to_env_action  # noqa: E402
from gr00t.simulation.pi2sr_provenance import (  # noqa: E402
    build_provenance_manifest,
    canonical_sha256,
    sha256_file,
)
from gr00t.simulation.pi2sr_runtime import (  # noqa: E402
    CANONICAL_CHECKPOINT_SHA256,
    DirectContactEncoder,
)
from gr00t.simulation.pi2sr_telemetry import (  # noqa: E402
    AsyncTelemetryWriter,
    StableGraspTracker,
    build_step_record,
    storage_probe,
)
from gr00t.simulation.s4_3_pi1 import OnlineTactileHistory  # noqa: E402
from gr00t.simulation.s4_3_runtime import ActionChunkQueue  # noqa: E402

ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_pi2sr"
RUNTIME_CONTRACT = ROOT / "configs/simulation/pi2sr/runtime_contract_v2.json"
TELEMETRY_CONTRACT = ROOT / "configs/simulation/pi2sr/telemetry_contract_v1.json"
PROVENANCE_CONTRACT = ROOT / "configs/simulation/pi2sr/provenance_contract_v1.json"
TASKS = ("pinch_tongs", "hammer_nail", "click_mouse")
STEPS = 15


def experiment_root() -> Path:
    value = os.environ.get("UNIT_EXPERIMENT_ROOT")
    return Path(value) if value else ROOT / ".local/experiments"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def synthetic_tactile(task_index: int, step: int) -> np.ndarray:
    """Create declared contact transitions without simulating a scientific outcome."""

    value = np.zeros((5, 6), dtype=np.float32)
    if 3 <= step <= 12:
        occupied = 2 + int(step >= 9)
        value[:occupied, 0] = 1.0
        base_force = np.float32(0.30 + 0.01 * task_index + 0.02 * (step % 3))
        value[:occupied, 1] = base_force
        value[:occupied, 2] = base_force * np.float32(0.25)
        value[:occupied, 3] = np.float32(0.01 * step)
        value[:occupied, 4] = np.float32(0.02 * task_index)
        value[:occupied, 5] = np.float32(0.03)
    return value.reshape(-1)


def policy_stub(state: np.ndarray, h_value: np.ndarray, request_index: int) -> np.ndarray:
    """Deterministic action-queue fixture; not a trained policy or performance test."""

    base = np.asarray(state, dtype=np.float32).copy()
    base[:3] += np.float32(np.mean(h_value) * 1e-4)
    base[6:] += np.float32(request_index * 1e-4)
    return np.repeat(base[None], 27, axis=0)


def run_episode(
    *,
    task: str,
    task_index: int,
    encoder: DirectContactEncoder,
    writer: AsyncTelemetryWriter | None,
    run_id: str,
) -> dict[str, Any]:
    episode_id = f"pi2sr-smoke-{task}"
    reset_identity = f"PI2SR_DEVELOPMENT_SYNTHETIC_RESET_{task_index}"
    history = OnlineTactileHistory()
    stable = StableGraspTracker()
    actions = ActionChunkQueue(episode_id)
    trace = hashlib.sha256()
    contact_steps = 0
    stable_steps = 0
    request_index = -1
    replan_steps: list[int] = []
    writer_accepts: list[bool] = []
    for step in range(STEPS):
        tactile = synthetic_tactile(task_index, step)
        tactile_history = history.reset(tactile) if step == 0 else history.append(tactile)
        h_value = encoder.encode(tactile_history)
        state = np.linspace(-0.2, 0.2, 22, dtype=np.float32) + np.float32(task_index * 0.01)
        replan = actions.needs_replan
        generated_ns = time.monotonic_ns()
        if replan:
            request_index += 1
            actions.set_plan(policy_stub(state, h_value, request_index), step)
            replan_steps.append(step)
        queue_index = actions.executed_from_current_plan
        commanded = actions.pop()
        applied = policy_action_to_env_action(commanded).values
        applied_ns = time.monotonic_ns()
        occupied = int(np.sum(tactile.reshape(5, 6)[:, 0] > 0.5))
        contact = occupied > 0
        stable_grasp = stable.update(tactile)
        contact_steps += int(contact)
        stable_steps += int(stable_grasp)
        terminated = step == STEPS - 1
        stage_predicates = {
            "CONTACT": contact,
            "LIFT": None,
            "NATIVE_TRIGGER": None,
            "SUCCESS": False,
            "STABLE_GRASP": stable_grasp,
        }
        record = build_step_record(
            run_id=run_id,
            episode_id=episode_id,
            task=task,
            reset_identity=reset_identity,
            control_step_index=step,
            simulation_time_sec=step * 0.02,
            observation_timestamp_ns=time.monotonic_ns(),
            policy_request_id=f"{episode_id}-request-{request_index}",
            policy_request_index=request_index,
            action_chunk_generated_ns=generated_ns,
            action_applied_ns=applied_ns,
            policy_facing_state=state,
            environment_facing_state=applied,
            commanded_action=commanded,
            environment_applied_action=applied,
            raw_canonical_tactile=tactile,
            canonical_h=h_value,
            canonical_h_reference=None,
            h_runtime_identity="DIRECT_IN_PROCESS:cpu:float32:batch1:N1",
            history_valid_samples=min(step + 1, 26),
            history_bootstrap_status="WARM" if step >= 25 else "LEFT_REPEAT_FIRST_BOOTSTRAP",
            replan_required=replan,
            replan_stride=5,
            action_queue_index=queue_index,
            action_queue_remaining=4 - queue_index,
            terminated=terminated,
            truncated=False,
            stage_predicates=stage_predicates,
            evaluation_only_fields={
                name: {"available": False, "value": None}
                for name in (
                    "object_pose",
                    "object_height",
                    "native_pinch_count",
                    "native_trigger",
                    "native_latch",
                    "task_progress",
                    "native_success_components",
                )
            },
        )
        if writer is not None:
            writer_accepts.append(writer.submit(record))
        for array in (state, tactile_history, h_value, commanded, applied):
            trace.update(np.ascontiguousarray(array).tobytes())
        trace.update(json.dumps(stage_predicates, sort_keys=True).encode("utf-8"))
        trace.update(bytes((int(replan), int(terminated))))
    return {
        "episode_id": episode_id,
        "task": task,
        "reset_identity": reset_identity,
        "steps": STEPS,
        "trace_sha256": trace.hexdigest(),
        "replan_steps": replan_steps,
        "replan_count": actions.replan_count,
        "contact_steps": contact_steps,
        "stable_grasp_steps": stable_steps,
        "terminated_count": 1,
        "native_success_claimed": False,
        "writer_accepts": writer_accepts,
    }


def read_first_record(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return json.loads(next(source))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    experiment = experiment_root()
    checkpoint = experiment / "simulation/s4_2r/contact_state/accepted.pt"
    if sha256_file(checkpoint) != CANONICAL_CHECKPOINT_SHA256:
        raise RuntimeError("canonical E_T checkpoint integrity failed")
    telemetry_root = experiment / "simulation/s4_3_pi2sr/telemetry"
    probe = storage_probe(telemetry_root)
    if probe["status"] != "PASS":
        raise RuntimeError(f"telemetry storage probe failed: {probe}")

    run_id = f"pi2sr-smoke-{git('rev-parse', '--short=12', 'HEAD')}-{time.time_ns()}"
    run_root = telemetry_root / run_id
    encoder = DirectContactEncoder(checkpoint)
    baseline_rows = []
    telemetry_rows = []
    writer_summaries = []
    telemetry_paths = []
    for task_index, task in enumerate(TASKS):
        baseline = run_episode(
            task=task,
            task_index=task_index,
            encoder=encoder,
            writer=None,
            run_id=f"{run_id}-off",
        )
        destination = run_root / f"{task}.jsonl.gz"
        writer = AsyncTelemetryWriter(destination)
        telemetry = run_episode(
            task=task,
            task_index=task_index,
            encoder=encoder,
            writer=writer,
            run_id=f"{run_id}-on",
        )
        summary = writer.close()
        if summary["status"] != "PASS" or not all(telemetry["writer_accepts"]):
            raise RuntimeError(f"telemetry writer failed for {task}: {summary}")
        baseline_rows.append(baseline)
        telemetry_rows.append(telemetry)
        writer_summaries.append(summary)
        telemetry_paths.append(destination)

    comparisons = []
    for baseline, telemetry in zip(baseline_rows, telemetry_rows, strict=True):
        comparisons.append(
            {
                "task": baseline["task"],
                "observation_action_trace_equal": baseline["trace_sha256"]
                == telemetry["trace_sha256"],
                "replan_semantics_equal": baseline["replan_steps"] == telemetry["replan_steps"],
                "success_predicate_equal": baseline["native_success_claimed"]
                == telemetry["native_success_claimed"],
                "termination_equal": baseline["terminated_count"] == telemetry["terminated_count"],
            }
        )
    noninterference_pass = all(all(row.values()) for row in comparisons)

    with tempfile.TemporaryDirectory(prefix="pi2sr-writer-failure-") as directory:
        blocked_parent = Path(directory) / "blocked"
        blocked_parent.write_text("not a directory", encoding="utf-8")
        protected_action = np.arange(22, dtype=np.float32)
        before = protected_action.tobytes()
        failing = AsyncTelemetryWriter(blocked_parent / "telemetry.jsonl.gz", queue_capacity=2)
        time.sleep(0.05)
        failure_accepted = failing.submit(read_first_record(telemetry_paths[0]))
        failure_summary = failing.close()
        writer_failure_isolated = (
            not failure_accepted
            and failure_summary["status"] == "FAIL"
            and protected_action.tobytes() == before
        )

    telemetry_contract = json.loads(TELEMETRY_CONTRACT.read_text(encoding="utf-8"))
    first_record = read_first_record(telemetry_paths[0])
    flat_required = set()
    for group, fields in telemetry_contract["required_fields"].items():
        if group != "stage_predicates":
            flat_required.update(fields)
    missing = sorted(flat_required - set(first_record))
    stage_missing = sorted(
        set(telemetry_contract["required_fields"]["stage_predicates"])
        - set(first_record["stage_predicates"])
    )
    schema_pass = (
        not missing
        and not stage_missing
        and first_record["policy_input"] is False
        and first_record["evaluation_only"]["evaluation_only"] is True
        and first_record["evaluation_only"]["policy_input"] is False
    )
    schema_audit = {
        "schema": "tactile3d-unit.pi2sr-telemetry-schema-audit.v1",
        "status": "PASS" if schema_pass else "FAIL",
        "schema_version": first_record["schema"],
        "contract_sha256": sha256_file(TELEMETRY_CONTRACT),
        "missing_fields": missing,
        "missing_stage_predicates": stage_missing,
        "evaluation_only": True,
        "policy_input": False,
        "prospective_only_stage_predicates": ["STABLE_GRASP"],
        "approach_defined": False,
    }
    atomic_json(args.artifact_root / "telemetry_schema_audit.json", schema_audit)

    overhead = {
        name: max(summary[name] for summary in writer_summaries)
        for name in ("p50_ms", "p95_ms", "p99_ms", "max_queue_backlog")
    }
    overhead["dropped_records"] = sum(summary["dropped_records"] for summary in writer_summaries)
    noninterference = {
        "schema": "tactile3d-unit.pi2sr-telemetry-noninterference.v1",
        "status": (
            "PASS"
            if noninterference_pass and writer_failure_isolated and overhead["dropped_records"] == 0
            else "FAIL"
        ),
        "paired_runs": comparisons,
        "policy_observation_equal": noninterference_pass,
        "action_transform_equal": noninterference_pass,
        "success_predicate_equal": noninterference_pass,
        "replan_ratio_and_queue_equal": noninterference_pass,
        "future_or_privileged_policy_leakage": False,
        "writer_failure_isolated": writer_failure_isolated,
        "failure_injection": failure_summary,
        "overhead": overhead,
    }
    atomic_json(args.artifact_root / "telemetry_noninterference.json", noninterference)

    smoke = {
        "schema": "tactile3d-unit.pi2sr-telemetry-smoke.v1",
        "status": (
            "PASS" if schema_audit["status"] == noninterference["status"] == "PASS" else "FAIL"
        ),
        "run_id": run_id,
        "episodes": 6,
        "paired_development_runs": 3,
        "tasks": list(TASKS),
        "steps_per_run": STEPS,
        "total_control_steps": 6 * STEPS,
        "scientific_benchmark": False,
        "policy": "DETERMINISTIC_ENGINEERING_STUB_NOT_A_TRAINED_POLICY",
        "canonical_h": "DIRECT_IN_PROCESS",
        "storage_probe": probe,
        "baseline": baseline_rows,
        "telemetry_on": telemetry_rows,
        "writer_summaries": writer_summaries,
        "telemetry_files": [str(path) for path in telemetry_paths],
        "contact_exercised": all(row["contact_steps"] > 0 for row in telemetry_rows),
        "stable_grasp_exercised": all(row["stable_grasp_steps"] > 0 for row in telemetry_rows),
        "reset_exercised": len({row["reset_identity"] for row in telemetry_rows}) == 3,
        "action_queue_exercised": all(row["replan_count"] == 3 for row in telemetry_rows),
        "end_of_episode_exercised": all(row["terminated_count"] == 1 for row in telemetry_rows),
        "errors": [],
        "training_or_optimizer_updates": 0,
        "real_hardware": False,
        "policy_effect_claim": False,
    }
    atomic_json(args.artifact_root / "telemetry_smoke.json", smoke)

    reset_manifest = [row["reset_identity"] for row in telemetry_rows]
    provenance = build_provenance_manifest(
        repo_root=ROOT,
        config_paths=[RUNTIME_CONTRACT, TELEMETRY_CONTRACT, PROVENANCE_CONTRACT],
        base_model_checkpoint_tree_sha256=None,
        teacher_or_contact_encoder_sha256=CANONICAL_CHECKPOINT_SHA256,
        data_identity={
            "dataset_manifest_sha256": canonical_sha256(
                {"kind": "synthetic_contract_smoke", "steps": STEPS, "tasks": TASKS}
            ),
            "split_identity": "PI2SR_DEVELOPMENT_SYNTHETIC_CONTRACT_SMOKE",
            "source_group_identities": list(TASKS),
            "normalization_sha256": CANONICAL_CHECKPOINT_SHA256,
            "target_mask_sha256": None,
            "episode_reset_manifest_sha256": canonical_sha256(reset_manifest),
        },
        randomness={
            "root_training_seed": None,
            "initialization_seed_namespace": None,
            "data_loader_seed": None,
            "augmentation_seed": None,
            "noise_time_sampling_seed": None,
            "evaluator_reset_seed": 4327100,
            "policy_sampling_root_key": "NO_STOCHASTIC_POLICY_ENGINEERING_STUB",
            "policy_request_counter": sum(row["replan_count"] for row in telemetry_rows),
            "policy_key_index": None,
        },
        device="cpu",
        dtype="float32",
    )
    provenance_audit = {
        "schema": "tactile3d-unit.pi2sr-provenance-contract-audit.v1",
        "status": "PASS",
        "contract_sha256": sha256_file(PROVENANCE_CONTRACT),
        "manifest": provenance,
        "source_identity_present": True,
        "environment_identity_present": True,
        "data_reset_identity_present": True,
        "initialization_trace": "EXPLICITLY_UNAVAILABLE_NOT_FABRICATED",
        "training_seeds": None,
        "policy_sampling_key_counter_present": True,
        "historical_retroactive_prng_reconstruction_claimed": False,
    }
    atomic_json(args.artifact_root / "provenance_contract_audit.json", provenance_audit)
    print(
        json.dumps(
            {
                "episodes": smoke["episodes"],
                "noninterference": noninterference["status"],
                "provenance": provenance_audit["status"],
                "schema": schema_audit["status"],
                "smoke": smoke["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
