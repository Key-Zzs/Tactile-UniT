#!/usr/bin/env python3
"""Recompute PI2S historical checkpoint, failure-stage, and H-norm evidence.

This is a read-only analysis of the frozen PI2B checkpoints and canonical 3,000
rollouts.  It deliberately does not load model weights, run a simulator, use a
GPU, or infer unrecorded task stages.  Checkpoint tree identities are consumed
from the prior live-hash freezes; the four large intermediate checkpoints are
bound to the separately completed PI2S hash artifact instead of being rescanned.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = (ROOT / ".local/experiments").resolve()
PI2B = EXPERIMENT_ROOT / "simulation/s4_3_pi2b_policy"
PI2S = EXPERIMENT_ROOT / "simulation/s4_3_pi2s"
OUTPUT_ROOT = PI2S / "artifacts"

MODELS = ("B0", "B_VA27", "B1", "B_HVA", "B2")
SEEDS = (42, 43, 44)
RESET_SEEDS = (16, 17, 18, 19)
FOCAL = {
    ("B0", 43),
    ("B_VA27", 43),
    ("B1", 43),
    ("B_HVA", 42),
    ("B_HVA", 43),
    ("B2", 43),
}
OUTPUTS = {
    "checkpoint_recipe_matrix": OUTPUT_ROOT / "checkpoint_recipe_matrix.json",
    "historical_failure_stage_analysis": OUTPUT_ROOT / "historical_failure_stage_analysis.json",
    "h_distribution_diagnostics": OUTPUT_ROOT / "h_distribution_diagnostics.json",
}

EXPECTED_TOTAL = 3_000
EXPECTED_SUCCESSES = 803
EXPECTED_FAILURES = 2_197
EXPECTED_INTEGRATION_COMMIT = "6602c05d78ff97a150255cc37aae83e1f15c9df6"


class AuditError(RuntimeError):
    """Raised when frozen evidence violates an analysis invariant."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value))


class InputRegistry:
    """Read each input once and retain a SHA256-bound provenance ledger."""

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}

    def _record(self, path: Path, data: bytes, role: str, expected_sha256: str | None) -> None:
        resolved = str(path.resolve())
        observed = sha256_bytes(data)
        if expected_sha256 is not None:
            require(observed == expected_sha256, f"input SHA mismatch: {path}")
        if resolved in self._records:
            previous = self._records[resolved]
            require(previous["sha256"] == observed, f"input changed while analyzing: {path}")
            if role not in previous["roles"]:
                previous["roles"].append(role)
                previous["roles"].sort()
            return
        self._records[resolved] = {
            "path": resolved,
            "bytes": len(data),
            "sha256": observed,
            "roles": [role],
            "expected_sha256": expected_sha256,
            "expected_sha256_status": "NOT_PROVIDED" if expected_sha256 is None else "MATCH",
        }

    def bytes(self, path: Path, role: str, expected_sha256: str | None = None) -> bytes:
        require(path.is_file(), f"missing input: {path}")
        data = path.read_bytes()
        self._record(path, data, role, expected_sha256)
        return data

    def json(self, path: Path, role: str, expected_sha256: str | None = None) -> Any:
        data = self.bytes(path, role, expected_sha256)
        try:
            return json.loads(data)
        except json.JSONDecodeError as error:
            raise AuditError(f"invalid JSON input {path}: {error}") from error

    def records(self) -> list[dict[str, Any]]:
        return [json_copy(self._records[key]) for key in sorted(self._records)]

    def observed_sha256(self, path: Path) -> str:
        resolved = str(path.resolve())
        require(resolved in self._records, f"input was not registered before SHA lookup: {path}")
        return str(self._records[resolved]["sha256"])

    def manifest_sha256(self) -> str:
        return sha256_bytes(canonical_bytes(self.records()))


def output_artifact(schema: str, body: dict[str, Any], inputs: InputRegistry) -> dict[str, Any]:
    input_records = inputs.records()
    expected_matches = sum(row["expected_sha256_status"] == "MATCH" for row in input_records)
    expected_missing = sum(row["expected_sha256_status"] == "NOT_PROVIDED" for row in input_records)
    semantic = {
        "schema": schema,
        **body,
        "producer": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_bytes(Path(__file__).read_bytes()),
            "execution": "CPU_INPUT_READ_ONLY_ARTIFACT_WRITE_ONLY_NO_MODEL_LOAD_NO_SIMULATOR_NO_GPU",
        },
        "input_manifest": input_records,
        "input_manifest_sha256": inputs.manifest_sha256(),
        "input_manifest_summary": {
            "observed_sha256_recorded": len(input_records),
            "provided_expected_sha256_match": expected_matches,
            "expected_sha256_not_provided": expected_missing,
        },
    }
    return {
        **semantic,
        "created_at_utc": now_utc(),
        "semantic_sha256": sha256_bytes(canonical_bytes(semantic)),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode() + b"\n"
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    observed = path.read_bytes()
    require(observed == data, f"atomic write readback mismatch: {path}")
    return sha256_bytes(observed)


def check_artifact(path: Path, expected: dict[str, Any]) -> None:
    require(path.is_file(), f"missing output for --check: {path}")
    current = json.loads(path.read_bytes())
    created = current.pop("created_at_utc", None)
    observed_semantic = current.pop("semantic_sha256", None)
    expected_semantic = json_copy(expected)
    expected_semantic.pop("created_at_utc", None)
    expected_semantic_sha = expected_semantic.pop("semantic_sha256", None)
    require(isinstance(created, str) and created, f"missing created_at_utc: {path}")
    require(observed_semantic == sha256_bytes(canonical_bytes(current)), f"bad semantic SHA: {path}")
    require(expected_semantic_sha == sha256_bytes(canonical_bytes(expected_semantic)), "internal semantic SHA error")
    require(current == expected_semantic, f"output is stale or drifted: {path}")


def quantiles(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    require(bool(ordered), "quantiles require at least one value")

    def at(probability: float) -> float:
        position = probability * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "min": ordered[0],
        "q01": at(0.01),
        "q05": at(0.05),
        "q25": at(0.25),
        "median": at(0.5),
        "q75": at(0.75),
        "q95": at(0.95),
        "q99": at(0.99),
        "max": ordered[-1],
    }


def parse_step_metrics(log_bytes: bytes) -> dict[str, Any]:
    text = log_bytes.decode(errors="replace").replace("\r", "\n")
    rows: list[tuple[int, dict[str, float]]] = []
    for match in re.finditer(r"Step\s+(\d+):\s*([^\n]+)", text):
        step = int(match.group(1))
        metrics: dict[str, float] = {}
        for name, value in re.findall(r"([A-Za-z0-9_]+)=([-+0-9.eE]+)", match.group(2)):
            metrics[name] = float(value)
        if metrics:
            rows.append((step, metrics))
    require(bool(rows), "training log has no parsed Step metrics")
    by_step: dict[int, dict[str, float]] = {}
    for step, metrics in rows:
        by_step[step] = metrics
    steps = sorted(by_step)
    return {
        "metric_records_parsed": len(by_step),
        "first_logged_step": steps[0],
        "last_logged_step": steps[-1],
        "first_metrics": by_step[steps[0]],
        "last_metrics": by_step[steps[-1]],
        "gradient_metric_names": sorted(
            name for name in set().union(*(row.keys() for row in by_step.values())) if "grad_norm" in name
        ),
        "all_parsed_metrics_finite": all(math.isfinite(value) for row in by_step.values() for value in row.values()),
    }


def seed42_evidence_paths() -> dict[str, dict[str, Any]]:
    local = ROOT / ".local/artifacts/simulation"
    return {
        "B0": {
            "stage": "S4.3-PI0",
            "entrypoint": "third_party/dexjoco/openpi/scripts/train.py",
            "freeze": local / "s4_3_pi0/official_training_config.json",
            "completion": local / "s4_3_pi0/training_completion.json",
            "launch": local / "s4_3_pi0/training_launch.json",
        },
        "B_VA27": {
            "stage": "S4.3-PI2N",
            "entrypoint": "scripts/simulation/train_s4_3_pi2n.py",
            "freeze": local / "s4_3_pi2n/b_va27_training_freeze.json",
            "completion": local / "s4_3_pi2n/b_va27_training_completion.json",
        },
        "B1": {
            "stage": "S4.3-PI1B",
            "entrypoint": "scripts/simulation/train_s4_3_pi1.py",
            "freeze": local / "s4_3_pi1/training_protocol_freeze.json",
            "completion": local / "s4_3_pi1/pi1b_training_completion.json",
            "launch": local / "s4_3_pi1/pi1b_launch.json",
        },
        "B_HVA": {
            "stage": "S4.3-PI2M",
            "entrypoint": "scripts/simulation/train_s4_3_pi2m_bhva.py",
            "freeze": local / "s4_3_pi2m/training_protocol_freeze.json",
            "completion": local / "s4_3_pi2m/training_completion.json",
            "launch": local / "s4_3_pi2m/training_launch.json",
        },
        "B2": {
            "stage": "S4.3-PI1C",
            "entrypoint": "scripts/simulation/train_s4_3_pi1.py",
            "freeze": local / "s4_3_pi1/training_protocol_freeze.json",
            "completion": local / "s4_3_pi1/pi1c_training_completion.json",
            "launch": local / "s4_3_pi1/pi1c_launch.json",
        },
    }


def completion_progress(model: str, completion: dict[str, Any]) -> tuple[int, int | None, str]:
    if model == "B0":
        step = int(completion["train_state_step"])
        return step, step, "EXPLICIT_TRAIN_STATE_STEP"
    completed_step = int(completion["optimizer_steps"])
    gates = completion.get("gates", {})
    if gates.get("restored_train_state_step_30000") == "PASS":
        return completed_step, completed_step, "EXPLICIT_COLD_RESTORE_STEP_GATE"
    require(gates.get("checkpoint_cold_load") == "PASS", f"seed42 {model} lacks cold-load gate")
    return completed_step, None, "OPTIMIZER_STEP_PLUS_COLD_LOAD_GATE_NO_SCALAR_RESTORE_FIELD"


def source_commit_from_seed42(model: str, freeze: dict[str, Any]) -> str:
    if model == "B0":
        return str(freeze["source_commit"])
    if model in {"B_HVA", "B_VA27"}:
        return str(freeze["git_head"])
    return "UNVERIFIABLE_NOT_RECORDED_IN_SELECTED_PI1_ARTIFACTS"


def recipe_matrix(inputs: InputRegistry) -> tuple[dict[str, Any], dict[tuple[str, int], str]]:
    pre_path = PI2B / "artifacts/pre_final_freeze.json"
    recipes_path = ROOT / "configs/simulation/pi2b_policy/model_recipes.json"
    protocol_path = ROOT / "configs/simulation/pi2b_policy/protocol.json"
    recipe_audit_path = PI2B / "artifacts/model_recipe_audit.json"
    import_path = PI2B / "artifacts/import_origins.json"
    runs_path = PI2B / "artifacts/training_runs.json"
    completion_path = PI2B / "artifacts/training_completion.json"
    resources_path = PI2B / "artifacts/training_resource_summary.json"
    timing_path = PI2B / "artifacts/time_target_mask_audit.json"
    intermediate_path = PI2S / "artifacts/intermediate_checkpoint_hashes.json"
    snapshot_manifest_path = PI2B / "artifacts/code_snapshot_manifest.json"

    pre = inputs.json(pre_path, "15-final-checkpoint live-hash freeze")
    recipes = inputs.json(recipes_path, "frozen five-model recipes")
    protocol = inputs.json(protocol_path, "PI2B training protocol")
    recipe_audit = inputs.json(recipe_audit_path, "Track A recipe audit")
    imports = inputs.json(import_path, "historical Track A import-origin audit")
    training_runs = inputs.json(runs_path, "frozen Track A per-run configs")
    training_completion = inputs.json(completion_path, "Track A cold-restore completion")
    resources = inputs.json(resources_path, "actual Track A device/runtime records")
    timing = inputs.json(timing_path, "target-time and mask audit")
    intermediate = inputs.json(intermediate_path, "four intermediate checkpoint content hashes")
    snapshot_manifest = inputs.json(snapshot_manifest_path, "Track A immutable training-source snapshot manifest")

    require(pre.get("status") == "PASS", "pre-final freeze not PASS")
    require(recipes.get("status") == "FROZEN_BEFORE_NEW_TRAINING", "recipes not frozen")
    require(protocol.get("optimizer_steps") == 30_000, "protocol optimizer steps drift")
    require(recipe_audit.get("status") == "PASS", "recipe audit not PASS")
    require(imports.get("status") == "PASS", "import-origin audit not PASS")
    require(training_completion.get("status") == "PASS", "training completion not PASS")
    require(resources.get("status") == "PASS", "resource summary not PASS")
    require(timing.get("status") == "PASS", "time/target audit not PASS")
    require(intermediate.get("status") == "PASS", "intermediate hash artifact not PASS")
    require(intermediate.get("source_commit") == EXPECTED_INTEGRATION_COMMIT, "intermediate source commit drift")
    require(all(value is True for value in intermediate.get("gates", {}).values()), "intermediate hash gate failure")
    require(snapshot_manifest.get("status") == "PASS", "Track A source snapshot manifest not PASS")
    inputs.bytes(
        Path(snapshot_manifest["path"]),
        "immutable Track A training-source snapshot tar",
        snapshot_manifest["sha256"],
    )

    # Bind the current integrated sources to the hashes frozen by Track A.  The
    # historical import-origin artifact remains historical; this is not a claim
    # that today's Python process imports from that worktree.
    integrated_source_comparison = {}
    for relative, expected in training_runs["rows"][0]["source_sha256"].items():
        current_bytes = inputs.bytes(ROOT / relative, "current integrated copy of Track A frozen training source")
        observed = sha256_bytes(current_bytes)
        integrated_source_comparison[relative] = {
            "track_a_frozen_sha256": expected,
            "current_integrated_sha256": observed,
            "match": observed == expected,
        }
    mismatched_integrated_sources = {
        relative for relative, row in integrated_source_comparison.items() if not row["match"]
    }
    require(
        mismatched_integrated_sources == {"gr00t/simulation/pi2b_policy/contract.py"},
        f"unexpected integrated-source mismatch set: {sorted(mismatched_integrated_sources)}",
    )
    imported_openpi = ROOT / ".local/external/simulation/s4_3_pi1/openpi"
    starting = inputs.json(PI2B / "artifacts/starting_integrity.json", "Track A source and dataset start audit")
    for relative, expected in starting["source"]["openpi_hashes"].items():
        inputs.bytes(imported_openpi / relative, "accepted historical OpenPI source", expected)

    run_rows = {(row["model_id"], int(row["seed"])): row for row in training_runs["rows"]}
    completion_rows = {
        (row["model"], int(row["training_seed"])): row for row in training_completion["runs"]
    }
    resource_rows = {(row["model"], int(row["training_seed"])): row for row in resources["runs"]}
    require(set(run_rows) == {(model, seed) for model in MODELS for seed in (43, 44)}, "Track A run matrix drift")
    require(set(completion_rows) == set(run_rows), "Track A completion matrix drift")
    require(set(resource_rows) == set(run_rows), "Track A resource matrix drift")

    seed42: dict[str, dict[str, Any]] = {}
    for model, spec in seed42_evidence_paths().items():
        freeze = inputs.json(spec["freeze"], f"seed42 {model} training freeze")
        completion = inputs.json(spec["completion"], f"seed42 {model} completion/cold-load evidence")
        launch = inputs.json(spec["launch"], f"seed42 {model} launch evidence") if "launch" in spec else None
        expected_recipe = recipes["models"][model]
        require(freeze.get("status") in {"PASS", "FROZEN_BEFORE_TRAINING"}, f"seed42 {model} freeze status")
        require(completion.get("status") == "PASS", f"seed42 {model} completion status")
        if "seed" in completion:
            require(int(completion["seed"]) == 42, f"seed42 {model} completion seed")
        if completion.get("model_id") is not None:
            require(completion["model_id"] == model, f"seed42 {model} completion model")
        if completion.get("mode") is not None:
            require(completion["mode"] == expected_recipe["mode"], f"seed42 {model} completion mode")
        if completion.get("lambda_phys") is not None:
            require(
                float(completion["lambda_phys"]) == float(expected_recipe["lambda_phys"]),
                f"seed42 {model} completion lambda",
            )
        if completion.get("global_batch_size") is not None:
            require(int(completion["global_batch_size"]) == 32, f"seed42 {model} completion batch")
        if freeze.get("seed") is not None:
            require(int(freeze["seed"]) == 42, f"seed42 {model} freeze seed")
        if freeze.get("model_id") is not None:
            require(freeze["model_id"] == model, f"seed42 {model} freeze model")
        if freeze.get("mode") is not None:
            require(freeze["mode"] == expected_recipe["mode"], f"seed42 {model} freeze mode")
        if freeze.get("lambda_phys") is not None:
            require(
                float(freeze["lambda_phys"]) == float(expected_recipe["lambda_phys"]),
                f"seed42 {model} freeze lambda",
            )
        if launch is not None:
            require(launch.get("status") in {"PASS", "LAUNCHING"}, f"seed42 {model} launch status")
            if launch.get("seed") is not None:
                require(int(launch["seed"]) == 42, f"seed42 {model} launch seed")
            if launch.get("mode") is not None:
                require(launch["mode"] == expected_recipe["mode"], f"seed42 {model} launch mode")
            if launch.get("lambda_phys") is not None:
                require(
                    float(launch["lambda_phys"]) == float(expected_recipe["lambda_phys"]),
                    f"seed42 {model} launch lambda",
                )
            if launch.get("global_batch_size") is not None:
                require(int(launch["global_batch_size"]) == 32, f"seed42 {model} launch batch")
        completed_step, restored_step, restored_kind = completion_progress(model, completion)
        require(completed_step == 30_000, f"seed42 {model} completed optimizer steps")
        seed42[model] = {
            "spec": spec,
            "freeze": freeze,
            "completion": completion,
            "launch": launch,
            "completed_step": completed_step,
            "restored_step": restored_step,
            "restored_kind": restored_kind,
            "historical_recipe_crosscheck": "PASS",
        }

    final_rows = pre["checkpoints"]
    require(len(final_rows) == 15, "expected exactly 15 final checkpoints")
    require(
        {(row["model"], int(row["training_seed"])) for row in final_rows}
        == {(model, seed) for model in MODELS for seed in SEEDS},
        "15-checkpoint model/seed matrix drift",
    )
    checkpoint_hashes: dict[tuple[str, int], str] = {}
    output_rows: list[dict[str, Any]] = []
    run_log_evidence: dict[tuple[str, int], dict[str, Any]] = {}

    for model in MODELS:
        for seed in (43, 44):
            run_id = run_rows[(model, seed)]["run_id"]
            launch_path = PI2B / f"artifacts/launches/{run_id}.json"
            status_path = PI2B / f"status/training/{run_id}.json"
            log_path = PI2B / f"logs/training/{run_id}.log"
            launch = inputs.json(launch_path, f"actual Track A launch {run_id}")
            status = inputs.json(status_path, f"terminal Track A job status {run_id}")
            log_bytes = inputs.bytes(log_path, f"actual Track A training metrics {run_id}")
            require(launch["config_sha256"] == run_rows[(model, seed)]["config_sha256"], f"launch config drift {run_id}")
            require(status.get("state") == "DONE" and status.get("exit_code") == 0, f"run not DONE/0 {run_id}")
            require(status.get("optimizer_steps") == 30_000, f"optimizer steps drift {run_id}")
            metric_evidence = parse_step_metrics(log_bytes)
            require(metric_evidence["first_logged_step"] == 0, f"first metric step drift {run_id}")
            require(metric_evidence["last_logged_step"] == 29_900, f"last metric step drift {run_id}")
            require(metric_evidence["all_parsed_metrics_finite"], f"non-finite metric {run_id}")
            run_log_evidence[(model, seed)] = {
                "launch": launch,
                "terminal_status": status,
                "metrics": metric_evidence,
            }

    for row in sorted(final_rows, key=lambda value: (MODELS.index(value["model"]), int(value["training_seed"]))):
        model = str(row["model"])
        seed = int(row["training_seed"])
        checkpoint_hashes[(model, seed)] = str(row["tree_sha256"])
        require(row.get("status") == "PASS", f"checkpoint freeze not PASS: {model}/seed{seed}")
        require(row.get("tree_sha256") == row.get("expected_tree_sha256"), f"frozen tree mismatch: {model}/seed{seed}")
        checkpoint_path = Path(row["path"])
        require(checkpoint_path.is_dir(), f"checkpoint missing: {checkpoint_path}")
        require(checkpoint_path.name == "29999", f"unexpected final directory label: {checkpoint_path}")
        require((checkpoint_path / "_CHECKPOINT_METADATA").is_file(), f"metadata missing: {checkpoint_path}")
        require((checkpoint_path / "params").is_dir(), f"params missing: {checkpoint_path}")
        require((checkpoint_path / "train_state").is_dir(), f"train_state missing: {checkpoint_path}")
        recipe = recipes["models"][model]
        common = recipes["common"]

        if seed == 42:
            evidence = seed42[model]
            completion = evidence["completion"]
            completed_step = evidence["completed_step"]
            restored_step = evidence["restored_step"]
            source = {
                "cohort": "HISTORICAL_SEED42_SEPARATE_STAGE",
                "stage": evidence["spec"]["stage"],
                "entrypoint": evidence["spec"]["entrypoint"],
                "source_commit": source_commit_from_seed42(model, evidence["freeze"]),
                "freeze_artifact": str(evidence["spec"]["freeze"].resolve()),
                "completion_artifact": str(evidence["spec"]["completion"].resolve()),
                "per_process_import_origin": "UNVERIFIABLE_NOT_PERSISTED_FOR_THIS_HISTORICAL_PROCESS",
            }
            optimizer = {
                "completed_optimizer_steps": completed_step,
                "restored_train_state_step": restored_step,
                "restore_evidence_kind": evidence["restored_kind"],
                "checkpoint_cold_load": completion.get("gates", {}).get(
                    "checkpoint_cold_load", "NOT_RECORDED_UNDER_THIS_GATE_NAME"
                ),
                "optimizer_state_leaf_membership_inventory": "NOT_PERSISTED",
                "group_gradient_evidence": {
                    key: value
                    for key, value in completion.get("last_logged_metrics", {}).items()
                    if "grad_norm" in key
                },
            }
            actual_devices = completion.get("physical_gpus", "NOT_UNIFORMLY_RECORDED")
            config_identity = {
                "kind": "FREEZE_ARTIFACT_SHA256",
                "sha256": inputs.observed_sha256(evidence["spec"]["freeze"]),
            }
        else:
            frozen_run = run_rows[(model, seed)]
            completed = completion_rows[(model, seed)]
            runtime = resource_rows[(model, seed)]
            log_evidence = run_log_evidence[(model, seed)]
            completed_step = int(runtime["optimizer_steps"])
            restored_step = int(completed["restored_train_state_step"])
            source = {
                "cohort": "TRACK_A_COMMON_SOURCE_SEEDS43_44",
                "stage": "S4.3-PI2B Track A",
                "entrypoint": frozen_run["entrypoint"],
                "source_commit": frozen_run["git_head"],
                "source_sha256": frozen_run["source_sha256"],
                "pretraining_import_audit": {
                    "python": imports["python"],
                    "modules": imports["modules"],
                    "scope_note": "Historical pretraining audit; not a claim about current main-worktree imports.",
                },
            }
            optimizer = {
                "completed_optimizer_steps": completed_step,
                "restored_train_state_step": restored_step,
                "restore_evidence_kind": "EXPLICIT_TRACK_A_COLD_RESTORE_SCALAR",
                "all_train_state_arrays_finite": completed["all_train_state_arrays_finite"],
                "cold_restore": completed["cold_restore"],
                "first_and_last_logged_metrics": log_evidence["metrics"],
                "group_gradient_evidence": {
                    name: "RECORDED_IN_TRAINING_LOG"
                    for name in log_evidence["metrics"]["gradient_metric_names"]
                },
                "optimizer_state_leaf_membership_inventory": "NOT_PERSISTED",
                "membership_evidence_scope": (
                    "The accepted OpenPI trainable_filter selects all nnx.Param leaves not matched by the freeze filter; "
                    "the log independently records finite nonzero named gradient groups when those groups exist."
                ),
            }
            actual_devices = {
                "fsdp_devices": runtime["fsdp_devices"],
                "physical_gpu_index": runtime["physical_gpu_index"],
                "global_batch_size": runtime["global_batch_size"],
            }
            config_identity = {
                "kind": "TRACK_A_RUN_CONFIG_SHA256",
                "sha256": frozen_run["config_sha256"],
            }

        require(completed_step == 30_000, f"completed optimizer steps mismatch: {model}/seed{seed}")
        if restored_step is not None:
            require(restored_step == 30_000, f"restored step mismatch: {model}/seed{seed}")
        group_gradient_evidence = optimizer.get("group_gradient_evidence", {})
        optimizer_conclusion = (
            "SUPPORTED_BY_STATIC_TRAINABLE_FILTER_AND_RECORDED_GROUP_GRADIENTS; "
            "EXACT_OPTIMIZER_STATE_LEAF_NAMES_NOT_PERSISTED"
            if group_gradient_evidence
            else "STATIC_TRAINABLE_FILTER_ONLY_NO_NAMED_GROUP_GRADIENT_PERSISTED; "
            "EXACT_OPTIMIZER_STATE_LEAF_NAMES_NOT_PERSISTED"
        )
        output_rows.append(
            {
                "checkpoint_id": f"{model}_seed{seed}_final",
                "focal_checkpoint": (model, seed) in FOCAL,
                "model": model,
                "training_seed": seed,
                "identity": {
                    "path": str(checkpoint_path),
                    "directory_step_label": 29_999,
                    "completed_optimizer_steps": completed_step,
                    "restored_train_state_step": restored_step,
                    "restored_train_state_step_status": (
                        "VERIFIED_DIRECT_FIELD_OR_EXPLICIT_GATE"
                        if restored_step is not None
                        else "UNVERIFIABLE_SCALAR_NOT_PERSISTED"
                    ),
                    "directory_vs_state_explanation": (
                        "Orbax checkpoint directory uses zero-based save key 29999 and persisted evidence reports "
                        "30000 completed optimizer steps. A scalar restored train-state step is separately reported "
                        "only when directly recorded or asserted by an explicit cold-restore gate."
                    ),
                    "tree_sha256": row["tree_sha256"],
                    "tree_hash_evidence": row["hash_source"],
                    "current_analysis_rescan": False,
                    "params_present": row["params_present"],
                    "train_state_present": row["train_state_present"],
                },
                "recipe": {
                    "official_parent": common["base"],
                    "mode": recipe["mode"],
                    "state_dim": common["state_dim"],
                    "dataset_action_dim": common["dataset_action_dim"],
                    "internal_action_dim": common["internal_action_dim"],
                    "action_horizon": common["action_horizon"],
                    "dtype": common["dtype"],
                    "physical_target_offset_control_ticks": common["physical_target_offset_control_ticks"],
                    "physical_target_horizon_seconds": common["physical_target_horizon_seconds"],
                    "target_valid_rows": common["target_valid_rows"],
                    "target_tail_masked_rows": common["target_tail_masked_rows"],
                    "contact_adapter_parameters": recipe.get("contact_adapter_parameters", 0),
                    "physical_auxiliary_parameters": recipe.get("physical_auxiliary_parameters", 0),
                    "auxiliary_target": recipe.get("auxiliary_target"),
                    "lambda_phys": recipe["lambda_phys"],
                    "lambda_rule": "FROZEN_SEED42_VALUE_REUSED_FOR_SEEDS43_44_NOT_RECALIBRATED_PER_SEED",
                    "config_identity": config_identity,
                },
                "optimizer_and_runtime_evidence": {
                    **optimizer,
                    "global_batch_size": 32,
                    "actual_device_record": actual_devices,
                    "optimizer_recipe": {
                        "name": "AdamW",
                        "peak_lr": 5e-5,
                        "warmup_steps": 10_000,
                        "clip_gradient_norm": 1.0,
                        "weight_decay": 1e-10,
                    },
                    "adapter_aux_lora_optimizer_conclusion": optimizer_conclusion,
                },
                "source_and_import_evidence": source,
                "unverifiable_from_persisted_history": [
                    "bytewise equality of model initialization across different modes sharing the same numeric seed",
                    "complete per-parameter optimizer-state leaf-name inventory",
                    "per-process historical import snapshot for seed42",
                    "dropout/shuffle/augmentation/flow-noise PRNG subkey trace",
                    *(
                        ["scalar restored train-state step for this checkpoint"]
                        if restored_step is None
                        else []
                    ),
                ],
            }
        )

    intermediate_rows = []
    for row in sorted(intermediate["checkpoints"], key=lambda value: value["label"]):
        require(row["directory_step"] in {10_000, 20_000}, f"bad intermediate step: {row['label']}")
        require(len(row["tree_sha256"]) == 64, f"bad intermediate hash: {row['label']}")
        intermediate_rows.append(
            {
                "label": row["label"],
                "path": row["path"],
                "directory_step": row["directory_step"],
                "tree_sha256": row["tree_sha256"],
                "files": row["files"],
                "bytes": row["bytes"],
                "identity_source": str(intermediate_path),
                "use_scope": "READ_ONLY_CHANGE_TRAJECTORY_ONLY_NOT_FINAL_SELECTION",
            }
        )
    require(len(intermediate_rows) == 4, "expected four intermediate checkpoint hashes")

    source_commits = defaultdict(set)
    entrypoints = defaultdict(set)
    for row in output_rows:
        cohort = row["source_and_import_evidence"]["cohort"]
        source_commits[cohort].add(row["source_and_import_evidence"]["source_commit"])
        entrypoints[cohort].add(row["source_and_import_evidence"]["entrypoint"])
    restored_step_verified = [
        row["checkpoint_id"]
        for row in output_rows
        if row["identity"]["restored_train_state_step"] == 30_000
    ]
    restored_step_unavailable = [
        row["checkpoint_id"]
        for row in output_rows
        if row["identity"]["restored_train_state_step"] is None
    ]
    require(len(restored_step_verified) == 13, "expected 13 direct-or-explicit restored-step records")
    require(
        restored_step_unavailable == ["B1_seed42_final", "B2_seed42_final"],
        f"unexpected restored-step evidence gaps: {restored_step_unavailable}",
    )

    body = {
        "status": "COMPLETE_WITH_DECLARED_UNVERIFIABLE_FIELDS",
        "integrated_evidence_commit": EXPECTED_INTEGRATION_COMMIT,
        "matrix_shape": {"models": 5, "training_seeds": 3, "final_checkpoints": 15},
        "focal_checkpoint_ids": [
            f"{model}_seed{seed}_final" for model, seed in sorted(FOCAL, key=lambda item: (MODELS.index(item[0]), item[1]))
        ],
        "checkpoint_directory_convention": {
            "directory_label": 29_999,
            "completed_optimizer_steps": 30_000,
            "restored_train_state_step_verified_checkpoint_ids": restored_step_verified,
            "restored_train_state_step_unavailable_checkpoint_ids": restored_step_unavailable,
            "status": "CONSISTENT_ZERO_BASED_SAVE_KEY_VS_COMPLETED_OPTIMIZER_PROGRESS",
        },
        "source_cohort_comparison": {
            "seed42": {
                "source_commits": sorted(source_commits["HISTORICAL_SEED42_SEPARATE_STAGE"]),
                "entrypoints": sorted(entrypoints["HISTORICAL_SEED42_SEPARATE_STAGE"]),
            },
            "seeds43_44": {
                "source_commits": sorted(source_commits["TRACK_A_COMMON_SOURCE_SEEDS43_44"]),
                "entrypoints": sorted(entrypoints["TRACK_A_COMMON_SOURCE_SEEDS43_44"]),
            },
            "conclusion": "NOT_FULL_SOURCE_PARITY_SEED42_VS_SEEDS43_44",
            "interpretation": (
                "The five seed42 checkpoints came from several earlier stage-specific entrypoints and commits, while "
                "seeds43/44 used one Track A source snapshot. Recipe intent is matched, but seed is not the only "
                "provenance difference; this does not by itself identify a performance cause."
            ),
        },
        "historical_track_a_import_origins": {
            **imports,
            "scope_note": (
                "This records the actual Track A audit-time imports. The current integrated main-worktree import audit "
                "is a separate fact and is not overwritten by this historical record."
            ),
        },
        "track_a_source_snapshot_and_integrated_copy": {
            "immutable_snapshot": snapshot_manifest,
            "current_source_comparison": integrated_source_comparison,
            "expected_integrated_difference": {
                "path": "gr00t/simulation/pi2b_policy/contract.py",
                "reason": "post-merge read-only integrated resolver; strict historical runtime loader remains preserved",
            },
        },
        "common_target_time_mask_contract": timing,
        "final_checkpoints": output_rows,
        "intermediate_checkpoints": intermediate_rows,
        "limitations": [
            "Final checkpoint trees are bound to prior live full-tree hashes and were not rescanned in this CPU analysis.",
            "The four 10k/20k trees are referenced from the separately fsynced full-content PI2S hash artifact.",
            "Persisted records do not prove byte-identical initialization or full PRNG subkey traces across modes.",
            "Historical seed42 training processes did not persist a uniform import-origin manifest.",
            "A static trainable-filter audit plus available named gradient logs supports optimizer participation; some rows lack named group-gradient persistence, and the exact optimizer-state leaf-name inventory was not persisted.",
            "B1/B2 seed42 persist 30000 completed optimizer steps and a cold-load PASS, but no scalar restored train-state step field.",
        ],
        "gates": {
            "fifteen_final_checkpoint_identities_exact": "PASS",
            "six_focal_checkpoint_identities_exact": "PASS",
            "four_intermediate_checkpoint_hashes_bound": "PASS",
            "directory_29999_vs_completed_optimizer_steps_30000_explicit": "PASS",
            "restored_step_direct_field_or_explicit_gate_13_of_15": "PASS",
            "seed42_b1_b2_scalar_restore_unavailable": "DECLARED",
            "central_recipe_contract_fields_recorded": "PASS",
            "seed42_historical_recipe_crosschecked": "PASS",
            "actual_track_a_launch_status_log_and_import_evidence_bound": "PASS",
            "immutable_track_a_source_snapshot_bound": "PASS",
            "only_expected_integrated_source_difference_present": "PASS",
            "seed42_vs_43_44_non_seed_source_differences_disclosed": "PASS",
            "optimizer_claim_matches_available_evidence": "PASS",
            "unverifiable_history_not_fabricated": "PASS",
        },
    }
    return body, checkpoint_hashes


def empty_group(model: str, seed: int) -> dict[str, Any]:
    return {
        "model": model,
        "training_seed": seed,
        "episodes": 0,
        "successes": 0,
        "failures": 0,
        "termination_counts": Counter(),
        "failure_max_pinch_counts": Counter(),
        "all_max_pinch_counts": Counter(),
        "matched_contact_positive_episodes": 0,
        "matched_contact_positive_failures": 0,
        "server_client_error_episodes": 0,
        "server_client_error_events": 0,
        "top_level_server_client_error_events": 0,
        "failure_steps": Counter(),
        "raw_artifact_sha256": [],
        "representatives": defaultdict(list),
        "h_rows": [],
        "tactile_active_samples": 0,
        "tactile_samples": 0,
    }


def coarse_bin(max_pinch_count: int) -> str:
    if max_pinch_count <= 0:
        return "NO_NATIVE_PINCH_OBSERVED"
    if max_pinch_count == 1:
        return "ONE_NATIVE_PINCH_MAX"
    if max_pinch_count == 2:
        return "TWO_NATIVE_PINCH_MAX"
    return "THREE_OR_MORE_NATIVE_PINCH_MAX"


def stable_counter(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter, key=lambda item: str(item))}


def historical_rollouts(
    inputs: InputRegistry, checkpoint_hashes: dict[tuple[str, int], str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    completeness_path = PI2B / "artifacts/rollout_completeness.json"
    result_path = PI2B / "artifacts/result_summary.json"
    pre_final_path = PI2B / "artifacts/pre_final_freeze.json"
    amendment_path = PI2B / "artifacts/runtime_amendment_cross_device_publish.json"
    completeness = inputs.json(completeness_path, "canonical rollout block/SHA index")
    result = inputs.json(result_path, "published Track A aggregate for independent comparison")
    pre_final = inputs.json(pre_final_path, "frozen formal-evaluator source hashes")
    amendment = inputs.json(amendment_path, "scoped runtime amendment and unchanged outcome rules")
    require(completeness.get("status") == "PASS", "rollout completeness not PASS")
    require(completeness.get("total_canonical_outcomes") == EXPECTED_TOTAL, "frozen canonical count drift")
    require(result.get("status") == "COMPLETE_VALID", "Track A result summary not complete")
    require(pre_final.get("status") == "PASS", "pre-final freeze not PASS during rollout audit")
    require(amendment.get("status") == "PASS", "runtime amendment not PASS")
    require(amendment.get("success_or_timeout_rules_changed") is False, "runtime amendment changed outcome rules")
    evaluator_relative = "scripts/simulation/evaluate_s4_3_pi1d_augmented.py"
    evaluator_sha = pre_final["sources_sha256"][evaluator_relative]
    inputs.bytes(ROOT / evaluator_relative, "frozen evaluator defining native termination labels", evaluator_sha)

    expected_raw = completeness["raw_artifacts"]
    require(len(expected_raw) == 60, "expected 60 canonical raw blocks")
    groups = {(model, seed): empty_group(model, seed) for model in MODELS for seed in SEEDS}
    canonical_tuples: set[tuple[str, int, str]] = set()
    all_reset_sequences: dict[tuple[str, int], list[str]] = defaultdict(list)
    raw_manifest_for_sha: list[dict[str, Any]] = []

    for reset_seed in RESET_SEEDS:
        for model in MODELS:
            lower_model = model.lower()
            for seed in SEEDS:
                key = f"r{reset_seed}_{lower_model}_s{seed}"
                require(key in expected_raw, f"missing raw index key: {key}")
                expected = expected_raw[key]
                path = PI2B / f"artifacts/final_raw/reset_seed_{reset_seed}/{lower_model}/train_seed_{seed}.json"
                payload = inputs.json(path, f"canonical rollout block {key}", expected["sha256"])
                raw_manifest_for_sha.append(
                    {"key": key, "path": str(path), "sha256": expected["sha256"], "episodes": expected["episodes"]}
                )
                require(payload.get("status") == "PASS", f"raw block not PASS: {key}")
                require(payload.get("model") == model, f"raw block model mismatch: {key}")
                require(int(payload.get("training_seed")) == seed, f"raw block seed mismatch: {key}")
                require(int(payload.get("reset_seed")) == reset_seed, f"raw block reset mismatch: {key}")
                require(payload.get("checkpoint_tree_sha256") == checkpoint_hashes[(model, seed)], f"raw block checkpoint mismatch: {key}")
                require(payload.get("episodes") == 50, f"raw block count mismatch: {key}")
                require(len(payload.get("episode_results", [])) == 50, f"episode rows mismatch: {key}")
                require(all(value == "PASS" for value in payload.get("gates", {}).values()), f"raw block gate failure: {key}")

                group = groups[(model, seed)]
                top_errors = payload.get("server_client_errors", [])
                group["top_level_server_client_error_events"] += len(top_errors)
                group["raw_artifact_sha256"].append(expected["sha256"])
                observed_successes = 0
                for episode in payload["episode_results"]:
                    identity = str(episode["reset_identity"])
                    canonical = (model, seed, identity)
                    require(canonical not in canonical_tuples, f"duplicate canonical tuple: {canonical}")
                    canonical_tuples.add(canonical)
                    all_reset_sequences[(model, seed)].append(identity)
                    require(episode.get("checkpoint_tree_sha256") == checkpoint_hashes[(model, seed)], f"episode checkpoint mismatch: {key}")
                    require(episode.get("model") == model, f"episode model mismatch: {key}")
                    require(int(episode.get("training_seed")) == seed, f"episode seed mismatch: {key}")
                    require(int(episode.get("reset_seed")) == reset_seed, f"episode reset mismatch: {key}")
                    success = episode.get("success")
                    require(isinstance(success, bool), f"non-boolean outcome: {key}")
                    termination = str(episode.get("termination"))
                    errors = episode.get("server_client_errors", [])
                    progress = episode.get("task_progress", {})
                    max_pinch = progress.get("max_pinch_count")
                    require(isinstance(max_pinch, int) and max_pinch >= 0, f"invalid max_pinch_count: {key}")
                    tactile = episode.get("tactile_diagnostics", {})
                    matched_sum = tactile.get("matched_contact_count_sum")
                    require(isinstance(matched_sum, int) and matched_sum >= 0, f"invalid matched contact sum: {key}")
                    contact_positive = matched_sum > 0

                    group["episodes"] += 1
                    group["successes"] += int(success)
                    group["failures"] += int(not success)
                    group["termination_counts"][termination] += 1
                    group["all_max_pinch_counts"][max_pinch] += 1
                    group["matched_contact_positive_episodes"] += int(contact_positive)
                    group["server_client_error_episodes"] += int(bool(errors))
                    group["server_client_error_events"] += len(errors)
                    group["tactile_active_samples"] += int(tactile.get("active_samples", 0))
                    group["tactile_samples"] += int(tactile.get("samples", 0))
                    observed_successes += int(success)

                    if not success:
                        group["failure_max_pinch_counts"][max_pinch] += 1
                        group["matched_contact_positive_failures"] += int(contact_positive)
                        group["failure_steps"][int(episode["steps"])] += 1
                        representative = {
                            "reset_identity": identity,
                            "reset_seed": reset_seed,
                            "episode_index": int(episode["episode_index"]),
                            "max_pinch_count": max_pinch,
                            "termination": termination,
                            "video": "N/A_NOT_RECORDED_IN_CANONICAL_RAW_ARTIFACT",
                        }
                        group["representatives"][coarse_bin(max_pinch)].append(representative)

                    h = episode.get("contact_state_diagnostics")
                    if model == "B_HVA":
                        require(isinstance(h, dict), f"missing H diagnostics: {key}")
                        group["h_rows"].append(
                            {
                                "success": success,
                                "finite": bool(h["finite"]),
                                "shape": list(h["shape"]),
                                "queries": int(h["queries"]),
                                "mean_l2": float(h["mean_l2"]),
                                "max_l2": float(h["max_l2"]),
                            }
                        )

                require(observed_successes == int(payload["successes"]), f"block success count mismatch: {key}")

    require(len(canonical_tuples) == EXPECTED_TOTAL, "canonical tuple count mismatch")
    reference_sequence = all_reset_sequences[(MODELS[0], SEEDS[0])]
    require(len(reference_sequence) == 200, "reference reset sequence length mismatch")
    require(all(sequence == reference_sequence for sequence in all_reset_sequences.values()), "shared reset order mismatch")

    group_outputs: list[dict[str, Any]] = []
    total = Counter()
    for model in MODELS:
        for seed in SEEDS:
            group = groups[(model, seed)]
            require(group["episodes"] == 200, f"group count mismatch: {model}/seed{seed}")
            require(group["successes"] + group["failures"] == 200, f"group outcome mismatch: {model}/seed{seed}")
            require(group["server_client_error_events"] == 0, f"episode server/client error: {model}/seed{seed}")
            require(group["top_level_server_client_error_events"] == 0, f"block server/client error: {model}/seed{seed}")
            require(group["termination_counts"].get("max_steps", 0) == group["failures"], f"failure termination mismatch: {model}/seed{seed}")
            require(group["termination_counts"].get("success", 0) == group["successes"], f"success termination mismatch: {model}/seed{seed}")
            require(group["failure_steps"] == Counter({1000: group["failures"]}), f"failure step mismatch: {model}/seed{seed}")
            selected_representatives = {
                name: min(rows, key=lambda value: value["reset_identity"])
                for name, rows in sorted(group["representatives"].items())
            }
            failure_max = stable_counter(group["failure_max_pinch_counts"])
            below_second = group["failure_max_pinch_counts"].get(0, 0) + group["failure_max_pinch_counts"].get(1, 0)
            below_third = below_second + group["failure_max_pinch_counts"].get(2, 0)
            group_outputs.append(
                {
                    "checkpoint_id": f"{model}_seed{seed}_final",
                    "focal_checkpoint": (model, seed) in FOCAL,
                    "model": model,
                    "training_seed": seed,
                    "episodes": group["episodes"],
                    "successes": group["successes"],
                    "failures": group["failures"],
                    "success_rate": group["successes"] / group["episodes"],
                    "termination_counts": stable_counter(group["termination_counts"]),
                    "server_client_error_episodes": group["server_client_error_episodes"],
                    "server_client_error_events": group["server_client_error_events"],
                    "top_level_server_client_error_events": group["top_level_server_client_error_events"],
                    "failure_step_counts": stable_counter(group["failure_steps"]),
                    "all_episode_max_pinch_counts": stable_counter(group["all_max_pinch_counts"]),
                    "failure_max_pinch_counts": failure_max,
                    "failures_before_second_native_pinch_count": below_second,
                    "failures_before_third_native_pinch_count": below_third,
                    "matched_contact_positive_episodes": group["matched_contact_positive_episodes"],
                    "matched_contact_positive_failures": group["matched_contact_positive_failures"],
                    "matched_contact_definition": "tactile_diagnostics.matched_contact_count_sum > 0 at some point in the episode",
                    "first_failure_stage": "N/A_MISSING_TIMESTAMPED_NATIVE_STAGE_TELEMETRY",
                    "detailed_stage_ladder": {
                        "INITIAL_APPROACH": "N/A",
                        "OBJECT_CONTACT": "COARSE_EPISODE_AGGREGATE_ONLY",
                        "STABLE_GRASP": "N/A",
                        "LIFT": "N/A",
                        "CYCLE_1_2_3": "COARSE_MAX_PINCH_COUNT_ONLY",
                        "NATIVE_SUCCESS": "OBSERVED_BOOLEAN_AND_TERMINATION",
                    },
                    "representative_selection": {
                        "rule": "lexicographically smallest reset_identity among failures in each observed coarse max-pinch bin",
                        "records": selected_representatives,
                        "video_availability": "N/A_NOT_RECORDED_IN_CANONICAL_RAW_ARTIFACTS",
                    },
                    "raw_artifact_sha256": sorted(group["raw_artifact_sha256"]),
                }
            )
            total["episodes"] += group["episodes"]
            total["successes"] += group["successes"]
            total["failures"] += group["failures"]
            total["server_client_error_events"] += group["server_client_error_events"]
            total["top_level_server_client_error_events"] += group["top_level_server_client_error_events"]
            for termination, count in group["termination_counts"].items():
                total[f"termination:{termination}"] += count

    require(total["episodes"] == EXPECTED_TOTAL, "recomputed total drift")
    require(total["successes"] == EXPECTED_SUCCESSES, "recomputed success drift")
    require(total["failures"] == EXPECTED_FAILURES, "recomputed failure drift")
    require(total["termination:max_steps"] == EXPECTED_FAILURES, "max_steps total drift")
    require(total["termination:success"] == EXPECTED_SUCCESSES, "success termination total drift")
    require(total["server_client_error_events"] == 0, "server/client error total nonzero")
    require(total["top_level_server_client_error_events"] == 0, "top-level server/client error total nonzero")
    published = result["formal_evaluation"]
    require(published["canonical_outcomes"] == total["episodes"], "published episode count mismatch")
    require(published["successes"] == total["successes"], "published success count mismatch")
    require(published["failures"] == total["failures"], "published failure count mismatch")
    require(published["timeouts"] == 0, "published timeout count nonzero")

    aborted = []
    for manifest_path in sorted((PI2B / "aborted_infrastructure").glob("*/manifest.json")):
        manifest = inputs.json(manifest_path, "preserved pre-scientific infrastructure abort")
        require(manifest.get("canonical_rollouts") == 0, f"abort consumed canonical rollout: {manifest_path}")
        aborted.append({"path": str(manifest_path), **manifest})

    historical_body = {
        "status": "COMPLETE_WITH_FIRST_FAILURE_STAGE_UNOBSERVABLE",
        "integrated_evidence_commit": EXPECTED_INTEGRATION_COMMIT,
        "analysis_kind": "READ_ONLY_RECOMPUTATION_OF_EXISTING_CANONICAL_ROLLOUTS",
        "outcome_recomputation": {
            "canonical_outcomes": total["episodes"],
            "successes": total["successes"],
            "failures": total["failures"],
            "termination_counts": {"max_steps": total["termination:max_steps"], "success": total["termination:success"]},
            "server_client_error_events": total["server_client_error_events"],
            "top_level_server_client_error_events": total["top_level_server_client_error_events"],
            "timeouts": 0,
            "raw_block_count": len(expected_raw),
            "raw_block_manifest_sha256": sha256_bytes(canonical_bytes(sorted(raw_manifest_for_sha, key=lambda value: value["key"]))),
            "independent_match_to_track_a_summary": True,
        },
        "termination_semantics": {
            "max_steps": (
                "Native evaluator outcome: success=false and steps==1000. It is distinct from IPC/server timeout; "
                "all episode-level and block-level server_client_errors arrays are empty."
            ),
            "ipc_or_server_timeout": "ZERO_OBSERVED",
            "caution": "Absence of recorded IPC errors is not evidence that a max_steps episode reached any unlogged task stage.",
            "evidence": {
                "frozen_evaluator_source": evaluator_relative,
                "frozen_evaluator_sha256": evaluator_sha,
                "runtime_amendment": str(amendment_path),
                "runtime_amendment_success_or_timeout_rules_changed": False,
            },
        },
        "stage_contract": {
            "desired_ladder": [
                "INITIAL_APPROACH",
                "OBJECT_CONTACT",
                "STABLE_GRASP",
                "LIFT",
                "CYCLE_1",
                "CYCLE_2",
                "CYCLE_3",
                "NATIVE_SUCCESS",
            ],
            "observable_fields": [
                "episode success/termination/steps",
                "task_progress.final_pinch_count",
                "task_progress.max_pinch_count",
                "episode-aggregate tactile_diagnostics.matched_contact_count_sum",
            ],
            "missing_fields": [
                "timestamped native stage transitions",
                "timestamped object pose/lift height",
                "timestamped stable-grasp predicate",
                "timestamped cycle completion events",
                "first-failure transition",
            ],
            "first_failure_rule": "N/A when the first uncompleted native stage is not directly observed; no inference from final frame or aggregate max count",
            "coarse_failure_bins": {
                "NO_NATIVE_PINCH_OBSERVED": "max_pinch_count == 0",
                "ONE_NATIVE_PINCH_MAX": "max_pinch_count == 1",
                "TWO_NATIVE_PINCH_MAX": "max_pinch_count == 2",
                "THREE_OR_MORE_NATIVE_PINCH_MAX": "max_pinch_count >= 3",
            },
        },
        "all_fifteen_checkpoints": group_outputs,
        "six_focal_checkpoint_ids": [row["checkpoint_id"] for row in group_outputs if row["focal_checkpoint"]],
        "selection_bias_caution": (
            "Success/failure-conditional aggregates are post-outcome descriptions. Different policies induce different "
            "closed-loop trajectories, so these associations are not causal localization."
        ),
        "preserved_negative_history": {
            "canonical_failures_retained": EXPECTED_FAILURES,
            "pre_scientific_infrastructure_aborts": aborted,
            "note": "Infrastructure aborts remain visible but are not counted as canonical scientific outcomes.",
        },
        "diagnostic_implication": {
            "historical_telemetry_sufficient_for_unique_root_cause": False,
            "first_failure_stage_identified": False,
            "limited_development_replay_needed_for_timestamped_stage_localization": True,
            "automatic_budget_expansion": False,
        },
        "gates": {
            "sixty_raw_sha256_values_match_frozen_index": "PASS",
            "three_thousand_unique_shared_reset_outcomes": "PASS",
            "recomputed_803_success_2197_failure": "PASS",
            "all_2197_failures_retained": "PASS",
            "all_failures_native_max_steps_at_1000_steps": "PASS",
            "server_client_error_events_zero": "PASS",
            "max_steps_not_relabelled_as_ipc_timeout": "PASS",
            "frozen_evaluator_and_unchanged_outcome_rules_bound": "PASS",
            "full_fifteen_checkpoint_coarse_summaries": "PASS",
            "six_focal_checkpoints_marked": "PASS",
            "missing_detailed_ladder_reported_na": "PASS",
            "representatives_selected_without_outcome_cherry_pick": "PASS",
        },
    }

    h_groups = []
    for seed in SEEDS:
        group = groups[("B_HVA", seed)]
        rows = group["h_rows"]
        require(len(rows) == 200, f"H diagnostic row count mismatch: seed{seed}")
        require(all(row["finite"] for row in rows), f"non-finite H diagnostic: seed{seed}")
        require(all(row["shape"] == [256] for row in rows), f"H shape mismatch: seed{seed}")
        require(all(row["queries"] > 0 for row in rows), f"zero H queries: seed{seed}")
        total_queries = sum(row["queries"] for row in rows)
        weighted_mean = sum(row["mean_l2"] * row["queries"] for row in rows) / total_queries
        success_rows = [row for row in rows if row["success"]]
        failure_rows = [row for row in rows if not row["success"]]
        h_groups.append(
            {
                "checkpoint_id": f"B_HVA_seed{seed}_final",
                "training_seed": seed,
                "episodes": len(rows),
                "successes": len(success_rows),
                "failures": len(failure_rows),
                "queries": total_queries,
                "finite_episodes": sum(int(row["finite"]) for row in rows),
                "shape_counts": {"[256]": len(rows)},
                "episode_mean_l2": {
                    "mean": statistics.fmean(row["mean_l2"] for row in rows),
                    "sample_sd": statistics.stdev(row["mean_l2"] for row in rows),
                    "query_weighted_mean": weighted_mean,
                    "quantiles": quantiles(row["mean_l2"] for row in rows),
                },
                "episode_max_l2": {
                    "mean": statistics.fmean(row["max_l2"] for row in rows),
                    "sample_sd": statistics.stdev(row["max_l2"] for row in rows),
                    "quantiles": quantiles(row["max_l2"] for row in rows),
                },
                "success_conditioned_episode_mean_l2": (
                    {
                        "n": len(success_rows),
                        "mean": statistics.fmean(row["mean_l2"] for row in success_rows),
                        "quantiles": quantiles(row["mean_l2"] for row in success_rows),
                    }
                    if success_rows
                    else {"n": 0, "mean": None, "quantiles": None}
                ),
                "failure_conditioned_episode_mean_l2": {
                    "n": len(failure_rows),
                    "mean": statistics.fmean(row["mean_l2"] for row in failure_rows),
                    "quantiles": quantiles(row["mean_l2"] for row in failure_rows),
                },
                "tactile_active_sample_fraction": group["tactile_active_samples"] / group["tactile_samples"],
                "association_caution": (
                    "Outcome-conditioned and active-sample differences are closed-loop associations and may be effects "
                    "of policy behavior rather than causes."
                ),
            }
        )

    weighted_means = {row["training_seed"]: row["episode_mean_l2"]["query_weighted_mean"] for row in h_groups}
    max_norms = {row["training_seed"]: row["episode_max_l2"]["quantiles"]["max"] for row in h_groups}
    mean_range = max(weighted_means.values()) - min(weighted_means.values())
    relative_range = mean_range / statistics.fmean(weighted_means.values())
    h_body = {
        "status": "COMPLETE_FOR_PERSISTED_EPISODE_AGGREGATES",
        "integrated_evidence_commit": EXPECTED_INTEGRATION_COMMIT,
        "scope": "Historical B_HVA seed42/43/44 episode-aggregate contact_state_diagnostics only",
        "metric_definition": {
            "mean_l2": "evaluator-recorded mean L2 norm over H queries within an episode",
            "max_l2": "evaluator-recorded maximum L2 norm over H queries within an episode",
            "query_weighted_mean": "sum(episode mean_l2 * episode queries) / sum(episode queries)",
            "tactile_active_sample_fraction": "sum(active_samples) / sum(samples) from episode tactile_diagnostics",
        },
        "by_training_seed": h_groups,
        "cross_seed_comparison": {
            "query_weighted_mean_l2": {str(seed): weighted_means[seed] for seed in SEEDS},
            "maximum_observed_l2": {str(seed): max_norms[seed] for seed in SEEDS},
            "weighted_mean_absolute_range": mean_range,
            "weighted_mean_relative_range": relative_range,
            "successes": {str(row["training_seed"]): row["successes"] for row in h_groups},
            "tactile_active_sample_fraction": {
                str(row["training_seed"]): row["tactile_active_sample_fraction"] for row in h_groups
            },
        },
        "conclusion": {
            "simple_norm_collapse_in_hva43": "NOT_SUPPORTED_BY_PERSISTED_AGGREGATE_NORM_EVIDENCE",
            "basis": [
                "all 200 HVA43 episode summaries are finite",
                "all report H shape [256] and positive query counts",
                "HVA43 query-weighted mean and maximum L2 norms are close to HVA42/HVA44 rather than near zero",
            ],
            "not_concluded": [
                "No claim that HVA43 representation content, rank, temporal alignment, or task information is healthy.",
                "No claim that H norms explain success or failure.",
                "No unique root cause is localized by episode-aggregate norms.",
                "The higher HVA43 tactile-active fraction is association-only because policies induce their own trajectories.",
            ],
        },
        "limitations": [
            "Only episode aggregate mean/max norms were persisted; per-query vectors and per-dimension distributions are absent here.",
            "Aggregate norms cannot detect rotations, low-rank collapse at constant norm, stale values, temporal misalignment, or semantic corruption.",
            "HVA43 has one success, so success-conditioned HVA43 summaries have n=1 and must not be generalized.",
            "Closed-loop H distributions are policy-dependent and cannot establish causation.",
        ],
        "gates": {
            "six_hundred_hva_episode_summaries": "PASS",
            "all_h_summaries_finite": "PASS",
            "all_h_shapes_256": "PASS",
            "all_h_query_counts_positive": "PASS",
            "seed42_43_44_separately_reported": "PASS",
            "negative_hva43_performance_retained": "PASS",
            "simple_norm_collapse_not_overclaimed": "PASS",
        },
    }
    return historical_body, h_body


def build_all() -> dict[str, dict[str, Any]]:
    inputs = InputRegistry()
    recipe_body, checkpoint_hashes = recipe_matrix(inputs)
    historical_body, h_body = historical_rollouts(inputs, checkpoint_hashes)
    # Build only after every input has been registered so every output binds the
    # identical complete input manifest rather than a phase-dependent subset.
    return {
        "checkpoint_recipe_matrix": output_artifact(
            "tactile3d-unit.s4-3-pi2s-checkpoint-recipe-matrix.v2", recipe_body, inputs
        ),
        "historical_failure_stage_analysis": output_artifact(
            "tactile3d-unit.s4-3-pi2s-historical-failure-stage-analysis.v2", historical_body, inputs
        ),
        "h_distribution_diagnostics": output_artifact(
            "tactile3d-unit.s4-3-pi2s-h-distribution-diagnostics.v2", h_body, inputs
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute from SHA-bound inputs and verify existing outputs without writing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = build_all()
    hashes: dict[str, str] = {}
    if args.check:
        for name, path in OUTPUTS.items():
            check_artifact(path, artifacts[name])
            hashes[name] = sha256_bytes(path.read_bytes())
        operation = "CHECK_PASS"
    else:
        for name, path in OUTPUTS.items():
            hashes[name] = atomic_json(path, artifacts[name])
        # Verify the exact persisted semantics immediately after all replaces.
        for name, path in OUTPUTS.items():
            check_artifact(path, artifacts[name])
        operation = "WRITE_AND_READBACK_PASS"
    print(
        json.dumps(
            {
                "status": operation,
                "outputs": {name: {"path": str(OUTPUTS[name]), "sha256": hashes[name]} for name in OUTPUTS},
                "canonical_outcomes": EXPECTED_TOTAL,
                "successes": EXPECTED_SUCCESSES,
                "failures": EXPECTED_FAILURES,
                "gpu_used": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except AuditError as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(1) from error
