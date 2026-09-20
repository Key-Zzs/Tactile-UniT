#!/usr/bin/env python3
"""Cold-load, verify, and immutably freeze one completed PI2N candidate."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from time import perf_counter
from typing import Any


os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[2]
OPENPI = ROOT / ".local/external/simulation/s4_3_pi1/openpi"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2n"
RUN_ROOT = Path(
    "/mnt/ugreen_nas/storage/UniT_storage/experiments/simulation/s4_3_pi2n"
)
BASE_MANIFEST = ROOT / ".local/artifacts/simulation/s4_3_pi0/official_base_model_manifest.json"
STARTING_IMMUTABILITY = ARTIFACTS / "checkpoint_immutability_before.json"
EXPECTED_DEXJOCO = "8d23b0fab23b17a58c4b55f3942e17013aaf8267"

PROTECTED_CHECKPOINTS = {
    "B0": ROOT
    / ".local/experiments/simulation/s4_3_pi0/training/pinch_tongs/"
    "s43_pi0_official_seed42/29999",
    "B1": ROOT
    / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/"
    "s43_pi1b_contact_tokens_seed42/29999",
    "B2": ROOT
    / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/"
    "s43_pi1c_contact_tokens_physical_aux_seed42/29999",
    "BVA": ROOT
    / ".local/experiments/simulation/s4_3_pi2u/bva/pinch_tongs/"
    "s43_pi2u_bva_seed42/29999",
    "B_HVA": ROOT
    / ".local/experiments/simulation/s4_3_pi2m/bhva/pinch_tongs/"
    "s43_pi2m_bhva_seed42/29999",
}

STEP_PATTERN = re.compile(r"(?:^|\n)Step (?P<step>\d+): (?P<metrics>[^\r\n]+)")
METRIC_PATTERN = re.compile(r"(?P<key>[a-z0-9_]+)=(?P<value>[-+0-9.eE]+)")
PROGRESS_PATTERN = re.compile(
    r"Progress on: (?P<current>[0-9.]+)(?P<suffix>[kM]?)it/30\.0kit "
    r"rate:(?P<rate>[0-9.]+)(?P<unit>s/it|it/s).*?elapsed:(?P<elapsed>[0-9:]+)"
)


@dataclass(frozen=True)
class CandidateSpec:
    model_id: str
    experiment: str
    mode_name: str
    physical_gpu_id: int
    physical_gpu_uuid: str
    contact_adapter_expected: bool

    @property
    def stem(self) -> str:
        return self.model_id.lower()


CANDIDATES = {
    "B_VA27": CandidateSpec(
        "B_VA27",
        "s43_pi2n_b_va27_seed42",
        "VA_PHYSICAL_AUX",
        1,
        "GPU-dc38e4af-8bd2-e334-2587-77fb72e4df67",
        False,
    ),
    "B_VAC_V": CandidateSpec(
        "B_VAC_V",
        "s43_pi2n_b_vac_v_seed42",
        "CONTACT_STATE_TOKENS_VAC_V_PHYSICAL_AUX",
        3,
        "GPU-92eab6a5-6b52-7750-ec56-edc0439bdb92",
        True,
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(32 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def reserve_failed_audit_attempt(spec: CandidateSpec) -> Path:
    """Reserve an immutable directory without touching canonical PASS outputs."""
    root = ARTIFACTS / "candidate_completion_attempts" / spec.stem
    root.mkdir(parents=True, exist_ok=True)
    for index in range(1, 10_000):
        attempt = root / f"attempt_{index:03d}"
        try:
            attempt.mkdir()
        except FileExistsError:
            continue
        return attempt
    raise RuntimeError(f"too many failed completion audits for {spec.model_id}")


def checkpoint_hashes(path: Path) -> tuple[str, str, list[dict[str, Any]]]:
    rows = [
        {
            "path": file.relative_to(path).as_posix(),
            "bytes": file.stat().st_size,
            "sha256": sha256_file(file),
        }
        for file in sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    ]

    def aggregate(selected: list[dict[str, Any]]) -> str:
        digest = hashlib.sha256()
        for row in selected:
            digest.update(row["path"].encode())
            digest.update(b"\0")
            digest.update(row["sha256"].encode())
            digest.update(b"\0")
            digest.update(str(row["bytes"]).encode())
            digest.update(b"\n")
        return digest.hexdigest()

    return (
        aggregate(rows),
        aggregate([row for row in rows if row["path"].startswith("params/")]),
        rows,
    )


def tree_hash(path: Path) -> str:
    return checkpoint_hashes(path)[0]


def parse_training_log(text: str) -> dict[str, Any]:
    normalized = text.replace("\r", "\n")
    metric_rows = []
    for match in STEP_PATTERN.finditer(normalized):
        metrics = {
            item.group("key"): float(item.group("value"))
            for item in METRIC_PATTERN.finditer(match.group("metrics"))
        }
        metric_rows.append({"step": int(match.group("step")), **metrics})
    progress_steps: list[int] = []
    rates: list[float] = []
    elapsed = None
    for match in PROGRESS_PATTERN.finditer(normalized):
        multiplier = {"": 1, "k": 1000, "M": 1_000_000}[match.group("suffix")]
        progress_steps.append(int(round(float(match.group("current")) * multiplier)))
        rate = float(match.group("rate"))
        rates.append(rate if match.group("unit") == "s/it" else 1.0 / rate)
        elapsed = match.group("elapsed")
    return {
        "metric_rows": metric_rows,
        "progress_steps": progress_steps,
        "rates_seconds_per_step": rates,
        "elapsed": elapsed,
    }


def elapsed_seconds(value: str | None) -> int | None:
    if value is None:
        return None
    fields = [int(item) for item in value.split(":")]
    if len(fields) != 3:
        return None
    hours, minutes, seconds = fields
    return hours * 3600 + minutes * 60 + seconds


def process_matches_candidate(spec: CandidateSpec) -> bool:
    result = subprocess.run(
        ["pgrep", "-af", "[t]rain_s4_3_pi2n.py"],
        text=True,
        capture_output=True,
        check=False,
    )
    return any(
        f"--model-id {spec.model_id}" in row
        and f"--exp-name {spec.experiment}" in row
        for row in result.stdout.splitlines()
    )


def restore_train_step(checkpoint: Path) -> int:
    import jax
    import numpy as np
    import orbax.checkpoint as ocp

    with ocp.PyTreeCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(checkpoint / "train_state")
        restore_args = jax.tree.map(
            lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), metadata
        )
        state = checkpointer.restore(
            checkpoint / "train_state",
            ocp.args.PyTreeRestore(item=metadata, restore_args=restore_args),
        )
    step = int(np.asarray(state["step"]).item())
    del state
    gc.collect()
    return step


def symbolic_checkpoint(spec: CandidateSpec) -> str:
    return (
        "$PI2N_RUN_ROOT/runs/"
        f"{spec.model_id}/pinch_tongs/{spec.experiment}/29999"
    )


def prerequisite_paths(spec: CandidateSpec) -> dict[str, Path]:
    common = {
        "lambda_calibration": ARTIFACTS / f"{spec.stem}_lambda_calibration.json",
        "loaded_base_gradient_gate": ARTIFACTS
        / f"{spec.stem}_loaded_base_gradient_gate.json",
        "target_audit": ARTIFACTS / f"{spec.stem}_target_audit.json",
    }
    if spec.model_id == "B_VA27":
        common.update(
            {
                "bva_mode_contract.json": ROOT
                / ".local/artifacts/simulation/s4_3_pi2u/bva_mode_contract.json",
                "bva_none_parity.json": ROOT
                / ".local/artifacts/simulation/s4_3_pi2u/bva_none_parity.json",
                "corrected_target_audit.json": ROOT
                / ".local/artifacts/simulation/s4_3_pi2m/corrected_target_audit.json",
            }
        )
    else:
        common["b_vac_v_mode_parity.json"] = ARTIFACTS / "b_vac_v_mode_parity.json"
    return common


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", choices=tuple(CANDIDATES), required=True)
    args = parser.parse_args()
    spec = CANDIDATES[args.model_id]
    experiment = RUN_ROOT / "runs" / spec.model_id / "pinch_tongs" / spec.experiment
    checkpoint = experiment / "29999"
    log_path = ROOT / f".local/logs/simulation/s4_3_pi2n/{spec.model_id}/train.log"
    state_dir = RUN_ROOT / "runtime_state" / spec.model_id
    job_path = state_dir / "job_status.json"
    heartbeat_path = state_dir / "heartbeat.json"
    freeze_path = ARTIFACTS / f"{spec.stem}_training_freeze.json"
    completion_path = ARTIFACTS / f"{spec.stem}_training_completion.json"
    manifest_path = ARTIFACTS / f"{spec.stem}_checkpoint_manifest.json"
    if completion_path.exists() or manifest_path.exists():
        raise SystemExit(f"refusing to overwrite completed {spec.model_id} audit")
    required = (
        experiment,
        checkpoint,
        log_path,
        job_path,
        heartbeat_path,
        freeze_path,
        BASE_MANIFEST,
        STARTING_IMMUTABILITY,
    )
    if any(not path.exists() for path in required):
        raise SystemExit(f"{spec.model_id}_FINAL_CHECKPOINT_OR_PROTOCOL_MISSING")
    job = json.loads(job_path.read_text())
    heartbeat = json.loads(heartbeat_path.read_text())
    if not (
        job.get("state") == "DONE"
        and heartbeat.get("state") == "DONE"
        and job.get("exit_code") == 0
        and heartbeat.get("exit_code") == 0
    ):
        raise SystemExit(f"{spec.model_id}_TRAINING_NOT_DONE_EXIT_ZERO")
    if process_matches_candidate(spec):
        raise SystemExit(f"{spec.model_id}_TRAINING_PROCESS_STILL_LIVE")

    freeze = json.loads(freeze_path.read_text())
    starting = json.loads(STARTING_IMMUTABILITY.read_text())
    log_text = log_path.read_text(errors="replace")
    parsed = parse_training_log(log_text)
    metric_rows = parsed["metric_rows"]
    progress_steps = parsed["progress_steps"]
    rates = parsed["rates_seconds_per_step"]
    expected_lambda = float(freeze["lambda_phys"])

    hash_started = perf_counter()
    checkpoint_sha, params_sha, files = checkpoint_hashes(checkpoint)
    protected_hashes = {
        name: tree_hash(path) for name, path in PROTECTED_CHECKPOINTS.items()
    }
    hash_elapsed = perf_counter() - hash_started

    train_state_started = perf_counter()
    train_state_step = restore_train_step(checkpoint)
    train_state_elapsed = perf_counter() - train_state_started

    os.chdir(OPENPI)
    sys.path[:0] = [str(ROOT), str(OPENPI / "src")]
    import flax.traverse_util
    import numpy as np
    from gr00t.simulation.pi05_tactile_unit import TactilePi0
    from gr00t.simulation.s4_3_pi1 import TactileUnitMode
    from openpi.models import model as openpi_model
    from scripts.simulation.train_s4_3_pi2n import build_config

    cold_load_started = perf_counter()
    params = openpi_model.restore_params(checkpoint / "params", restore_type=np.ndarray)
    arrays = {
        name: np.asarray(value)
        for name, value in flax.traverse_util.flatten_dict(params, sep="/").items()
    }
    config = build_config(spec.model_id, spec.experiment, expected_lambda, 1)
    model = config.model.load(params, remove_extra_params=False)
    cold_load_elapsed = perf_counter() - cold_load_started

    metadata = json.loads((checkpoint / "_CHECKPOINT_METADATA").read_text())
    commit_time = datetime.fromtimestamp(
        int(metadata["commit_timestamp_nsecs"]) / 1e9, tz=timezone.utc
    ).isoformat()
    nonfinite = sum(
        int(array.size - np.count_nonzero(np.isfinite(array)))
        for array in arrays.values()
    )
    contact_arrays = {
        name: value for name, value in arrays.items() if "contact_adapter" in name
    }
    physical_arrays = {
        name: value for name, value in arrays.items() if "physical_auxiliary" in name
    }
    numeric_dirs = sorted(
        int(path.name)
        for path in experiment.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    temporary_dirs = sorted(
        path.relative_to(experiment).as_posix()
        for path in experiment.rglob("*")
        if path.is_dir() and "orbax-checkpoint-tmp" in path.name
    )
    source_unchanged = all(
        sha256_file(ROOT / relative) == expected
        for relative, expected in freeze["source_files_sha256"].items()
    )
    snapshot = RUN_ROOT / "code_snapshots" / spec.model_id / "prelaunch_source.tar.gz"
    prerequisites = prerequisite_paths(spec)
    mode_hashes = freeze["mode_prerequisites_sha256"]
    prerequisite_unchanged = bool(
        sha256_file(prerequisites["lambda_calibration"])
        == freeze["lambda_calibration_sha256"]
        and sha256_file(prerequisites["loaded_base_gradient_gate"])
        == freeze["loaded_base_gradient_gate_sha256"]
        and sha256_file(prerequisites["target_audit"])
        == freeze["target_audit_sha256"]
        and all(
            sha256_file(prerequisites[name]) == expected
            for name, expected in mode_hashes.items()
        )
    )
    common_metrics = (
        "loss",
        "official_loss",
        "physical_loss",
        "lambda_phys",
        "grad_norm",
        "param_norm",
        "physical_auxiliary_grad_norm",
        "pi05_trainable_grad_norm",
    )
    required_metrics = common_metrics + (
        ("contact_adapter_grad_norm",) if spec.contact_adapter_expected else ()
    )
    expected_protected = {
        name: row["tree_sha256"]
        for name, row in starting["live_migrated_policy_checkpoints"].items()
    }
    logical_checkpoint = symbolic_checkpoint(spec)
    job_identity = bool(
        job.get("model_id") == spec.model_id
        and heartbeat.get("model_id") == spec.model_id
        and job.get("physical_gpu_id") == spec.physical_gpu_id
        and heartbeat.get("physical_gpu_id") == spec.physical_gpu_id
        and job.get("physical_gpu_uuid") == spec.physical_gpu_uuid
        and heartbeat.get("physical_gpu_uuid") == spec.physical_gpu_uuid
        and job.get("checkpoint_directory") == str(experiment)
        and heartbeat.get("checkpoint_directory") == str(experiment)
        and job.get("expected_final_checkpoint") == str(checkpoint)
        and heartbeat.get("expected_final_checkpoint") == str(checkpoint)
    )
    freeze_contract = bool(
        freeze.get("status") == "FROZEN_BEFORE_TRAINING"
        and freeze.get("model_id") == spec.model_id
        and freeze.get("seed") == 42
        and freeze.get("steps") == 30_000
        and freeze.get("global_batch_size") == 32
        and freeze.get("final_checkpoint_step") == 29_999
        and freeze.get("checkpoint_selection") == "FINAL_ONLY"
        and freeze.get("resume") is False
        and freeze.get("overwrite") is False
    )
    expected_mode = getattr(TactileUnitMode, spec.mode_name)
    gates = {
        "job_and_heartbeat_done_exit_zero": True,
        "job_heartbeat_run_identity_exact": job_identity,
        "training_process_exited": not process_matches_candidate(spec),
        "training_freeze_contract_exact": freeze_contract,
        "progress_reached_30000": max(progress_steps, default=-1) == 30_000,
        "restored_train_state_step_30000": train_state_step == 30_000,
        "metric_records_300": len(metric_rows) == 300,
        "metric_steps_exact_0_to_29900": [row["step"] for row in metric_rows]
        == list(range(0, 30_000, 100)),
        "final_save_requested": "Saving checkpoint at step 29999" in log_text,
        "final_save_finalized": "Finished saving checkpoint (finalized tmp dir)" in log_text
        and f"/{spec.experiment}/29999`" in log_text,
        "checkpoint_steps_exact": numeric_dirs == [10_000, 20_000, 29_999],
        "no_temporary_checkpoint_directories": not temporary_dirs,
        "final_checkpoint_metadata_present": (checkpoint / "_CHECKPOINT_METADATA").is_file(),
        "final_params_committed": (checkpoint / "params/manifest.ocdbt").is_file(),
        "final_train_state_committed": (checkpoint / "train_state/manifest.ocdbt").is_file(),
        "checkpoint_cold_load": bool(arrays),
        "exact_checkpoint_parameter_tree": type(model) is TactilePi0,
        "mode_identity_exact": model.tactile_unit_mode is expected_mode,
        "seed42": config.seed == 42,
        "lambda_identity": float(model.lambda_phys) == expected_lambda,
        "contact_adapter_identity": bool(contact_arrays)
        == spec.contact_adapter_expected,
        "physical_auxiliary_present": bool(physical_arrays),
        "all_checkpoint_arrays_nonempty": bool(arrays)
        and all(array.size > 0 for array in arrays.values()),
        "all_checkpoint_arrays_finite": nonfinite == 0,
        "all_logged_metrics_finite": bool(metric_rows)
        and all(
            name in row and math.isfinite(float(row[name]))
            for row in metric_rows
            for name in required_metrics
        ),
        "no_logged_nan_or_inf": re.search(
            r"=\s*(?:nan|[-+]?inf)(?:[,\s]|$)", log_text, re.I
        )
        is None,
        "lambda_constant_in_log": bool(metric_rows)
        and all(
            abs(float(row["lambda_phys"]) - expected_lambda) <= 5e-5
            for row in metric_rows
        ),
        "frozen_training_sources_unchanged": source_unchanged,
        "frozen_source_snapshot_unchanged": snapshot.is_file()
        and sha256_file(snapshot) == freeze["source_snapshot_sha256"],
        "frozen_base_manifest_unchanged": sha256_file(BASE_MANIFEST)
        == freeze["base_manifest_sha256"],
        "frozen_target_unchanged": sha256_file(Path(freeze["target_path"]))
        == freeze["target_sha256"],
        "frozen_prerequisites_unchanged": prerequisite_unchanged,
        "protected_policy_checkpoints_unchanged": protected_hashes
        == expected_protected,
        "dexjoco_revision_unchanged": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT / "third_party/dexjoco",
            text=True,
        ).strip()
        == EXPECTED_DEXJOCO,
        "dev_and_final_performance_still_unseen": all(
            json.loads((ARTIFACTS / name).read_text()).get("policy_performance_seen")
            is False
            for name in ("development_manifest.json", "final_reset_manifest.json")
        ),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    wall_seconds = elapsed_seconds(parsed["elapsed"])
    completion = {
        "schema": "tactile3d-unit.s4-3-pi2n-candidate-training-completion.v1",
        "status": status,
        "model_id": spec.model_id,
        "mode": spec.mode_name,
        "seed": 42,
        "lambda_phys": expected_lambda,
        "optimizer_steps": train_state_step,
        "global_batch_size": 32,
        "physical_gpu_id": spec.physical_gpu_id,
        "physical_gpu_uuid": spec.physical_gpu_uuid,
        "training_duration": parsed["elapsed"],
        "training_wall_seconds": wall_seconds,
        "gpu_hours": wall_seconds / 3600 if wall_seconds is not None else None,
        "median_recent_seconds_per_step": statistics.median(rates[-50:])
        if rates
        else None,
        "last_logged_metrics": metric_rows[-1] if metric_rows else None,
        "logged_metric_records": len(metric_rows),
        "checkpoint": logical_checkpoint,
        "checkpoint_commit_utc": commit_time,
        "checkpoint_tree_sha256": checkpoint_sha,
        "params_tree_sha256": params_sha,
        "checkpoint_files": len(files),
        "checkpoint_bytes": sum(row["bytes"] for row in files),
        "hash_elapsed_seconds": hash_elapsed,
        "train_state_restore_seconds": train_state_elapsed,
        "cold_load_elapsed_seconds": cold_load_elapsed,
        "parameter_array_leaves": len(arrays),
        "parameter_elements": sum(array.size for array in arrays.values()),
        "parameter_bytes": sum(array.nbytes for array in arrays.values()),
        "dtype_counts": dict(
            sorted(Counter(str(array.dtype) for array in arrays.values()).items())
        ),
        "contact_adapter_arrays": len(contact_arrays),
        "physical_auxiliary_arrays": len(physical_arrays),
        "nonfinite_checkpoint_elements": nonfinite,
        "training_freeze_sha256": sha256_file(freeze_path),
        "protected_checkpoint_tree_sha256": protected_hashes,
        "log": "$REPO_ROOT/" + log_path.relative_to(ROOT).as_posix(),
        "policy_evaluation_performed": False,
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
    }
    failed_attempt = None
    if status == "PASS":
        completion_output = completion_path
        manifest_output = manifest_path
    else:
        failed_attempt = reserve_failed_audit_attempt(spec)
        completion_output = failed_attempt / "training_completion.json"
        manifest_output = failed_attempt / "checkpoint_manifest.json"
    atomic_json(completion_output, completion)
    manifest = {
        "schema": "tactile3d-unit.s4-3-pi2n-candidate-checkpoint-manifest.v1",
        "status": status,
        "frozen": status == "PASS",
        "model_id": spec.model_id,
        "mode": spec.mode_name,
        "seed": 42,
        "lambda_phys": expected_lambda,
        "optimizer_steps": train_state_step,
        "checkpoint": logical_checkpoint,
        "checkpoint_tree_sha256": checkpoint_sha,
        "params_tree_sha256": params_sha,
        "tree_hash_algorithm": "sha256(sorted(relative_path NUL file_sha256 NUL bytes newline))",
        "file_manifest": files,
        "training_completion": "$REPO_ROOT/"
        + completion_output.relative_to(ROOT).as_posix(),
        "training_completion_sha256": sha256_file(completion_output),
        "policy_evaluation_performed": False,
        "gates": completion["gates"],
    }
    atomic_json(manifest_output, manifest)
    print(
        json.dumps(
            {
                "status": status,
                "model_id": spec.model_id,
                "optimizer_steps": train_state_step,
                "checkpoint_tree_sha256": checkpoint_sha,
                "params_tree_sha256": params_sha,
                "nonfinite": nonfinite,
                "failed_attempt": str(failed_attempt) if failed_attempt else None,
            },
            sort_keys=True,
        )
    )
    if status != "PASS":
        failed = [name for name, value in gates.items() if not value]
        raise SystemExit(
            f"{spec.model_id}_COMPLETION_AUDIT_FAIL_PRESERVED: "
            + ",".join(failed)
        )


if __name__ == "__main__":
    main()
