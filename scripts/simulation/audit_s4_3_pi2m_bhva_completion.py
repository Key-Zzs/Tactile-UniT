#!/usr/bin/env python3
"""Cold-load, verify, and freeze the completed S4.3-PI2M B_HVA checkpoint."""

from __future__ import annotations

from collections import Counter
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
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2m"
EXPERIMENT = (
    ROOT
    / ".local/experiments/simulation/s4_3_pi2m/bhva/pinch_tongs/"
    "s43_pi2m_bhva_seed42"
)
CHECKPOINT = EXPERIMENT / "29999"
LOG = ROOT / ".local/logs/simulation/s4_3_pi2m/bhva/train.log"
TRAINING_FREEZE = ARTIFACTS / "training_protocol_freeze.json"
JOB_STATUS = ARTIFACTS / "job_status.json"
COMPLETION = ARTIFACTS / "training_completion.json"
MANIFEST = ARTIFACTS / "bhva_checkpoint_manifest.json"
EXPECTED_DEXJOCO = "8d23b0fab23b17a58c4b55f3942e17013aaf8267"
EXPECTED_PROTECTED = {
    "B0": "1e7a6ace5d69a988a8b258e1c56a24d88b077580a05b27be3f510df9ac3864f3",
    "B1": "04b59cbc3491bf4e88dc75558a08a94e5588e2fbe398d413e1766a87c7ab6f4f",
    "B2": "86c4908533ca24da06ae66f943e3605b5ccbae455441290ca8fb35514359e949",
    "BVA": "04b609d7cc89e5fffdfab8219bf34da362117a15d0c7e3d5cd4ee20a9ee4770d",
}
PROTECTED = {
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
}
EXPECTED_S42 = {
    "s4_2": "7da07e8f5babe705d1fd643fc54dcce6db25e3636edf043aae2f7fec940a35ba",
    "s4_2_formal": "4404d9bd732999a6bec98bfcd9ea072d6215c642c60f66bfa0aa6671319035e9",
    "s4_2dr": "48bdf8a02773a898c25725d3afa46d083371e5113dee71a1f8ed34e596fa5e75",
    "s4_2ds": "528c2ecd55e529cba4604c5ecfd57466b56f9fcdeff900f86714ee9e44624d7e",
    "s4_2r": "90d801dfd541bc3d8065fb8e3a9d88d443377b51a653188290ea4fbd6a37b3eb",
}
E_T = ROOT / ".local/experiments/simulation/s4_2r/contact_state/accepted.pt"
CONTACT_SIDECAR = (
    ROOT / ".local/datasets/simulation/s4_3_pi1/pinch_tongs_official_tactile/sidecar.npz"
)
TARGET_SIDECAR = (
    ROOT / ".local/datasets/simulation/s4_3_pi2m/pinch_tongs_va_t27/sidecar.npz"
)
STEP_PATTERN = re.compile(r"Step (?P<step>\d+): (?P<metrics>[^\r\n]+)")
METRIC_PATTERN = re.compile(r"(?P<key>[a-z0-9_]+)=(?P<value>[-+0-9.eE]+)")
PROGRESS_PATTERN = re.compile(
    r"Progress on: (?P<current>[0-9.]+)(?P<suffix>[kM]?)it/30\.0kit "
    r"rate:(?P<rate>[0-9.]+)(?P<unit>s/it|it/s).*?elapsed:(?P<elapsed>[0-9:]+)"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(32 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def symbolic(path: Path) -> str:
    return "$REPO_ROOT/" + path.resolve().relative_to(ROOT).as_posix()


def command(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        args, cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def checkpoint_hashes(path: Path) -> tuple[str, str, list[dict[str, Any]]]:
    rows = [
        {
            "path": file.relative_to(path).as_posix(),
            "bytes": file.stat().st_size,
            "sha256": sha256_file(file),
        }
        for file in sorted(item for item in path.rglob("*") if item.is_file())
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


def restore_train_step() -> int:
    import jax
    import numpy as np
    import orbax.checkpoint as ocp

    with ocp.PyTreeCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(CHECKPOINT / "train_state")
        restore_args = jax.tree.map(
            lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), metadata
        )
        state = checkpointer.restore(
            CHECKPOINT / "train_state",
            ocp.args.PyTreeRestore(item=metadata, restore_args=restore_args),
        )
    step = int(np.asarray(state["step"]).item())
    del state
    gc.collect()
    return step


def elapsed_seconds(value: str | None) -> int | None:
    if value is None:
        return None
    hours, minutes, seconds = (int(item) for item in value.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def main() -> None:
    if COMPLETION.exists() or MANIFEST.exists():
        raise SystemExit("refusing to overwrite completed PI2M B_HVA audit")
    required = (CHECKPOINT, LOG, TRAINING_FREEZE, JOB_STATUS)
    if any(not path.exists() for path in required):
        raise SystemExit("PI2M_BHVA_FINAL_CHECKPOINT_OR_PROTOCOL_MISSING")

    freeze = json.loads(TRAINING_FREEZE.read_text())
    job = json.loads(JOB_STATUS.read_text())
    log_text = LOG.read_text(errors="replace")
    metric_rows = []
    for match in STEP_PATTERN.finditer(log_text):
        metrics = {
            item.group("key"): float(item.group("value"))
            for item in METRIC_PATTERN.finditer(match.group("metrics"))
        }
        metric_rows.append({"step": int(match.group("step")), **metrics})
    progress_steps, rates, elapsed = [], [], None
    for match in PROGRESS_PATTERN.finditer(log_text):
        multiplier = {"": 1, "k": 1000, "M": 1_000_000}[match.group("suffix")]
        progress_steps.append(int(round(float(match.group("current")) * multiplier)))
        rate = float(match.group("rate"))
        rates.append(rate if match.group("unit") == "s/it" else 1.0 / rate)
        elapsed = match.group("elapsed")

    hash_started = perf_counter()
    checkpoint_sha, params_sha, files = checkpoint_hashes(CHECKPOINT)
    protected_hashes = {name: tree_hash(path) for name, path in PROTECTED.items()}
    s42_hashes = {
        name: tree_hash(ROOT / ".local/experiments/simulation" / name)
        for name in EXPECTED_S42
    }
    hash_seconds = perf_counter() - hash_started

    train_state_started = perf_counter()
    train_state_step = restore_train_step()
    train_state_restore_seconds = perf_counter() - train_state_started

    os.chdir(OPENPI)
    sys.path[:0] = [str(ROOT), str(OPENPI / "src")]
    import flax.traverse_util
    import numpy as np
    from gr00t.simulation.pi05_tactile_unit import TactilePi0
    from gr00t.simulation.s4_3_pi1 import TactileUnitMode
    from openpi.models import model as openpi_model
    from scripts.simulation.train_s4_3_pi2m_bhva import build_config

    cold_load_started = perf_counter()
    params = openpi_model.restore_params(CHECKPOINT / "params", restore_type=np.ndarray)
    flat = flax.traverse_util.flatten_dict(params, sep="/")
    arrays = {name: np.asarray(value) for name, value in flat.items()}
    config = build_config("s43_pi2m_bhva_completion", float(freeze["lambda_phys"]), 1)
    model = config.model.load(params, remove_extra_params=False)
    cold_load_seconds = perf_counter() - cold_load_started

    metadata = json.loads((CHECKPOINT / "_CHECKPOINT_METADATA").read_text())
    commit_time = datetime.fromtimestamp(
        int(metadata["commit_timestamp_nsecs"]) / 1e9, tz=timezone.utc
    ).isoformat()
    nonfinite = sum(
        int(array.size - np.count_nonzero(np.isfinite(array))) for array in arrays.values()
    )
    contact_arrays = {name: value for name, value in arrays.items() if "contact_adapter" in name}
    physical_arrays = {
        name: value for name, value in arrays.items() if "physical_auxiliary" in name
    }
    required_metrics = (
        "loss",
        "official_loss",
        "physical_loss",
        "lambda_phys",
        "grad_norm",
        "param_norm",
        "contact_adapter_grad_norm",
        "physical_auxiliary_grad_norm",
        "pi05_trainable_grad_norm",
    )
    expected_lambda = float(freeze["lambda_phys"])
    numeric_dirs = sorted(
        int(path.name)
        for path in EXPERIMENT.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    temporary = sorted(
        path.relative_to(EXPERIMENT).as_posix()
        for path in EXPERIMENT.rglob("*")
        if path.is_dir() and "orbax-checkpoint-tmp" in path.name
    )
    code_unchanged = all(
        sha256_file(ROOT / relative) == expected
        for relative, expected in freeze["code_sha256"].items()
    )
    prerequisites = {
        "target": ARTIFACTS / "corrected_target_audit.json",
        "mode": ARTIFACTS / "mode_parity.json",
        "gradient": ARTIFACTS / "loaded_base_gradient_gate.json",
        "lambda": ARTIFACTS / "lambda_calibration.json",
        "reset": ARTIFACTS / "fresh_reset_manifest.json",
    }
    protocol_inputs_unchanged = (
        sha256_file(ROOT / "configs/simulation/s4_3_pi2m_bhva_protocol.json")
        == freeze["config_sha256"]
        and all(
            sha256_file(path) == freeze["prerequisite_sha256"][name]
            for name, path in prerequisites.items()
        )
    )
    gates = {
        "job_status_done_exit_zero": job.get("state") == "DONE" and job.get("exit_code") == 0,
        "training_process_exited": subprocess.run(
            ["pgrep", "-f", "[t]rain_s4_3_pi2m_bhva.py"], capture_output=True
        ).returncode
        != 0,
        "optimizer_steps_30000": max(progress_steps, default=-1) == 30_000,
        "restored_train_state_step_30000": train_state_step == 30_000,
        "metric_records_300": len(metric_rows) == 300,
        "first_metric_step_0": bool(metric_rows) and metric_rows[0]["step"] == 0,
        "last_metric_step_29900": bool(metric_rows) and metric_rows[-1]["step"] == 29_900,
        "final_save_requested": "Saving checkpoint at step 29999" in log_text,
        "final_save_finalized": "Finished saving checkpoint (finalized tmp dir)" in log_text
        and str(CHECKPOINT) in log_text,
        "checkpoint_steps_exact": numeric_dirs == [10_000, 20_000, 29_999],
        "final_checkpoint_committed": CHECKPOINT.is_dir() and not temporary,
        "checkpoint_cold_load": bool(arrays),
        "exact_checkpoint_parameter_tree": type(model) is TactilePi0,
        "mode_identity_B_HVA": model.tactile_unit_mode
        is TactileUnitMode.CONTACT_STATE_TOKENS_VA_PHYSICAL_AUX,
        "seed42": config.seed == 42,
        "lambda_identity": float(model.lambda_phys) == expected_lambda,
        "contact_adapter_present": bool(contact_arrays),
        "physical_auxiliary_present": bool(physical_arrays),
        "all_checkpoint_arrays_nonempty": all(array.size > 0 for array in arrays.values()),
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
        "lambda_constant_in_log": all(
            abs(float(row["lambda_phys"]) - expected_lambda) <= 5e-5
            for row in metric_rows
        ),
        "frozen_training_code_unchanged": code_unchanged,
        "frozen_protocol_inputs_unchanged": protocol_inputs_unchanged,
        "contact_sidecar_unchanged": sha256_file(CONTACT_SIDECAR)
        == freeze["contact_sidecar_sha256"],
        "corrected_target_sidecar_unchanged": sha256_file(TARGET_SIDECAR)
        == freeze["target_sidecar_sha256"],
        "protected_B0_B1_B2_BVA_unchanged": protected_hashes == EXPECTED_PROTECTED,
        "protected_E_T_unchanged": sha256_file(E_T)
        == "3d4519195e7a0d9f5af63399840a48121d5f614ea587c80b9461fc696a372a19",
        "protected_S4_2_trees_unchanged": s42_hashes == EXPECTED_S42,
        "dexjoco_revision_unchanged": command(
            "git", "rev-parse", "HEAD", cwd=ROOT / "third_party/dexjoco"
        )
        == EXPECTED_DEXJOCO,
        "no_formal_evaluation_before_completion": not any(
            (ARTIFACTS / f"{name}_raw_rollouts.json").exists()
            for name in ("b1", "b_hva", "bhva", "b2")
        ),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    wall_seconds = elapsed_seconds(elapsed)
    completion = {
        "schema": "tactile3d-unit.s4-3-pi2m-bhva-training-completion.v1",
        "status": status,
        "model_id": "B_HVA",
        "mode": "CONTACT_STATE_TOKENS_VA_PHYSICAL_AUX",
        "seed": 42,
        "lambda_phys": expected_lambda,
        "optimizer_steps": train_state_step,
        "last_periodic_step": metric_rows[-1]["step"] if metric_rows else None,
        "last_logged_metrics": metric_rows[-1] if metric_rows else None,
        "logged_metric_records": len(metric_rows),
        "global_batch_size": 32,
        "physical_gpus": [3],
        "training_duration": elapsed,
        "training_wall_seconds": wall_seconds,
        "gpu_hours": wall_seconds / 3600 if wall_seconds is not None else None,
        "median_recent_seconds_per_step": statistics.median(rates[-50:]) if rates else None,
        "checkpoint": symbolic(CHECKPOINT),
        "checkpoint_commit_utc": commit_time,
        "checkpoint_tree_sha256": checkpoint_sha,
        "params_tree_sha256": params_sha,
        "checkpoint_files": len(files),
        "checkpoint_bytes": sum(row["bytes"] for row in files),
        "hash_elapsed_seconds": hash_seconds,
        "train_state_restore_seconds": train_state_restore_seconds,
        "cold_load_elapsed_seconds": cold_load_seconds,
        "parameter_array_leaves": len(arrays),
        "parameter_elements": sum(array.size for array in arrays.values()),
        "parameter_bytes": sum(array.nbytes for array in arrays.values()),
        "dtype_counts": dict(
            sorted(Counter(str(array.dtype) for array in arrays.values()).items())
        ),
        "contact_adapter_arrays": len(contact_arrays),
        "physical_auxiliary_arrays": len(physical_arrays),
        "nonfinite_checkpoint_elements": nonfinite,
        "protected_checkpoint_tree_sha256": protected_hashes,
        "protected_S4_2_tree_sha256": s42_hashes,
        "log": symbolic(LOG),
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
    }
    atomic_json(COMPLETION, completion)
    manifest = {
        "schema": "tactile3d-unit.s4-3-pi2m-bhva-checkpoint-manifest.v1",
        "status": status,
        "frozen": status == "PASS",
        "model_identity": "B_HVA CONTACT_STATE_TOKENS_VA_PHYSICAL_AUX",
        "seed": 42,
        "lambda_phys": expected_lambda,
        "optimizer_steps": train_state_step,
        "checkpoint": symbolic(CHECKPOINT),
        "checkpoint_tree_sha256": checkpoint_sha,
        "params_tree_sha256": params_sha,
        "tree_hash_algorithm": "sha256(sorted(relative_path NUL file_sha256 NUL bytes newline))",
        "file_manifest": files,
        "training_completion": symbolic(COMPLETION),
        "training_completion_sha256": sha256_file(COMPLETION),
        "formal_evaluation_performed": False,
        "gates": completion["gates"],
    }
    atomic_json(MANIFEST, manifest)
    print(
        json.dumps(
            {
                "status": status,
                "optimizer_steps": train_state_step,
                "checkpoint_tree_sha256": checkpoint_sha,
                "params_tree_sha256": params_sha,
                "nonfinite": nonfinite,
            },
            sort_keys=True,
        )
    )
    if status != "PASS":
        failed = [name for name, value in gates.items() if not value]
        raise SystemExit("PI2M_BHVA_COMPLETION_AUDIT_FAIL: " + ",".join(failed))


if __name__ == "__main__":
    main()
