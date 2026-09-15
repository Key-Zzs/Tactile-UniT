#!/usr/bin/env python3
"""Cold-load and freeze the completed formal S4.3-PI2U BVA checkpoint."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
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
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2u"
EXPERIMENT = ROOT / ".local/experiments/simulation/s4_3_pi2u/bva/pinch_tongs/s43_pi2u_bva_seed42"
CHECKPOINT = EXPERIMENT / "29999"
LOG = ROOT / ".local/logs/simulation/s4_3_pi2u/bva/train.log"
FROZEN = ARTIFACTS / "bva_training_protocol.json"
TRAINING_PROTOCOL = ROOT / "configs/simulation/s4_3_pi2u_bva_protocol.json"
EXPECTED_DEXJOCO = "8d23b0fab23b17a58c4b55f3942e17013aaf8267"
STEP_PATTERN = re.compile(r"Step (?P<step>\d+): (?P<metrics>[^\r\n]+)")
METRIC_PATTERN = re.compile(r"(?P<key>[a-z0-9_]+)=(?P<value>[-+0-9.eE]+)")
PROGRESS_PATTERN = re.compile(
    r"Progress on: (?P<current>[0-9.]+)(?P<suffix>[kM]?)it/30\.0kit "
    r"rate:(?P<rate>[0-9.]+)(?P<unit>s/it|it/s).*?elapsed:(?P<elapsed>[0-9:]+)"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def symbolic(path: Path) -> str:
    absolute = path if path.is_absolute() else ROOT / path
    return "$REPO_ROOT/" + absolute.absolute().relative_to(ROOT).as_posix()


def command(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(args, cwd=cwd, text=True, check=True, capture_output=True).stdout.strip()


def atomic_json(name: str, payload: Any) -> None:
    target = ARTIFACTS / name
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)


def checkpoint_hashes(path: Path) -> tuple[str, str, list[dict[str, Any]]]:
    records = []
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        records.append(
            {
                "path": file.relative_to(path).as_posix(),
                "bytes": file.stat().st_size,
                "sha256": sha256_file(file),
            }
        )

    def aggregate(rows: list[dict[str, Any]]) -> str:
        digest = hashlib.sha256()
        for row in rows:
            digest.update(row["path"].encode())
            digest.update(b"\0")
            digest.update(row["sha256"].encode())
            digest.update(b"\0")
            digest.update(str(row["bytes"]).encode())
            digest.update(b"\n")
        return digest.hexdigest()

    return aggregate(records), aggregate([row for row in records if row["path"].startswith("params/")]), records


def main() -> None:
    completion_path = ARTIFACTS / "bva_training_completion.json"
    manifest_path = ARTIFACTS / "bva_checkpoint_manifest.json"
    if completion_path.exists() or manifest_path.exists():
        raise SystemExit("refusing to overwrite completed BVA audit")
    if not CHECKPOINT.is_dir() or not LOG.is_file() or not FROZEN.is_file():
        raise SystemExit("S4_3_PI2U_BVA_FINAL_CHECKPOINT_OR_PROTOCOL_MISSING")
    frozen = json.loads(FROZEN.read_text())
    if frozen.get("status") != "FROZEN_BEFORE_TRAINING":
        raise SystemExit("BVA training protocol was not frozen")
    log_text = LOG.read_text(errors="replace")
    records = []
    for match in STEP_PATTERN.finditer(log_text):
        metrics = {item.group("key"): float(item.group("value")) for item in METRIC_PATTERN.finditer(match.group("metrics"))}
        records.append({"step": int(match.group("step")), **metrics})
    progress_steps, rates, elapsed = [], [], None
    for match in PROGRESS_PATTERN.finditer(log_text):
        multiplier = {"": 1, "k": 1000, "M": 1_000_000}[match.group("suffix")]
        progress_steps.append(int(round(float(match.group("current")) * multiplier)))
        rate = float(match.group("rate"))
        rates.append(rate if match.group("unit") == "s/it" else 1.0 / rate)
        elapsed = match.group("elapsed")

    started = perf_counter()
    checkpoint_sha, params_sha, files = checkpoint_hashes(CHECKPOINT)
    hash_seconds = perf_counter() - started
    os.chdir(OPENPI)
    sys.path[:0] = [str(ROOT), str(OPENPI / "src")]
    import flax.traverse_util
    import numpy as np
    from gr00t.simulation.pi05_tactile_unit import TactilePi0
    from gr00t.simulation.s4_3_pi1 import TactileUnitMode
    from openpi.models import model as openpi_model
    from scripts.simulation.train_s4_3_pi2u_bva import build_config

    restore_started = perf_counter()
    params = openpi_model.restore_params(CHECKPOINT / "params", restore_type=np.ndarray)
    flat = flax.traverse_util.flatten_dict(params, sep="/")
    arrays = {name: np.asarray(value) for name, value in flat.items()}
    config = build_config("s43_pi2u_bva_completion", float(frozen["lambda_phys"]), 1)
    model = config.model.load(params, remove_extra_params=False)
    restore_seconds = perf_counter() - restore_started
    metadata = json.loads((CHECKPOINT / "_CHECKPOINT_METADATA").read_text())
    commit_time = datetime.fromtimestamp(
        int(metadata["commit_timestamp_nsecs"]) / 1e9, tz=timezone.utc
    ).isoformat()
    nonfinite = sum(int(array.size - np.count_nonzero(np.isfinite(array))) for array in arrays.values())
    contact_arrays = {name: value for name, value in arrays.items() if "contact_adapter" in name}
    physical_arrays = {name: value for name, value in arrays.items() if "physical_auxiliary" in name}
    required_metrics = (
        "loss", "official_loss", "physical_loss", "lambda_phys", "grad_norm", "param_norm",
        "physical_auxiliary_grad_norm", "pi05_trainable_grad_norm",
    )
    frozen_inputs = {
        "contact_leakage_audit": ARTIFACTS / "contact_leakage_audit.json",
        "lambda_calibration": ARTIFACTS / "bva_lambda_calibration.json",
        "mode_contract": ARTIFACTS / "bva_mode_contract.json",
        "none_parity": ARTIFACTS / "bva_none_parity.json",
        "target_manifest": ARTIFACTS / "bva_target_manifest.json",
        "temporal_remediation": ROOT / "configs/simulation/s4_3_pi2u_bva_temporal_remediation.json",
        "tracked_protocol": TRAINING_PROTOCOL,
        "va_bridge_manifest": ARTIFACTS / "va_bridge_checkpoint_manifest.json",
    }
    implementation = {
        "target_builder": ROOT / "scripts/simulation/build_s4_3_pi2u_bva_targets.py",
        "model": ROOT / "gr00t/simulation/pi05_tactile_unit.py",
        "mode": ROOT / "gr00t/simulation/s4_3_pi1.py",
        "entrypoint": ROOT / "scripts/simulation/train_s4_3_pi2u_bva.py",
        "launcher": ROOT / "scripts/simulation/launch_s4_3_pi2u_bva.sh",
    }
    temporary = sorted(path.name for path in EXPERIMENT.glob("29999*tmp*"))
    gates = {
        "training_process_exited": subprocess.run(["pgrep", "-f", "[t]rain_s4_3_pi2u_bva.py"], capture_output=True).returncode != 0,
        "optimizer_steps_30000": max(progress_steps, default=-1) == 30_000,
        "final_checkpoint_committed": CHECKPOINT.is_dir() and not temporary,
        "checkpoint_cold_load": bool(arrays),
        "exact_checkpoint_parameter_tree": type(model) is TactilePi0,
        "mode_VA_PHYSICAL_AUX": model.tactile_unit_mode is TactileUnitMode.VA_PHYSICAL_AUX,
        "seed42": config.seed == 42,
        "lambda_identity": float(model.lambda_phys) == float(frozen["lambda_phys"]),
        "physical_auxiliary_present": bool(physical_arrays),
        "no_contact_adapter": not contact_arrays,
        "all_checkpoint_arrays_finite": nonfinite == 0,
        "all_checkpoint_arrays_nonempty": all(value.size > 0 for value in arrays.values()),
        "all_logged_metrics_finite": bool(records) and all(
            name in row and math.isfinite(float(row[name])) for row in records for name in required_metrics
        ),
        "no_logged_nan_or_inf": re.search(r"=\s*(?:nan|[-+]?inf)(?:[,\s]|$)", log_text, re.I) is None,
        "lambda_constant_in_log": all(abs(float(row["lambda_phys"]) - float(frozen["lambda_phys"])) <= 5e-5 for row in records),
        "frozen_inputs_unchanged": all(
            sha256_file(path) == frozen["inputs_sha256"][name] for name, path in frozen_inputs.items()
        ),
        "frozen_implementation_unchanged": all(
            sha256_file(path) == frozen["implementation_sha256"][name] for name, path in implementation.items()
        ),
        "dexjoco_revision_unchanged": command("git", "rev-parse", "HEAD", cwd=ROOT / "third_party/dexjoco") == EXPECTED_DEXJOCO,
        "no_evaluation_before_completion": not (ARTIFACTS / "bva_eval.json").exists(),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    completion = {
        "schema": "tactile3d-unit.s4-3-pi2u-bva-training-completion.v1",
        "status": status,
        "mode": "VA_PHYSICAL_AUX",
        "seed": 42,
        "lambda_phys": float(frozen["lambda_phys"]),
        "optimizer_steps": 30000,
        "last_periodic_step": max((row["step"] for row in records), default=None),
        "last_logged_metrics": records[-1] if records else None,
        "logged_metric_records": len(records),
        "global_batch_size": 32,
        "training_duration": elapsed,
        "median_recent_seconds_per_step": statistics.median(rates[-50:]) if rates else None,
        "checkpoint": symbolic(CHECKPOINT),
        "checkpoint_commit_utc": commit_time,
        "checkpoint_tree_sha256": checkpoint_sha,
        "params_tree_sha256": params_sha,
        "checkpoint_files": len(files),
        "checkpoint_bytes": sum(row["bytes"] for row in files),
        "cold_load_elapsed_seconds": restore_seconds,
        "hash_elapsed_seconds": hash_seconds,
        "parameter_array_leaves": len(arrays),
        "parameter_elements": sum(value.size for value in arrays.values()),
        "parameter_bytes": sum(value.nbytes for value in arrays.values()),
        "dtype_counts": dict(sorted(Counter(str(value.dtype) for value in arrays.values()).items())),
        "contact_adapter_arrays": len(contact_arrays),
        "physical_auxiliary_arrays": len(physical_arrays),
        "nonfinite_checkpoint_elements": nonfinite,
        "log": symbolic(LOG),
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
    }
    manifest = {
        "schema": "tactile3d-unit.s4-3-pi2u-bva-checkpoint-manifest.v1",
        "status": status,
        "frozen": status == "PASS",
        "model_identity": "BVA VA_PHYSICAL_AUX",
        "seed": 42,
        "lambda_phys": float(frozen["lambda_phys"]),
        "optimizer_steps": 30000,
        "checkpoint": symbolic(CHECKPOINT),
        "checkpoint_tree_sha256": checkpoint_sha,
        "params_tree_sha256": params_sha,
        "tree_hash_algorithm": "sha256(sorted(relative_path NUL file_sha256 NUL bytes newline))",
        "file_manifest": files,
        "training_completion": "$REPO_ROOT/.local/artifacts/simulation/s4_3_pi2u/bva_training_completion.json",
        "evaluation_performed": False,
        "gates": completion["gates"],
    }
    atomic_json("bva_training_completion.json", completion)
    manifest["training_completion_sha256"] = sha256_file(completion_path)
    atomic_json("bva_checkpoint_manifest.json", manifest)
    print(json.dumps({"status": status, "checkpoint_tree_sha256": checkpoint_sha, "nonfinite": nonfinite}, sort_keys=True))
    if status != "PASS":
        raise SystemExit("S4_3_PI2U_BVA_COMPLETION_AUDIT_FAIL: " + ",".join(name for name, value in gates.items() if not value))


if __name__ == "__main__":
    main()
