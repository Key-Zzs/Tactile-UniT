#!/usr/bin/env python3
"""Cold-load, hash, and freeze the completed S4.3-PI1B checkpoint."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter
from typing import Any


os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[2]
OPENPI = ROOT / ".local/external/simulation/s4_3_pi1/openpi"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi1"
EXPERIMENT = (
    ROOT
    / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/"
    "s43_pi1b_contact_tokens_seed42"
)
CHECKPOINT = EXPERIMENT / "29999"
LOG = ROOT / ".local/logs/simulation/s4_3_pi1/pi1b/train.log"
EXPECTED_DEXJOCO = "8d23b0fab23b17a58c4b55f3942e17013aaf8267"
STEP_PATTERN = re.compile(r"Step (?P<step>\d+): (?P<metrics>[^\r\n]+)")
METRIC_PATTERN = re.compile(r"(?P<key>[a-z0-9_]+)=(?P<value>[-+0-9.eE]+)")
PROGRESS_PATTERN = re.compile(r"Progress on: (?P<current>[0-9.]+)(?P<suffix>[kM]?)it/30\.0kit")


def symbolic(path: Path) -> str:
    return "$REPO_ROOT/" + path.resolve().relative_to(ROOT).as_posix()


def command(*args: str, cwd: Path = ROOT) -> str:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_hashes(path: Path) -> tuple[str, str, list[dict[str, Any]]]:
    records = []
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file.relative_to(path).as_posix()
        records.append({"path": relative, "bytes": file.stat().st_size, "sha256": sha256_file(file)})

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
        aggregate(records),
        aggregate([row for row in records if row["path"].startswith("params/")]),
        records,
    )


def atomic_json(name: str, payload: Any) -> None:
    target = ARTIFACTS / name
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)


def current_s4_2() -> tuple[dict[str, str], dict[str, str]]:
    frozen = json.loads((ARTIFACTS / "s4_2_immutability.json").read_text())
    checkpoints = {
        name: sha256_file(ROOT / relative)
        for name, relative in frozen["checkpoint_paths"].items()
    }
    configs = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in sorted((ROOT / "configs/simulation").glob("s4_2*.json"))
    }
    return checkpoints, configs


def main() -> None:
    if not CHECKPOINT.is_dir() or not LOG.is_file():
        raise SystemExit("S4_3_PI1B_FINAL_CHECKPOINT_OR_LOG_MISSING")

    text = LOG.read_text(errors="replace")
    steps = []
    for match in STEP_PATTERN.finditer(text):
        metrics = {
            item.group("key"): float(item.group("value"))
            for item in METRIC_PATTERN.finditer(match.group("metrics"))
        }
        steps.append({"step": int(match.group("step")), **metrics})
    progress = []
    for match in PROGRESS_PATTERN.finditer(text):
        multiplier = {"": 1, "k": 1000, "M": 1_000_000}[match.group("suffix")]
        progress.append(int(round(float(match.group("current")) * multiplier)))

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
    from scripts.simulation.train_s4_3_pi1 import build_config

    restore_started = perf_counter()
    params = openpi_model.restore_params(CHECKPOINT / "params", restore_type=np.ndarray)
    flat = flax.traverse_util.flatten_dict(params, sep="/")
    arrays = {name: np.asarray(value) for name, value in flat.items()}
    model_config = build_config("CONTACT_STATE_TOKENS", "pi1b_completion_audit", 0.0)
    model = model_config.model.load(params, remove_extra_params=False)
    restore_seconds = perf_counter() - restore_started

    frozen = json.loads((ARTIFACTS / "s4_2_immutability.json").read_text())
    current_checkpoints, current_configs = current_s4_2()
    metadata = json.loads((CHECKPOINT / "_CHECKPOINT_METADATA").read_text())
    nonfinite = sum(int(array.size - np.count_nonzero(np.isfinite(array))) for array in arrays.values())
    dtype_counts = Counter(str(array.dtype) for array in arrays.values())
    adapter_arrays = {name: array for name, array in arrays.items() if "contact_adapter" in name}
    physical_arrays = {name: array for name, array in arrays.items() if "physical_auxiliary" in name}
    required_metrics = (
        "loss",
        "official_loss",
        "grad_norm",
        "param_norm",
        "contact_adapter_grad_norm",
        "pi05_trainable_grad_norm",
    )
    all_metrics_finite = bool(steps) and all(
        name in row and math.isfinite(float(row[name])) for row in steps for name in required_metrics
    )
    temporary_siblings = sorted(path.name for path in EXPERIMENT.glob("29999*tmp*"))
    commit_ns = int(metadata["commit_timestamp_nsecs"])
    commit_time = datetime.fromtimestamp(commit_ns / 1e9, tz=timezone.utc).isoformat()
    gates = {
        "training_process_exited": subprocess.run(
            ["pgrep", "-f", "[t]rain_s4_3_pi1.py.*CONTACT_STATE_TOKENS"], capture_output=True
        ).returncode != 0,
        "optimizer_steps_30000": max(progress, default=-1) == 30_000,
        "final_checkpoint_committed": CHECKPOINT.is_dir() and not temporary_siblings,
        "checkpoint_cold_load": bool(arrays),
        "exact_checkpoint_parameter_tree": model is not None,
        "mode_identity_B1": (
            type(model) is TactilePi0
            and model.tactile_unit_mode is TactileUnitMode.CONTACT_STATE_TOKENS
            and float(model.lambda_phys) == 0.0
        ),
        "seed42": model_config.seed == 42,
        "all_checkpoint_arrays_nonempty": all(array.size > 0 for array in arrays.values()),
        "all_checkpoint_arrays_finite": nonfinite == 0,
        "all_logged_metrics_finite": all_metrics_finite,
        "no_logged_nan_or_inf": re.search(r"=\s*(?:nan|[-+]?inf)(?:[,\s]|$)", text, re.I) is None,
        "contact_adapter_present": bool(adapter_arrays),
        "physical_auxiliary_absent": not physical_arrays,
        "checkpoint_sha256_recorded": len(checkpoint_sha) == 64 and len(params_sha) == 64,
        "training_manifest_inputs_present": all(
            item in text
            for item in (
                "DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets/pinch_tongs",
                "pinch_tongs_official_tactile/sidecar.npz",
                "DexJoCo-Pi05/pi05_base/params",
                "local_batch_size: 32",
            )
        ),
        "s4_2_checkpoints_unchanged": current_checkpoints == frozen["checkpoints"],
        "s4_2_configs_unchanged": current_configs == frozen["tracked_configs"],
        "dexjoco_revision_unchanged": command("git", "rev-parse", "HEAD", cwd=ROOT / "third_party/dexjoco")
        == EXPECTED_DEXJOCO,
        "pi1d_not_run": not (ARTIFACTS / "pi1d_results.json").exists(),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    summary = {
        "schema": "tactile3d-unit.s4-3-pi1b-training-manifest.v1",
        "status": status,
        "mode": "CONTACT_STATE_TOKENS",
        "seed": 42,
        "optimizer_steps": 30_000,
        "last_periodic_step": max((row["step"] for row in steps), default=None),
        "logged_metric_records": len(steps),
        "last_logged_metrics": steps[-1] if steps else None,
        "global_batch_size": 32,
        "physical_gpus": [1, 2],
        "checkpoint": symbolic(CHECKPOINT),
        "params": symbolic(CHECKPOINT / "params"),
        "checkpoint_commit_utc": commit_time,
        "checkpoint_tree_sha256": checkpoint_sha,
        "params_tree_sha256": params_sha,
        "tree_hash_algorithm": "sha256(sorted(relative_path NUL file_sha256 NUL bytes newline))",
        "checkpoint_files": len(files),
        "checkpoint_bytes": sum(row["bytes"] for row in files),
        "hash_elapsed_seconds": hash_seconds,
        "cold_load_elapsed_seconds": restore_seconds,
        "parameter_array_leaves": len(arrays),
        "parameter_elements": sum(array.size for array in arrays.values()),
        "parameter_bytes": sum(array.nbytes for array in arrays.values()),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "contact_adapter_arrays": len(adapter_arrays),
        "physical_auxiliary_arrays": len(physical_arrays),
        "nonfinite_checkpoint_elements": nonfinite,
        "log": symbolic(LOG),
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
    }
    freeze = {
        "schema": "tactile3d-unit.s4-3-pi1b-checkpoint-freeze.v1",
        "status": status,
        "frozen": status == "PASS",
        "model_identity": "B1 CONTACT_STATE_TOKENS",
        "seed": 42,
        "optimizer_steps": 30_000,
        "checkpoint": symbolic(CHECKPOINT),
        "checkpoint_tree_sha256": checkpoint_sha,
        "params_tree_sha256": params_sha,
        "training_manifest": "$REPO_ROOT/.local/artifacts/simulation/s4_3_pi1/pi1b_training_manifest.json",
        "training_manifest_sha256": None,
        "pi1d_evaluation_performed": False,
        "gates": summary["gates"],
    }
    atomic_json("pi1b_training_manifest.json", summary)
    freeze["training_manifest_sha256"] = sha256_file(ARTIFACTS / "pi1b_training_manifest.json")
    atomic_json("pi1b_checkpoint_freeze.json", freeze)
    print(
        json.dumps(
            {
                "status": status,
                "optimizer_steps": 30_000,
                "checkpoint_tree_sha256": checkpoint_sha,
                "parameter_array_leaves": len(arrays),
                "nonfinite_checkpoint_elements": nonfinite,
            },
            sort_keys=True,
        )
    )
    if status != "PASS":
        failed = [name for name, value in gates.items() if not value]
        raise SystemExit("S4_3_PI1B_COMPLETION_AUDIT_FAIL: " + ",".join(failed))


if __name__ == "__main__":
    main()
