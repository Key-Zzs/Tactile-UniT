#!/usr/bin/env python3
"""Audit completion of the locked official 30k pi0.5 LoRA run."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import numpy as np  # noqa: E402
import orbax.checkpoint as ocp  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / ".local/experiments/simulation/s4_3_pi0/training/pinch_tongs/s43_pi0_official_seed42"
FINAL_CHECKPOINT = RUN_ROOT / "29999"
LOG = ROOT / ".local/logs/simulation/s4_3_pi0/train_official_seed42.log"
ARTIFACT = ROOT / ".local/artifacts/simulation/s4_3_pi0/training_completion.json"
TRAINING_CONFIG = ROOT / ".local/artifacts/simulation/s4_3_pi0/official_training_config.json"
TRAINING_LAUNCH = ROOT / ".local/artifacts/simulation/s4_3_pi0/training_launch.json"

EXPECTED_CHECKPOINT_STEPS = [10_000, 20_000, 29_999]
EXPECTED_FINAL_FILE_COUNT = 28
EXPECTED_FINAL_BYTES = 9_562_228_955
STEP_PATTERN = re.compile(
    r"Step (?P<step>\d+): grad_norm=(?P<grad>[-+0-9.eE]+), "
    r"loss=(?P<loss>[-+0-9.eE]+), param_norm=(?P<param>[-+0-9.eE]+)"
)
DURATION_PATTERN = re.compile(
    r"Progress on: 30\.0kit/30\.0kit.*remaining:00:00.*elapsed:(?P<duration>\d\d:\d\d:\d\d)"
)


def symbolic(path: Path) -> str:
    return "$REPO_ROOT/" + path.resolve().relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(payload: Any) -> None:
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    temporary = ARTIFACT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(ARTIFACT)


def process_alive() -> bool:
    result = subprocess.run(
        ["pgrep", "-f", "^.*python scripts/train.py pinch_tongs"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def tmux_alive() -> bool:
    return (
        subprocess.run(
            ["tmux", "has-session", "-t", "s43_pi0_pinch"],
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


def restore_train_step() -> int:
    path = FINAL_CHECKPOINT / "train_state"
    with ocp.PyTreeCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(path)
        restore_args = jax.tree.map(lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), metadata)
        state = checkpointer.restore(
            path,
            ocp.args.PyTreeRestore(item=metadata, restore_args=restore_args),
        )
    step = int(np.asarray(state["step"]).item())
    del state
    gc.collect()
    return step


def main() -> None:
    log_text = LOG.read_text(encoding="utf-8", errors="replace")
    metrics = [
        {
            "step": int(match.group("step")),
            "grad_norm": float(match.group("grad")),
            "loss": float(match.group("loss")),
            "param_norm": float(match.group("param")),
        }
        for match in STEP_PATTERN.finditer(log_text)
    ]
    duration_matches = list(DURATION_PATTERN.finditer(log_text))
    duration = duration_matches[-1].group("duration") if duration_matches else None
    numeric_dirs = sorted(int(path.name) for path in RUN_ROOT.iterdir() if path.is_dir() and path.name.isdigit())
    temporary_dirs = sorted(
        path.relative_to(RUN_ROOT).as_posix()
        for path in RUN_ROOT.rglob("*")
        if path.is_dir() and "orbax-checkpoint-tmp" in path.name
    )
    files = sorted(path for path in FINAL_CHECKPOINT.rglob("*") if path.is_file())
    file_manifest = {
        path.relative_to(FINAL_CHECKPOINT).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    }
    total_bytes = sum(item["bytes"] for item in file_manifest.values())
    checkpoint_metadata = json.loads((FINAL_CHECKPOINT / "_CHECKPOINT_METADATA").read_text(encoding="utf-8"))
    commit_ns = checkpoint_metadata["commit_timestamp_nsecs"]
    commit_time = datetime.fromtimestamp(commit_ns / 1e9, tz=timezone.utc).isoformat()
    train_step = restore_train_step()
    config = json.loads(TRAINING_CONFIG.read_text(encoding="utf-8"))
    launch = json.loads(TRAINING_LAUNCH.read_text(encoding="utf-8"))
    all_finite = bool(metrics) and all(
        math.isfinite(row[key]) for row in metrics for key in ("grad_norm", "loss", "param_norm")
    )
    gates = {
        "training_process_exited": not process_alive(),
        "tmux_session_exited": not tmux_alive(),
        "no_traceback": "Traceback (most recent call last)" not in log_text,
        "metric_records_300": len(metrics) == 300,
        "metrics_all_finite": all_finite,
        "first_metric_step_0": bool(metrics) and metrics[0]["step"] == 0,
        "last_logged_metric_step_29900": bool(metrics) and metrics[-1]["step"] == 29_900,
        "progress_30000_complete": duration is not None,
        "final_save_requested": "Saving checkpoint at step 29999" in log_text,
        "final_save_finalized": "Finished saving checkpoint (finalized tmp dir)" in log_text
        and str(FINAL_CHECKPOINT) in log_text,
        "checkpoint_steps_exact": numeric_dirs == EXPECTED_CHECKPOINT_STEPS,
        "no_temporary_checkpoint_dirs": not temporary_dirs,
        "final_file_count": len(files) == EXPECTED_FINAL_FILE_COUNT,
        "final_total_bytes": total_bytes == EXPECTED_FINAL_BYTES,
        "final_checkpoint_metadata": (FINAL_CHECKPOINT / "_CHECKPOINT_METADATA").is_file(),
        "final_params": (FINAL_CHECKPOINT / "params/manifest.ocdbt").is_file(),
        "final_train_state": (FINAL_CHECKPOINT / "train_state/manifest.ocdbt").is_file(),
        "final_assets": (FINAL_CHECKPOINT / "assets/local_repo/norm_stats.json").is_file(),
        "train_state_step_30000": train_step == 30_000,
        "official_pi05": config["model"]["family"] == "pi0.5" and config["model"]["pi05"],
        "official_lora_variants": config["model"]["paligemma_variant"] == "gemma_2b_lora"
        and config["model"]["action_expert_variant"] == "gemma_300m_lora",
        "official_steps": config["training"]["num_train_steps"] == 30_000,
        "algorithm_defaults_unchanged": not config["official_algorithmic_defaults_overridden"],
        "two_gpu_data_parallel": launch["physical_gpus"] == [1, 2]
        and launch["process_environment"]["CUDA_VISIBLE_DEVICES"] == "1,2",
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi0-training-completion.v1",
        "stage": "PI0-9",
        "task": "pinch_tongs",
        "run_root": symbolic(RUN_ROOT),
        "final_checkpoint": symbolic(FINAL_CHECKPOINT),
        "checkpoint_steps": numeric_dirs,
        "training_duration": duration,
        "physical_gpus": launch["physical_gpus"],
        "final_logged_metrics": metrics[-1] if metrics else None,
        "metric_record_count": len(metrics),
        "train_state_step": train_step,
        "final_checkpoint_commit_utc": commit_time,
        "final_checkpoint_file_count": len(files),
        "final_checkpoint_bytes": total_bytes,
        "hash_algorithm": "sha256",
        "checkpoint_manifest": file_manifest,
        "temporary_checkpoint_dirs": temporary_dirs,
        "gates": {name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
        "status": "PASS" if all(gates.values()) else "FAIL",
    }
    write_json(payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "train_state_step": train_step,
                "duration": duration,
                "checkpoint_files": len(files),
                "checkpoint_bytes": total_bytes,
                "final_metrics": payload["final_logged_metrics"],
            },
            sort_keys=True,
        )
    )
    if payload["status"] != "PASS":
        raise SystemExit("S4_3_PI0_TRAINING_COMPLETION_FAIL")


if __name__ == "__main__":
    main()
