#!/usr/bin/env python3
"""Audit the stable persistent launch of the unique B_HVA seed-42 run."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2m"
OUTPUT = ARTIFACTS / "training_launch_audit.json"
STEP_PATTERN = re.compile(r"Step (?P<step>\d+): (?P<metrics>[^\r\n]+)")
METRIC_PATTERN = re.compile(r"(?P<key>[a-z0-9_]+)=(?P<value>[-+0-9.eE]+)")
PROGRESS_PATTERN = re.compile(
    r"Progress on: (?P<current>[0-9.]+)(?P<suffix>[kM]?)it/30\.0kit "
    r"rate:(?P<rate>[0-9.]+)(?P<unit>s/it|it/s)"
)


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, check=False, capture_output=True, text=True)


def find_pid() -> int | None:
    candidates = []
    for child in Path("/proc").iterdir():
        if not child.name.isdigit():
            continue
        try:
            command = (child / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeDecodeError):
            continue
        if "train_s4_3_pi2m_bhva.py" in command and "--exp-name s43_pi2m_bhva_seed42" in command:
            candidates.append(int(child.name))
    return max(candidates) if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-timing-samples", type=int, default=50)
    args = parser.parse_args()
    launch = json.loads((ARTIFACTS / "training_launch.json").read_text())
    freeze = json.loads((ARTIFACTS / "training_protocol_freeze.json").read_text())
    status = json.loads((ARTIFACTS / "job_status.json").read_text())
    log_path = Path(launch["log_path"])
    checkpoint_dir = Path(launch["experiment_directory"])
    text = log_path.read_text(errors="replace")
    step_records = []
    for match in STEP_PATTERN.finditer(text):
        metrics = {
            item.group("key"): float(item.group("value"))
            for item in METRIC_PATTERN.finditer(match.group("metrics"))
        }
        step_records.append({"step": int(match.group("step")), **metrics})
    timing = []
    progress_steps = []
    for match in PROGRESS_PATTERN.finditer(text):
        rate = float(match.group("rate"))
        timing.append(rate if match.group("unit") == "s/it" else 1.0 / rate)
        multiplier = {"": 1, "k": 1000, "M": 1_000_000}[match.group("suffix")]
        progress_steps.append(int(round(float(match.group("current")) * multiplier)))
    steady = timing[-args.minimum_timing_samples :]
    median_seconds = statistics.median(steady) if steady else math.nan
    current_step = max(
        [int(step_records[-1].get("step", -1)) if step_records else -1, *progress_steps],
        default=-1,
    )
    last = step_records[-1] if step_records else {}
    pid = find_pid()
    command = ""
    environment = {}
    if pid is not None:
        try:
            command = (Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()).strip()
            for entry in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
                if b"=" not in entry:
                    continue
                key, value = entry.split(b"=", 1)
                if key in {
                    b"CUDA_DEVICE_ORDER",
                    b"CUDA_VISIBLE_DEVICES",
                    b"WANDB_MODE",
                    b"CUBLAS_WORKSPACE_CONFIG",
                }:
                    environment[key.decode()] = value.decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pid = None

    physical_gpus = launch["physical_gpu_ids"]
    gpu_rows = run(
        "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"
    ).stdout.splitlines()
    uuids = {
        int(parts[0].strip()): parts[1].strip()
        for row in gpu_rows
        if len(parts := row.split(",", 1)) == 2
    }
    process_rows = run(
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid",
        "--format=csv,noheader,nounits",
    ).stdout.splitlines()
    active = {
        (int(parts[0].strip()), parts[1].strip())
        for row in process_rows
        if len(parts := row.split(",", 1)) == 2 and parts[0].strip().isdigit()
    }
    selected_active = pid is not None and all((pid, uuids[gpu]) in active for gpu in physical_gpus)
    common = Path(run("git", "rev-parse", "--path-format=absolute", "--git-common-dir").stdout.strip())
    locks_held = bool(physical_gpus) and all(
        run("flock", "-n", str(common / f"tactile3d_unit_gpu{gpu}.lock"), "true").returncode
        != 0
        for gpu in physical_gpus
    )
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
    finite_metrics = bool(last) and all(
        name in last and math.isfinite(float(last[name])) for name in required_metrics
    )
    loss_decomposition = bool(last) and abs(
        float(last.get("loss", math.nan))
        - (
            float(last.get("official_loss", math.nan))
            + expected_lambda * float(last.get("physical_loss", math.nan))
        )
    ) <= 1e-3
    gates = {
        "frozen_protocol": freeze.get("status") == "FROZEN_BEFORE_TRAINING",
        "job_status_running": status.get("state") == "RUNNING",
        "tmux_session_alive": run("tmux", "has-session", "-t", launch["session"]).returncode == 0,
        "training_process_alive": pid is not None,
        "correct_entrypoint_mode_and_run": "train_s4_3_pi2m_bhva.py" in command
        and "--exp-name s43_pi2m_bhva_seed42" in command,
        "correct_lambda_command": f"--lambda-phys {repr(expected_lambda)}" in command,
        "one_two_or_four_devices": len(physical_gpus) in {1, 2, 4},
        "global_batch32_divisible": 32 % len(physical_gpus) == 0,
        "selected_gpus_active": selected_active,
        "gpu_locks_held": locks_held,
        "visible_devices_match": environment.get("CUDA_VISIBLE_DEVICES")
        == ",".join(map(str, physical_gpus)),
        "pci_bus_order": environment.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
        "wandb_offline": environment.get("WANDB_MODE") == "offline",
        "deterministic_cublas": environment.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
        "correct_dataset": "DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets/pinch_tongs" in text,
        "correct_contact_sidecar": "pinch_tongs_official_tactile/sidecar.npz" in text,
        "correct_VA_target_sidecar": "pinch_tongs_va_t27/sidecar.npz" in text,
        "correct_pi05_base": "DexJoCo-Pi05/pi05_base/params" in text,
        "optimizer_step_logged": current_step >= 1 and bool(step_records),
        "finite_required_metrics": finite_metrics,
        "loss_decomposition": loss_decomposition,
        "contact_adapter_gradient_nonzero": float(last.get("contact_adapter_grad_norm", 0)) > 0,
        "physical_head_gradient_nonzero": float(last.get("physical_auxiliary_grad_norm", 0)) > 0,
        "pi05_trainable_gradient_nonzero": float(last.get("pi05_trainable_grad_norm", 0)) > 0,
        "minimum_50_steady_timing_samples": len(steady) >= args.minimum_timing_samples,
        "finite_positive_throughput": math.isfinite(median_seconds) and median_seconds > 0,
        "checkpoint_directory_writable": checkpoint_dir.is_dir()
        and os.access(checkpoint_dir, os.W_OK),
    }
    eta_seconds = (
        max(0, 30_000 - current_step) * median_seconds
        if math.isfinite(median_seconds)
        else math.nan
    )
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2m-training-launch-audit.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "stage": "B_HVA",
        "seed": 42,
        "global_batch_size": 32,
        "lambda_phys": expected_lambda,
        "physical_gpus": physical_gpus,
        "physical_gpu_uuids": launch["physical_gpu_uuids"],
        "logical_gpus": launch["logical_cuda_devices"],
        "pid": pid,
        "supervisor_pid": status.get("supervisor_pid"),
        "process_command": command.replace(str(ROOT), "$REPO_ROOT"),
        "process_environment": environment,
        "tmux_session": launch["session"],
        "current_progress_step": current_step,
        "steady_timing_samples": len(steady),
        "median_steady_state_seconds_per_step": median_seconds,
        "eta_hours": eta_seconds / 3600 if math.isfinite(eta_seconds) else None,
        "expected_completion_utc": (
            datetime.now(timezone.utc) + timedelta(seconds=eta_seconds)
        ).isoformat()
        if math.isfinite(eta_seconds)
        else None,
        "last_logged_metrics": last,
        "log": launch["log_path"],
        "checkpoint_directory": launch["experiment_directory"],
        "expected_final_checkpoint": launch["expected_final_checkpoint"],
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUTPUT)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "step": current_step,
                "seconds_per_step": median_seconds,
                "eta_hours": payload["eta_hours"],
                "failed": [name for name, value in gates.items() if not value],
            },
            sort_keys=True,
        )
    )
    if payload["status"] != "PASS":
        raise SystemExit("PI2M_BHVA_LAUNCH_AUDIT_FAIL")


if __name__ == "__main__":
    main()
