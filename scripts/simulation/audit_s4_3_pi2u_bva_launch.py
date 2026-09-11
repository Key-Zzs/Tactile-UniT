#!/usr/bin/env python3
"""Audit a stable persistent launch of the frozen BVA seed-42 run."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import statistics
import subprocess


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / ".local/artifacts/simulation/s4_3_pi2u/bva_launch.json"
STEP_PATTERN = re.compile(r"Step (?P<step>\d+): (?P<metrics>[^\r\n]+)")
METRIC_PATTERN = re.compile(r"(?P<key>[a-z0-9_]+)=(?P<value>[-+0-9.eE]+)")
PROGRESS_PATTERN = re.compile(
    r"Progress on: (?P<current>[0-9.]+)(?P<suffix>[kM]?)it/30\.0kit "
    r"rate:(?P<rate>[0-9.]+)(?P<unit>s/it|it/s)"
)
SELECTED_PATTERN = re.compile(
    r"BVA_SELECTED_PHYSICAL_GPUS=(?P<gpu>[0-3](?:,[0-3])*) FSDP_DEVICES=(?P<count>[12]) GLOBAL_BATCH=32"
)


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, check=False, capture_output=True, text=True)


def symbolic(path: Path) -> str:
    absolute = path if path.is_absolute() else ROOT / path
    return "$REPO_ROOT/" + absolute.absolute().relative_to(ROOT).as_posix()


def find_pid() -> int | None:
    candidates = []
    for child in Path("/proc").iterdir():
        if not child.name.isdigit():
            continue
        try:
            command = (child / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeDecodeError):
            continue
        if "train_s4_3_pi2u_bva.py" in command:
            candidates.append(int(child.name))
    return max(candidates) if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="s43_pi2u_bva_seed42")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--minimum-timing-samples", type=int, default=50)
    args = parser.parse_args()
    protocol = json.loads(
        (ROOT / ".local/artifacts/simulation/s4_3_pi2u/bva_training_protocol.json").read_text()
    )
    text = args.log.read_text(errors="replace")
    selection = SELECTED_PATTERN.search(text)
    physical_gpus = [int(value) for value in selection.group("gpu").split(",")] if selection else []
    device_count = int(selection.group("count")) if selection else 0
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
    current_step = max([int(step_records[-1].get("step", -1)) if step_records else -1, *progress_steps])
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
                if key in {b"CUDA_DEVICE_ORDER", b"CUDA_VISIBLE_DEVICES", b"WANDB_MODE"}:
                    environment[key.decode()] = value.decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pid = None
    gpu_rows = run("nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits").stdout.splitlines()
    uuids = {int(row.split(",", 1)[0].strip()): row.split(",", 1)[1].strip() for row in gpu_rows}
    process_rows = run(
        "nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"
    ).stdout.splitlines()
    active = {
        (int(parts[0].strip()), parts[1].strip())
        for row in process_rows
        if len(parts := row.split(",", 1)) == 2 and parts[0].strip().isdigit()
    }
    selected_active = pid is not None and all((pid, uuids[gpu]) in active for gpu in physical_gpus)
    common_git_dir = Path(run("git", "rev-parse", "--path-format=absolute", "--git-common-dir").stdout.strip())
    locks_held = bool(physical_gpus) and all(
        run("flock", "-n", str(common_git_dir / f"tactile3d_unit_gpu{gpu}.lock"), "true").returncode != 0
        for gpu in physical_gpus
    )
    last = step_records[-1] if step_records else {}
    required_metrics = (
        "loss",
        "official_loss",
        "physical_loss",
        "grad_norm",
        "pi05_trainable_grad_norm",
        "physical_auxiliary_grad_norm",
    )
    session_alive = run("tmux", "has-session", "-t", args.session).returncode == 0
    gates = {
        "frozen_BVA_protocol": protocol.get("status") == "FROZEN_BEFORE_TRAINING",
        "training_process_alive": pid is not None,
        "tmux_session_alive": session_alive,
        "selection_recorded": selection is not None,
        "one_or_two_devices": device_count in {1, 2} and len(physical_gpus) == device_count,
        "global_batch32_divisible": device_count > 0 and 32 % device_count == 0,
        "only_scanned_physical_gpu_range": all(gpu in range(4) for gpu in physical_gpus),
        "selected_gpus_active": selected_active,
        "gpu_locks_held": locks_held,
        "pci_bus_order": environment.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
        "visible_devices_match": environment.get("CUDA_VISIBLE_DEVICES") == ",".join(map(str, physical_gpus)),
        "wandb_offline": environment.get("WANDB_MODE") == "offline",
        "correct_seed42_entrypoint": "train_s4_3_pi2u_bva.py" in command and "--exp-name s43_pi2u_bva_seed42" in command,
        "correct_lambda": f"--lambda-phys {protocol.get('lambda_phys')}" in command,
        "optimizer_step_logged": current_step >= 1 and bool(step_records),
        "finite_required_metrics": all(key in last and math.isfinite(float(last[key])) for key in required_metrics),
        "pi05_trainable_gradient_nonzero": float(last.get("pi05_trainable_grad_norm", 0)) > 0,
        "physical_head_gradient_nonzero": float(last.get("physical_auxiliary_grad_norm", 0)) > 0,
        "minimum_50_steady_timing_samples": len(steady) >= args.minimum_timing_samples,
        "finite_positive_throughput": math.isfinite(median_seconds) and median_seconds > 0,
        "checkpoint_directory_writable": args.checkpoint_dir.parent.is_dir(),
    }
    eta_hours = (30000 - current_step) * median_seconds / 3600 if gates["finite_positive_throughput"] else math.nan
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2u-bva-launch.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "stage": "BVA",
        "seed": 42,
        "global_batch_size": 32,
        "physical_gpus": physical_gpus,
        "logical_gpus": list(range(device_count)),
        "pid": pid,
        "process_command": command,
        "process_environment": environment,
        "tmux_session": args.session,
        "current_progress_step": current_step,
        "steady_timing_samples": len(steady),
        "median_steady_state_seconds_per_step": median_seconds,
        "eta_hours": eta_hours,
        "expected_completion_utc": (
            datetime.now(timezone.utc) + timedelta(hours=eta_hours)
        ).isoformat() if math.isfinite(eta_hours) else None,
        "last_logged_metrics": last,
        "log": symbolic(args.log),
        "checkpoint_directory": symbolic(args.checkpoint_dir),
        "expected_final_checkpoint": symbolic(args.checkpoint_dir / "29999"),
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    temporary = ARTIFACT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(ARTIFACT)
    print(json.dumps({"status": payload["status"], "step": current_step, "eta_hours": eta_hours}, sort_keys=True))
    if payload["status"] != "PASS":
        failed = [name for name, value in gates.items() if not value]
        raise SystemExit("BVA launch audit failed: " + ",".join(failed))


if __name__ == "__main__":
    main()
