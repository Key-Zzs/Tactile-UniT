#!/usr/bin/env python3
"""Audit the stable persistent launch of the frozen PI1B run."""

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
ARTIFACT = ROOT / ".local/artifacts/simulation/s4_3_pi1/pi1b_launch.json"
EXPECTED_MODE = "CONTACT_STATE_TOKENS"
EXPECTED_EXP = "s43_pi1b_contact_tokens_seed42"
STEP_PATTERN = re.compile(r"Step (?P<step>\d+): (?P<metrics>[^\r\n]+)")
METRIC_PATTERN = re.compile(r"(?P<key>[a-z0-9_]+)=(?P<value>[-+0-9.eE]+)")
PROGRESS_PATTERN = re.compile(
    r"Progress on: (?P<current>[0-9.]+)(?P<suffix>[kM]?)it/30\.0kit "
    r"rate:(?P<rate>[0-9.]+)(?P<unit>s/it|it/s)"
)


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, check=False, capture_output=True, text=True)


def symbolic(path: Path) -> str:
    return "$REPO_ROOT/" + path.resolve().relative_to(ROOT).as_posix()


def find_pid() -> int | None:
    candidates = []
    for child in Path("/proc").iterdir():
        if not child.name.isdigit():
            continue
        try:
            command = (child / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeDecodeError):
            continue
        if "train_s4_3_pi1.py" in command and f"--mode {EXPECTED_MODE}" in command:
            candidates.append(int(child.name))
    return max(candidates) if candidates else None


def gpu_snapshot(pid: int | None, physical_gpus: list[int]) -> tuple[dict[int, str], bool]:
    rows = run("nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits").stdout.splitlines()
    uuids = {
        int(parts[0].strip()): parts[1].strip()
        for row in rows
        if len(parts := row.split(",", 1)) == 2
    }
    process_rows = run(
        "nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"
    ).stdout.splitlines()
    processes = {
        (int(parts[0].strip()), parts[1].strip())
        for row in process_rows
        if len(parts := row.split(",", 1)) == 2 and parts[0].strip().isdigit()
    }
    return uuids, pid is not None and all((pid, uuids[gpu]) in processes for gpu in physical_gpus)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--session", default="s43_pi1b_pinch")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--minimum-timing-samples", type=int, default=50)
    args = parser.parse_args()
    physical_gpus = [int(value) for value in args.gpu.split(",")]
    if not physical_gpus or len(set(physical_gpus)) != len(physical_gpus):
        raise SystemExit("S4_3_PI1_INVALID_GPU_SELECTION")
    frozen_protocol = json.loads(
        (ROOT / "configs/simulation/s4_3_pi1_training_protocol.json").read_text()
    )

    text = args.log.read_text(errors="replace")
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
    # Drop early compile-contaminated progress samples if more than the requested steady window exists.
    steady = timing[-args.minimum_timing_samples :]
    median_seconds = statistics.median(steady) if steady else math.nan
    last = step_records[-1] if step_records else {}
    current_step = max([int(last.get("step", -1)), *progress_steps], default=-1)

    pid = find_pid()
    command = ""
    environment = {}
    if pid is not None:
        try:
            command = (Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()).strip()
            entries = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            environment = {
                key.decode(): value.decode()
                for entry in entries
                if b"=" in entry
                for key, value in [entry.split(b"=", 1)]
                if key in {b"CUDA_DEVICE_ORDER", b"CUDA_VISIBLE_DEVICES", b"WANDB_MODE"}
            }
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pid = None
    _, selected_gpus_active = gpu_snapshot(pid, physical_gpus)
    common_git_dir = Path(run("git", "rev-parse", "--path-format=absolute", "--git-common-dir").stdout.strip())
    locks_held = all(
        run("flock", "-n", str(common_git_dir / f"tactile3d_unit_gpu{gpu}.lock"), "true").returncode != 0
        for gpu in physical_gpus
    )
    required_metrics = (
        "loss",
        "official_loss",
        "grad_norm",
        "param_norm",
        "contact_adapter_grad_norm",
        "pi05_trainable_grad_norm",
    )
    finite_metrics = bool(last) and all(
        name in last and math.isfinite(float(last[name])) for name in required_metrics
    )
    eta_seconds = max(0, 30_000 - current_step) * median_seconds
    expected_completion = datetime.now(timezone.utc) + timedelta(seconds=eta_seconds)
    gates = {
        "tmux_session_alive": run("tmux", "has-session", "-t", args.session).returncode == 0,
        "training_process_alive": pid is not None,
        "correct_mode": (
            f"--mode {EXPECTED_MODE}" in command
            and frozen_protocol["B1"]["mode"] == EXPECTED_MODE
        ),
        "correct_experiment": f"--exp-name {EXPECTED_EXP}" in command,
        "correct_dataset": "DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets/pinch_tongs" in text,
        "correct_sidecar": "pinch_tongs_official_tactile/sidecar.npz" in text,
        "correct_pi05_base": "DexJoCo-Pi05/pi05_base/params" in text,
        "seed42_frozen_entrypoint": "seed=42" in (ROOT / "scripts/simulation/train_s4_3_pi1.py").read_text(),
        "global_batch32": "local_batch_size: 32" in text,
        "two_visible_devices": environment.get("CUDA_VISIBLE_DEVICES") == args.gpu,
        "selected_gpus_active": selected_gpus_active,
        "gpu_locks_held": locks_held,
        "pci_bus_order": environment.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
        "wandb_offline": environment.get("WANDB_MODE") == "offline",
        "optimizer_step_logged": bool(step_records),
        "finite_required_metrics": finite_metrics,
        "adapter_gradient_nonzero": float(last.get("contact_adapter_grad_norm", 0.0)) > 0,
        "pi05_trainable_gradient_nonzero": float(last.get("pi05_trainable_grad_norm", 0.0)) > 0,
        "minimum_50_steady_timing_samples": len(steady) >= args.minimum_timing_samples,
        "finite_positive_throughput": math.isfinite(median_seconds) and median_seconds > 0,
        "checkpoint_directory_writable": args.checkpoint_dir.is_dir() and os.access(args.checkpoint_dir, os.W_OK),
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi1b-launch.v1",
        "stage": "PI1B",
        "mode": EXPECTED_MODE,
        "seed": 42,
        "global_batch_size": 32,
        "physical_gpus": physical_gpus,
        "logical_gpus": list(range(len(physical_gpus))),
        "pid": pid,
        "tmux_session": args.session,
        "log": symbolic(args.log),
        "checkpoint_directory": symbolic(args.checkpoint_dir),
        "expected_final_checkpoint": symbolic(args.checkpoint_dir / "29999"),
        "last_logged_metrics": last,
        "current_progress_step": current_step,
        "steady_timing_samples": len(steady),
        "median_steady_state_seconds_per_step": median_seconds,
        "eta_hours": eta_seconds / 3600,
        "expected_completion_utc": expected_completion.isoformat(),
        "process_command": command.replace(str(ROOT), "$REPO_ROOT"),
        "process_environment": environment,
        "monitor_commands": {
            "tmux_attach": f"tmux attach -t {args.session}",
            "tmux_capture": f"tmux capture-pane -pt {args.session} -S -80",
            "log": f"tail -f {symbolic(args.log)}",
            "gpu": "watch -n 2 nvidia-smi",
            "training_step": f"grep -aE 'Step [0-9]+:' {symbolic(args.log)} | tail -20",
            "checkpoints": f"find {symbolic(args.checkpoint_dir)} -maxdepth 1 -mindepth 1 -type d -printf '%f\\n' | sort -n",
        },
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "status": "PASS" if all(gates.values()) else "FAIL",
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    temporary = ARTIFACT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(ARTIFACT)
    print(json.dumps({key: payload[key] for key in ("status", "current_progress_step", "steady_timing_samples", "median_steady_state_seconds_per_step", "eta_hours")}, sort_keys=True))
    if payload["status"] != "PASS":
        failed = [name for name, value in payload["gates"].items() if value == "FAIL"]
        raise SystemExit("S4_3_PI1B_STARTUP_GATE_FAIL: " + ",".join(failed))


if __name__ == "__main__":
    main()
