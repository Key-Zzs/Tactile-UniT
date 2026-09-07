#!/usr/bin/env python3
"""Validate the mandatory first finite step of the official PI0 training run."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / ".local/artifacts/simulation/s4_3_pi0/training_launch.json"
STEP_PATTERN = re.compile(r"Step (?P<step>\d+): (?P<metrics>[^\r\n]+)")
METRIC_PATTERN = re.compile(r"(?P<key>grad_norm|loss|param_norm)=(?P<value>[-+0-9.eE]+)")


def command(*args: str, allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, cwd=ROOT, check=False, capture_output=True, text=True)
    if result.returncode and not allow_failure:
        raise RuntimeError(result.stderr.strip() or "command failed: " + " ".join(args))
    return result


def symbolic(path: Path) -> str:
    return "$REPO_ROOT/" + str(path.resolve().relative_to(ROOT))


def write_json(payload: Any) -> None:
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    temporary = ARTIFACT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(ARTIFACT)


def find_training_pid() -> int | None:
    proc = Path("/proc")
    matches = []
    for child in proc.iterdir():
        if not child.name.isdigit():
            continue
        try:
            cmdline = (child / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "scripts/train.py pinch_tongs" in cmdline:
            matches.append(int(child.name))
    return max(matches) if matches else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        required=True,
        help="Comma-separated physical GPU indices in CUDA_VISIBLE_DEVICES order.",
    )
    parser.add_argument("--session", default="s43_pi0_pinch")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    physical_gpus = [int(value) for value in args.gpu.split(",")]
    if not physical_gpus or len(set(physical_gpus)) != len(physical_gpus):
        raise SystemExit("S4_3_PI0_INVALID_GPU_SELECTION")
    gpu_mask = ",".join(str(value) for value in physical_gpus)

    log_text = args.log.read_text(encoding="utf-8", errors="replace")
    matches = list(STEP_PATTERN.finditer(log_text))
    first = matches[0] if matches else None
    parsed_metrics = (
        {item.group("key"): float(item.group("value")) for item in METRIC_PATTERN.finditer(first.group("metrics"))}
        if first
        else {}
    )
    metrics = {"step": int(first.group("step")), **parsed_metrics} if len(parsed_metrics) == 3 and first else None
    finite = metrics is not None and all(
        math.isfinite(metrics[key]) for key in ("grad_norm", "loss", "param_norm")
    )

    pid = find_training_pid()
    alive = pid is not None and Path(f"/proc/{pid}").exists()
    if alive and pid is not None:
        cmdline = (Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()).strip()
        environ = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        environment = {
            item.split(b"=", 1)[0].decode(): item.split(b"=", 1)[1].decode()
            for item in environ
            if b"=" in item
            and item.split(b"=", 1)[0] in {b"CUDA_DEVICE_ORDER", b"CUDA_VISIBLE_DEVICES", b"WANDB_MODE"}
        }
    else:
        cmdline = ""
        environment = {}

    tmux_alive = command("tmux", "has-session", "-t", args.session, allow_failure=True).returncode == 0
    gpu_rows = command(
        "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits", allow_failure=True
    ).stdout.splitlines()
    gpu_uuids = {
        int(index.strip()): uuid.strip()
        for row in gpu_rows
        if len(parts := row.split(",", 1)) == 2
        for index, uuid in [parts]
    }
    process_rows = command(
        "nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits", allow_failure=True
    ).stdout.splitlines()
    gpu_processes = {
        (int(process_pid.strip()), uuid.strip())
        for row in process_rows
        if len(parts := row.split(",", 1)) == 2 and parts[0].strip().isdigit()
        for process_pid, uuid in [parts]
    }
    selected_gpu_in_use = pid is not None and all(
        (pid, gpu_uuids.get(gpu, "")) in gpu_processes for gpu in physical_gpus
    )
    gates = {
        "tmux_session_alive": tmux_alive,
        "training_process_alive": alive,
        "official_train_script": "scripts/train.py pinch_tongs" in cmdline,
        "official_dataset": "DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets/pinch_tongs" in cmdline,
        "official_base_model": "DexJoCo-Pi05/pi05_base/params" in cmdline,
        "official_config": "pinch_tongs" in cmdline,
        "selected_gpu_mask": environment.get("CUDA_VISIBLE_DEVICES") == gpu_mask,
        "selected_gpu_in_use": selected_gpu_in_use,
        "pci_bus_order": environment.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
        "wandb_offline": environment.get("WANDB_MODE") == "offline",
        "first_step_logged": metrics is not None,
        "finite_first_step": finite,
        "checkpoint_directory_writable": args.checkpoint_dir.is_dir() and os.access(args.checkpoint_dir, os.W_OK),
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi0-training-launch.v1",
        "stage": "PI0-7",
        "task": "pinch_tongs",
        "config": "official pinch_tongs",
        "physical_gpus": physical_gpus,
        "logical_gpus": list(range(len(physical_gpus))),
        "pid": pid,
        "tmux_session": args.session,
        "log": symbolic(args.log),
        "checkpoint_directory": symbolic(args.checkpoint_dir),
        "first_logged_metrics": metrics,
        "process_command": cmdline.replace(str(ROOT), "$REPO_ROOT"),
        "process_environment": environment,
        "expected_steps": 30000,
        "expected_final_checkpoint": symbolic(args.checkpoint_dir / "29999"),
        "monitor_commands": {
            "tmux_attach": f"tmux attach -t {args.session}",
            "tmux_capture": f"tmux capture-pane -pt {args.session} -S -50",
            "log": f"tail -f {symbolic(args.log)}",
            "gpu": "watch -n 2 nvidia-smi",
            "training_step": f"grep -aE 'Step [0-9]+:' {symbolic(args.log)} | tail -20",
            "checkpoints": f"find {symbolic(args.checkpoint_dir)} -maxdepth 1 -mindepth 1 -type d -printf '%f\\n' | sort -n",
        },
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "status": "PASS" if all(gates.values()) else "FAIL",
    }
    write_json(payload)
    print(json.dumps({"training_launch": payload["status"], "pid": pid, "metrics": metrics}, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit("S4_3_PI0_TRAINING_STARTUP_GATE_FAIL")


if __name__ == "__main__":
    main()
