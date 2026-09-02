#!/usr/bin/env python3
"""Launch the unit policy server and DexJoCo client for one 30-reset job."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_3_training import atomic_json, sha256_file  # noqa: E402

TMP_ROOT = ROOT / ".local/tmp/simulation/s4_3_restart/rollout_sockets"
LOG_ROOT = ROOT / ".local/logs/simulation/s4_3_restart/rollout_jobs"
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_restart/rollout_jobs"
ROLLOUT_ROOT = ROOT / ".local/logs/simulation/s4_3_restart/closed_loop"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dex_python = os.environ.get("DEXJOCO_PYTHON")
    unit_checkpoint = os.environ.get("UNIT_FULLDATA_CKPT")
    if not dex_python or not unit_checkpoint:
        raise RuntimeError("DEXJOCO_PYTHON and UNIT_FULLDATA_CKPT are required")
    if sha256_file(args.checkpoint) != args.checkpoint_sha256:
        raise RuntimeError("rollout job checkpoint identity mismatch")
    job_id = f"{args.task}_{args.variant}_seed{args.seed}"
    socket_path = TMP_ROOT / f"{job_id}.sock"
    server_log_path = LOG_ROOT / f"{job_id}_server.log"
    client_log_path = LOG_ROOT / f"{job_id}_client.log"
    for path in (TMP_ROOT, LOG_ROOT, ARTIFACT_ROOT):
        path.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    server_log = server_log_path.open("w", encoding="utf-8")
    server_command = [
        sys.executable,
        str(ROOT / "scripts/simulation/serve_s4_3_act_policy.py"),
        "--socket",
        str(socket_path),
        "--checkpoint",
        str(args.checkpoint),
        "--checkpoint-sha256",
        args.checkpoint_sha256,
        "--task",
        args.task,
        "--variant",
        args.variant,
        "--seed",
        str(args.seed),
        "--unit-checkpoint",
        unit_checkpoint,
        "--device",
        "cuda:0",
    ]
    server_environment = os.environ.copy()
    server_environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    server = subprocess.Popen(
        server_command,
        cwd=ROOT,
        env=server_environment,
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + 180
        while not socket_path.exists():
            if server.poll() is not None:
                raise RuntimeError("policy server exited before socket readiness")
            if time.monotonic() >= deadline:
                raise TimeoutError("policy server readiness timeout")
            time.sleep(1)
        client_environment = os.environ.copy()
        client_environment.pop("DISPLAY", None)
        client_environment.update({"MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "0"})
        client_command = [
            dex_python,
            str(ROOT / "scripts/simulation/run_s4_3_policy_rollouts_dex.py"),
            "--socket",
            str(socket_path),
            "--task",
            args.task,
            "--variant",
            args.variant,
            "--seed",
            str(args.seed),
            "--checkpoint-sha256",
            args.checkpoint_sha256,
        ]
        with client_log_path.open("w", encoding="utf-8") as client_log:
            client_result = subprocess.run(
                client_command,
                cwd=ROOT,
                env=client_environment,
                stdout=client_log,
                stderr=subprocess.STDOUT,
                timeout=8 * 60 * 60,
            )
        if client_result.returncode != 0:
            raise RuntimeError(f"DexJoCo rollout client exit {client_result.returncode}")
        server.wait(timeout=120)
        if server.returncode != 0:
            raise RuntimeError(f"policy server exit {server.returncode}")
    except Exception:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        raise
    finally:
        server_log.close()
        socket_path.unlink(missing_ok=True)

    metadata_paths = sorted(
        (ROLLOUT_ROOT / args.task / args.variant / f"seed_{args.seed}").glob("*/metadata.json")
    )
    if len(metadata_paths) != 30:
        raise RuntimeError("rollout job does not contain exactly 30 reset results")
    rollouts = [json.loads(path.read_text(encoding="utf-8")) for path in metadata_paths]
    if any(
        row["task"] != args.task
        or row["variant"] != args.variant
        or row["training_seed"] != args.seed
        or row["checkpoint_sha256"] != args.checkpoint_sha256
        or row["logging_status"] != "PASS"
        for row in rollouts
    ):
        raise RuntimeError("rollout job metadata identity mismatch")
    summary = {
        "schema": "tactile3d-unit.s4-3-rollout-job.v1",
        "stage": "R12",
        "task": args.task,
        "variant": args.variant,
        "training_seed": args.seed,
        "physical_gpu": int(os.environ["S4_3_PHYSICAL_GPU"]),
        "logical_device": "cuda:0",
        "checkpoint": str(args.checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": args.checkpoint_sha256,
        "rollouts": 30,
        "successes": sum(row["success"] for row in rollouts),
        "runtime_exception_rollouts": sum(bool(row["runtime_exceptions"]) for row in rollouts),
        "server_log": str(server_log_path.relative_to(ROOT)),
        "server_log_sha256": sha256_file(server_log_path),
        "client_log": str(client_log_path.relative_to(ROOT)),
        "client_log_sha256": sha256_file(client_log_path),
        "metadata": [
            {
                "rollout_id": row["rollout_id"],
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            for row, path in zip(rollouts, metadata_paths)
        ],
        "status": "PASS",
    }
    atomic_json(ARTIFACT_ROOT / f"{job_id}.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
