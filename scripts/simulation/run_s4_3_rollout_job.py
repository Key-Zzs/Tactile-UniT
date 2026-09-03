#!/usr/bin/env python3
"""Launch the unit policy server and DexJoCo client for one 30-reset job."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_3_training import atomic_json, sha256_file  # noqa: E402
from gr00t.simulation.s4_3_transport import (  # noqa: E402
    EndpointContractError,
    build_runtime_endpoint,
    cleanup_server_endpoint,
    write_endpoint_manifest,
)

TMP_ROOT = ROOT / ".local/tmp/simulation/s4_3_rr/endpoint_manifests"
LOG_ROOT = ROOT / ".local/logs/simulation/s4_3_rr/rollout_jobs"
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_rr/rollout_jobs"
EVAL_CONFIG = ROOT / "configs/simulation/s4_3_policy_eval_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument(
        "--run-kind", choices=("scientific", "rr_smoke"), default="scientific"
    )
    parser.add_argument("--evaluation-config", type=Path, default=EVAL_CONFIG)
    return parser.parse_args()


def run_roots(run_kind: str) -> tuple[Path, Path, Path]:
    if run_kind == "scientific":
        return (
            LOG_ROOT,
            ARTIFACT_ROOT,
            ROOT / ".local/logs/simulation/s4_3_rr/closed_loop",
        )
    return (
        ROOT / ".local/logs/simulation/s4_3_rr/production_smoke/jobs",
        ROOT / ".local/artifacts/simulation/s4_3_rr/production_smoke/jobs",
        ROOT / ".local/logs/simulation/s4_3_rr/production_smoke/closed_loop",
    )


def main() -> None:
    args = parse_args()
    dex_python = os.environ.get("DEXJOCO_PYTHON")
    unit_checkpoint = os.environ.get("UNIT_FULLDATA_CKPT")
    if not dex_python or not unit_checkpoint:
        raise RuntimeError("DEXJOCO_PYTHON and UNIT_FULLDATA_CKPT are required")
    if sha256_file(args.checkpoint) != args.checkpoint_sha256:
        raise RuntimeError("rollout job checkpoint identity mismatch")
    job_id = f"{args.task}_{args.variant}_seed{args.seed}"
    attempt = int(os.environ.get("S4_3_ROLLOUT_ATTEMPT", "1"))
    if attempt < 1:
        raise RuntimeError("S4_3_ROLLOUT_ATTEMPT must be positive")
    log_root, artifact_root, rollout_root = run_roots(args.run_kind)
    server_log_path = log_root / f"{job_id}_attempt{attempt}_server.log"
    client_log_path = log_root / f"{job_id}_attempt{attempt}_client.log"
    endpoint_manifest = TMP_ROOT / f"{args.run_kind}_{job_id}_attempt{attempt}_{os.getpid()}.json"
    endpoint = build_runtime_endpoint(
        experiment_identity="S4.3-RR-ACT-rollout-v2",
        task=args.task,
        variant=args.variant,
        training_seed=args.seed,
        worker_identity=f"{args.run_kind}:{job_id}",
    )
    write_endpoint_manifest(
        endpoint_manifest,
        endpoint,
        {
            "run_kind": args.run_kind,
            "task": args.task,
            "variant": args.variant,
            "training_seed": args.seed,
            "checkpoint_sha256": args.checkpoint_sha256,
            "evaluation_config_sha256": sha256_file(args.evaluation_config),
        },
    )
    for path in (TMP_ROOT, log_root, artifact_root):
        path.mkdir(parents=True, exist_ok=True)
    server_log = server_log_path.open("w", encoding="utf-8")
    server_command = [
        sys.executable,
        str(ROOT / "scripts/simulation/serve_s4_3_act_policy.py"),
        "--endpoint-manifest",
        str(endpoint_manifest),
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
        while not endpoint.socket_path.exists() or not endpoint.lease_path.exists():
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
            "--endpoint-manifest",
            str(endpoint_manifest),
            "--task",
            args.task,
            "--variant",
            args.variant,
            "--seed",
            str(args.seed),
            "--checkpoint-sha256",
            args.checkpoint_sha256,
            "--run-kind",
            args.run_kind,
            "--evaluation-config",
            str(args.evaluation_config),
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
        try:
            cleanup_server_endpoint(endpoint, allow_unregistered_own_socket=True)
        except EndpointContractError:
            pass

    metadata_paths = sorted(
        (rollout_root / args.task / args.variant / f"seed_{args.seed}").glob("*/metadata.json")
    )
    expected_resets = sum(
        row["task"] == args.task
        for row in json.loads(args.evaluation_config.read_text(encoding="utf-8"))["resets"]
    )
    if args.run_kind == "scientific" and expected_resets != 30:
        raise RuntimeError("scientific rollout job did not receive 30 frozen resets")
    if len(metadata_paths) != expected_resets:
        raise RuntimeError("rollout job reset-result cardinality mismatch")
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
        "stage": "RR4" if args.run_kind == "rr_smoke" else "RR6",
        "run_kind": args.run_kind,
        "scientific_result": args.run_kind == "scientific",
        "attempt": attempt,
        "task": args.task,
        "variant": args.variant,
        "training_seed": args.seed,
        "physical_gpu": int(os.environ["S4_3_PHYSICAL_GPU"]),
        "logical_device": "cuda:0",
        "checkpoint": str(args.checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": args.checkpoint_sha256,
        "rollouts": expected_resets,
        "successes": sum(row["success"] for row in rollouts),
        "runtime_exception_rollouts": sum(bool(row["runtime_exceptions"]) for row in rollouts),
        "server_log": str(server_log_path.relative_to(ROOT)),
        "server_log_sha256": sha256_file(server_log_path),
        "client_log": str(client_log_path.relative_to(ROOT)),
        "client_log_sha256": sha256_file(client_log_path),
        "endpoint": {
            "algorithm": "$RUNTIME_TMP/tu3d_<short_hash>_<pid>_<nonce>.sock",
            "basename": endpoint.socket_path.name,
            "job_hash": endpoint.job_hash,
            "encoded_length": endpoint.encoded_length,
            "ceiling_bytes": endpoint.ceiling_bytes,
            "cleanup_pass": not endpoint.socket_path.exists() and not endpoint.lease_path.exists(),
        },
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
    atomic_json(artifact_root / f"{job_id}.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        traceback.print_exc()
        print(
            json.dumps(
                {
                    "classification": "INFRASTRUCTURE_FAILURE",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "status": "FAIL",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        raise SystemExit(75) from error
