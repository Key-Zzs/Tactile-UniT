#!/usr/bin/env python3
"""Run the frozen seed-7 matched-input PI2M evaluation."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEXJOCO = ROOT / "third_party/dexjoco"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2m"
LOGS = ROOT / ".local/logs/simulation/s4_3_pi2m/evaluation_seed7"
CACHE = ROOT / ".local/cache/simulation/s4_3_pi2m/evaluation/seed7"
TMP = ROOT / ".local/tmp/s43m7"
PRE_FREEZE = ARTIFACTS / "pre_eval_freeze.json"
CONDA_ROOT = Path(sys.executable).resolve().parents[3]
OPENPI_PYTHON = CONDA_ROOT / "envs/openpi/bin/python"
UNIT_PYTHON = CONDA_ROOT / "envs/unit/bin/python"
EVAL_PYTHON = ROOT / ".local/external/s4_3_pi0/eval-venv/bin/python"
MODELS = ("B1", "B_HVA", "B2")
RUNTIME_MODES = {model: "CONTACT_STATE_TOKENS" for model in MODELS}
EPISODES = 200
EVALUATOR_SEED = 7


def configure_frozen_runtime():
    from scripts.simulation import run_s4_3_pi2u_eval as frozen

    frozen.ARTIFACTS = ARTIFACTS
    frozen.LOGS = LOGS
    frozen.CACHE = CACHE
    frozen.TMP = TMP
    frozen.PRE_FREEZE = PRE_FREEZE
    frozen.MODELS = MODELS
    frozen.RUNTIME_MODES = RUNTIME_MODES
    frozen.EPISODES = EPISODES
    frozen.EVALUATOR_SEED = EVALUATOR_SEED
    return frozen


def run_one(model: str, gpu: int, port: int) -> dict[str, Any]:
    frozen = configure_frozen_runtime()
    lower = model.lower()
    socket_path = TMP / f"{lower}_contact_state.sock"
    output = CACHE / lower
    diagnostics = TMP / f"{lower}_inference.jsonl"
    contact_artifact = TMP / f"{lower}_contact_state_service.json"
    contact_log = LOGS / f"{lower}_contact_state_service.log"
    server_log = LOGS / f"{lower}_server.log"
    client_log = LOGS / f"{lower}_client.log"
    protected = (
        socket_path,
        output,
        diagnostics,
        contact_artifact,
        ARTIFACTS / f"{lower}_raw_rollouts.json",
    )
    if any(path.exists() for path in protected):
        raise RuntimeError(f"refusing to overwrite PI2M output for {model}")
    started = time.time()
    contact = policy = None
    handles = []
    try:
        contact_handle = contact_log.open("w")
        server_handle = server_log.open("w")
        client_handle = client_log.open("w")
        handles.extend((contact_handle, server_handle, client_handle))
        base_env = os.environ | {"PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"}
        contact = subprocess.Popen(
            [
                str(UNIT_PYTHON),
                str(ROOT / "scripts/simulation/run_s4_3_pi2u_eval.py"),
                "contact-service",
                "--artifact",
                str(contact_artifact),
                "--socket",
                str(socket_path),
            ],
            cwd=ROOT,
            env=base_env,
            stdout=contact_handle,
            stderr=subprocess.STDOUT,
        )
        frozen.wait_for_log(contact, contact_log, "CONTACT_STATE_SERVICE_READY", 120)
        gpu_env = base_env | {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
        }
        policy = subprocess.Popen(
            [
                str(OPENPI_PYTHON),
                str(ROOT / "scripts/simulation/serve_s4_3_pi2m_policy.py"),
                "--model",
                model,
                "--port",
                str(port),
            ],
            cwd=ROOT,
            env=gpu_env,
            stdout=server_handle,
            stderr=subprocess.STDOUT,
        )
        frozen.wait_for_log(policy, server_log, f"PI2M_POLICY_SERVER_READY model={model}", 300)
        eval_env = gpu_env | {
            "MUJOCO_GL": "egl",
            "PYTHONPATH": f"{ROOT}:{DEXJOCO / 'dexjoco'}",
        }
        result = subprocess.run(
            [
                str(EVAL_PYTHON),
                str(Path(__file__).resolve()),
                "evaluate-model",
                "--model",
                model,
                "--contact-socket",
                str(socket_path),
                "--output",
                str(output),
                "--diagnostics",
                str(diagnostics),
                "--port",
                str(port),
            ],
            cwd=ROOT,
            env=eval_env,
            stdout=client_handle,
            stderr=subprocess.STDOUT,
        )
        if result.returncode:
            raise RuntimeError(f"{model} evaluator exited {result.returncode}")
        raw = frozen.read_json(ARTIFACTS / f"{lower}_raw_rollouts.json")
        return {
            "model": model,
            "physical_gpu": gpu,
            "port": port,
            "status": raw["status"],
            "episodes": raw["episodes"],
            "successes_frozen_until_completion": "RECORDED_IN_RAW_ARTIFACT",
            "started_unix": started,
            "elapsed_seconds": time.time() - started,
            "runtime_mode": RUNTIME_MODES[model],
            "infrastructure_retries": 0,
        }
    finally:
        frozen.stop_process(policy)
        frozen.stop_process(contact)
        for handle in handles:
            handle.close()


def orchestrate(gpus: list[int]) -> None:
    frozen = configure_frozen_runtime()
    if frozen.read_json(PRE_FREEZE).get("status") != "PASS":
        raise SystemExit("PI2M pre-evaluation freeze is not PASS")
    if not 1 <= len(gpus) <= 3 or len(gpus) != len(set(gpus)):
        raise SystemExit("PI2M evaluation requires one to three distinct GPUs")
    if gpus != sorted(gpus) or any(gpu not in range(4) for gpu in gpus):
        raise SystemExit("PI2M GPUs must be sorted physical IDs in range 0-3")
    if any((ARTIFACTS / f"{model.lower()}_raw_rollouts.json").exists() for model in MODELS):
        raise SystemExit("refusing to overwrite or resume canonical PI2M outcomes")
    snapshot1 = frozen.gpu_snapshot()
    time.sleep(2)
    snapshot2 = frozen.gpu_snapshot()
    if not all(
        frozen.gpu_is_idle(gpu, snapshot1) and frozen.gpu_is_idle(gpu, snapshot2)
        for gpu in gpus
    ):
        raise SystemExit("one or more selected PI2M GPUs is not genuinely idle in both snapshots")
    locks = []
    try:
        locks = [frozen.acquire_gpu_lock(gpu) for gpu in gpus]
        snapshot3 = frozen.gpu_snapshot()
        if not all(frozen.gpu_is_idle(gpu, snapshot3) for gpu in gpus):
            raise SystemExit("one or more selected PI2M GPUs became busy after lock acquisition")
        LOGS.mkdir(parents=True, exist_ok=False)
        CACHE.mkdir(parents=True, exist_ok=False)
        TMP.mkdir(parents=True, exist_ok=False)
        first_wave = list(zip(MODELS[: len(gpus)], gpus, range(8370, 8370 + len(gpus))))
        rows: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(first_wave)) as executor:
            rows.extend(
                future.result()
                for future in [executor.submit(run_one, *assignment) for assignment in first_wave]
            )
        for offset, model in enumerate(MODELS[len(gpus) :]):
            rows.append(run_one(model, gpus[offset % len(gpus)], 8380 + offset))
        payload = {
            "schema": "tactile3d-unit.s4-3-pi2m-gpu-execution.v1",
            "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
            "snapshots": [snapshot1, snapshot2, snapshot3, frozen.gpu_snapshot()],
            "workers": rows,
            "maximum_heavy_workers": len(gpus),
            "unrelated_gpu_processes_killed_or_preempted": False,
            "wave_plan": [[row[0] for row in first_wave], list(MODELS[len(gpus) :])],
        }
        frozen.atomic_json(ARTIFACTS / "gpu_execution.json", payload)
        print(json.dumps({"status": payload["status"], "workers": rows}, sort_keys=True))
        if payload["status"] != "PASS":
            raise SystemExit("PI2M_EVALUATION_INCOMPLETE")
    finally:
        for handle in locks:
            handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("contact-service").add_argument("--unused", action="store_true")
    evaluate = sub.add_parser("evaluate-model")
    evaluate.add_argument("--model", choices=MODELS, required=True)
    evaluate.add_argument("--contact-socket", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--diagnostics", type=Path, required=True)
    evaluate.add_argument("--port", type=int, required=True)
    launch = sub.add_parser("orchestrate")
    launch.add_argument("--gpus", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frozen = configure_frozen_runtime()
    if args.command == "evaluate-model":
        frozen.evaluate_model(
            args.model, args.contact_socket, args.output, args.diagnostics, args.port
        )
    elif args.command == "orchestrate":
        try:
            gpus = [int(value) for value in args.gpus.split(",")]
        except ValueError as error:
            raise SystemExit("--gpus must be comma-separated physical IDs") from error
        orchestrate(gpus)
    else:
        raise SystemExit("contact-service is provided by the frozen PI2U entrypoint")


if __name__ == "__main__":
    main()
