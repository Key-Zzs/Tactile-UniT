#!/usr/bin/env python3
"""Run the frozen PI1D runtime for the paired S4.3-PI2A confirmation."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEXJOCO = ROOT / "third_party/dexjoco"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2a"
LOGS = ROOT / ".local/logs/simulation/s4_3_pi2a"
CACHE = ROOT / ".local/cache/simulation/s4_3_pi2a"
TMP = ROOT / ".local/tmp/simulation/s4_3_pi2a"
PRE_FREEZE = ARTIFACTS / "pre_eval_freeze.json"
CONDA_ROOT = Path(sys.executable).resolve().parents[3]
OPENPI_PYTHON = CONDA_ROOT / "envs/openpi/bin/python"
UNIT_PYTHON = CONDA_ROOT / "envs/unit/bin/python"
EVAL_PYTHON = ROOT / ".local/external/s4_3_pi0/eval-venv/bin/python"
MODELS = {
    "B0": "NONE",
    "B1": "CONTACT_STATE_TOKENS",
    "B2": "CONTACT_STATE_TOKENS_PHYSICAL_AUX",
}
EPISODES = 200
EVALUATOR_SEED = 2


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def stop_process(process: subprocess.Popen[Any] | None, timeout: float = 10.0) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"process {process.pid} ignored SIGINT and SIGTERM")


def wait_for_log(process: subprocess.Popen[Any], path: Path, needle: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and needle in path.read_text(errors="replace"):
            return
        code = process.poll()
        if code is not None:
            tail = path.read_text(errors="replace")[-12000:] if path.exists() else ""
            raise RuntimeError(f"process exited {code} before {needle}:\n{tail}")
        time.sleep(1.0)
    raise TimeoutError(f"timed out waiting for {needle} in {path}")


def gpu_snapshot() -> dict[str, Any]:
    inventory = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    applications = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    return {"inventory": inventory, "compute_applications": applications}


def gpu_is_idle(index: int, snapshot: dict[str, Any]) -> bool:
    row = next(
        (line for line in snapshot["inventory"] if int(line.split(",", 1)[0].strip()) == index),
        None,
    )
    if row is None:
        return False
    fields = [field.strip() for field in row.split(",")]
    uuid = fields[1]
    memory_used = int(fields[3])
    has_compute_process = any(
        line.split(",", 1)[0].strip() == uuid for line in snapshot["compute_applications"]
    )
    return memory_used <= 64 and not has_compute_process


def contact_service(artifact: Path, socket_path: Path) -> None:
    from scripts.simulation import serve_s4_3_pi1_contact_state as frozen

    frozen.ARTIFACT = artifact
    sys.argv = [sys.argv[0], "--socket", str(socket_path)]
    frozen.main()


def evaluate_model(
    model: str, socket_path: Path, output: Path, diagnostics: Path, port: int
) -> None:
    """Invoke the frozen PI1D augmented environment with PI2A constants only."""

    from gr00t.simulation.s4_3_pi1 import TactileUnitMode
    from scripts.simulation import evaluate_s4_3_pi1d_augmented as frozen

    if model not in MODELS:
        raise SystemExit(f"unknown PI2A model: {model}")
    raw_artifact = ARTIFACTS / f"{model.lower()}_raw_rollouts.json"
    temporary_artifact = ARTIFACTS / f"pi1d_{model.lower()}_eval.json"
    if (
        output.exists()
        or diagnostics.exists()
        or raw_artifact.exists()
        or temporary_artifact.exists()
    ):
        raise SystemExit(f"refusing to overwrite PI2A output for {model}")
    expected_mode = TactileUnitMode(MODELS[model])
    frozen.MODEL_ID = model
    frozen.MODE = expected_mode
    frozen.DIAGNOSTICS_JSONL = diagnostics
    frozen.OUTPUT_ROOT = output
    frozen.CONTACT_SOCKET = socket_path
    frozen.EXPECTED_EPISODES = EPISODES
    frozen.EVALUATOR_SEED = EVALUATOR_SEED
    frozen.ARTIFACTS = ARTIFACTS
    diagnostics.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(DEXJOCO / "dexjoco"))
    from dexjoco_openpi_client import eval_dexjoco_openpi as official
    from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv

    official.DexJoCoOpenPIEnv = frozen.build_augmented_environment(DexJoCoOpenPIEnv)
    official.inference_process = frozen.instrumented_inference_process
    official.main(
        config=DEXJOCO / "configs/rand_obj/pinch_tongs.yaml",
        seed=EVALUATOR_SEED,
        rand_full=False,
        randomize_dynamics=False,
        port=port,
        host="127.0.0.1",
        output=output,
        render_mode="rgb_array",
        replan_ratio=0.8,
        episodes=EPISODES,
        pad_state_dim46=False,
        record_pressed_digits=False,
    )
    if not temporary_artifact.is_file():
        raise RuntimeError(f"frozen runtime did not emit {temporary_artifact}")
    temporary_artifact.replace(raw_artifact)
    payload = read_json(raw_artifact)
    if payload["status"] != "PASS" or payload["episodes"] != EPISODES:
        raise RuntimeError(f"PI2A raw artifact failed for {model}")


def acquire_gpu_lock(index: int):
    common = subprocess.check_output(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=ROOT, text=True
    ).strip()
    handle = open(Path(common) / f"tactile3d_unit_gpu{index}.lock", "a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"GPU {index} advisory lock is held")
    return handle


def run_one(model: str, gpu: int, port: int) -> dict[str, Any]:
    lower = model.lower()
    socket_path = TMP / f"{lower}_contact_state.sock"
    output = CACHE / "evaluation/seed2" / lower
    diagnostics = TMP / f"{lower}_inference.jsonl"
    contact_artifact = ARTIFACTS / f"{lower}_contact_state_service.json"
    contact_log = LOGS / f"{lower}_contact_state_service.log"
    server_log = LOGS / f"{lower}_server.log"
    client_log = LOGS / f"{lower}_client.log"
    for path in (
        socket_path,
        output,
        diagnostics,
        contact_artifact,
        ARTIFACTS / f"{lower}_raw_rollouts.json",
    ):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite {path}")
    started = time.time()
    contact = policy = None
    handles = []
    try:
        contact_handle = contact_log.open("w")
        server_handle = server_log.open("w")
        client_handle = client_log.open("w")
        handles.extend((contact_handle, server_handle, client_handle))
        base_env = os.environ.copy()
        base_env.update({"PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"})
        contact = subprocess.Popen(
            [
                str(UNIT_PYTHON),
                str(Path(__file__).resolve()),
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
        wait_for_log(contact, contact_log, "CONTACT_STATE_SERVICE_READY", 120)

        gpu_env = base_env | {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
        }
        policy = subprocess.Popen(
            [
                str(OPENPI_PYTHON),
                str(ROOT / "scripts/simulation/serve_s4_3_pi1_policy.py"),
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
        wait_for_log(policy, server_log, f"PI1D_POLICY_SERVER_READY model={model}", 240)

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
        raw = read_json(ARTIFACTS / f"{lower}_raw_rollouts.json")
        return {
            "model": model,
            "gpu": gpu,
            "port": port,
            "status": raw["status"],
            "episodes": raw["episodes"],
            "successes_frozen_until_completion": "RECORDED_IN_RAW_ARTIFACT",
            "started_unix": started,
            "elapsed_seconds": time.time() - started,
            "concurrency": "B0/B1/B2 parallel on separate GPUs",
            "infrastructure_retries": 0,
        }
    finally:
        stop_process(policy)
        stop_process(contact)
        for handle in handles:
            handle.close()


def orchestrate(gpus: list[int]) -> None:
    if read_json(PRE_FREEZE).get("status") != "PASS":
        raise SystemExit("PI2A pre-evaluation freeze is not PASS")
    if len(gpus) != 3 or len(set(gpus)) != 3:
        raise SystemExit("parallel PI2A launch requires exactly three distinct idle GPUs")
    if any((ARTIFACTS / f"{model.lower()}_raw_rollouts.json").exists() for model in MODELS):
        raise SystemExit("refusing to overwrite or resume canonical PI2A raw outcomes")
    snapshot = gpu_snapshot()
    if not all(gpu_is_idle(index, snapshot) for index in gpus):
        raise SystemExit("one or more selected PI2A GPUs is not genuinely idle")
    locks = []
    try:
        locks = [acquire_gpu_lock(index) for index in gpus]
        LOGS.mkdir(parents=True, exist_ok=False)
        (CACHE / "evaluation/seed2").mkdir(parents=True, exist_ok=False)
        TMP.mkdir(parents=True, exist_ok=False)
        assignments = [
            (model, gpu, 8230 + offset) for offset, (model, gpu) in enumerate(zip(MODELS, gpus))
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_one, *assignment) for assignment in assignments]
            rows = [future.result() for future in futures]
        final_snapshot = gpu_snapshot()
        payload = {
            "schema": "tactile3d-unit.s4-3-pi2a-gpu-execution.v1",
            "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
            "initial_snapshot": snapshot,
            "final_snapshot": final_snapshot,
            "workers": rows,
            "one_heavy_worker_per_physical_gpu": True,
            "unrelated_gpu_processes_killed_or_preempted": False,
        }
        atomic_json(ARTIFACTS / "gpu_execution.json", payload)
        print(json.dumps({"status": payload["status"], "workers": rows}, sort_keys=True))
        if payload["status"] != "PASS":
            raise SystemExit("S4_3_PI2A_EVALUATION_INCOMPLETE")
    finally:
        for handle in locks:
            handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    contact = subparsers.add_parser("contact-service")
    contact.add_argument("--artifact", type=Path, required=True)
    contact.add_argument("--socket", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate-model")
    evaluate.add_argument("--model", choices=tuple(MODELS), required=True)
    evaluate.add_argument("--contact-socket", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--diagnostics", type=Path, required=True)
    evaluate.add_argument("--port", type=int, required=True)
    launch = subparsers.add_parser("launch")
    launch.add_argument("--gpus", default="1,2,3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "contact-service":
        contact_service(args.artifact, args.socket)
    elif args.command == "evaluate-model":
        evaluate_model(args.model, args.contact_socket, args.output, args.diagnostics, args.port)
    else:
        orchestrate([int(value) for value in args.gpus.split(",")])


if __name__ == "__main__":
    main()
