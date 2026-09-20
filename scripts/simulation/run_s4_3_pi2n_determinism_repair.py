#!/usr/bin/env python3
"""Version the PI2N runtime after an exact fixed-input GPU repeat failure."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/simulation/s4_3_pi2n_runtime_determinism_repair.json"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2n"
ATTEMPT_2 = ARTIFACTS / "runtime_randomness_audit.json"
RESULT = ARTIFACTS / "runtime_determinism_repair.json"
RUNTIME_V2 = ARTIFACTS / "runtime_protocol_v2.json"
RAW = ARTIFACTS / "runtime_determinism_repair_raw"
LOGS = ROOT / ".local/logs/simulation/s4_3_pi2n/runtime_determinism_repair"
CACHE = ROOT / ".local/cache/simulation/s4_3_pi2n/runtime_determinism_repair"
OPENPI_PYTHON = Path("/home/wbcd/miniconda3/envs/openpi/bin/python")
EVAL_PYTHON = ROOT / ".local/external/s4_3_pi0/eval-venv/bin/python"
SERVER = ROOT / "scripts/simulation/serve_s4_3_pi2m_policy.py"
MODELS = ("B1", "B_HVA", "B2")
REPEATS = 3
DETERMINISTIC_XLA_FLAGS = (
    "--xla_gpu_deterministic_ops=true",
    "--xla_gpu_exclude_nondeterministic_ops=true",
    "--xla_gpu_autotune_level=0",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": base64.b64encode(array.tobytes()).decode(),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def canonical_payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def action_sha256(action: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(action, dtype=np.float32).tobytes()).hexdigest()


def fixture_payload() -> dict[str, Any]:
    flat = np.arange(640 * 640 * 3, dtype=np.uint32)
    base = np.asarray((flat * 37 + 11) % 256, dtype=np.uint8).reshape(640, 640, 3)
    return {
        "base": base,
        "wrist": np.ascontiguousarray(base[:, ::-1]),
        "state": np.linspace(-1.0, 1.0, 23, dtype=np.float32),
        "contact_state": np.linspace(-0.5, 0.5, 256, dtype=np.float32),
        "prompt": "Grasp the tongs and perform three consecutive open-close motions.",
    }


def infer(model: str, port: int, output: Path, action_output: Path) -> None:
    from openpi_client import websocket_client_policy

    payload = fixture_payload()
    client = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=port)
    result = client.infer(payload)
    action = np.asarray(result["actions"], dtype=np.float32)
    action_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(action_output, action, allow_pickle=False)
    artifact = {
        "schema": "tactile3d-unit.s4-3-pi2n-runtime-determinism-raw.v1",
        "status": "PASS" if action.shape == (30, 22) and np.isfinite(action).all() else "FAIL",
        "model": model,
        "input_sha256": canonical_payload_sha256(payload),
        "action_sha256": action_sha256(action),
        "action_shape": list(action.shape),
        "action_finite": bool(np.isfinite(action).all()),
        "action_minimum": float(action.min()),
        "action_maximum": float(action.max()),
        "action_mean_l2": float(np.linalg.norm(action, axis=-1).mean()),
        "policy_timing": result.get("policy_timing"),
        "task_rollout_performed": False,
    }
    atomic_json(output, artifact)


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
    return int(fields[3]) <= 64 and not any(
        line.split(",", 1)[0].strip() == uuid for line in snapshot["compute_applications"]
    )


def acquire_gpu_lock(index: int):
    common = subprocess.check_output(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=ROOT,
        text=True,
    ).strip()
    handle = open(Path(common) / f"tactile3d_unit_gpu{index}.lock", "a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"GPU {index} advisory lock is held")
    return handle


def wait_for_log(process: subprocess.Popen[Any], path: Path, needle: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and needle in path.read_text(errors="replace"):
            return
        if process.poll() is not None:
            tail = path.read_text(errors="replace")[-12000:] if path.exists() else ""
            raise RuntimeError(f"server exited before {needle}:\n{tail}")
        time.sleep(1)
    raise TimeoutError(f"timed out waiting for {needle}")


def stop_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=10)


def run_one(model: str, repeat: int, gpu: int, port: int) -> dict[str, Any]:
    raw = RAW / f"{model.lower()}_repeat_{repeat}.json"
    action = CACHE / f"{model.lower()}_repeat_{repeat}_action.npy"
    server_log = LOGS / f"{model.lower()}_repeat_{repeat}_server.log"
    client_log = LOGS / f"{model.lower()}_repeat_{repeat}_client.log"
    if any(path.exists() for path in (raw, action, server_log, client_log)):
        raise RuntimeError(f"refusing to overwrite determinism fixture {model} repeat {repeat}")
    xla_flags = " ".join(
        [os.environ.get("XLA_FLAGS", "").strip(), *DETERMINISTIC_XLA_FLAGS]
    ).strip()
    environment = os.environ | {
        "PYTHONPATH": str(ROOT),
        "PYTHONUNBUFFERED": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "XLA_FLAGS": xla_flags,
    }
    server = None
    started = time.time()
    with server_log.open("w") as server_output, client_log.open("w") as client_output:
        try:
            server = subprocess.Popen(
                [str(OPENPI_PYTHON), str(SERVER), "--model", model, "--port", str(port)],
                cwd=ROOT,
                env=environment,
                stdout=server_output,
                stderr=subprocess.STDOUT,
            )
            wait_for_log(server, server_log, f"PI2M_POLICY_SERVER_READY model={model}", 300)
            result = subprocess.run(
                [
                    str(EVAL_PYTHON),
                    str(Path(__file__).resolve()),
                    "infer",
                    "--model",
                    model,
                    "--port",
                    str(port),
                    "--output",
                    str(raw),
                    "--action-output",
                    str(action),
                ],
                cwd=ROOT,
                env=environment,
                stdout=client_output,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if result.returncode:
                raise RuntimeError(
                    f"fixture client exited {result.returncode}:\n"
                    f"{client_log.read_text(errors='replace')[-12000:]}"
                )
            payload = json.loads(raw.read_text())
            return {
                **payload,
                "repeat": repeat,
                "physical_gpu": gpu,
                "port": port,
                "fresh_server": True,
                "elapsed_seconds": time.time() - started,
                "server_log": str(server_log),
                "client_log": str(client_log),
                "action_npy": str(action),
            }
        finally:
            stop_process(server)


def orchestrate(gpu: int) -> None:
    protocol = json.loads(PROTOCOL.read_text())
    attempt = json.loads(ATTEMPT_2.read_text())
    if protocol.get("status") != "FROZEN_AFTER_ATTEMPT_2_BEFORE_REPAIR_FIXTURE":
        raise SystemExit("determinism repair protocol is not frozen")
    expected_attempt = (
        attempt.get("classification") == "R_C_UNEXPLAINED_MODEL_UNFAIRNESS"
        and attempt.get("fairness_gates", {}).get(
            "fixed_observation_action_repeatable_within_each_model"
        )
        == "FAIL"
        and all(
            value == "PASS"
            for name, value in attempt.get("fairness_gates", {}).items()
            if name != "fixed_observation_action_repeatable_within_each_model"
        )
        and all(value == "PASS" for value in attempt.get("structural_gates", {}).values())
    )
    if not expected_attempt:
        raise SystemExit("attempt-2 evidence does not match the frozen repair trigger")
    protected = (RESULT, RUNTIME_V2, RAW, LOGS, CACHE)
    if any(path.exists() for path in protected):
        raise SystemExit("refusing to overwrite or resume the determinism repair fixture")
    first = gpu_snapshot()
    time.sleep(2)
    second = gpu_snapshot()
    if not (gpu_is_idle(gpu, first) and gpu_is_idle(gpu, second)):
        raise SystemExit("selected repair-fixture GPU is not genuinely idle in two snapshots")
    lock = acquire_gpu_lock(gpu)
    try:
        third = gpu_snapshot()
        if not gpu_is_idle(gpu, third):
            raise SystemExit("selected repair-fixture GPU became busy after locking")
        RAW.mkdir(parents=True, exist_ok=False)
        LOGS.mkdir(parents=True, exist_ok=False)
        CACHE.mkdir(parents=True, exist_ok=False)
        rows = []
        for model_index, model in enumerate(MODELS):
            for repeat in range(1, REPEATS + 1):
                rows.append(run_one(model, repeat, gpu, 8490 + model_index * 4 + repeat))
        input_hashes = {row["input_sha256"] for row in rows}
        action_hashes = {
            model: {row["action_sha256"] for row in rows if row["model"] == model}
            for model in MODELS
        }
        gates = {
            "attempt_2_preserved_and_trigger_exact": expected_attempt,
            "same_input_sha256_all_nine": len(input_hashes) == 1,
            "same_float32_action_sha256_within_each_model": all(
                len(values) == 1 for values in action_hashes.values()
            ),
            "finite_30x22_actions_all_nine": all(
                row["status"] == "PASS"
                and row["action_finite"] is True
                and row["action_shape"] == [30, 22]
                for row in rows
            ),
            "fresh_server_each_inference": all(row["fresh_server"] is True for row in rows),
            "no_task_rollouts": all(row["task_rollout_performed"] is False for row in rows),
        }
        passed = all(gates.values())
        payload = {
            "schema": "tactile3d-unit.s4-3-pi2n-runtime-determinism-repair.v1",
            "status": "PASS" if passed else "FAIL",
            "created_at": now(),
            "protocol_sha256": sha256_file(PROTOCOL),
            "attempt_2_sha256": sha256_file(ATTEMPT_2),
            "engineering_fix_index": 1,
            "runtime_environment": {
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "XLA_FLAGS": list(DETERMINISTIC_XLA_FLAGS),
            },
            "input_sha256": next(iter(input_hashes)) if len(input_hashes) == 1 else sorted(input_hashes),
            "action_sha256_by_model": {
                model: next(iter(values)) if len(values) == 1 else sorted(values)
                for model, values in action_hashes.items()
            },
            "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
            "rows": rows,
            "gpu_snapshots": [first, second, third, gpu_snapshot()],
            "candidate_policy_results_read": False,
            "formal_results_read": False,
            "task_rollouts": 0,
        }
        atomic_json(RESULT, payload)
        runtime = {
            "schema": "tactile3d-unit.s4-3-pi2n-runtime-protocol.v2",
            "status": "PASS" if passed else "BLOCKED",
            "classification": (
                "R_B_LEGAL_RANDOMNESS_COMPARABLE_AFTER_DETERMINISM_REPAIR"
                if passed
                else "R_C_UNEXPLAINED_MODEL_UNFAIRNESS"
            ),
            "scientific_rollouts_authorized": passed,
            "supersedes_for_future_scientific_rollouts": str(ARTIFACTS / "runtime_protocol.json"),
            "attempt_2_raw_rollout_evidence_retained": True,
            "attempt_2_performance_role": "DEVELOPMENT_EXPOSED_RUNTIME_ONLY",
            "official_async_evaluator_retained": True,
            "policy_sampling_changed": False,
            "per_episode_policy_rng_reset": False,
            "synchronous_replacement": False,
            "required_server_environment": payload["runtime_environment"],
            "fixed_input_repair_sha256": sha256_file(RESULT),
            "retry_contract": json.loads(
                (ARTIFACTS / "runtime_protocol.json").read_text()
            )["retry_contract"],
            "remaining_variability": [
                "asynchronous observation/action queue scheduling",
                "request-count-dependent policy RNG position",
                "host/GPU timing"
            ],
        }
        atomic_json(RUNTIME_V2, runtime)
        print(json.dumps({"status": payload["status"], "gates": payload["gates"]}, sort_keys=True))
        if not passed:
            raise SystemExit("PI2N_RUNTIME_DETERMINISM_REPAIR_FAILED")
    finally:
        lock.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    infer_parser = sub.add_parser("infer")
    infer_parser.add_argument("--model", choices=MODELS, required=True)
    infer_parser.add_argument("--port", type=int, required=True)
    infer_parser.add_argument("--output", type=Path, required=True)
    infer_parser.add_argument("--action-output", type=Path, required=True)
    run_parser = sub.add_parser("orchestrate")
    run_parser.add_argument("--gpu", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "infer":
        infer(args.model, args.port, args.output, args.action_output)
    else:
        orchestrate(args.gpu)


if __name__ == "__main__":
    main()
