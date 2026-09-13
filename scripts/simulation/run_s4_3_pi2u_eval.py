#!/usr/bin/env python3
"""Run the fresh seed-4 retry after the invalidated seed-3 infrastructure abort."""

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
from queue import Empty


ROOT = Path(__file__).resolve().parents[2]
DEXJOCO = ROOT / "third_party/dexjoco"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2u"
LOGS = ROOT / ".local/logs/simulation/s4_3_pi2u/evaluation_seed4"
CACHE = ROOT / ".local/cache/simulation/s4_3_pi2u/evaluation/seed4"
# AF_UNIX socket paths are limited to 108 bytes on this host; keep this short.
TMP = ROOT / ".local/tmp/s43u4"
PRE_FREEZE = ARTIFACTS / "pre_eval_retry_seed4.json"
CONDA_ROOT = Path(sys.executable).resolve().parents[3]
OPENPI_PYTHON = CONDA_ROOT / "envs/openpi/bin/python"
UNIT_PYTHON = CONDA_ROOT / "envs/unit/bin/python"
EVAL_PYTHON = ROOT / ".local/external/s4_3_pi0/eval-venv/bin/python"
MODELS = ("B0", "BVA", "B1", "B2")
RUNTIME_MODES = {"B0": "NONE", "BVA": "NONE", "B1": "CONTACT_STATE_TOKENS", "B2": "CONTACT_STATE_TOKENS_PHYSICAL_AUX"}
EPISODES = 200
EVALUATOR_SEED = 4


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
        if process.poll() is not None:
            tail = path.read_text(errors="replace")[-12000:] if path.exists() else ""
            raise RuntimeError(f"process exited {process.returncode} before {needle}:\n{tail}")
        time.sleep(1.0)
    raise TimeoutError(f"timed out waiting for {needle} in {path}")


def gpu_snapshot() -> dict[str, Any]:
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    return {"inventory": inventory, "compute_applications": apps}


def gpu_is_idle(index: int, snapshot: dict[str, Any]) -> bool:
    row = next((line for line in snapshot["inventory"] if int(line.split(",", 1)[0].strip()) == index), None)
    if row is None:
        return False
    fields = [field.strip() for field in row.split(",")]
    uuid, memory = fields[1], int(fields[3])
    return memory <= 64 and not any(line.split(",", 1)[0].strip() == uuid for line in snapshot["compute_applications"])


def acquire_gpu_lock(index: int):
    common = subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=ROOT, text=True).strip()
    handle = open(Path(common) / f"tactile3d_unit_gpu{index}.lock", "a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"GPU {index} advisory lock is held")
    return handle


def contact_service(artifact: Path, socket_path: Path) -> None:
    from scripts.simulation import serve_s4_3_pi1_contact_state as frozen

    frozen.ARTIFACT = artifact
    sys.argv = [sys.argv[0], "--socket", str(socket_path)]
    frozen.main()


def install_cross_episode_action_quarantine(official: Any, diagnostics: Path, model: str) -> None:
    """Preserve the official merge logic while discarding impossible future chunks.

    An action chunk is generated only from an observation at or before the current
    simulator timestamp. A future timestamp can therefore only be a late chunk
    from the preceding episode after the evaluator reset its counter to zero.
    The upstream Queue.empty-based episode cleanup can race with that producer.
    """
    def guarded_receive_actions(action_queue, actions_buffer, now_timestamp: int, dual_arm: bool) -> None:
        interp_action_fn = official._interp_dual_arm_action if dual_arm else official._interp_single_arm_action
        while actions_buffer and actions_buffer[0].timestamp < now_timestamp:
            actions_buffer.popleft()
        while True:
            try:
                action_chunk = action_queue.get_nowait()
                if action_chunk.timestamp > now_timestamp:
                    append = {
                        "type": "stale_cross_episode_action_discarded",
                        "model": model,
                        "action_timestamp": int(action_chunk.timestamp),
                        "current_timestamp": int(now_timestamp),
                        "reason": "future timestamp is impossible within one causal episode",
                    }
                    with diagnostics.open("a", encoding="utf-8") as output:
                        output.write(json.dumps(append, sort_keys=True) + "\n")
                    continue
                action_chunk_timestamp_range = (now_timestamp, action_chunk.timestamp + action_chunk.action.shape[0])
                if action_chunk_timestamp_range[1] <= now_timestamp:
                    continue
                action = action_chunk.action[(action_chunk_timestamp_range[0] - action_chunk.timestamp):(action_chunk_timestamp_range[1] - action_chunk.timestamp)]
                if actions_buffer:
                    buffer_timestamp_range = (actions_buffer[0].timestamp, actions_buffer[-1].timestamp + 1)
                    assert buffer_timestamp_range[1] - buffer_timestamp_range[0] == len(actions_buffer), "Buffer timestamps must be continuous"
                else:
                    buffer_timestamp_range = (now_timestamp, now_timestamp)
                overlap_range = (max(action_chunk_timestamp_range[0], buffer_timestamp_range[0]), min(action_chunk_timestamp_range[1], buffer_timestamp_range[1]))
                overlap_len = overlap_range[1] - overlap_range[0]
                for ts in range(overlap_range[0], overlap_range[1]):
                    buffer_idx = ts - buffer_timestamp_range[0]
                    action_idx = ts - action_chunk_timestamp_range[0]
                    interp_t = (ts - overlap_range[0] + 1) / (overlap_len + 1)
                    actions_buffer[buffer_idx] = official.Action(action=interp_action_fn(actions_buffer[buffer_idx].action, action[action_idx], interp_t), timestamp=ts)
                for ts in range(buffer_timestamp_range[1], action_chunk_timestamp_range[1]):
                    action_idx = ts - action_chunk_timestamp_range[0]
                    actions_buffer.append(official.Action(action=action[action_idx], timestamp=ts))
            except Empty:
                break
    official.receive_actions = guarded_receive_actions


def evaluate_model(model: str, socket_path: Path, output: Path, diagnostics: Path, port: int) -> None:
    """Use the byte-frozen PI2A evaluator and vary only model/mode/seed/output."""

    from gr00t.simulation.s4_3_pi1 import TactileUnitMode
    from scripts.simulation import evaluate_s4_3_pi1d_augmented as frozen

    raw_artifact = ARTIFACTS / f"{model.lower()}_raw_rollouts.json"
    temporary_artifact = ARTIFACTS / f"pi1d_{model.lower()}_eval.json"
    if any(path.exists() for path in (output, diagnostics, raw_artifact, temporary_artifact)):
        raise SystemExit(f"refusing to overwrite canonical PI2U output for {model}")
    frozen.MODEL_ID = model
    # BVA deliberately shares B0's exact observation payload. The frozen evaluator
    # may compute causal telemetry for integrity diagnostics, but it never transmits
    # tactile/contact data to B0 or BVA because MODE is NONE.
    frozen.MODE = TactileUnitMode(RUNTIME_MODES[model])
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
    install_cross_episode_action_quarantine(official, diagnostics, model)
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
        raise RuntimeError(f"PI2U raw artifact failed for {model}")


def run_one(model: str, gpu: int, port: int) -> dict[str, Any]:
    lower = model.lower()
    socket_path = TMP / f"{lower}_contact_state.sock"
    output = CACHE / lower
    diagnostics = TMP / f"{lower}_inference.jsonl"
    contact_artifact = ARTIFACTS / f"{lower}_contact_state_service.json"
    contact_log = LOGS / f"{lower}_contact_state_service.log"
    server_log = LOGS / f"{lower}_server.log"
    client_log = LOGS / f"{lower}_client.log"
    if any(path.exists() for path in (socket_path, output, diagnostics, contact_artifact, ARTIFACTS / f"{lower}_raw_rollouts.json")):
        raise RuntimeError(f"refusing to overwrite PI2U output for {model}")
    started = time.time()
    contact = policy = None
    handles = []
    try:
        contact_handle, server_handle, client_handle = contact_log.open("w"), server_log.open("w"), client_log.open("w")
        handles.extend((contact_handle, server_handle, client_handle))
        base_env = os.environ | {"PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"}
        contact = subprocess.Popen(
            [str(UNIT_PYTHON), str(Path(__file__).resolve()), "contact-service", "--artifact", str(contact_artifact), "--socket", str(socket_path)],
            cwd=ROOT, env=base_env, stdout=contact_handle, stderr=subprocess.STDOUT,
        )
        wait_for_log(contact, contact_log, "CONTACT_STATE_SERVICE_READY", 120)
        gpu_env = base_env | {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(gpu),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
        }
        policy = subprocess.Popen(
            [str(OPENPI_PYTHON), str(ROOT / "scripts/simulation/serve_s4_3_pi2u_policy.py"), "--model", model, "--port", str(port)],
            cwd=ROOT, env=gpu_env, stdout=server_handle, stderr=subprocess.STDOUT,
        )
        wait_for_log(policy, server_log, f"PI2U_POLICY_SERVER_READY model={model}", 300)
        eval_env = gpu_env | {"MUJOCO_GL": "egl", "PYTHONPATH": f"{ROOT}:{DEXJOCO / 'dexjoco'}"}
        result = subprocess.run(
            [str(EVAL_PYTHON), str(Path(__file__).resolve()), "evaluate-model", "--model", model,
             "--contact-socket", str(socket_path), "--output", str(output), "--diagnostics", str(diagnostics), "--port", str(port)],
            cwd=ROOT, env=eval_env, stdout=client_handle, stderr=subprocess.STDOUT,
        )
        if result.returncode:
            raise RuntimeError(f"{model} evaluator exited {result.returncode}")
        raw = read_json(ARTIFACTS / f"{lower}_raw_rollouts.json")
        return {
            "model": model, "physical_gpu": gpu, "port": port, "status": raw["status"],
            "episodes": raw["episodes"], "successes_frozen_until_completion": "RECORDED_IN_RAW_ARTIFACT",
            "started_unix": started, "elapsed_seconds": time.time() - started,
            "runtime_mode": RUNTIME_MODES[model], "infrastructure_retries": 0,
        }
    finally:
        stop_process(policy)
        stop_process(contact)
        for handle in handles:
            handle.close()


def orchestrate(gpus: list[int]) -> None:
    if read_json(PRE_FREEZE).get("status") != "PASS":
        raise SystemExit("PI2U pre-evaluation freeze is not PASS")
    if not 1 <= len(gpus) <= 3 or len(gpus) != len(set(gpus)):
        raise SystemExit("PI2U launch requires one to three distinct idle GPUs")
    if any((ARTIFACTS / f"{model.lower()}_raw_rollouts.json").exists() for model in MODELS):
        raise SystemExit("refusing to overwrite or resume canonical PI2U raw outcomes")
    snapshot = gpu_snapshot()
    if not all(gpu_is_idle(index, snapshot) for index in gpus):
        raise SystemExit("one or more selected PI2U GPUs is not genuinely idle")
    locks = []
    try:
        locks = [acquire_gpu_lock(index) for index in gpus]
        LOGS.mkdir(parents=True, exist_ok=False)
        CACHE.mkdir(parents=True, exist_ok=False)
        TMP.mkdir(parents=True, exist_ok=False)
        first_wave = list(zip(MODELS[: len(gpus)], gpus, range(8330, 8330 + len(gpus))))
        rows: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(first_wave)) as executor:
            rows.extend(future.result() for future in [executor.submit(run_one, *assignment) for assignment in first_wave])
        for offset, model in enumerate(MODELS[len(gpus) :]):
            rows.append(run_one(model, gpus[offset % len(gpus)], 8340 + offset))
        final_snapshot = gpu_snapshot()
        payload = {
            "schema": "tactile3d-unit.s4-3-pi2u-gpu-execution.v1",
            "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
            "initial_snapshot": snapshot, "final_snapshot": final_snapshot, "workers": rows,
            "maximum_heavy_workers": len(gpus), "unrelated_gpu_processes_killed_or_preempted": False,
            "wave_plan": [[row[0] for row in first_wave], list(MODELS[len(gpus) :])],
        }
        atomic_json(ARTIFACTS / "gpu_execution.json", payload)
        print(json.dumps({"status": payload["status"], "workers": rows}, sort_keys=True))
        if payload["status"] != "PASS":
            raise SystemExit("S4_3_PI2U_EVALUATION_INCOMPLETE")
    finally:
        for handle in locks:
            handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    contact = sub.add_parser("contact-service")
    contact.add_argument("--artifact", type=Path, required=True)
    contact.add_argument("--socket", type=Path, required=True)
    evaluate = sub.add_parser("evaluate-model")
    evaluate.add_argument("--model", choices=MODELS, required=True)
    evaluate.add_argument("--contact-socket", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--diagnostics", type=Path, required=True)
    evaluate.add_argument("--port", type=int, required=True)
    launch = sub.add_parser("launch")
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
