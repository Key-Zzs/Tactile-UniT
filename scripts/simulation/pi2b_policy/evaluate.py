#!/usr/bin/env python3
"""Run the frozen 15-checkpoint Track-A FINAL cohort without interpretation."""

from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from gr00t.simulation.pi2b_policy.contract import MODEL_ORDER, Workspace
from gr00t.simulation.pi2b_policy.coordination import (
    coordination_paths,
    gpu_inventory,
    gpu_is_idle,
    gpu_snapshot,
    open_lock,
    teacher_has_pending_request,
)
from gr00t.simulation.pi2b_policy.integrity import sha256_file


ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2b_policy"
PRE_FREEZE = ARTIFACTS / "pre_final_freeze.json"
RESET_MANIFEST = ARTIFACTS / "reset_manifest.json"
RAW = ARTIFACTS / "final_raw"
EXECUTION = ARTIFACTS / "final_gpu_execution.json"
COMPLETENESS = ARTIFACTS / "rollout_completeness.json"
PROGRESS = ARTIFACTS / "final_progress.json"
LOGS = ROOT / ".local/logs/simulation/s4_3_pi2b_policy/final"
CACHE = ROOT / ".local/cache/simulation/s4_3_pi2b_policy/final"
TMP = Path("/tmp/pi2ba_final")
CONDA_ROOT = Path(sys.executable).resolve().parents[3]
OPENPI_PYTHON = CONDA_ROOT / "envs/openpi/bin/python"
UNIT_PYTHON = CONDA_ROOT / "envs/unit/bin/python"
EVAL_PYTHON = Workspace.load(ROOT).main_root / ".local/external/s4_3_pi0/eval-venv/bin/python"
TRAINING_SEEDS = (42, 43, 44)
RESET_SEEDS = (16, 17, 18, 19)
RUNTIME_MODES = {
    "B0": "NONE",
    "B_VA27": "NONE",
    "B1": "CONTACT_STATE_TOKENS",
    "B_HVA": "CONTACT_STATE_TOKENS",
    "B2": "CONTACT_STATE_TOKENS",
}
XLA_FLAGS = (
    "--xla_gpu_deterministic_ops=true",
    "--xla_gpu_exclude_nondeterministic_ops=true",
    "--xla_gpu_autotune_level=0",
)
JOBS = tuple(
    (model, training_seed, reset_seed)
    for reset_seed in RESET_SEEDS
    for training_seed in TRAINING_SEEDS
    for model in MODEL_ORDER
)
DEXJOCO = Workspace.load(ROOT).main_root / "third_party/dexjoco"


class ContractError(RuntimeError):
    """A scientific integrity failure that must never be retried."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def job_key(model: str, training_seed: int, reset_seed: int) -> str:
    return f"r{reset_seed}_{model.lower()}_s{training_seed}"


def raw_path(model: str, training_seed: int, reset_seed: int) -> Path:
    return RAW / f"reset_seed_{reset_seed}" / model.lower() / f"train_seed_{training_seed}.json"


def configure_runtime(model: str, training_seed: int, reset_seed: int, attempt: int):
    from scripts.simulation import run_s4_3_pi2u_eval as runtime

    key = job_key(model, training_seed, reset_seed)
    runtime.ARTIFACTS = ARTIFACTS / "final_attempts" / key / f"attempt_{attempt}"
    runtime.LOGS = LOGS / key / f"attempt_{attempt}"
    runtime.CACHE = CACHE / key / f"attempt_{attempt}"
    runtime.TMP = TMP / key / f"attempt_{attempt}"
    runtime.MODELS = MODEL_ORDER
    runtime.RUNTIME_MODES = RUNTIME_MODES
    runtime.EPISODES = 50
    runtime.EVALUATOR_SEED = reset_seed
    return runtime


def contact_service(artifact: Path, socket_path: Path) -> None:
    from scripts.simulation import run_s4_3_pi2u_eval as runtime

    runtime.contact_service(artifact, socket_path)


def evaluate_model(model: str, training_seed: int, reset_seed: int, attempt: int, socket_path: Path, output: Path, diagnostics: Path, port: int) -> None:
    runtime = configure_runtime(model, training_seed, reset_seed, attempt)
    runtime.evaluate_model(model, socket_path, output, diagnostics, port)


def checkpoint_row(freeze: dict[str, Any], model: str, seed: int) -> dict[str, Any]:
    return next(row for row in freeze["checkpoints"] if row["model"] == model and int(row["training_seed"]) == seed)


def run_attempt(freeze: dict[str, Any], model: str, training_seed: int, reset_seed: int, gpu: int, port: int, attempt: int) -> dict[str, Any]:
    runtime = configure_runtime(model, training_seed, reset_seed, attempt)
    key = job_key(model, training_seed, reset_seed)
    socket_path = TMP / key / f"a{attempt}.sock"
    output = runtime.CACHE / "environment"
    diagnostics = runtime.TMP / "inference.jsonl"
    contact_artifact = runtime.TMP / "contact_state_service.json"
    contact_log = runtime.LOGS / "contact.log"
    server_log = runtime.LOGS / "server.log"
    client_log = runtime.LOGS / "client.log"
    canonical = raw_path(model, training_seed, reset_seed)
    if canonical.exists() or runtime.ARTIFACTS.exists() or runtime.LOGS.exists() or runtime.CACHE.exists() or runtime.TMP.exists():
        raise RuntimeError(f"refusing to overwrite {key} attempt {attempt}")
    for path in (runtime.ARTIFACTS, runtime.LOGS, runtime.CACHE, runtime.TMP):
        path.mkdir(parents=True, exist_ok=False)
    started = time.time()
    contact = policy = None
    handles = []
    try:
        contact_handle = contact_log.open("x")
        server_handle = server_log.open("x")
        client_handle = client_log.open("x")
        handles.extend((contact_handle, server_handle, client_handle))
        base_env = os.environ | {"PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"}
        contact = subprocess.Popen(
            [str(UNIT_PYTHON), str(Path(__file__).resolve()), "contact-service", "--artifact", str(contact_artifact), "--socket", str(socket_path)],
            cwd=ROOT, env=base_env, stdout=contact_handle, stderr=subprocess.STDOUT,
        )
        runtime.wait_for_log(contact, contact_log, "CONTACT_STATE_SERVICE_READY", 120)
        gpu_env = base_env | {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "XLA_FLAGS": " ".join(XLA_FLAGS),
        }
        policy = subprocess.Popen(
            [str(OPENPI_PYTHON), str(ROOT / "scripts/simulation/pi2b_policy/serve.py"), "--model", model, "--training-seed", str(training_seed), "--port", str(port)],
            cwd=ROOT, env=gpu_env, stdout=server_handle, stderr=subprocess.STDOUT,
        )
        runtime.wait_for_log(policy, server_log, f"PI2B_POLICY_SERVER_READY model={model} training_seed={training_seed}", 600)
        eval_env = gpu_env | {"MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "0", "PYTHONPATH": f"{ROOT}:{DEXJOCO / 'dexjoco'}"}
        result = subprocess.run(
            [str(EVAL_PYTHON), str(Path(__file__).resolve()), "evaluate-model", "--model", model, "--training-seed", str(training_seed), "--reset-seed", str(reset_seed), "--attempt", str(attempt), "--contact-socket", str(socket_path), "--output", str(output), "--diagnostics", str(diagnostics), "--port", str(port)],
            cwd=ROOT, env=eval_env, stdout=client_handle, stderr=subprocess.STDOUT,
        )
        if result.returncode:
            raise RuntimeError(f"evaluator exited {result.returncode}")
        generated = runtime.ARTIFACTS / f"{model.lower()}_raw_rollouts.json"
        payload = read_json(generated)
        if payload.get("status") != "PASS" or len(payload.get("episode_results", [])) != 50:
            raise ContractError("frozen evaluator integrity gate failed")
        bound = checkpoint_row(freeze, model, training_seed)
        payload.update({
            "training_seed": training_seed,
            "reset_seed": reset_seed,
            "checkpoint_path": bound["path"],
            "checkpoint_tree_sha256": bound["tree_sha256"],
            "server_sampling_contract": "PI2N_ACCEPTED_FRESH_SERVER_REQUEST_ADVANCE",
            "determinism_env": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "XLA_FLAGS": list(XLA_FLAGS)},
            "infrastructure_attempt": attempt,
        })
        for row in payload["episode_results"]:
            row.update({"model": model, "training_seed": training_seed, "reset_seed": reset_seed, "checkpoint_tree_sha256": bound["tree_sha256"], "runtime_mode": RUNTIME_MODES[model]})
        canonical.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(canonical, payload)
        return {"model": model, "training_seed": training_seed, "reset_seed": reset_seed, "gpu": gpu, "port": port, "attempt": attempt, "episodes": 50, "elapsed_seconds": time.time() - started, "status": "PASS", "raw_sha256": sha256_file(canonical)}
    finally:
        runtime.stop_process(policy)
        runtime.stop_process(contact)
        for handle in handles:
            handle.close()


def run_job(freeze: dict[str, Any], job: tuple[str, int, int], gpu: int, port: int) -> dict[str, Any]:
    errors = []
    for attempt in (1, 2, 3):
        try:
            row = run_attempt(freeze, *job, gpu, port, attempt)
            row["prior_infrastructure_errors"] = errors
            return row
        except Exception as error:
            errors.append({"attempt": attempt, "type": type(error).__name__, "message": str(error), "at": now()})
            if isinstance(error, ContractError) or attempt == 3:
                raise
    raise AssertionError("unreachable")


def expected_resets(manifest: dict[str, Any], seed: int) -> list[str]:
    return [row["reset_identity"] for row in manifest["reset_specs"] if int(row["seed"]) == seed]


def audit_completeness(workers: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = read_json(RESET_MANIFEST)
    reference = manifest["ordered_reset_identities"]
    tuples = []
    raw_index = {}
    gates = {}
    for model, training_seed, reset_seed in JOBS:
        path = raw_path(model, training_seed, reset_seed)
        payload = read_json(path)
        rows = payload.get("episode_results", [])
        observed = [row.get("reset_identity") for row in rows]
        name = job_key(model, training_seed, reset_seed)
        gate = payload.get("status") == "PASS" and len(rows) == 50 and observed == expected_resets(manifest, reset_seed) and all(value == "PASS" for value in payload.get("gates", {}).values())
        gates[name] = "PASS" if gate else "FAIL"
        tuples.extend((model, training_seed, value) for value in observed)
        raw_index[name] = {"path": str(path), "sha256": sha256_file(path), "episodes": len(rows)}
    global_gates = {
        "all_60_blocks_pass": len(gates) == 60 and all(value == "PASS" for value in gates.values()),
        "exactly_3000_canonical_tuples": len(tuples) == 3000 and len(set(tuples)) == 3000,
        "all_15_share_frozen_200_resets": all([identity for m, s, identity in tuples if m == model and s == seed] == reference for seed in TRAINING_SEEDS for model in MODEL_ORDER),
        "worker_records_exact": len(workers) == 60 and all(row["status"] == "PASS" for row in workers),
        "no_unresolved_runtime_error": all(not row.get("unresolved_runtime_error") for row in workers),
    }
    return {
        "schema": "tactile3d-unit.s4-3-pi2b-policy-rollout-completeness.v1",
        "created_at_utc": now(),
        "status": "PASS" if all(global_gates.values()) else "FAIL",
        "total_canonical_outcomes": len(tuples),
        "models": list(MODEL_ORDER),
        "training_seeds": list(TRAINING_SEEDS),
        "ordered_reset_sequence_sha256": hashlib.sha256("\n".join(reference).encode()).hexdigest(),
        "raw_artifacts": raw_index,
        "per_block_gates": gates,
        "gates": {key: "PASS" if value else "FAIL" for key, value in global_gates.items()},
        "native_failures_and_timeouts_retained": True,
        "performance_interpreted_before_completion": False,
        "statistics_computed": False,
    }


def update_coordination(workspace: Workspace, status: str, **extra: Any) -> None:
    paths = coordination_paths(workspace)
    with open_lock(paths["scheduler"], shared=False, nonblocking=False):
        atomic_json(paths["request"], {"schema": "tactile3d-unit.pi2b-coordination-policy-request.v1", "track": "policy", "kind": "FORMAL_EVALUATION", "runtime_barrier": "exclusive", "status": status, "updated_at_utc": now(), **extra})


def orchestrate(max_workers: int) -> None:
    workspace = Workspace.load(ROOT)
    required_executables = (OPENPI_PYTHON, UNIT_PYTHON, EVAL_PYTHON)
    if any(not path.is_file() for path in required_executables):
        raise SystemExit(f"required runtime executable missing: {required_executables}")
    if not (DEXJOCO / "dexjoco/dexjoco_openpi_client/eval_dexjoco_openpi.py").is_file():
        raise SystemExit(f"audited DexJoCo source missing: {DEXJOCO}")
    freeze = read_json(PRE_FREEZE)
    if freeze.get("status") != "PASS" or freeze.get("cohort_performance_seen") is not False:
        raise SystemExit("pre-FINAL freeze is not eligible")
    if any(path.exists() for path in (RAW, EXECUTION, COMPLETENESS, PROGRESS, LOGS, CACHE, TMP)):
        raise SystemExit("refusing to overwrite or resume canonical FINAL state")
    if any(sha256_file(ROOT / relative) != expected for relative, expected in freeze["sources_sha256"].items()):
        raise SystemExit("frozen source drifted")
    if any(sha256_file(Path(path)) != expected for path, expected in freeze["external_sources_sha256"].items()):
        raise SystemExit("frozen external source drifted")
    if teacher_has_pending_request(workspace):
        raise SystemExit("teacher has a pending heavy request")
    paths = coordination_paths(workspace)
    barrier = open_lock(paths["runtime_barrier"], shared=False, nonblocking=False)
    gpu_locks = []
    workers = []
    started = time.time()
    try:
        update_coordination(workspace, "EVAL_SELECTING_GPU", supervisor_pid=os.getpid())
        snap1 = gpu_snapshot()
        time.sleep(2)
        snap2 = gpu_snapshot()
        candidates = [index for index in sorted(gpu_inventory(snap2)) if gpu_is_idle(index, snap1, snap2)]
        selected = candidates[:max_workers]
        if not selected:
            raise RuntimeError("no GPU is idle twice")
        for gpu in selected:
            try:
                gpu_locks.append(open_lock(workspace.common_git_dir / f"tactile3d_unit_gpu{gpu}.lock", shared=False))
            except BlockingIOError:
                for handle in gpu_locks:
                    handle.close()
                gpu_locks = []
                raise RuntimeError("selected GPU advisory lock became busy")
        snap3 = gpu_snapshot()
        if not all(gpu_is_idle(index, snap3) for index in selected):
            raise RuntimeError("selected GPU became busy after locking")
        for path in (RAW, LOGS, CACHE, TMP):
            path.mkdir(parents=True, exist_ok=False)
        update_coordination(workspace, "EVAL_RUNNING", supervisor_pid=os.getpid(), gpus=selected, total_blocks=60, total_rollouts=3000)
        pending = list(enumerate(JOBS))
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected)) as pool:
            active: dict[concurrent.futures.Future, tuple[int, tuple[str, int, int], int]] = {}
            while pending or active:
                while pending and len(active) < len(selected):
                    index, job = pending.pop(0)
                    used = {value[2] for value in active.values()}
                    gpu = next(value for value in selected if value not in used)
                    future = pool.submit(run_job, freeze, job, gpu, 8800 + gpu)
                    active[future] = (index, job, gpu)
                done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    active.pop(future)
                    workers.append(future.result())
                    elapsed = time.time() - started
                    completed = len(workers)
                    atomic_json(PROGRESS, {"schema": "tactile3d-unit.s4-3-pi2b-policy-progress.v1", "status": "RUNNING", "updated_at_utc": now(), "completed_blocks": completed, "total_blocks": 60, "completed_rollouts": completed * 50, "total_rollouts": 3000, "elapsed_seconds": elapsed, "rollouts_per_hour": completed * 50 / elapsed * 3600, "eta_seconds": elapsed / completed * (60 - completed), "gpus": selected, "performance_values_read": False})
        completeness = audit_completeness(workers)
        atomic_json(COMPLETENESS, completeness)
        atomic_json(EXECUTION, {"schema": "tactile3d-unit.s4-3-pi2b-policy-final-execution.v1", "status": completeness["status"], "created_at_utc": now(), "workers": workers, "gpu_snapshots": [snap1, snap2, snap3, gpu_snapshot()], "exclusive_barrier_held_for_full_wave": True, "performance_interpreted_during_execution": False})
        atomic_json(PROGRESS, {"status": completeness["status"], "updated_at_utc": now(), "completed_blocks": len(workers), "total_blocks": 60, "completed_rollouts": len(workers) * 50, "total_rollouts": 3000, "elapsed_seconds": time.time() - started, "performance_values_read": False})
        update_coordination(workspace, "COMPLETE" if completeness["status"] == "PASS" else "FAILED", completed_blocks=len(workers), total_blocks=60)
        if completeness["status"] != "PASS":
            raise SystemExit(2)
    except BaseException as error:
        update_coordination(workspace, "FAILED", supervisor_pid=os.getpid(), error_type=type(error).__name__, error=str(error))
        raise
    finally:
        for handle in gpu_locks:
            handle.close()
        barrier.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    contact = sub.add_parser("contact-service")
    contact.add_argument("--artifact", type=Path, required=True)
    contact.add_argument("--socket", type=Path, required=True)
    evaluate = sub.add_parser("evaluate-model")
    evaluate.add_argument("--model", choices=MODEL_ORDER, required=True)
    evaluate.add_argument("--training-seed", choices=TRAINING_SEEDS, type=int, required=True)
    evaluate.add_argument("--reset-seed", choices=RESET_SEEDS, type=int, required=True)
    evaluate.add_argument("--attempt", choices=(1, 2, 3), type=int, required=True)
    evaluate.add_argument("--contact-socket", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--diagnostics", type=Path, required=True)
    evaluate.add_argument("--port", type=int, required=True)
    launch = sub.add_parser("orchestrate")
    launch.add_argument("--max-workers", type=int, choices=(1, 2, 3, 4), default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "contact-service":
        contact_service(args.artifact, args.socket)
    elif args.command == "evaluate-model":
        evaluate_model(args.model, args.training_seed, args.reset_seed, args.attempt, args.contact_socket, args.output, args.diagnostics, args.port)
    else:
        orchestrate(args.max_workers)


if __name__ == "__main__":
    main()
