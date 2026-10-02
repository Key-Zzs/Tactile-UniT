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
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
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
AMENDMENT = ARTIFACTS / "runtime_amendment_cross_device_publish.json"
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
    with temporary.open("w", encoding="utf-8") as output:
        output.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def publish_file(source: Path, destination: Path) -> dict[str, Any]:
    """Durably publish a file across mounts without moving the source.

    The copy is written and fsynced inside the destination directory, then
    renamed atomically on that same filesystem.  Keeping the source preserves
    the first-complete-attempt evidence required by the retry contract.
    """

    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.publish-{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    source_sha = sha256_file(source)
    try:
        with source.open("rb") as input_file, temporary.open("xb") as output_file:
            shutil.copyfileobj(input_file, output_file, length=8 * 1024 * 1024)
            output_file.flush()
            os.fsync(output_file.fileno())
        copied_sha = sha256_file(temporary)
        if copied_sha != source_sha or temporary.stat().st_size != source.stat().st_size:
            raise RuntimeError("cross-filesystem publish copy verification failed")
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "source": str(source),
        "destination": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": source_sha,
        "source_preserved": source.is_file(),
        "destination_fsynced_before_atomic_rename": True,
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def job_key(model: str, training_seed: int, reset_seed: int) -> str:
    return f"r{reset_seed}_{model.lower()}_s{training_seed}"


def raw_path(model: str, training_seed: int, reset_seed: int) -> Path:
    return RAW / f"reset_seed_{reset_seed}" / model.lower() / f"train_seed_{training_seed}.json"


def attempt_paths(model: str, training_seed: int, reset_seed: int, attempt: int):
    key = job_key(model, training_seed, reset_seed)
    return SimpleNamespace(
        ARTIFACTS=ARTIFACTS / "final_attempts" / key / f"attempt_{attempt}",
        LOGS=LOGS / key / f"attempt_{attempt}",
        CACHE=CACHE / key / f"attempt_{attempt}",
        TMP=TMP / key / f"attempt_{attempt}",
    )


def contact_service(artifact: Path, socket_path: Path) -> None:
    from scripts.simulation import run_s4_3_pi2u_eval as runtime

    runtime.contact_service(artifact, socket_path)


def evaluate_model(model: str, training_seed: int, reset_seed: int, attempt: int, socket_path: Path, output: Path, diagnostics: Path, port: int) -> None:
    """Parameterize the accepted evaluator without shared mutable module paths."""

    from gr00t.simulation.s4_3_pi1 import TactileUnitMode
    from scripts.simulation import evaluate_s4_3_pi1d_augmented as frozen
    from scripts.simulation import run_s4_3_pi2u_eval as accepted

    paths = attempt_paths(model, training_seed, reset_seed, attempt)
    generated = paths.ARTIFACTS / f"{model.lower()}_raw_rollouts.json"
    worker_artifacts = paths.TMP / "worker_artifacts"
    temporary_artifact = worker_artifacts / f"pi1d_{model.lower()}_eval.json"
    if any(path.exists() for path in (output, diagnostics, generated, worker_artifacts)):
        raise SystemExit("refusing to overwrite evaluator attempt state")
    frozen.MODEL_ID = model
    frozen.MODE = TactileUnitMode(RUNTIME_MODES[model])
    frozen.DIAGNOSTICS_JSONL = diagnostics
    frozen.OUTPUT_ROOT = output
    frozen.CONTACT_SOCKET = socket_path
    frozen.EXPECTED_EPISODES = 50
    frozen.EVALUATOR_SEED = reset_seed
    worker_artifacts.mkdir(parents=True, exist_ok=False)
    frozen.ARTIFACTS = worker_artifacts
    diagnostics.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(DEXJOCO / "dexjoco"))
    from dexjoco_openpi_client import eval_dexjoco_openpi as official
    from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv

    official.DexJoCoOpenPIEnv = frozen.build_augmented_environment(DexJoCoOpenPIEnv)
    official.inference_process = frozen.instrumented_inference_process
    accepted.install_cross_episode_action_quarantine(official, diagnostics, model)
    official.main(
        config=DEXJOCO / "configs/rand_obj/pinch_tongs.yaml",
        seed=reset_seed,
        rand_full=False,
        randomize_dynamics=False,
        port=port,
        host="127.0.0.1",
        output=output,
        render_mode="rgb_array",
        replan_ratio=0.8,
        episodes=50,
        pad_state_dim46=False,
        record_pressed_digits=False,
    )
    if not temporary_artifact.is_file():
        raise ContractError(f"accepted evaluator did not emit {temporary_artifact}")
    publish_file(temporary_artifact, generated)
    payload = read_json(generated)
    if payload.get("status") != "PASS" or payload.get("episodes") != 50:
        raise ContractError("accepted evaluator integrity gate failed")


def checkpoint_row(freeze: dict[str, Any], model: str, seed: int) -> dict[str, Any]:
    return next(row for row in freeze["checkpoints"] if row["model"] == model and int(row["training_seed"]) == seed)


def run_attempt(freeze: dict[str, Any], model: str, training_seed: int, reset_seed: int, gpu: int, port: int, attempt: int) -> dict[str, Any]:
    from scripts.simulation import run_s4_3_pi2u_eval as runtime

    paths = attempt_paths(model, training_seed, reset_seed, attempt)
    key = job_key(model, training_seed, reset_seed)
    socket_path = TMP / key / f"a{attempt}.sock"
    output = paths.CACHE / "environment"
    diagnostics = paths.TMP / "inference.jsonl"
    contact_artifact = paths.TMP / "contact_state_service.json"
    contact_log = paths.LOGS / "contact.log"
    server_log = paths.LOGS / "server.log"
    client_log = paths.LOGS / "client.log"
    canonical = raw_path(model, training_seed, reset_seed)
    if canonical.exists() or paths.ARTIFACTS.exists() or paths.LOGS.exists() or paths.CACHE.exists() or paths.TMP.exists():
        raise RuntimeError(f"refusing to overwrite {key} attempt {attempt}")
    for path in (paths.ARTIFACTS, paths.LOGS, paths.CACHE, paths.TMP):
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
        generated = paths.ARTIFACTS / f"{model.lower()}_raw_rollouts.json"
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


def validate_amendment(freeze: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not AMENDMENT.is_file():
        raise SystemExit("resume amendment is missing")
    amendment = read_json(AMENDMENT)
    if amendment.get("status") != "PASS" or amendment.get("performance_seen") is not False:
        raise SystemExit("resume amendment is not eligible")
    if amendment.get("pre_final_freeze_sha256") != sha256_file(PRE_FREEZE):
        raise SystemExit("resume amendment does not bind the current pre-FINAL freeze")
    current_source = sha256_file(Path(__file__).resolve())
    if amendment.get("amended_evaluate_sha256") != current_source:
        raise SystemExit("amended evaluator source drifted")
    recovery_script = ROOT / "scripts/simulation/pi2b_policy/recover_cross_device_publish.py"
    if amendment.get("recovery_script_sha256") != sha256_file(recovery_script):
        raise SystemExit("recovery script drifted")
    unchanged = {
        relative: expected
        for relative, expected in freeze["sources_sha256"].items()
        if relative != "scripts/simulation/pi2b_policy/evaluate.py"
    }
    if any(sha256_file(ROOT / relative) != expected for relative, expected in unchanged.items()):
        raise SystemExit("a non-amended frozen source drifted")
    workers = amendment.get("recovered_workers", [])
    expected_jobs = {(str(row["model"]), int(row["training_seed"]), int(row["reset_seed"])) for row in workers}
    if len(workers) != 3 or expected_jobs != {
        ("B0", 42, 16), ("B_VA27", 42, 16), ("B1", 42, 16)
    }:
        raise SystemExit("resume amendment recovered-job set is not exact")
    for row in workers:
        path = raw_path(str(row["model"]), int(row["training_seed"]), int(row["reset_seed"]))
        if not path.is_file() or sha256_file(path) != row["raw_sha256"]:
            raise SystemExit(f"recovered canonical raw drifted: {path}")
    return amendment, workers


def orchestrate(max_workers: int, *, resume_amendment: bool = False) -> None:
    workspace = Workspace.load(ROOT)
    required_executables = (OPENPI_PYTHON, UNIT_PYTHON, EVAL_PYTHON)
    if any(not path.is_file() for path in required_executables):
        raise SystemExit(f"required runtime executable missing: {required_executables}")
    if not (DEXJOCO / "dexjoco/dexjoco_openpi_client/eval_dexjoco_openpi.py").is_file() or not (DEXJOCO / "configs/rand_obj/pinch_tongs.yaml").is_file():
        raise SystemExit(f"audited DexJoCo source missing: {DEXJOCO}")
    freeze = read_json(PRE_FREEZE)
    if freeze.get("status") != "PASS" or freeze.get("cohort_performance_seen") is not False:
        raise SystemExit("pre-FINAL freeze is not eligible")
    protected = (RAW, EXECUTION, COMPLETENESS, PROGRESS, LOGS, CACHE, TMP)
    if not resume_amendment and any(path.exists() for path in protected):
        raise SystemExit("refusing to overwrite or resume canonical FINAL state")
    workers: list[dict[str, Any]] = []
    amendment = None
    if resume_amendment:
        if any(path.exists() for path in (EXECUTION, COMPLETENESS, PROGRESS)):
            raise SystemExit("resume refuses final/progress outputs")
        amendment, workers = validate_amendment(freeze)
    elif any(sha256_file(ROOT / relative) != expected for relative, expected in freeze["sources_sha256"].items()):
        raise SystemExit("frozen source drifted")
    if any(sha256_file(Path(path)) != expected for path, expected in freeze["external_sources_sha256"].items()):
        raise SystemExit("frozen external source drifted")
    if teacher_has_pending_request(workspace):
        raise SystemExit("teacher has a pending heavy request")
    paths = coordination_paths(workspace)
    barrier = open_lock(paths["runtime_barrier"], shared=False, nonblocking=False)
    gpu_locks = []
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
            path.mkdir(parents=True, exist_ok=resume_amendment)
        update_coordination(workspace, "EVAL_RUNNING", supervisor_pid=os.getpid(), gpus=selected, total_blocks=60, total_rollouts=3000, recovered_blocks=len(workers), infrastructure_amendment=resume_amendment)
        completed_jobs = {
            (str(row["model"]), int(row["training_seed"]), int(row["reset_seed"]))
            for row in workers
        }
        pending = [
            (index, job) for index, job in enumerate(JOBS) if job not in completed_jobs
        ]
        recovered_count = len(workers)
        atomic_json(PROGRESS, {"schema": "tactile3d-unit.s4-3-pi2b-policy-progress.v1", "status": "RUNNING", "updated_at_utc": now(), "completed_blocks": recovered_count, "total_blocks": 60, "completed_rollouts": recovered_count * 50, "total_rollouts": 3000, "elapsed_seconds": 0.0, "rollouts_per_hour": None, "eta_seconds": None, "gpus": selected, "performance_values_read": False, "recovered_first_complete_blocks": recovered_count})
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
                    new_completed = completed - recovered_count
                    atomic_json(PROGRESS, {"schema": "tactile3d-unit.s4-3-pi2b-policy-progress.v1", "status": "RUNNING", "updated_at_utc": now(), "completed_blocks": completed, "total_blocks": 60, "completed_rollouts": completed * 50, "total_rollouts": 3000, "elapsed_seconds": elapsed, "rollouts_per_hour": new_completed * 50 / elapsed * 3600, "eta_seconds": elapsed / new_completed * (60 - completed), "gpus": selected, "performance_values_read": False, "recovered_first_complete_blocks": recovered_count})
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
    launch.add_argument("--resume-amendment", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "contact-service":
        contact_service(args.artifact, args.socket)
    elif args.command == "evaluate-model":
        evaluate_model(args.model, args.training_seed, args.reset_seed, args.attempt, args.contact_socket, args.output, args.diagnostics, args.port)
    else:
        orchestrate(args.max_workers, resume_amendment=args.resume_amendment)


if __name__ == "__main__":
    main()
