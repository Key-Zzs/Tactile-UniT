#!/usr/bin/env python3
"""Run the frozen PI2N 3x3x10 production-runtime variability audit."""

from __future__ import annotations

import argparse
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
DEXJOCO = ROOT / "third_party/dexjoco"
PROTOCOL = ROOT / "configs/simulation/s4_3_pi2n_runtime_audit_protocol.json"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2n"
RAW = ARTIFACTS / "runtime_audit_raw"
RESULT = ARTIFACTS / "runtime_randomness_audit.json"
RUNTIME_CONTRACT = ARTIFACTS / "runtime_protocol.json"
RESET_MANIFEST = ARTIFACTS / "runtime_reset_manifest.json"
PROGRESS = ARTIFACTS / "runtime_audit_progress.json"
LOGS = ROOT / ".local/logs/simulation/s4_3_pi2n/runtime_audit"
CACHE = ROOT / ".local/cache/simulation/s4_3_pi2n/runtime_audit"
TMP = ROOT / ".local/tmp/s43n_runtime"
CONDA_ROOT = Path(sys.executable).resolve().parents[3]
OPENPI_PYTHON = CONDA_ROOT / "envs/openpi/bin/python"
UNIT_PYTHON = CONDA_ROOT / "envs/unit/bin/python"
EVAL_PYTHON = ROOT / ".local/external/s4_3_pi0/eval-venv/bin/python"
MODELS = ("B1", "B_HVA", "B2")
RUNTIME_MODES = {model: "CONTACT_STATE_TOKENS" for model in MODELS}
EPISODES = 10
EVALUATOR_SEED = 8
CURRENT_MODEL = ""


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_payload_sha256(payload: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(payload):
        digest.update(key.encode())
        digest.update(b"\0")
        value = payload[key]
        if isinstance(value, str):
            digest.update(b"str\0")
            digest.update(value.encode())
        else:
            array = np.ascontiguousarray(np.asarray(value))
            digest.update(str(array.dtype).encode())
            digest.update(b"\0")
            digest.update(str(array.shape).encode())
            digest.update(b"\0")
            digest.update(array.tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def action_sha256(action: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(action, dtype=np.float32).tobytes()).hexdigest()


def instrumented_inference_process(
    obs_queue,
    action_queue,
    stop_event,
    port: int,
    inferencing_event,
    seed: int,
    host: str,
):
    """Official worker semantics plus fixed-input/output identity telemetry."""

    from dexjoco_openpi_client import eval_dexjoco_openpi as official
    from openpi_client import websocket_client_policy
    from scripts.simulation.evaluate_s4_3_pi1d_augmented import append_jsonl

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    official._set_seed(seed)
    client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    request_index = 0
    while not stop_event.is_set():
        observation = official.get_latest(obs_queue)
        if observation is None:
            stop_event.wait(0.01)
            continue
        payload = dict(observation.obs)
        episode_index = int(payload.pop("_pi1_episode_index"))
        reset_identity = str(payload.pop("_pi1_reset_identity"))
        forbidden = sorted(
            set(payload)
            & {
                "contact_shared_target",
                "physical_aux_valid",
                "va_shared_target",
                "va_aux_valid",
                "vac_vision_target",
                "vac_aux_valid",
            }
        )
        try:
            input_sha256 = canonical_payload_sha256(payload)
            result = client.infer(payload)
            action_chunk = np.asarray(result["actions"], dtype=np.float32)
            record = {
                "type": "action_chunk",
                "model": CURRENT_MODEL,
                "request_index": request_index,
                "episode_index": episode_index,
                "reset_identity": reset_identity,
                "observation_timestamp": int(observation.timestamp),
                "policy_input_sha256": input_sha256,
                "action_chunk_sha256": action_sha256(action_chunk),
                "shape": list(action_chunk.shape),
                "finite": bool(np.isfinite(action_chunk).all()),
                "minimum": float(action_chunk.min()),
                "maximum": float(action_chunk.max()),
                "mean_l2": float(np.linalg.norm(action_chunk, axis=-1).mean()),
                "contact_state_sent": "contact_state" in payload,
                "contact_state_sha256": (
                    hashlib.sha256(
                        np.ascontiguousarray(payload["contact_state"], dtype=np.float32).tobytes()
                    ).hexdigest()
                    if "contact_state" in payload
                    else None
                ),
                "training_only_fields_sent": forbidden,
                "policy_timing": result.get("policy_timing"),
                "server_timing": result.get("server_timing"),
            }
            from scripts.simulation import evaluate_s4_3_pi1d_augmented as augmented

            append_jsonl(augmented.DIAGNOSTICS_JSONL, record)
            action_queue.put(official.ActionChunk(action=action_chunk, timestamp=observation.timestamp))
            inferencing_event.clear()
            request_index += 1
        except Exception as error:
            from scripts.simulation import evaluate_s4_3_pi1d_augmented as augmented

            append_jsonl(
                augmented.DIAGNOSTICS_JSONL,
                {
                    "type": "server_client_error",
                    "model": CURRENT_MODEL,
                    "request_index": request_index,
                    "episode_index": episode_index,
                    "observation_timestamp": int(observation.timestamp),
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise


def configure_runtime(artifact_root: Path, log_root: Path, cache_root: Path, tmp_root: Path):
    from scripts.simulation import run_s4_3_pi2u_eval as runtime

    runtime.ARTIFACTS = artifact_root
    runtime.LOGS = log_root
    runtime.CACHE = cache_root
    runtime.TMP = tmp_root
    runtime.MODELS = MODELS
    runtime.RUNTIME_MODES = RUNTIME_MODES
    runtime.EPISODES = EPISODES
    runtime.EVALUATOR_SEED = EVALUATOR_SEED
    return runtime


def evaluate_model(
    model: str,
    replicate: int,
    socket_path: Path,
    output: Path,
    diagnostics: Path,
    port: int,
) -> None:
    global CURRENT_MODEL

    artifact_root = RAW / f"replicate_{replicate}"
    log_root = LOGS / f"replicate_{replicate}"
    cache_root = CACHE / f"replicate_{replicate}"
    tmp_root = TMP / f"replicate_{replicate}"
    runtime = configure_runtime(artifact_root, log_root, cache_root, tmp_root)
    from scripts.simulation import evaluate_s4_3_pi1d_augmented as augmented

    CURRENT_MODEL = model
    augmented.instrumented_inference_process = instrumented_inference_process
    runtime.evaluate_model(model, socket_path, output, diagnostics, port)


def run_one(model: str, replicate: int, gpu: int, port: int) -> dict[str, Any]:
    artifact_root = RAW / f"replicate_{replicate}"
    log_root = LOGS / f"replicate_{replicate}"
    cache_root = CACHE / f"replicate_{replicate}"
    tmp_root = TMP / f"replicate_{replicate}"
    runtime = configure_runtime(artifact_root, log_root, cache_root, tmp_root)
    lower = model.lower()
    socket_path = tmp_root / f"{lower}_contact_state.sock"
    output = cache_root / lower
    diagnostics = tmp_root / f"{lower}_inference.jsonl"
    contact_artifact = tmp_root / f"{lower}_contact_state_service.json"
    contact_log = log_root / f"{lower}_contact_state_service.log"
    server_log = log_root / f"{lower}_server.log"
    client_log = log_root / f"{lower}_client.log"
    protected = (
        socket_path,
        output,
        diagnostics,
        contact_artifact,
        artifact_root / f"{lower}_raw_rollouts.json",
    )
    if any(path.exists() for path in protected):
        raise RuntimeError(f"refusing to overwrite runtime audit output for replicate={replicate} model={model}")
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
        runtime.wait_for_log(contact, contact_log, "CONTACT_STATE_SERVICE_READY", 120)
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
        runtime.wait_for_log(policy, server_log, f"PI2M_POLICY_SERVER_READY model={model}", 300)
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
                "--replicate",
                str(replicate),
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
            raise RuntimeError(f"replicate={replicate} model={model} evaluator exited {result.returncode}")
        raw = json.loads((artifact_root / f"{lower}_raw_rollouts.json").read_text())
        return {
            "model": model,
            "replicate": replicate,
            "physical_gpu": gpu,
            "port": port,
            "status": raw["status"],
            "episodes": raw["episodes"],
            "started_unix": started,
            "elapsed_seconds": time.time() - started,
            "runtime_mode": RUNTIME_MODES[model],
            "fresh_policy_server": True,
            "infrastructure_retries": 0,
        }
    finally:
        runtime.stop_process(policy)
        runtime.stop_process(contact)
        for handle in handles:
            handle.close()


def read_diagnostics(replicate: int, model: str) -> list[dict[str, Any]]:
    path = TMP / f"replicate_{replicate}/{model.lower()}_inference.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit(protocol: dict[str, Any], workers: list[dict[str, Any]], snapshots) -> tuple[dict, dict, dict]:
    raw = {
        (replicate, model): json.loads(
            (RAW / f"replicate_{replicate}/{model.lower()}_raw_rollouts.json").read_text()
        )
        for replicate in range(1, 4)
        for model in MODELS
    }
    diagnostics = {
        key: read_diagnostics(*key) for key in raw
    }
    reset_sequences = {
        key: [row["reset_identity"] for row in payload["episode_results"]]
        for key, payload in raw.items()
    }
    reference_resets = reset_sequences[(1, "B1")]
    reset_equal = all(value == reference_resets for value in reset_sequences.values())
    first_chunks = {}
    for key, rows in diagnostics.items():
        chunks = [row for row in rows if row.get("type") == "action_chunk"]
        first_chunks[key] = next(
            (row for row in chunks if row["episode_index"] == 0 and row["request_index"] == 0),
            None,
        )
    fixed_opportunity = all(value is not None for value in first_chunks.values())
    first_input_hashes = {
        value["policy_input_sha256"] for value in first_chunks.values() if value is not None
    }
    fixed_input_equal = fixed_opportunity and len(first_input_hashes) == 1
    action_repeat_equal = {
        model: len(
            {
                first_chunks[(replicate, model)]["action_chunk_sha256"]
                for replicate in range(1, 4)
                if first_chunks[(replicate, model)] is not None
            }
        )
        == 1
        for model in MODELS
    }
    raw_pass = {key: payload.get("status") == "PASS" for key, payload in raw.items()}
    chunks_valid = all(
        row.get("finite") is True
        and row.get("shape") == [30, 22]
        and not row.get("training_only_fields_sent")
        and row.get("contact_state_sent") is True
        for rows in diagnostics.values()
        for row in rows
        if row.get("type") == "action_chunk"
    )
    stale_discards = {
        f"replicate_{replicate}/{model}": sum(
            row.get("type") == "stale_cross_episode_action_discarded"
            for row in diagnostics[(replicate, model)]
        )
        for replicate in range(1, 4)
        for model in MODELS
    }
    errors = {
        f"replicate_{replicate}/{model}": sum(
            row.get("type") == "server_client_error" for row in diagnostics[(replicate, model)]
        )
        for replicate in range(1, 4)
        for model in MODELS
    }
    per_model_disagreement = {}
    for model in MODELS:
        outcomes = [
            [bool(row["success"]) for row in raw[(replicate, model)]["episode_results"]]
            for replicate in range(1, 4)
        ]
        disagreements = [
            index for index in range(EPISODES) if len({outcomes[replicate][index] for replicate in range(3)}) > 1
        ]
        per_model_disagreement[model] = {
            "successes_by_replicate": [sum(row) for row in outcomes],
            "outcome_disagreement_reset_indices": disagreements,
            "outcome_disagreement_count": len(disagreements),
            "pairwise_disagreement_counts": {
                "1_vs_2": sum(a != b for a, b in zip(outcomes[0], outcomes[1], strict=True)),
                "1_vs_3": sum(a != b for a, b in zip(outcomes[0], outcomes[2], strict=True)),
                "2_vs_3": sum(a != b for a, b in zip(outcomes[1], outcomes[2], strict=True)),
            },
        }
    structural_gates = {
        "ordered_reset_identity_equal_all_nine_runs": reset_equal,
        "ten_unique_resets": len(reference_resets) == EPISODES and len(set(reference_resets)) == EPISODES,
        "fixed_observation_input_equal_all_models_replicates": fixed_input_equal,
        "finite_30x22_chunks_and_no_target_leakage": chunks_valid,
    }
    fairness_gates = {
        "all_nine_raw_runtime_audits_PASS": all(raw_pass.values()),
        "fixed_observation_opportunity_all_nine_runs": fixed_opportunity,
        "fixed_observation_action_repeatable_within_each_model": all(action_repeat_equal.values()),
        "no_server_client_errors": sum(errors.values()) == 0,
        "same_episode_timeout_replan_runtime_contract": all(
            payload.get("episodes") == EPISODES
            and payload.get("replan_ratio") == 0.8
            and payload.get("mode") == "CONTACT_STATE_TOKENS"
            for payload in raw.values()
        ),
    }
    if not all(structural_gates.values()):
        classification = "R_A_STRUCTURAL_ERROR"
    elif not all(fairness_gates.values()):
        classification = "R_C_UNEXPLAINED_MODEL_UNFAIRNESS"
    else:
        classification = "R_B_LEGAL_RANDOMNESS_COMPARABLE"
    scientific_authorized = classification == "R_B_LEGAL_RANDOMNESS_COMPARABLE"
    reset_manifest = {
        "schema": "tactile3d-unit.s4-3-pi2n-runtime-reset-manifest.v1",
        "status": "PASS" if reset_equal else "FAIL",
        "seed_namespace": "PI2N_RUNTIME_DIAG_DEVELOPMENT_EXPOSED",
        "task": "pinch_tongs",
        "regime": "rand_obj",
        "evaluator_seed": EVALUATOR_SEED,
        "ordered_resets": [
            {
                "episode_index": index,
                "reset_identity": identity,
                "identity_definition": "sha256(mjSTATE_INTEGRATION float64 bytes + official processed state float64 bytes)",
            }
            for index, identity in enumerate(reference_resets)
        ],
        "sequence_sha256": hashlib.sha256("\n".join(reference_resets).encode()).hexdigest(),
        "all_model_replicate_sequences_equal": reset_equal,
        "formal_or_development_candidate_selection_use": False,
    }
    contract = {
        "schema": "tactile3d-unit.s4-3-pi2n-runtime-protocol.v1",
        "status": "PASS" if scientific_authorized else "BLOCKED",
        "classification": classification,
        "scientific_rollouts_authorized": scientific_authorized,
        "official_async_evaluator_retained": True,
        "synchronous_replacement": False,
        "policy_sampling": protocol["randomness"]["policy_sampling"],
        "per_episode_policy_rng_reset": False,
        "remaining_variability": protocol["randomness"]["remaining_variability"],
        "reset_contract": "full ordered reset identity must match across every compared model",
        "retry_contract": protocol["retry"],
        "native_success_semantics_modified": False,
        "development_exposure": "RUNTIME_DIAG outcomes excluded from DEV candidate selection and FINAL statistics",
    }
    result = {
        "schema": "tactile3d-unit.s4-3-pi2n-runtime-randomness-audit.v1",
        "status": "PASS" if scientific_authorized else "FAIL",
        "classification": classification,
        "protocol_sha256": sha256_file(PROTOCOL),
        "workers": workers,
        "gpu_snapshots": snapshots,
        "structural_gates": {name: "PASS" if value else "FAIL" for name, value in structural_gates.items()},
        "fairness_gates": {name: "PASS" if value else "FAIL" for name, value in fairness_gates.items()},
        "fixed_observation_fixture": {
            "input_sha256": next(iter(first_input_hashes)) if len(first_input_hashes) == 1 else sorted(first_input_hashes),
            "action_repeatable_within_model": action_repeat_equal,
            "first_action_sha256_by_model": {
                model: first_chunks[(1, model)]["action_chunk_sha256"] if first_chunks[(1, model)] else None
                for model in MODELS
            },
        },
        "stale_cross_episode_action_discards_quarantined": stale_discards,
        "server_client_errors": errors,
        "per_model_outcome_disagreement": per_model_disagreement,
        "outcome_interpretation": "disagreement count only; not a success-rate effect and not used for policy selection",
        "unrelated_gpu_processes_killed_or_preempted": False,
        "candidate_policy_results_read": False,
        "formal_results_read": False,
    }
    return result, contract, reset_manifest


def orchestrate(gpu: int) -> None:
    protocol = json.loads(PROTOCOL.read_text())
    if protocol.get("status") != "FROZEN_BEFORE_RUNTIME_AUDIT":
        raise SystemExit("runtime audit protocol is not frozen")
    protected = (RAW, RESULT, RUNTIME_CONTRACT, RESET_MANIFEST, PROGRESS, LOGS, CACHE, TMP)
    if any(path.exists() for path in protected):
        raise SystemExit("refusing to overwrite or resume a PI2N runtime audit output")
    from scripts.simulation import run_s4_3_pi2u_eval as runtime

    snapshot1 = runtime.gpu_snapshot()
    time.sleep(2)
    snapshot2 = runtime.gpu_snapshot()
    if not runtime.gpu_is_idle(gpu, snapshot1) or not runtime.gpu_is_idle(gpu, snapshot2):
        raise SystemExit("selected runtime-audit GPU is not genuinely idle in two snapshots")
    lock = runtime.acquire_gpu_lock(gpu)
    try:
        snapshot3 = runtime.gpu_snapshot()
        if not runtime.gpu_is_idle(gpu, snapshot3):
            raise SystemExit("selected runtime-audit GPU became busy after locking")
        for path in (RAW, LOGS, CACHE, TMP):
            path.mkdir(parents=True, exist_ok=False)
        execution_order = {
            int(name.rsplit("_", 1)[1]): models
            for name, models in protocol["cohort"]["execution_order"].items()
        }
        workers = []
        total = EPISODES * len(MODELS) * 3
        completed = 0
        for replicate in range(1, 4):
            for offset, model in enumerate(execution_order[replicate]):
                for path in (
                    RAW / f"replicate_{replicate}",
                    LOGS / f"replicate_{replicate}",
                    CACHE / f"replicate_{replicate}",
                    TMP / f"replicate_{replicate}",
                ):
                    path.mkdir(parents=True, exist_ok=True)
                atomic_json(
                    PROGRESS,
                    {
                        "schema": "tactile3d-unit.s4-3-pi2n-runtime-audit-progress.v1",
                        "state": "RUNNING",
                        "current_model": model,
                        "current_replicate": replicate,
                        "completed_rollouts": completed,
                        "total_rollouts": total,
                        "completed_jobs": len(workers),
                        "total_jobs": 9,
                        "updated_unix": time.time(),
                    },
                )
                row = run_one(model, replicate, gpu, 8460 + (replicate - 1) * 10 + offset)
                workers.append(row)
                completed += EPISODES
        snapshots = [snapshot1, snapshot2, snapshot3, runtime.gpu_snapshot()]
        result, contract, reset_manifest = audit(protocol, workers, snapshots)
        atomic_json(RESULT, result)
        atomic_json(RUNTIME_CONTRACT, contract)
        atomic_json(RESET_MANIFEST, reset_manifest)
        atomic_json(
            PROGRESS,
            {
                "schema": "tactile3d-unit.s4-3-pi2n-runtime-audit-progress.v1",
                "state": "DONE" if result["status"] == "PASS" else "FAILED",
                "completed_rollouts": completed,
                "total_rollouts": total,
                "completed_jobs": len(workers),
                "total_jobs": 9,
                "classification": result["classification"],
                "updated_unix": time.time(),
            },
        )
        print(json.dumps({"status": result["status"], "classification": result["classification"]}, sort_keys=True))
        if result["status"] != "PASS":
            raise SystemExit("PI2N_RUNTIME_AUDIT_BLOCKS_SCIENTIFIC_ROLLOUTS")
    finally:
        lock.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate-model")
    evaluate.add_argument("--model", choices=MODELS, required=True)
    evaluate.add_argument("--replicate", type=int, choices=(1, 2, 3), required=True)
    evaluate.add_argument("--contact-socket", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--diagnostics", type=Path, required=True)
    evaluate.add_argument("--port", type=int, required=True)
    launch = sub.add_parser("orchestrate")
    launch.add_argument("--gpu", type=int, choices=(0, 1, 2, 3), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "evaluate-model":
        evaluate_model(
            args.model,
            args.replicate,
            args.contact_socket,
            args.output,
            args.diagnostics,
            args.port,
        )
    else:
        orchestrate(args.gpu)


if __name__ == "__main__":
    main()
