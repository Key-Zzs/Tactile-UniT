#!/usr/bin/env python3
"""Run one disjoint non-scientific production-path smoke for each PI2M policy."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEXJOCO = ROOT / "third_party/dexjoco"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2m"
RAW = ARTIFACTS / "production_smoke_raw"
LOGS = ROOT / ".local/logs/simulation/s4_3_pi2m/production_smoke"
CACHE = ROOT / ".local/cache/simulation/s4_3_pi2m/production_smoke"
TMP = ROOT / ".local/tmp/s43m_smoke"
RESULT = ARTIFACTS / "production_smoke.json"
MODELS = ("B1", "B_HVA", "B2")
RUNTIME_MODES = {model: "CONTACT_STATE_TOKENS" for model in MODELS}
SMOKE_SEED = 700_042
EPISODES = 1
CHECKPOINTS = {
    "B1": ROOT
    / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/"
    "s43_pi1b_contact_tokens_seed42/29999",
    "B_HVA": ROOT
    / ".local/experiments/simulation/s4_3_pi2m/bhva/pinch_tongs/"
    "s43_pi2m_bhva_seed42/29999",
    "B2": ROOT
    / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/"
    "s43_pi1c_contact_tokens_physical_aux_seed42/29999",
}


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


def configure_smoke():
    from scripts.simulation import run_s4_3_pi2m_eval as formal

    formal.ARTIFACTS = RAW
    formal.LOGS = LOGS
    formal.CACHE = CACHE
    formal.TMP = TMP
    formal.PRE_FREEZE = ARTIFACTS / "production_smoke_precondition.json"
    formal.MODELS = MODELS
    formal.RUNTIME_MODES = RUNTIME_MODES
    formal.EPISODES = EPISODES
    formal.EVALUATOR_SEED = SMOKE_SEED
    formal.configure_frozen_runtime()
    return formal


def run_one(model: str, gpu: int, port: int) -> dict[str, Any]:
    formal = configure_smoke()
    frozen = formal.configure_frozen_runtime()
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
        RAW / f"{lower}_raw_rollouts.json",
    )
    if any(path.exists() for path in protected):
        raise RuntimeError(f"refusing to overwrite PI2M smoke output for {model}")
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
                str(formal.UNIT_PYTHON),
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
                str(formal.OPENPI_PYTHON),
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
        frozen.wait_for_log(
            policy, server_log, f"PI2M_POLICY_SERVER_READY model={model}", 300
        )
        eval_env = gpu_env | {
            "MUJOCO_GL": "egl",
            "PYTHONPATH": f"{ROOT}:{DEXJOCO / 'dexjoco'}",
        }
        result = subprocess.run(
            [
                str(formal.EVAL_PYTHON),
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
            raise RuntimeError(f"{model} smoke evaluator exited {result.returncode}")
        raw = frozen.read_json(RAW / f"{lower}_raw_rollouts.json")
        return {
            "model": model,
            "physical_gpu": gpu,
            "port": port,
            "status": raw["status"],
            "episodes": raw["episodes"],
            "started_unix": started,
            "elapsed_seconds": time.time() - started,
            "runtime_mode": RUNTIME_MODES[model],
        }
    finally:
        frozen.stop_process(policy)
        frozen.stop_process(contact)
        for handle in handles:
            handle.close()


def audit(rows: list[dict[str, Any]], snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    raw = {
        model: json.loads((RAW / f"{model.lower()}_raw_rollouts.json").read_text())
        for model in MODELS
    }
    reset_ids = {
        model: payload["episode_results"][0]["reset_identity"]
        for model, payload in raw.items()
    }
    formal_resets = json.loads((ARTIFACTS / "fresh_reset_manifest.json").read_text())[
        "ordered_reset_identities"
    ]
    model_audits = {}
    for row in rows:
        model = row["model"]
        payload = raw[model]
        episode = payload["episode_results"][0]
        chunks = episode["action_chunks"]
        inference_ms = [
            float(chunk["policy_timing"]["infer_ms"])
            for chunk in chunks
            if chunk.get("policy_timing", {}).get("infer_ms") is not None
        ]
        norm_stats = json.loads(
            (CHECKPOINTS[model] / "assets/local_repo/norm_stats.json").read_text()
        )["norm_stats"]
        server_text = (LOGS / f"{model.lower()}_server.log").read_text(errors="replace")
        gates = {
            "raw_runtime_PASS": payload.get("status") == "PASS",
            "one_dedicated_episode": payload.get("episodes") == 1
            and len(payload.get("episode_results", [])) == 1,
            "runtime_contact_state_tokens": payload.get("mode") == "CONTACT_STATE_TOKENS"
            and payload.get("contact_state_sent_to_policy") is True,
            "server_cold_load_pass": f"PI2M_POLICY_COLD_LOAD_PASS model={model}" in server_text,
            "server_ready": f"PI2M_POLICY_SERVER_READY model={model}" in server_text,
            "state23": len(norm_stats["state"]["mean"]) == 23,
            "action22": len(norm_stats["actions"]["mean"]) == 22
            and episode["executed_action_stats"]["shape"][1] == 22,
            "action_horizon30": bool(chunks)
            and all(chunk["shape"] == [30, 22] for chunk in chunks),
            "actions_finite": episode["executed_action_stats"]["finite"]
            and all(chunk["finite"] for chunk in chunks),
            "replan_ratio_0p8": payload.get("replan_ratio") == 0.8,
            "tactile_history_26x30": episode["tactile_diagnostics"]["history_shape"]
            == [26, 30],
            "contact_state_256_finite_nonzero": episode["contact_state_diagnostics"]["shape"]
            == [256]
            and episode["contact_state_diagnostics"]["finite"] is True
            and episode["contact_state_diagnostics"]["queries"] > 0
            and episode["contact_state_diagnostics"]["max_l2"] > 0,
            "training_targets_never_sent": payload["gates"].get(
                "training_only_targets_never_sent"
            )
            == "PASS"
            and all(not chunk["training_only_fields_sent"] for chunk in chunks),
            "native_success_and_physics_unchanged": payload.get(
                "physics_action_success_reset_camera_prompt_modified"
            )
            is False,
            "no_server_client_errors": payload["gates"].get("no_server_client_errors")
            == "PASS",
            "inference_timing_recorded": bool(inference_ms)
            and all(math.isfinite(value) and value > 0 for value in inference_ms),
        }
        model_audits[model] = {
            "status": "PASS" if all(gates.values()) else "FAIL",
            "physical_gpu": row["physical_gpu"],
            "wall_seconds": row["elapsed_seconds"],
            "physical_simulation_seconds": episode["steps"] * 0.02,
            "environment_steps": episode["steps"],
            "action_chunks": len(chunks),
            "first_inference_ms": inference_ms[0] if inference_ms else None,
            "median_inference_ms_excluding_first": statistics.median(inference_ms[1:])
            if len(inference_ms) > 1
            else None,
            "reset_identity": episode["reset_identity"],
            "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        }
    shared_gates = {
        "three_models_exact": set(model_audits) == set(MODELS),
        "same_disjoint_reset_identity": len(set(reset_ids.values())) == 1
        and next(iter(reset_ids.values())) not in formal_resets,
        "dedicated_non_scientific_seed": all(
            payload.get("evaluator_seed") == SMOKE_SEED for payload in raw.values()
        )
        and SMOKE_SEED != 7,
        "formal_seed7_outputs_absent": not any(
            (ARTIFACTS / f"{model.lower()}_raw_rollouts.json").exists()
            for model in MODELS
        ),
        "exact_formal_sources_unchanged": all(
            sha256_file(ROOT / relative)
            == json.loads((ARTIFACTS / "evaluation_protocol_freeze.json").read_text())[
                "code_sha256"
            ][relative]
            for relative in (
                "scripts/simulation/run_s4_3_pi2m_eval.py",
                "scripts/simulation/serve_s4_3_pi2m_policy.py",
                "scripts/simulation/evaluate_s4_3_pi1d_augmented.py",
                "scripts/simulation/serve_s4_3_pi1_contact_state.py",
                "third_party/dexjoco/configs/rand_obj/pinch_tongs.yaml",
            )
        ),
        "all_model_smokes_PASS": all(
            value["status"] == "PASS" for value in model_audits.values()
        ),
    }
    return {
        "schema": "tactile3d-unit.s4-3-pi2m-production-smoke.v1",
        "status": "PASS" if all(shared_gates.values()) else "FAIL",
        "scientific_result": False,
        "seed_namespace": "PI2M_PRODUCTION_SMOKE",
        "evaluator_seed": SMOKE_SEED,
        "formal_evaluator_seed": 7,
        "models": model_audits,
        "shared_gates": {
            name: "PASS" if value else "FAIL" for name, value in shared_gates.items()
        },
        "gpu_snapshots": snapshots,
        "unrelated_gpu_processes_killed_or_preempted": False,
        "formal_performance_inspected": False,
        "raw_smoke_artifacts_retained_but_excluded_from_statistics": True,
    }


def orchestrate(gpus: list[int]) -> None:
    if RESULT.exists() or RAW.exists() or LOGS.exists() or CACHE.exists() or TMP.exists():
        raise SystemExit("refusing to overwrite PI2M production smoke")
    completion = json.loads((ARTIFACTS / "training_completion.json").read_text())
    manifest = json.loads((ARTIFACTS / "bhva_checkpoint_manifest.json").read_text())
    if completion.get("status") != "PASS" or manifest.get("status") != "PASS":
        raise SystemExit("PI2M B_HVA completion must PASS before production smoke")
    if not 1 <= len(gpus) <= 3 or gpus != sorted(set(gpus)):
        raise SystemExit("PI2M smoke requires one to three sorted distinct GPUs")
    formal = configure_smoke()
    frozen = formal.configure_frozen_runtime()
    snapshot1 = frozen.gpu_snapshot()
    time.sleep(2)
    snapshot2 = frozen.gpu_snapshot()
    if not all(
        frozen.gpu_is_idle(gpu, snapshot1) and frozen.gpu_is_idle(gpu, snapshot2)
        for gpu in gpus
    ):
        raise SystemExit("one or more selected PI2M smoke GPUs is not genuinely idle")
    locks = []
    try:
        locks = [frozen.acquire_gpu_lock(gpu) for gpu in gpus]
        snapshot3 = frozen.gpu_snapshot()
        if not all(frozen.gpu_is_idle(gpu, snapshot3) for gpu in gpus):
            raise SystemExit("a selected PI2M smoke GPU became busy after locking")
        RAW.mkdir(parents=True, exist_ok=False)
        LOGS.mkdir(parents=True, exist_ok=False)
        CACHE.mkdir(parents=True, exist_ok=False)
        TMP.mkdir(parents=True, exist_ok=False)
        assignments = list(zip(MODELS[: len(gpus)], gpus, range(8390, 8390 + len(gpus))))
        rows: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(assignments)) as executor:
            rows.extend(
                future.result()
                for future in [executor.submit(run_one, *item) for item in assignments]
            )
        for offset, model in enumerate(MODELS[len(gpus) :]):
            rows.append(run_one(model, gpus[offset % len(gpus)], 8400 + offset))
        payload = audit(rows, [snapshot1, snapshot2, snapshot3, frozen.gpu_snapshot()])
        atomic_json(RESULT, payload)
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "models": {
                        model: {
                            "wall_seconds": value["wall_seconds"],
                            "action_chunks": value["action_chunks"],
                        }
                        for model, value in payload["models"].items()
                    },
                },
                sort_keys=True,
            )
        )
        if payload["status"] != "PASS":
            raise SystemExit("PI2M_PRODUCTION_SMOKE_FAIL")
    finally:
        for handle in locks:
            handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
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
    formal = configure_smoke()
    if args.command == "evaluate-model":
        formal.evaluate_model(
            args.model, args.contact_socket, args.output, args.diagnostics, args.port
        )
    else:
        try:
            gpus = [int(value) for value in args.gpus.split(",")]
        except ValueError as error:
            raise SystemExit("--gpus must be comma-separated physical IDs") from error
        orchestrate(gpus)


if __name__ == "__main__":
    main()
