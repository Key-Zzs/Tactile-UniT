#!/usr/bin/env python3
"""Run the frozen six-policy PI2N FINAL cohort without interim interpretation."""

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


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEXJOCO = ROOT / "third_party/dexjoco"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2n"
PRE_FREEZE = ARTIFACTS / "pre_final_freeze.json"
RESET_MANIFEST = ARTIFACTS / "final_reset_manifest.json"
RAW = ARTIFACTS / "final_raw"
EXECUTION = ARTIFACTS / "final_gpu_execution.json"
COMPLETENESS = ARTIFACTS / "rollout_completeness.json"
LOGS = ROOT / ".local/logs/simulation/s4_3_pi2n/final"
CACHE = ROOT / ".local/cache/simulation/s4_3_pi2n/final"
TMP = ROOT / ".local/tmp/s43n_final"
CONDA_ROOT = Path(sys.executable).resolve().parents[3]
OPENPI_PYTHON = CONDA_ROOT / "envs/openpi/bin/python"
UNIT_PYTHON = CONDA_ROOT / "envs/unit/bin/python"
EVAL_PYTHON = ROOT / ".local/external/s4_3_pi0/eval-venv/bin/python"
MODELS = ("B0", "B_VA27", "B1", "B_HVA", "B2", "B_VAC_V")
RUNTIME_MODES = {
    "B0": "NONE",
    "B_VA27": "NONE",
    "B1": "CONTACT_STATE_TOKENS",
    "B_HVA": "CONTACT_STATE_TOKENS",
    "B2": "CONTACT_STATE_TOKENS",
    "B_VAC_V": "CONTACT_STATE_TOKENS",
}
SEED_BLOCKS = (12, 13, 14, 15)
EPISODES_PER_BLOCK = 50
EXECUTION_ORDER = {
    12: ("B0", "B_VA27", "B1", "B_HVA", "B2", "B_VAC_V"),
    13: ("B_VA27", "B1", "B_HVA", "B2", "B_VAC_V", "B0"),
    14: ("B1", "B_HVA", "B2", "B_VAC_V", "B0", "B_VA27"),
    15: ("B_HVA", "B2", "B_VAC_V", "B0", "B_VA27", "B1"),
}
XLA_FLAGS = (
    "--xla_gpu_deterministic_ops=true",
    "--xla_gpu_exclude_nondeterministic_ops=true",
    "--xla_gpu_autotune_level=0",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def block_paths(seed: int) -> tuple[Path, Path, Path, Path]:
    name = f"seed_{seed}"
    return RAW / name, LOGS / name, CACHE / name, TMP / name


def configure_runtime(seed: int):
    from scripts.simulation import run_s4_3_pi2u_eval as runtime

    artifact_root, log_root, cache_root, tmp_root = block_paths(seed)
    runtime.ARTIFACTS = artifact_root
    runtime.LOGS = log_root
    runtime.CACHE = cache_root
    runtime.TMP = tmp_root
    runtime.MODELS = MODELS
    runtime.RUNTIME_MODES = RUNTIME_MODES
    runtime.EPISODES = EPISODES_PER_BLOCK
    runtime.EVALUATOR_SEED = seed
    return runtime


def contact_service(artifact: Path, socket_path: Path) -> None:
    from scripts.simulation import run_s4_3_pi2u_eval as runtime

    runtime.contact_service(artifact, socket_path)


def evaluate_model(
    model: str,
    seed: int,
    socket_path: Path,
    output: Path,
    diagnostics: Path,
    port: int,
) -> None:
    runtime = configure_runtime(seed)
    runtime.evaluate_model(model, socket_path, output, diagnostics, port)


def frozen_inputs_unchanged(freeze: dict[str, Any]) -> bool:
    sources = all(
        sha256_file(ROOT / symbolic.removeprefix("$REPO_ROOT/")) == expected
        for symbolic, expected in freeze["sources_sha256"].items()
    )
    inputs = all(
        sha256_file(ROOT / row["path"].removeprefix("$REPO_ROOT/"))
        == row["sha256"]
        for row in freeze["frozen_input_files"].values()
    )
    return sources and inputs


def run_one(model: str, seed: int, gpu: int, port: int) -> dict[str, Any]:
    runtime = configure_runtime(seed)
    artifact_root, log_root, cache_root, tmp_root = block_paths(seed)
    lower = model.lower()
    socket_path = tmp_root / f"{lower}_contact_state.sock"
    output = cache_root / lower
    diagnostics = tmp_root / f"{lower}_inference.jsonl"
    contact_artifact = tmp_root / f"{lower}_contact_state_service.json"
    contact_log = log_root / f"{lower}_contact_state_service.log"
    server_log = log_root / f"{lower}_server.log"
    client_log = log_root / f"{lower}_client.log"
    raw_artifact = artifact_root / f"{lower}_raw_rollouts.json"
    protected = (
        socket_path,
        output,
        diagnostics,
        contact_artifact,
        raw_artifact,
        contact_log,
        server_log,
        client_log,
    )
    if any(path.exists() for path in protected):
        raise RuntimeError(
            f"refusing to overwrite PI2N FINAL output seed={seed} model={model}"
        )
    started = time.time()
    contact = policy = None
    handles = []
    try:
        contact_handle = contact_log.open("x")
        server_handle = server_log.open("x")
        client_handle = client_log.open("x")
        handles.extend((contact_handle, server_handle, client_handle))
        base_env = os.environ | {
            "PYTHONPATH": str(ROOT),
            "PYTHONUNBUFFERED": "1",
        }
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
        runtime.wait_for_log(
            contact, contact_log, "CONTACT_STATE_SERVICE_READY", 120
        )
        gpu_env = base_env | {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "XLA_FLAGS": " ".join(XLA_FLAGS),
        }
        policy = subprocess.Popen(
            [
                str(OPENPI_PYTHON),
                str(ROOT / "scripts/simulation/serve_s4_3_pi2n_policy.py"),
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
        runtime.wait_for_log(
            policy, server_log, f"PI2N_POLICY_SERVER_READY model={model}", 600
        )
        eval_env = gpu_env | {
            "MUJOCO_GL": "egl",
            "MUJOCO_EGL_DEVICE_ID": "0",
            "PYTHONPATH": f"{ROOT}:{DEXJOCO / 'dexjoco'}",
        }
        result = subprocess.run(
            [
                str(EVAL_PYTHON),
                str(Path(__file__).resolve()),
                "evaluate-model",
                "--model",
                model,
                "--seed",
                str(seed),
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
            raise RuntimeError(
                f"seed={seed} model={model} evaluator exited {result.returncode}"
            )
        raw = read_json(raw_artifact)
        return {
            "model": model,
            "seed": seed,
            "physical_gpu": gpu,
            "port": port,
            "status": raw["status"],
            "episodes": raw["episodes"],
            "started_unix": started,
            "elapsed_seconds": time.time() - started,
            "runtime_mode": RUNTIME_MODES[model],
            "fresh_policy_server": True,
            "fresh_contact_service": True,
            "infrastructure_retries": 0,
            "performance_interpreted_during_execution": False,
        }
    finally:
        runtime.stop_process(policy)
        runtime.stop_process(contact)
        for handle in handles:
            handle.close()


def expected_resets_by_seed(manifest: dict[str, Any]) -> dict[int, list[str]]:
    return {
        seed: [
            row["reset_identity"]
            for row in manifest["reset_specs"]
            if int(row["seed"]) == seed
        ]
        for seed in SEED_BLOCKS
    }


def audit_completeness(
    freeze: dict[str, Any], workers: list[dict[str, Any]], snapshots: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(RESET_MANIFEST)
    expected = expected_resets_by_seed(manifest)
    per_job_gates: dict[str, bool] = {}
    identities: dict[str, list[str]] = {model: [] for model in MODELS}
    raw_index: dict[str, dict[str, Any]] = {}
    for seed in SEED_BLOCKS:
        for model in MODELS:
            path = RAW / f"seed_{seed}" / f"{model.lower()}_raw_rollouts.json"
            payload = read_json(path)
            rows = payload.get("episode_results", [])
            observed = [row.get("reset_identity") for row in rows]
            should_receive_contact = RUNTIME_MODES[model] != "NONE"
            gate = bool(
                payload.get("status") == "PASS"
                and payload.get("model") == model
                and payload.get("mode") == RUNTIME_MODES[model]
                and payload.get("evaluator_seed") == seed
                and payload.get("episodes") == EPISODES_PER_BLOCK
                and len(rows) == EPISODES_PER_BLOCK
                and [row.get("episode_index") for row in rows]
                == list(range(EPISODES_PER_BLOCK))
                and observed == expected[seed]
                and len(set(observed)) == EPISODES_PER_BLOCK
                and payload.get("contact_state_sent_to_policy")
                is should_receive_contact
                and all(
                    value == "PASS" for value in payload.get("gates", {}).values()
                )
            )
            per_job_gates[f"seed_{seed}/{model}"] = gate
            identities[model].extend(observed)
            raw_index[f"seed_{seed}/{model}"] = {
                "path": "$REPO_ROOT/" + path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
                "episodes": len(rows),
            }
    reference = identities[MODELS[0]]
    reset_equal = all(identities[model] == reference for model in MODELS)
    expected_full = manifest["ordered_reset_identities"]
    gates = {
        "all_24_jobs_pass": all(per_job_gates.values())
        and len(per_job_gates) == len(MODELS) * len(SEED_BLOCKS),
        "all_models_exact_same_ordered_resets": reset_equal,
        "all_models_match_frozen_final_manifest": all(
            identities[model] == expected_full for model in MODELS
        ),
        "200_unique_resets_per_model": all(
            len(rows) == 200 and len(set(rows)) == 200
            for rows in identities.values()
        ),
        "exactly_1200_canonical_outcomes": sum(
            len(rows) for rows in identities.values()
        )
        == 1200,
        "worker_records_exact": len(workers) == 24
        and all(row.get("status") == "PASS" for row in workers)
        and {
            (int(row["seed"]), str(row["model"])) for row in workers
        }
        == {
            (seed, model) for seed in SEED_BLOCKS for model in MODELS
        },
        "vac_star_binding_unchanged": freeze.get("vac_star") == "B_VAC_V",
    }
    completeness = {
        "schema": "tactile3d-unit.s4-3-pi2n-rollout-completeness.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "created_at": now(),
        "models": list(MODELS),
        "vac_star": freeze["vac_star"],
        "seed_blocks": list(SEED_BLOCKS),
        "episodes_per_model": 200,
        "counts": {model: len(identities[model]) for model in MODELS},
        "total_canonical_outcomes": sum(
            len(rows) for rows in identities.values()
        ),
        "ordered_reset_sequence_sha256": hashlib.sha256(
            "\n".join(reference).encode()
        ).hexdigest(),
        "raw_artifacts": raw_index,
        "per_job_integrity_gates": {
            name: "PASS" if value else "FAIL"
            for name, value in per_job_gates.items()
        },
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "native_failures_and_timeouts_retained": True,
        "performance_interpreted_before_all_outcomes_complete": False,
        "statistics_computed": False,
    }
    execution = {
        "schema": "tactile3d-unit.s4-3-pi2n-final-gpu-execution.v1",
        "status": completeness["status"],
        "created_at": now(),
        "workers": workers,
        "snapshots": snapshots,
        "maximum_heavy_workers": len(
            {row["physical_gpu"] for row in workers}
        ),
        "wave_order": [
            {"seed": seed, "models": list(EXECUTION_ORDER[seed])}
            for seed in SEED_BLOCKS
        ],
        "unrelated_gpu_processes_killed_or_preempted": False,
        "performance_interpreted_during_execution": False,
    }
    return completeness, execution


def orchestrate(gpus: list[int]) -> None:
    freeze = read_json(PRE_FREEZE)
    if freeze.get("status") != "PASS":
        raise SystemExit("PI2N pre-FINAL freeze is not PASS")
    if freeze.get("formal_models") != list(MODELS):
        raise SystemExit("PI2N pre-FINAL formal model binding is not exact")
    if not 1 <= len(gpus) <= 4 or len(gpus) != len(set(gpus)):
        raise SystemExit("PI2N FINAL requires one to four distinct GPUs")
    if gpus != sorted(gpus) or any(gpu not in range(4) for gpu in gpus):
        raise SystemExit("PI2N FINAL GPUs must be sorted physical IDs in range 0-3")
    protected = (RAW, EXECUTION, COMPLETENESS, LOGS, CACHE, TMP)
    if any(path.exists() for path in protected):
        raise SystemExit("refusing to overwrite or resume canonical PI2N FINAL outcomes")
    if not frozen_inputs_unchanged(freeze):
        raise SystemExit("PI2N pre-FINAL frozen source or input file drifted")
    manifest = read_json(RESET_MANIFEST)
    if manifest.get("policy_performance_seen") is not False:
        raise SystemExit("PI2N FINAL performance-seen guard failed")
    runtime = configure_runtime(SEED_BLOCKS[0])
    snapshot1 = runtime.gpu_snapshot()
    time.sleep(2)
    snapshot2 = runtime.gpu_snapshot()
    if not all(
        runtime.gpu_is_idle(gpu, snapshot1)
        and runtime.gpu_is_idle(gpu, snapshot2)
        for gpu in gpus
    ):
        raise SystemExit("one or more selected PI2N FINAL GPUs is not idle twice")
    locks = []
    try:
        locks = [runtime.acquire_gpu_lock(gpu) for gpu in gpus]
        snapshot3 = runtime.gpu_snapshot()
        if not all(runtime.gpu_is_idle(gpu, snapshot3) for gpu in gpus):
            raise SystemExit("one or more PI2N FINAL GPUs became busy after locking")
        for path in (RAW, LOGS, CACHE, TMP):
            path.mkdir(parents=True, exist_ok=False)
        workers: list[dict[str, Any]] = []
        for seed_index, seed in enumerate(SEED_BLOCKS):
            for path in block_paths(seed):
                path.mkdir(parents=True, exist_ok=False)
            order = EXECUTION_ORDER[seed]
            for wave_start in range(0, len(order), len(gpus)):
                wave = order[wave_start : wave_start + len(gpus)]
                assignments = [
                    (
                        model,
                        seed,
                        gpus[offset],
                        8700 + seed_index * 30 + wave_start + offset,
                    )
                    for offset, model in enumerate(wave)
                ]
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=len(assignments)
                ) as executor:
                    futures = [
                        executor.submit(run_one, *assignment)
                        for assignment in assignments
                    ]
                    workers.extend(future.result() for future in futures)
        snapshots = [snapshot1, snapshot2, snapshot3, runtime.gpu_snapshot()]
        completeness, execution = audit_completeness(freeze, workers, snapshots)
        atomic_json(COMPLETENESS, completeness)
        atomic_json(EXECUTION, execution)
        print(
            json.dumps(
                {
                    "status": completeness["status"],
                    "workers": len(workers),
                    "outcomes": completeness["total_canonical_outcomes"],
                    "statistics_computed": False,
                },
                sort_keys=True,
            )
        )
        if completeness["status"] != "PASS":
            raise SystemExit("PI2N_FINAL_COMPLETENESS_FAIL")
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
    evaluate.add_argument("--seed", type=int, choices=SEED_BLOCKS, required=True)
    evaluate.add_argument("--contact-socket", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--diagnostics", type=Path, required=True)
    evaluate.add_argument("--port", type=int, required=True)
    launch = sub.add_parser("orchestrate")
    launch.add_argument("--gpus", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "contact-service":
        contact_service(args.artifact, args.socket)
    elif args.command == "evaluate-model":
        evaluate_model(
            args.model,
            args.seed,
            args.contact_socket,
            args.output,
            args.diagnostics,
            args.port,
        )
    else:
        try:
            gpus = [int(value) for value in args.gpus.split(",")]
        except ValueError as error:
            raise SystemExit("--gpus must be comma-separated physical IDs") from error
        orchestrate(gpus)


if __name__ == "__main__":
    main()
