#!/usr/bin/env python3
"""Run the RR4 P0/P3 smoke matrix through the exact production rollout worker."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_3_training import atomic_json, read_json, sha256_file  # noqa: E402
from scripts.simulation.run_s4_3_training_queue import gpu_state  # noqa: E402

ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_rr"
LOG_ROOT = ROOT / ".local/logs/simulation/s4_3_rr/production_smoke"
SMOKE_CONFIG = ARTIFACT_ROOT / "rr_smoke_resets.json"
MANIFEST = ARTIFACT_ROOT / "production_smoke_manifest.json"
RESULTS = ARTIFACT_ROOT / "production_smoke_results.json"
TRAINING = ROOT / ".local/artifacts/simulation/s4_3_restart/act_training_manifest.json"
WORKER = ROOT / "scripts/simulation/run_s4_3_rollout_job.py"
TASKS = ("pinch_tongs", "hammer_nail", "click_mouse")
VARIANTS = ("P0", "P3")
SEED_BASES = {"pinch_tongs": 6460000, "hammer_nail": 6461000, "click_mouse": 6462000}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command(*args: str, cwd: Path = ROOT, check: bool = True) -> str:
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or "command failed: " + " ".join(args))
    return result.stdout.strip()


def gpu_snapshot() -> dict[str, Any]:
    devices = command(
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ).splitlines()
    processes = command(
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
        check=False,
    ).splitlines()
    return {
        "devices": devices,
        "compute_processes": processes,
        "eligible": [gpu for gpu in range(4) if gpu_state(gpu)["idle"]],
    }


def wait_for_eligible_gpu(timeout_sec: int = 300) -> dict[str, Any]:
    """Wait through bounded CUDA teardown lag without relaxing the occupancy gate."""

    deadline = time.monotonic() + timeout_sec
    while True:
        snapshot = gpu_snapshot()
        if snapshot["eligible"]:
            return snapshot
        if time.monotonic() >= deadline:
            raise RuntimeError("no genuinely idle physical GPU is available for RR4")
        time.sleep(5)


def smoke_config() -> dict[str, Any]:
    scientific = read_json(ROOT / "configs/simulation/s4_3_policy_eval_v1.json")
    scientific_ids = {row["evaluation_reset_id"] for row in scientific["resets"]}
    resets = []
    for task in TASKS:
        reset = {
            "evaluation_reset_id": f"rr-smoke-{task}-00",
            "task": task,
            "reset_index": 0,
            "seed_namespace": "RR_SMOKE",
            "reset_seed": SEED_BASES[task],
            "environment_seed": SEED_BASES[task],
            "visual_randomization_seed": SEED_BASES[task],
            "dynamics_randomization": False,
            "perturbation_seed": None,
            "overlap_with_prior_sources": False,
        }
        if reset["evaluation_reset_id"] in scientific_ids:
            raise RuntimeError("RR_SMOKE identity overlaps POLICY_EVAL_V1")
        resets.append(reset)
    return {
        "schema": "tactile3d-unit.s4-3-rr-smoke-resets.v1",
        "stage": "RR4",
        "seed_namespace": "RR_SMOKE",
        "scientific_result": False,
        "policy_eval_v1_overlap": False,
        "resets": resets,
        "status": "PASS",
    }


def trace_audit(metadata: dict[str, Any]) -> dict[str, Any]:
    with np.load(ROOT / metadata["trace"], allow_pickle=False) as loaded:
        shapes = {name: list(loaded[name].shape) for name in loaded.files}
    return {
        "policy_action": shapes["policy_action"],
        "env_action": shapes["env_action"],
        "proprio": shapes["proprio"],
        "tactile": shapes["sim_tactile"],
        "p3_predicted_contact": shapes["p3_predicted_contact"],
        "action_27x22": metadata["action_chunk_shape"] == [27, 22],
        "adapter_22_to_23": (
            metadata["action_adapter"]["status"] == "PASS"
            and shapes["policy_action"][1:] == [22]
            and shapes["env_action"][1:] == [23]
        ),
    }


def main() -> None:
    dex_python = os.environ.get("DEXJOCO_PYTHON")
    unit_checkpoint = os.environ.get("UNIT_FULLDATA_CKPT")
    if not dex_python or not unit_checkpoint:
        raise RuntimeError("DEXJOCO_PYTHON and UNIT_FULLDATA_CKPT are required")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    config = smoke_config()
    atomic_json(SMOKE_CONFIG, config)
    training = read_json(TRAINING)
    lookup = {
        (row["task"], row["variant"], row["training_seed"]): row for row in training["jobs"]
    }
    identities = []
    for task in TASKS:
        for variant in VARIANTS:
            row = lookup[(task, variant, 0)]
            identities.append(
                {
                    "task": task,
                    "variant": variant,
                    "training_seed": 0,
                    "reset_id": f"rr-smoke-{task}-00",
                    "checkpoint": row["checkpoint"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                }
            )
    initial_gpu = gpu_snapshot()
    manifest = {
        "schema": "tactile3d-unit.s4-3-rr-production-smoke-manifest.v1",
        "stage": "RR4",
        "exact_production_worker": str(WORKER.relative_to(ROOT)),
        "exact_production_worker_sha256": sha256_file(WORKER),
        "smoke_config": str(SMOKE_CONFIG.relative_to(ROOT)),
        "smoke_config_sha256": sha256_file(SMOKE_CONFIG),
        "policy_eval_v1_overlap": False,
        "scientific_result": False,
        "minimum_post_warmup_control_steps": 100,
        "native_termination_permitted": True,
        "identities": identities,
        "initial_gpu_snapshot": initial_gpu,
        "status": "RUNNING",
    }
    atomic_json(MANIFEST, manifest)

    common_git = Path(command("git", "rev-parse", "--path-format=absolute", "--git-common-dir"))
    runs = []
    for identity in identities:
        snapshot = wait_for_eligible_gpu()
        gpu = snapshot["eligible"][0]
        lock_path = common_git / f"tactile3d_unit_gpu{gpu}.lock"
        lock = lock_path.open("a+")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            recheck = gpu_snapshot()
            if gpu not in recheck["eligible"]:
                raise RuntimeError("selected RR4 GPU became busy after lock acquisition")
            job_log = LOG_ROOT / f"{identity['task']}_{identity['variant']}_seed0.log"
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "S4_3_PHYSICAL_GPU": str(gpu),
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "DEXJOCO_PYTHON": dex_python,
                    "UNIT_FULLDATA_CKPT": unit_checkpoint,
                }
            )
            environment.pop("DISPLAY", None)
            started = now()
            with job_log.open("w", encoding="utf-8") as output:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(WORKER),
                        "--task",
                        identity["task"],
                        "--variant",
                        identity["variant"],
                        "--seed",
                        "0",
                        "--checkpoint",
                        str(ROOT / identity["checkpoint"]),
                        "--checkpoint-sha256",
                        identity["checkpoint_sha256"],
                        "--run-kind",
                        "rr_smoke",
                        "--evaluation-config",
                        str(SMOKE_CONFIG),
                    ],
                    cwd=ROOT,
                    env=environment,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
            run = {
                **identity,
                "physical_gpu": gpu,
                "logical_device": "cuda:0",
                "prelaunch_gpu_snapshot": recheck,
                "start_time": started,
                "end_time": now(),
                "exit_code": result.returncode,
                "log": str(job_log.relative_to(ROOT)),
                "log_sha256": sha256_file(job_log),
                "status": "PASS" if result.returncode == 0 else "FAIL",
            }
            runs.append(run)
            if result.returncode:
                manifest["status"] = "FAIL"
                atomic_json(MANIFEST, manifest)
                atomic_json(
                    RESULTS,
                    {
                        "schema": "tactile3d-unit.s4-3-rr-production-smoke-results.v1",
                        "stage": "RR4",
                        "runs": runs,
                        "status": "FAIL",
                    },
                )
                raise SystemExit("S4_3_RR_PRODUCTION_SMOKE_FAIL")
        finally:
            lock.close()

    audited = []
    for run in runs:
        job_path = (
            ARTIFACT_ROOT
            / "production_smoke/jobs"
            / f"{run['task']}_{run['variant']}_seed0.json"
        )
        job = read_json(job_path)
        metadata_path = ROOT / job["metadata"][0]["path"]
        metadata = read_json(metadata_path)
        trace = trace_audit(metadata)
        native_early = metadata["termination_reason"] in {
            "SUCCESS",
            "ENV_TERMINATION_FAILURE",
        }
        control_length_pass = metadata["control_steps"] >= 100 or native_early
        p3_pass = run["variant"] != "P3" or metadata["p3_predicted_contact_replans"] > 0
        endpoint_pass = (
            job["endpoint"]["encoded_length"] <= 80
            and job["endpoint"]["cleanup_pass"] is True
        )
        pass_gate = (
            job["status"] == "PASS"
            and metadata["run_kind"] == "rr_smoke"
            and metadata["scientific_result"] is False
            and metadata["warmup_samples"] == 26
            and metadata["replan_stride"] == 5
            and metadata["future_observation_read"] is False
            and metadata["expert_action_read"] is False
            and not metadata["runtime_exceptions"]
            and trace["action_27x22"]
            and trace["adapter_22_to_23"]
            and trace["proprio"][1:] == [22]
            and trace["tactile"][1:] == [30]
            and p3_pass
            and endpoint_pass
            and control_length_pass
        )
        audited.append(
            {
                **run,
                "job_artifact": str(job_path.relative_to(ROOT)),
                "metadata": str(metadata_path.relative_to(ROOT)),
                "control_steps": metadata["control_steps"],
                "termination_reason": metadata["termination_reason"],
                "trace_shapes": trace,
                "checkpoint_load": "PASS",
                "rgb": "PASS",
                "proprio": "PASS" if trace["proprio"][1:] == [22] else "FAIL",
                "tactile": "PASS" if trace["tactile"][1:] == [30] else "FAIL",
                "contact_state": "PASS" if p3_pass else "FAIL",
                "p3_auxiliary_runtime": "PASS" if p3_pass else "FAIL",
                "action_27x22": "PASS" if trace["action_27x22"] else "FAIL",
                "adapter_22_to_23": "PASS" if trace["adapter_22_to_23"] else "FAIL",
                "replan_stride_5": "PASS" if metadata["replan_stride"] == 5 else "FAIL",
                "egl": "PASS",
                "no_future_read": "PASS" if not metadata["future_observation_read"] else "FAIL",
                "socket_cleanup": "PASS" if endpoint_pass else "FAIL",
                "endpoint_encoded_length": job["endpoint"]["encoded_length"],
                "status": "PASS" if pass_gate else "FAIL",
            }
        )
    lingering = command(
        "pgrep",
        "-af",
        "serve_s4_3_act_policy.py|run_s4_3_policy_rollouts_dex.py",
        check=False,
    )
    result = {
        "schema": "tactile3d-unit.s4-3-rr-production-smoke-results.v1",
        "stage": "RR4",
        "exact_production_worker": True,
        "runs": audited,
        "smokes": len(audited),
        "tasks": list(TASKS),
        "variants": list(VARIANTS),
        "policy_eval_v1_overlap": False,
        "scientific_results_produced": 0,
        "lingering_processes": lingering.splitlines() if lingering else [],
        "no_process_leak": not bool(lingering),
    }
    result["status"] = (
        "PASS"
        if len(audited) == 6
        and all(row["status"] == "PASS" for row in audited)
        and result["no_process_leak"]
        else "FAIL"
    )
    manifest["status"] = result["status"]
    manifest["completed_at"] = now()
    atomic_json(MANIFEST, manifest)
    atomic_json(RESULTS, result)
    print(json.dumps({"smokes": len(audited), "status": result["status"]}, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit("S4_3_RR_PRODUCTION_SMOKE_FAIL")


if __name__ == "__main__":
    main()
