#!/usr/bin/env python3
"""Run the official OOD evaluator with isolated smoke-scale jobs.

The wrapper only assigns resources, invokes ``examples/run_eval.sh``, relocates
the evaluator's generated artifacts to ``.local/``, and validates the output.
It does not alter policy, actions, environment semantics, success criteria, or
episode length.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_EVALUATOR = PROJECT_ROOT / "examples/run_eval.sh"
PHASE7_DIR = PROJECT_ROOT / ".local/artifacts/reproduction/phase7"
LOG_DIR = PROJECT_ROOT / ".local/logs/reproduction/phase7"
PID_DIR = PROJECT_ROOT / ".local/tmp/phase7"
PROTOCOLS = ("ood_object_appearance", "ood_container_combination", "ood_object_type", "unseen_close")


@dataclass(frozen=True)
class Job:
    protocol: str
    gpu: int
    port: int
    tag: str
    tasks: tuple[str, ...]


def duplicate_rejecting_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def tasks_from_official_script(protocol: str) -> tuple[str, ...]:
    source = OFFICIAL_EVALUATOR.read_text(encoding="utf-8")
    match = re.search(rf"(?ms)^  {re.escape(protocol)}\)\s*.*?task_names=\((.*?)^    \) ;;", source)
    if not match:
        raise ValueError(f"could not extract {protocol} task list from {OFFICIAL_EVALUATOR}")
    tasks = tuple(re.findall(r"^\s*(gr1_unified/\S+_Env)\s*$", match.group(1), re.MULTILINE))
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError(f"invalid {protocol} task list in official evaluator")
    return tasks


def read_json_strict(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle, object_pairs_hook=duplicate_rejecting_object)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data


def safe_move(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing local evidence: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))


def validate(job: Job, artifact_dir: Path, episodes: int, returncode: int) -> dict[str, object]:
    results_path = artifact_dir / "results.json"
    client_log = artifact_dir / "test_robocasa_gr1_client.log"
    errors: list[str] = []
    try:
        results = read_json_strict(results_path)
    except Exception as error:  # retain complete failure information in summary
        results = {}
        errors.append(f"unparseable_results: {error}")
    log_text = client_log.read_text(encoding="utf-8", errors="replace") if client_log.is_file() else ""
    task_values = results.get("tasks") if isinstance(results.get("tasks"), dict) else {}
    expected = set(job.tasks)
    observed = set(task_values)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    nonfinite = [name for name, value in task_values.items() if not isinstance(value, (int, float)) or not math.isfinite(value)]
    out_of_range = [name for name, value in task_values.items() if isinstance(value, (int, float)) and math.isfinite(value) and not 0 <= value <= 1]
    complete_markers = re.findall(r"^Successfully executed:\s*(gr1_unified/\S+_Env)", log_text, re.MULTILINE)
    failed_markers = re.findall(r"^FAILED:\s*(gr1_unified/\S+_Env)", log_text, re.MULTILINE)
    result_blocks = len(re.findall(r"^Results for gr1_unified/", log_text, re.MULTILINE))
    pattern_errors = re.findall(r"Traceback|Segmentation fault|ConnectionRefusedError|Reset.*(?:error|fail)|inference.*(?:error|exception)|\\bNaN\\b|\\bInf\\b", log_text, re.IGNORECASE)
    if returncode != 0:
        errors.append(f"official_evaluator_returncode={returncode}")
    if missing:
        errors.append(f"missing_tasks={missing}")
    if unexpected:
        errors.append(f"unexpected_tasks={unexpected}")
    if len(task_values) != len(job.tasks):
        errors.append(f"result_task_count={len(task_values)} expected={len(job.tasks)}")
    if len(complete_markers) != len(job.tasks):
        errors.append(f"completed_task_markers={len(complete_markers)} expected={len(job.tasks)}")
    if result_blocks != len(job.tasks):
        errors.append(f"result_blocks={result_blocks} expected={len(job.tasks)}")
    if failed_markers:
        errors.append(f"failed_task_markers={failed_markers}")
    if nonfinite:
        errors.append(f"nonfinite_result_values={nonfinite}")
    if out_of_range:
        errors.append(f"out_of_range_result_values={out_of_range}")
    if pattern_errors:
        errors.append(f"log_error_markers={len(pattern_errors)}")
    successes = sum(round(float(value) * episodes) for value in task_values.values() if isinstance(value, (int, float)) and math.isfinite(value))
    expected_rollouts = len(job.tasks) * episodes
    return {
        "protocol": job.protocol,
        "tasks": len(job.tasks),
        "episodes_per_task": episodes,
        "expected_rollouts": expected_rollouts,
        "completed_rollouts": expected_rollouts if not errors else len(complete_markers) * episodes,
        "success": successes,
        "failure": expected_rollouts - successes,
        "smoke_success_rate": successes / expected_rollouts if expected_rollouts else None,
        "result_blocks": result_blocks,
        "completed_task_markers": len(complete_markers),
        "infrastructure_errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }


def select_videos(protocol: str, artifact_dir: Path) -> list[dict[str, object]]:
    root = artifact_dir / "videos"
    records: list[dict[str, object]] = []
    for task_dir in sorted(path for path in root.rglob("*") if path.is_dir()):
        files = sorted(path for path in task_dir.iterdir() if path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mkv"})
        for episode, path in enumerate(files, start=1):
            match = re.search(r"(?:^|_)success([01])_", path.name)
            if match is None:
                continue
            records.append({
                "protocol": protocol,
                "task": str(task_dir.relative_to(root)),
                "episode": episode,
                "success": match.group(1) == "1",
                "path": str(path.relative_to(PROJECT_ROOT)),
            })
    success = next((record for record in records if record["success"] is True), None)
    failure = next((record for record in records if record["success"] is False), None)
    if success is not None and failure is not None:
        success["selection"] = "success"
        failure["selection"] = "failure"
        return [success, failure]
    selected: list[dict[str, object]] = []
    seen_tasks: set[str] = set()
    for record in records:
        task = str(record["task"])
        if task in seen_tasks:
            continue
        record["selection"] = "representative_no_both_outcomes"
        selected.append(record)
        seen_tasks.add(task)
        if len(selected) == 2:
            break
    return selected


def launch(job: Job, model_path: Path, episodes: int, hf_home: Path | None) -> tuple[subprocess.Popen[str], Path]:
    log_path = LOG_DIR / f"{job.protocol}.orchestrator.log"
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(job.gpu),
        "PORT": str(job.port),
        "EVAL_TAG": job.tag,
        "N_ENVS": "1",
        "N_EPISODES": str(episodes),
        # The official shell entrypoint invokes python3.  Preserve the selected
        # interpreter's environment even when this wrapper is called by an
        # absolute interpreter path rather than from an activated shell.
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{environment.get('PATH', '')}",
        "PYTHONUNBUFFERED": "1",
    })
    if hf_home is not None:
        environment["HF_HOME"] = str(hf_home)
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(["bash", str(OFFICIAL_EVALUATOR), str(model_path), job.protocol], cwd=PROJECT_ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, text=True)
    handle.close()
    (PID_DIR / f"{job.protocol}.pid").write_text(f"{process.pid}\n", encoding="utf-8")
    return process, log_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path, help="checkpoint directory consumed by the official evaluator")
    parser.add_argument("--episodes", type=int, default=2, help="smoke episodes per OOD task")
    parser.add_argument("--sequential", action="store_true", help="run jobs one at a time after collision audit")
    parser.add_argument("--gpu", type=int, help="GPU to use for all jobs in --sequential mode")
    parser.add_argument("--hf-home", type=Path, help="local Hugging Face cache root needed by the official offline evaluator")
    parser.add_argument("--resume", action="store_true", help="validate and retain completed local protocol artifacts, then run only missing protocols")
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if args.gpu is not None and not args.sequential:
        raise ValueError("--gpu is only valid together with --sequential")
    model_path = args.model_path.resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    hf_home = args.hf_home.resolve() if args.hf_home is not None else None
    if hf_home is not None and not hf_home.is_dir():
        raise FileNotFoundError(hf_home)
    if PHASE7_DIR.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite existing Phase 7 evidence: {PHASE7_DIR}")
    PHASE7_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PID_DIR.mkdir(parents=True, exist_ok=True)
    job_gpus = [args.gpu if args.gpu is not None else 0] * len(PROTOCOLS) if args.sequential else list(range(len(PROTOCOLS)))
    jobs = [Job(protocol, gpu, 5900 + gpu, f"_phase7_{protocol}", tasks_from_official_script(protocol)) for protocol, gpu in zip(PROTOCOLS, job_gpus)]
    collision_audit = {
        "execution_mode": "SEQUENTIAL" if args.sequential else "PARALLEL",
        "ports": {job.protocol: job.port for job in jobs},
        "gpus": {job.protocol: job.gpu for job in jobs},
        "tags": {job.protocol: job.tag for job in jobs},
        "outputs": {job.protocol: f"evaluation_sim_{job.protocol}_1envs{job.tag}" for job in jobs},
        "task_counts_from_current_official_script": {job.protocol: len(job.tasks) for job in jobs},
        "status": "PASS",
    }
    (PHASE7_DIR / "collision_audit.json").write_text(json.dumps(collision_audit, indent=2) + "\n", encoding="utf-8")
    active: list[tuple[Job, subprocess.Popen[str]]] = []
    job_results: list[dict[str, object]] = []
    try:
        for job in jobs:
            existing = PHASE7_DIR / job.protocol
            if existing.is_dir():
                job_results.append(validate(job, existing, args.episodes, 0))
                (PID_DIR / f"{job.protocol}.pid").unlink(missing_ok=True)
                continue
            process, _ = launch(job, model_path, args.episodes, hf_home)
            active.append((job, process))
            if args.sequential:
                job, process = active.pop()
                returncode = process.wait()
                source = model_path / f"evaluation_sim_{job.protocol}_1envs{job.tag}"
                destination = PHASE7_DIR / job.protocol
                safe_move(source, destination)
                job_results.append(validate(job, destination, args.episodes, returncode))
                (PID_DIR / f"{job.protocol}.pid").unlink(missing_ok=True)
        for job, process in active:
            returncode = process.wait()
            source = model_path / f"evaluation_sim_{job.protocol}_1envs{job.tag}"
            destination = PHASE7_DIR / job.protocol
            safe_move(source, destination)
            job_results.append(validate(job, destination, args.episodes, returncode))
            (PID_DIR / f"{job.protocol}.pid").unlink(missing_ok=True)
    finally:
        # Do not terminate processes owned by other users or services.  This wrapper
        # only reports live PIDs it created if it is interrupted.
        still_running = {job.protocol: process.pid for job, process in active if process.poll() is None}
        (PID_DIR / "active_on_exit.json").write_text(json.dumps(still_running, indent=2) + "\n", encoding="utf-8")

    job_results.sort(key=lambda item: PROTOCOLS.index(str(item["protocol"])))
    summaries = PHASE7_DIR / "summaries"
    summaries.mkdir(exist_ok=True)
    total_expected = sum(int(item["expected_rollouts"]) for item in job_results)
    total_completed = sum(int(item["completed_rollouts"]) for item in job_results)
    failures = {str(item["protocol"]): item["infrastructure_errors"] for item in job_results if item["infrastructure_errors"]}
    summary = {
        "phase": "phase7",
        "purpose": "OOD evaluation surface functional closure; smoke-scale only, not a formal OOD metric reproduction.",
        "execution_mode": collision_audit["execution_mode"],
        "protocols": job_results,
        "expected_rollouts": total_expected,
        "completed_rollouts": total_completed,
        "infrastructure_failures": failures,
        "status": "PASS" if len(job_results) == len(jobs) and total_expected == total_completed and not failures else "FAIL",
    }
    (summaries / "phase7_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (summaries / "protocol_matrix.csv").open("w", encoding="utf-8") as handle:
        handle.write("protocol,tasks,episodes_per_task,expected_rollouts,completed_rollouts,success,failure,smoke_success_rate,status\n")
        for item in job_results:
            handle.write(",".join(str(item[key]) for key in ("protocol", "tasks", "episodes_per_task", "expected_rollouts", "completed_rollouts", "success", "failure", "smoke_success_rate", "status")) + "\n")
    (summaries / "infrastructure_failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    videos = [entry for job in jobs for entry in select_videos(job.protocol, PHASE7_DIR / job.protocol)]
    (summaries / "video_manifest.json").write_text(json.dumps(videos, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
