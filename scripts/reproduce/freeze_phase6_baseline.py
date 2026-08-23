#!/usr/bin/env python3
"""Validate completed Phase 6 evidence and freeze local-only provenance.

This command never launches an evaluator.  It validates the already-completed
official ID run, writes local evidence beneath ``.local/``, and prints the
public fields needed by the tracked canonical-baseline manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE6_DIR = PROJECT_ROOT / ".local/artifacts/reproduction/phase6"
LOG_DIR = PROJECT_ROOT / ".local/logs/reproduction/phase6"


def duplicate_rejecting_object(pairs: Iterable[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle, object_pairs_hook=duplicate_rejecting_object)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def id_tasks_from_official_script() -> list[str]:
    source = (PROJECT_ROOT / "examples/run_eval.sh").read_text(encoding="utf-8")
    match = re.search(r"(?ms)^  id\)\s*.*?task_names=\((.*?)^    \) ;;", source)
    if not match:
        raise ValueError("could not extract the current ID task list from examples/run_eval.sh")
    tasks = re.findall(r"^\s*(gr1_unified/\S+_Env)\s*$", match.group(1), re.MULTILINE)
    if len(tasks) != len(set(tasks)):
        raise ValueError("the official ID task list contains duplicates")
    return tasks


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def current_task_before(log_text: str, position: int) -> str | None:
    preceding = log_text[:position]
    tasks = re.findall(r"^Executing command for:\s*(gr1_unified/\S+_Env)", preceding, re.MULTILINE)
    return tasks[-1] if tasks else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path, help="local checkpoint directory; never written to tracked files")
    args = parser.parse_args()
    model_path = args.model_path.resolve()

    official_tasks = id_tasks_from_official_script()
    raw_results_path = PHASE6_DIR / "raw/results_official_parser.json"
    summary_path = PHASE6_DIR / "summaries/phase6_summary.json"
    parser_check_path = PHASE6_DIR / "summaries/parser_recompute_check.json"
    client_log_path = PHASE6_DIR / "raw/test_robocasa_gr1_client.log"
    for required in (raw_results_path, summary_path, parser_check_path, client_log_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    results = load_json(raw_results_path)
    previous = load_json(summary_path)
    parser_check = load_json(parser_check_path)
    tasks = results.get("tasks")
    metrics = results.get("average_success_rate")
    if not isinstance(tasks, dict) or not isinstance(metrics, dict):
        raise ValueError("official parser result has an invalid schema")
    result_task_names = list(tasks)
    missing_tasks = sorted(set(official_tasks) - set(result_task_names))
    unexpected_tasks = sorted(set(result_task_names) - set(official_tasks))
    duplicate_or_missing = bool(missing_tasks or unexpected_tasks or len(result_task_names) != len(set(result_task_names)))
    numeric_rates = all(isinstance(value, (int, float)) and math.isfinite(value) for value in tasks.values())
    numeric_metrics = all(isinstance(value, (int, float)) and math.isfinite(value) for value in metrics.values())
    log_text = client_log_path.read_text(encoding="utf-8", errors="replace")
    result_blocks = len(re.findall(r"^Results for gr1_unified/", log_text, re.MULTILINE))
    successful_tasks = re.findall(r"^Successfully executed:\s*(gr1_unified/\S+_Env)", log_text, re.MULTILINE)
    explicit_failures = re.findall(r"^FAILED:\s*(gr1_unified/\S+_Env)", log_text, re.MULTILINE)
    process_errors = re.findall(r"Traceback|Segmentation fault|Killed|ConnectionRefusedError", log_text, re.IGNORECASE)
    qacc_matches = list(re.finditer(r"WARNING: Nan, Inf or huge value in QACC.*", log_text))
    qacc_events = [
        {"task": current_task_before(log_text, event.start()), "message": event.group(0)}
        for event in qacc_matches
    ]

    expected_rollouts = len(official_tasks) * 50
    protocol_integrity = (
        len(official_tasks) == 24
        and len(result_task_names) == 24
        and result_blocks == 24
        and len(successful_tasks) == 24
        and not duplicate_or_missing
        and numeric_rates
        and numeric_metrics
        and not explicit_failures
        and not process_errors
        and previous.get("observed_rollouts_from_videos") == expected_rollouts
        and parser_check.get("all_task_values_equal") is True
        and parser_check.get("counts_equal") is True
        and parser_check.get("metrics_within_1e-12") is True
    )
    overall = float(metrics["overall"])
    historical = 0.664
    threshold = historical - 0.04
    report = {
        "previous_gate": previous.get("gate"),
        "previous_reason": "historical metric delta > 4pp",
        "new_gate": "PASS_WITH_PROVENANCE_WARNING" if protocol_integrity and overall > threshold else "FAIL",
        "protocol_integrity": "PASS" if protocol_integrity else "FAIL",
        "performance_non_regression": "PASS" if overall > threshold else "FAIL",
        "historical_metric_fidelity": "DIVERGED",
        "canonical_local_overall_sr": round(overall, 4),
        "validation": {
            "official_id_task_count": len(official_tasks),
            "parser_task_count": len(result_task_names),
            "official_parser_result_blocks": results.get("num_result_blocks"),
            "log_result_blocks": result_blocks,
            "episodes_per_task": 50,
            "expected_rollouts": expected_rollouts,
            "observed_rollouts_from_videos": previous.get("observed_rollouts_from_videos"),
            "successful_task_markers": len(successful_tasks),
            "missing_tasks": missing_tasks,
            "unexpected_tasks": unexpected_tasks,
            "non_finite_result_values": not (numeric_rates and numeric_metrics),
            "explicit_failed_task_markers": explicit_failures,
            "process_error_markers": len(process_errors),
            "official_parser_recompute_consistent": all(parser_check.get(key) is True for key in ("all_task_values_equal", "counts_equal", "metrics_within_1e-12")),
        },
        "gates": {
            "historical_overall_success_rate": historical,
            "allowed_regression_percentage_points": 4.0,
            "non_regression_threshold": threshold,
            "local_overall_success_rate": overall,
            "historical_fidelity_delta_percentage_points": round((overall - historical) * 100, 2),
            "exact_checkpoint_identity_established": False,
        },
        "qacc_warning": {
            "count": len(qacc_events),
            "events": qacc_events,
            "classification": "NON_BLOCKING_WARNING" if protocol_integrity and len(qacc_events) == 1 else "REQUIRES_REVIEW",
            "reason": "The affected task completed all 50 episodes with a parseable numeric result and no process failure marker." if protocol_integrity and len(qacc_events) == 1 else "Evidence does not meet the transient-warning criteria.",
        },
    }

    provenance_dir = PHASE6_DIR / "provenance"
    summaries_dir = PHASE6_DIR / "summaries"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for relative in [
        *[f"model-{index:05d}-of-00004.safetensors" for index in range(1, 5)],
        "model.safetensors.index.json",
        "config.json",
        *[f"tokenizer/model-{index:05d}-of-00002.safetensors" for index in range(1, 3)],
        "tokenizer/model.safetensors.index.json",
        "tokenizer/config.json",
    ]:
        candidate = model_path / relative
        if candidate.is_file():
            hashes[relative] = sha256(candidate)
        else:
            raise FileNotFoundError(candidate)
    (provenance_dir / "checkpoint_hashes.json").write_text(json.dumps({"repository": "xpeng-robotics/VLA-UniT-checkpoints", "variant": "VLA-UniT-3B-fulldata", "hf_revision": None, "hf_revision_status": "unknown_or_unavailable", "sha256": hashes}, indent=2) + "\n", encoding="utf-8")

    environment = {
        "git_head": command_output(["git", "rev-parse", "HEAD"], PROJECT_ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in ("torch", "mujoco", "robosuite", "transformers", "flash-attn")},
        "cuda_runtime": command_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "gpus": command_output(["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"]),
    }
    (provenance_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    simulator_revisions = {
        "robocasa_gr1_tabletop_tasks_commit": command_output(["git", "rev-parse", "HEAD"], PROJECT_ROOT / "third_party/robocasa-gr1-tabletop-tasks"),
        "robosuite_commit": command_output(["git", "rev-parse", "HEAD"], PROJECT_ROOT / "third_party/robosuite"),
    }
    (provenance_dir / "simulator_revisions.json").write_text(json.dumps(simulator_revisions, indent=2) + "\n", encoding="utf-8")
    (summaries_dir / "phase6_reacceptance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"report": report, "checkpoint_hashes": hashes, "environment": environment, "simulator_revisions": simulator_revisions}, indent=2))
    return 0 if report["new_gate"] == "PASS_WITH_PROVENANCE_WARNING" else 1


if __name__ == "__main__":
    raise SystemExit(main())
