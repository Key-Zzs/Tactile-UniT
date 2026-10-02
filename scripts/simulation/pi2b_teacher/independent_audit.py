#!/usr/bin/env python3
"""Independently audit paired effects from frozen per-group metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.simulation.pi2b_teacher.common import atomic_json, load_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    args = parser.parse_args()
    runtime = load_json(args.runtime_manifest)
    root = Path(runtime["write_root"])
    evaluation = load_json(root / "artifacts/evaluation_result.json")
    analysis = load_json(root / "artifacts/stratified_group_statistics.json")
    groups = evaluation["group_metrics"]
    tasks = evaluation["group_tasks"]
    task_names = sorted(set(tasks.values()))
    checked = {}
    failures = []
    for metric, expected in analysis["paired_effects"].items():
        common_groups = sorted(set(groups["T_VA_match"]) & set(groups["T_VAC_match"]))
        difference = {
            group: float(groups["T_VAC_match"][group][metric] - groups["T_VA_match"][group][metric])
            for group in common_groups
        }
        point = float(np.mean(list(difference.values())))
        task_points = {
            task: float(np.mean([value for group, value in difference.items() if tasks[group] == task]))
            for task in task_names
        }
        point_equal = abs(point - float(expected["estimate"])) < 1e-12
        tasks_equal = all(abs(task_points[task] - float(expected["per_task"][task])) < 1e-12 for task in task_names)
        checked[metric] = {"estimate": point, "per_task": task_points, "estimate_equal": point_equal, "per_task_equal": tasks_equal}
        if not point_equal or not tasks_equal:
            failures.append(metric)
    result = {
        "schema": "tactile3d-unit.s4-3-pi2b-teacher-independent-statistics-audit.v1",
        "status": "PASS" if not failures else "FAIL",
        "implementation": "independent direct NumPy group-difference recomputation",
        "checked": checked,
        "failures": failures,
    }
    atomic_json(root / "artifacts/statistics_independent_audit.json", result)
    if failures:
        raise SystemExit("independent paired statistics audit failed")
    print(json.dumps({"status": "PASS", "metrics": len(checked)}, indent=2))


if __name__ == "__main__":
    main()
