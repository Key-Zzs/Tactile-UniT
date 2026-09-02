#!/usr/bin/env python3
"""Run frozen paired hierarchical S4.3 closed-loop analyses."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_3_training import TASKS, VARIANTS, atomic_json, read_json  # noqa: E402

ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_restart"
ROLLOUTS = ARTIFACT_ROOT / "closed_loop_rollouts.json"
PROTOCOL = ROOT / "configs/simulation/s4_3_restart_policy_protocol.json"
COMPARISONS = {
    "P1-P0": ("P1", "P0", "RAW_TACTILE_EFFECT"),
    "P2-P1": ("P2", "P1", "CONTACT_STATE_VS_RAW_EFFECT"),
    "P3-P2": ("P3", "P2", "TACTILE_UNIT_AUXILIARY_EFFECT"),
    "P3-P0": ("P3", "P0", "FULL_TACTILE_UNIT_EFFECT"),
}


def row_key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        row["task"],
        row["variant"],
        int(row["training_seed"]),
        row["evaluation_reset_id"],
    )


def paired_cube(
    lookup: dict[tuple[str, str, int, str], dict[str, Any]],
    tasks: tuple[str, ...],
    high: str,
    low: str,
    value: Callable[[dict[str, Any]], float],
) -> np.ndarray:
    cube = np.empty((len(tasks), 3, 30), dtype=np.float64)
    for task_index, task in enumerate(tasks):
        for seed in range(3):
            scoped = sorted(
                [
                    row
                    for key, row in lookup.items()
                    if key[0] == task and key[1] == high and key[2] == seed
                ],
                key=lambda row: row["reset_index"],
            )
            other = sorted(
                [
                    row
                    for key, row in lookup.items()
                    if key[0] == task and key[1] == low and key[2] == seed
                ],
                key=lambda row: row["reset_index"],
            )
            if len(scoped) != 30 or len(other) != 30:
                raise RuntimeError("paired reset matrix is incomplete")
            if [row["evaluation_reset_id"] for row in scoped] != [
                row["evaluation_reset_id"] for row in other
            ]:
                raise RuntimeError("paired reset IDs differ across variants")
            cube[task_index, seed] = np.asarray(
                [value(left) - value(right) for left, right in zip(scoped, other)]
            )
    return cube


def hierarchical_bootstrap(cube: np.ndarray, samples: int, seed: int) -> tuple[float, list[float]]:
    rng = np.random.default_rng(seed)
    task_count, seed_count, reset_count = cube.shape
    draws = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        task_indices = rng.integers(0, task_count, size=task_count)
        task_values = []
        for task_index in task_indices:
            seed_indices = rng.integers(0, seed_count, size=seed_count)
            seed_values = []
            for seed_index in seed_indices:
                reset_indices = rng.integers(0, reset_count, size=reset_count)
                seed_values.append(cube[task_index, seed_index, reset_indices].mean())
            task_values.append(np.mean(seed_values))
        draws[sample] = np.mean(task_values)
    return float(cube.mean()), np.quantile(draws, [0.025, 0.975]).astype(float).tolist()


def classify(delta: float, interval: list[float]) -> str:
    if delta >= 0.05 and interval[0] > 0:
        return "MATERIAL_IMPROVEMENT"
    if delta <= -0.05 and interval[1] < 0:
        return "MATERIAL_HURT"
    return "NO_MATERIAL_DIFFERENCE"


def success_comparisons(
    lookup: dict[tuple[str, str, int, str], dict[str, Any]],
    tasks: tuple[str, ...],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    results = {}
    for index, (name, (high, low, meaning)) in enumerate(COMPARISONS.items()):
        cube = paired_cube(lookup, tasks, high, low, lambda row: float(row["success"]))
        delta, interval = hierarchical_bootstrap(cube, samples, seed + index)
        per_task = {}
        for task_index, task in enumerate(tasks):
            scoped_delta, scoped_interval = hierarchical_bootstrap(
                cube[task_index : task_index + 1], samples, seed + 100 + index * 10 + task_index
            )
            per_task[task] = {
                "delta": scoped_delta,
                "ci95": scoped_interval,
                "classification": classify(scoped_delta, scoped_interval),
            }
        results[name] = {
            "high": high,
            "low": low,
            "meaning": meaning,
            "delta": delta,
            "ci95": interval,
            "classification": classify(delta, interval),
            "per_task": per_task,
        }
    return results


def absolute_success(
    lookup: dict[tuple[str, str, int, str], dict[str, Any]],
    tasks: tuple[str, ...],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    results = {}
    for variant_index, variant in enumerate(VARIANTS):
        cube = np.empty((len(tasks), 3, 30), dtype=np.float64)
        for task_index, task in enumerate(tasks):
            for training_seed in range(3):
                scoped = sorted(
                    [
                        row
                        for key, row in lookup.items()
                        if key[0] == task and key[1] == variant and key[2] == training_seed
                    ],
                    key=lambda row: row["reset_index"],
                )
                if len(scoped) != 30:
                    raise RuntimeError("absolute success matrix is incomplete")
                cube[task_index, training_seed] = [row["success"] for row in scoped]
        rate, interval = hierarchical_bootstrap(cube, samples, seed + variant_index)
        results[variant] = {
            "macro_success": rate,
            "ci95": interval,
            "per_task": {
                task: float(cube[task_index].mean()) for task_index, task in enumerate(tasks)
            },
        }
    return results


def summarize_rollouts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for task in TASKS:
        tasks[task] = {}
        for variant in VARIANTS:
            scoped = [row for row in rows if row["task"] == task and row["variant"] == variant]
            success = np.asarray([row["success"] for row in scoped], dtype=np.float64)
            successful_times = [row["time_to_success_sec"] for row in scoped if row["success"]]
            tasks[task][variant] = {
                "rollouts": len(scoped),
                "successes": int(success.sum()),
                "success_rate": float(success.mean()),
                "time_to_success_successes_mean_sec": (
                    float(np.mean(successful_times)) if successful_times else None
                ),
                "termination_reasons": dict(Counter(row["termination_reason"] for row in scoped)),
                "runtime_exception_rollouts": sum(
                    bool(row["runtime_exceptions"]) for row in scoped
                ),
                "mean_peak_normal_force": float(
                    np.mean([row["force_metrics"]["peak_normal_force"] for row in scoped])
                ),
                "mean_integrated_normal_force": float(
                    np.mean([row["force_metrics"]["integrated_normal_force"] for row in scoped])
                ),
                "mean_peak_tangential_force": float(
                    np.mean([row["force_metrics"]["peak_tangential_force"] for row in scoped])
                ),
            }
    maxima = {
        task: max(tasks[task][variant]["success_rate"] for variant in VARIANTS) for task in TASKS
    }
    healthy_tasks = sum(value >= 0.2 for value in maxima.values())
    if all(value == 0 for value in maxima.values()):
        learning = "ZERO_LEARNING"
    elif healthy_tasks >= 2:
        learning = "HEALTHY"
    else:
        learning = "WEAK"
    ceilings = [
        task
        for task in TASKS
        if all(tasks[task][variant]["success_rate"] > 0.95 for variant in VARIANTS)
    ]
    return {
        "schema": "tactile3d-unit.s4-3-closed-loop-summary.v1",
        "stage": "R13",
        "tasks": tasks,
        "maximum_success_by_task": maxima,
        "benchmark_learning_floor": learning,
        "healthy_tasks_at_or_above_20_percent": healthy_tasks,
        "task_ceiling_warnings": ceilings,
        "status": "PASS",
    }


def secondary_value(row: dict[str, Any], metric: str) -> float:
    if metric == "time_to_success_sec_timeout_imputed":
        return float(
            row["time_to_success_sec"]
            if row["time_to_success_sec"] is not None
            else row["timeout_steps"] * 0.02
        )
    if metric == "timeout_rate":
        return float(row["termination_reason"] == "TIMEOUT")
    if metric == "environment_termination_failure_rate":
        return float(row["termination_reason"] == "ENV_TERMINATION_FAILURE")
    if metric in {
        "peak_normal_force",
        "integrated_normal_force",
        "peak_tangential_force",
        "free_to_contact",
        "contact_to_free",
    }:
        return float(row["force_metrics"][metric])
    if metric in {
        "action_step_norm",
        "action_acceleration",
        "tcp_jerk",
        "hand_total_variation",
    }:
        return float(row["action_metrics"][metric])
    if metric == "replan_count":
        return float(row["replan_count"])
    raise ValueError(metric)


def secondary_analysis(
    lookup: dict[tuple[str, str, int, str], dict[str, Any]],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    metrics = (
        "time_to_success_sec_timeout_imputed",
        "timeout_rate",
        "environment_termination_failure_rate",
        "peak_normal_force",
        "integrated_normal_force",
        "peak_tangential_force",
        "free_to_contact",
        "contact_to_free",
        "action_step_norm",
        "action_acceleration",
        "tcp_jerk",
        "hand_total_variation",
        "replan_count",
    )
    scopes = {"all_tasks": TASKS, "tactile_active": ("pinch_tongs", "click_mouse")}
    result: dict[str, Any] = {}
    for scope_index, (scope, tasks) in enumerate(scopes.items()):
        result[scope] = {}
        for comparison_index, (name, (high, low, _)) in enumerate(COMPARISONS.items()):
            result[scope][name] = {}
            for metric_index, metric in enumerate(metrics):
                cube = paired_cube(
                    lookup,
                    tasks,
                    high,
                    low,
                    lambda row, metric=metric: secondary_value(row, metric),
                )
                delta, interval = hierarchical_bootstrap(
                    cube,
                    samples,
                    seed + 1000 + scope_index * 1000 + comparison_index * 100 + metric_index,
                )
                result[scope][name][metric] = {"delta": delta, "ci95": interval}
    return {
        "schema": "tactile3d-unit.s4-3-secondary-metrics.v1",
        "stage": "R14.5",
        "time_to_success_definition": "timeout-imputed paired completion time; timeout uses frozen task timeout",
        "contrasts": result,
        "primary_success_reclassified": False,
        "status": "PASS",
    }


def seed_analysis(rows: list[dict[str, Any]], primary: dict[str, Any]) -> dict[str, Any]:
    rates = {}
    for task in TASKS:
        rates[task] = {}
        for variant in VARIANTS:
            values = []
            for seed in range(3):
                scoped = [
                    row
                    for row in rows
                    if row["task"] == task
                    and row["variant"] == variant
                    and row["training_seed"] == seed
                ]
                values.append(float(np.mean([row["success"] for row in scoped])))
            rates[task][variant] = {
                "per_training_seed": values,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "range": [float(np.min(values)), float(np.max(values))],
            }
    macro_rates = {}
    for variant in VARIANTS:
        values = []
        for seed in range(3):
            per_task = []
            for task in TASKS:
                scoped = [
                    row
                    for row in rows
                    if row["task"] == task
                    and row["variant"] == variant
                    and row["training_seed"] == seed
                ]
                if len(scoped) != 30:
                    raise RuntimeError("training-seed success matrix is incomplete")
                per_task.append(float(np.mean([row["success"] for row in scoped])))
            values.append(float(np.mean(per_task)))
        macro_rates[variant] = {
            "per_training_seed": values,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "range": [float(np.min(values)), float(np.max(values))],
        }
    dominance = {}
    for name, comparison in primary.items():
        high, low = comparison["high"], comparison["low"]
        per_seed = []
        for seed in range(3):
            high_values = [
                row["success"]
                for row in rows
                if row["variant"] == high and row["training_seed"] == seed
            ]
            low_values = [
                row["success"]
                for row in rows
                if row["variant"] == low and row["training_seed"] == seed
            ]
            per_seed.append(float(np.mean(high_values) - np.mean(low_values)))
        overall = comparison["delta"]
        dominated = (
            max(abs(value) for value in per_seed) > 2 * max(abs(overall), 1 / 90)
            and sum(np.sign(value) == np.sign(overall) for value in per_seed) <= 1
        )
        dominance[name] = {
            "effect_per_training_seed": per_seed,
            "effect_dominated_by_one_seed": bool(dominated),
        }
    return {
        "schema": "tactile3d-unit.s4-3-training-seed-analysis.v1",
        "stage": "R14.6",
        "success_rates": rates,
        "three_task_macro_success_rates": macro_rates,
        "contrast_seed_dominance": dominance,
        "bad_training_seeds_deleted": False,
        "status": "PASS",
    }


def main() -> None:
    source = read_json(ROLLOUTS)
    protocol = read_json(PROTOCOL)
    if source.get("status") != "PASS" or source.get("completed_rollouts") != 1080:
        raise RuntimeError("complete R12 rollout matrix is required")
    rows = source["rollouts"]
    lookup = {row_key(row): row for row in rows}
    if len(lookup) != 1080:
        raise RuntimeError("rollout identities are not unique")
    samples = int(protocol["statistics"]["samples"])
    seed = int(protocol["statistics"]["seed"])
    summary = summarize_rollouts(rows)
    primary_comparisons = success_comparisons(lookup, TASKS, samples, seed)
    primary_variants = absolute_success(lookup, TASKS, samples, seed + 20_000)
    tactile_comparisons = success_comparisons(
        lookup, ("pinch_tongs", "click_mouse"), samples, seed + 10_000
    )
    tactile_variants = absolute_success(
        lookup, ("pinch_tongs", "click_mouse"), samples, seed + 30_000
    )
    primary = {
        "schema": "tactile3d-unit.s4-3-primary-statistics.v1",
        "stage": "R14",
        "endpoint": "equal-weight three-task macro success",
        "method": protocol["statistics"],
        "material_effect": protocol["material_effect"],
        "variants": primary_variants,
        "comparisons": primary_comparisons,
        "status": "PASS",
    }
    tactile = {
        "schema": "tactile3d-unit.s4-3-tactile-active-statistics.v1",
        "stage": "R14.3",
        "label": "SECONDARY_TACTILE_ACTIVE_ANALYSIS",
        "tasks": ["pinch_tongs", "click_mouse"],
        "replaces_primary": False,
        "variants": tactile_variants,
        "comparisons": tactile_comparisons,
        "status": "PASS",
    }
    hammer = {
        "schema": "tactile3d-unit.s4-3-hammer-control-analysis.v1",
        "stage": "R14.4",
        "label": "MAPPED_TACTILE_INACTIVE_CONTROL_TASK",
        "warning": "HAMMER_NAIL_MAPPED_TACTILE_INACTIVE",
        "variants_disabled": False,
        "success_rates": {
            variant: summary["tasks"]["hammer_nail"][variant]["success_rate"]
            for variant in VARIANTS
        },
        "comparisons": {
            name: value["per_task"]["hammer_nail"] for name, value in primary_comparisons.items()
        },
        "status": "PASS",
    }
    secondary = secondary_analysis(lookup, samples, seed)
    training_seed = seed_analysis(rows, primary_comparisons)
    p3_rows = [row for row in rows if row["variant"] == "P3"]
    uncertainty = {
        "schema": "tactile3d-unit.s4-3-uncertainty-diagnostics.v1",
        "stage": "R14.7",
        "policy_variant": "P3",
        "rollouts": len(p3_rows),
        "status": "UNAVAILABLE_CAUSAL_INPUT_MISMATCH",
        "reason": "Frozen full S4.2 uncertainty requires forbidden future-derived Vision transition latent; no frozen A+H-only estimator exists.",
        "invocations": sum(row["uncertainty"]["invoked"] for row in p3_rows),
        "interventions": sum(row["uncertainty"]["intervention"] for row in p3_rows),
        "success_failure_correlation": None,
        "contact_onset_correlation": None,
        "large_force_correlation": None,
        "timeout_correlation": None,
        "scientific_status": "DIAGNOSTIC ONLY",
        "scientific_rollout_status": "PASS",
    }
    for name, value in {
        "closed_loop_summary.json": summary,
        "primary_statistics.json": primary,
        "tactile_active_statistics.json": tactile,
        "hammer_control_analysis.json": hammer,
        "secondary_metrics.json": secondary,
        "training_seed_analysis.json": training_seed,
        "uncertainty_diagnostics.json": uncertainty,
    }.items():
        atomic_json(ARTIFACT_ROOT / name, value)
    print(
        json.dumps(
            {
                "learning_floor": summary["benchmark_learning_floor"],
                "primary": {
                    name: value["classification"] for name, value in primary_comparisons.items()
                },
                "status": "PASS",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
