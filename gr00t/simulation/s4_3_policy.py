"""Frozen causal policy-benchmark contracts for restarted S4.3."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

TASKS = ("pinch_tongs", "hammer_nail", "click_mouse")
VARIANTS = ("P0", "P1", "P2", "P3")
TRAINING_SEEDS = (0, 1, 2)
FAILURE_CLASSES = (
    "SUCCESS",
    "TIMEOUT",
    "ENV_TERMINATION_FAILURE",
    "INVALID_ACTION",
    "NUMERIC_FAILURE",
    "SIMULATION_EXCEPTION",
)


def valid_bc_windows(length: int, *, history_steps: int = 26, action_steps: int = 27) -> int:
    """Count anchors with complete history and future Action target."""

    return max(0, int(length) - history_steps - action_steps + 2)


def timeout_statistics(lengths: Sequence[int]) -> dict[str, float | int]:
    """Apply the preregistered successful-TRAIN timeout rule."""

    values = np.asarray(lengths, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("timeout lengths must be a non-empty finite vector")
    if np.any(values <= 0):
        raise ValueError("episode lengths must be positive")
    p50, p90, p95 = np.percentile(values, (50, 90, 95), method="linear")
    timeout = min(1000, max(250, math.ceil(1.5 * float(p95))))
    return {
        "p50": float(p50),
        "p90": float(p90),
        "p95": float(p95),
        "timeout_steps": int(timeout),
    }


def validate_policy_eval_resets(config: Mapping[str, Any]) -> None:
    """Fail closed unless POLICY_EVAL_V1 contains 30 unique resets per task."""

    if config.get("status") != "FROZEN_BEFORE_ACT_TRAINING":
        raise ValueError("POLICY_EVAL_V1 is not frozen before training")
    rows = list(config.get("resets", ()))
    if len(rows) != 90:
        raise ValueError("POLICY_EVAL_V1 must contain exactly 90 reset specs")
    identities = {row.get("evaluation_reset_id") for row in rows}
    seeds = {row.get("reset_seed") for row in rows}
    if len(identities) != 90 or len(seeds) != 90:
        raise ValueError("evaluation reset identities and seeds must be unique")
    for task in TASKS:
        scoped = [row for row in rows if row.get("task") == task]
        if len(scoped) != 30 or {row.get("reset_index") for row in scoped} != set(range(30)):
            raise ValueError(f"{task} does not have the exact 30-reset index set")
        for row in scoped:
            if row.get("seed_namespace") != "POLICY_EVAL_V1":
                raise ValueError("unexpected reset seed namespace")
            if row.get("overlap_with_prior_sources") is not False:
                raise ValueError("evaluation reset overlaps a prior source")
            seed = row.get("reset_seed")
            if not (seed == row.get("environment_seed") == row.get("visual_randomization_seed")):
                raise ValueError("reset/environment/visual seeds must be identical")
            if row.get("dynamics_randomization") is not False:
                raise ValueError("POLICY_EVAL_V1 dynamics randomization must remain disabled")


def material_effect(delta: float, ci: Sequence[float]) -> str:
    """Classify success-rate contrasts with the frozen five-point rule."""

    if len(ci) != 2 or not np.isfinite([delta, *ci]).all() or ci[0] > ci[1]:
        raise ValueError("material-effect inputs must define a finite ordered CI")
    if delta >= 0.05 and ci[0] > 0.0:
        return "MATERIAL_IMPROVEMENT"
    if delta <= -0.05 and ci[1] < 0.0:
        return "MATERIAL_HURT"
    return "NO_MATERIAL_DIFFERENCE"


def macro_task_success(task_success: Mapping[str, float], tasks: Iterable[str] = TASKS) -> float:
    """Equal-weight task macro; never weight by rollout count."""

    selected = tuple(tasks)
    if not selected:
        raise ValueError("macro success needs at least one task")
    values = np.asarray([task_success[task] for task in selected], dtype=np.float64)
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("task success rates must be finite probabilities")
    return float(values.mean())


def learning_sanity(max_success_by_task: Mapping[str, float]) -> str:
    """Apply the frozen HEALTHY/WEAK/ZERO_LEARNING classification."""

    values = np.asarray([max_success_by_task[task] for task in TASKS], dtype=np.float64)
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("maximum task success must be finite probabilities")
    if np.all(values == 0.0):
        return "ZERO_LEARNING"
    if int(np.sum(values >= 0.20)) >= 2:
        return "HEALTHY"
    return "WEAK"


@dataclass(frozen=True)
class TensorProvenance:
    """Timestamp provenance attached to causal runtime tensors."""

    episode_id: str
    current_control_step: int
    source_min_step: int
    source_max_step: int
    role: str

    def validate(self, *, inference: bool) -> None:
        if self.role not in {"OBSERVATION", "PLAN", "TRAINING_TARGET", "DIAGNOSTIC"}:
            raise ValueError(f"unknown tensor role {self.role!r}")
        if self.source_min_step > self.source_max_step:
            raise ValueError("tensor provenance has an inverted source interval")
        if self.role == "OBSERVATION" and self.source_max_step > self.current_control_step:
            raise ValueError("future observation leakage")
        if inference and self.role == "TRAINING_TARGET":
            raise ValueError("training target is unavailable at inference")


def assert_inference_provenance(items: Iterable[TensorProvenance]) -> None:
    for item in items:
        item.validate(inference=True)
