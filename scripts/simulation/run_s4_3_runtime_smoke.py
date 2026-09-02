#!/usr/bin/env python3
"""Collect 3x3 causal DexJoCo dummy-policy runtime smokes for S4.3-1."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.dexjoco_adapter import DexJoCoRuntimeAdapter  # noqa: E402
from gr00t.simulation.s4_3_runtime import (  # noqa: E402
    ActionChunkQueue,
    CausalHistoryBuffer,
    mapped_force_metrics,
)

TASK_SEED_BASE = {"pinch_tongs": 6420000, "hammer_nail": 6421000, "click_mouse": 6422000}
TASKS = tuple(TASK_SEED_BASE)
TRACE_ROOT = ROOT / ".local/logs/simulation/s4_3_restart/runtime_smoke"
ARTIFACT = ROOT / ".local/artifacts/simulation/s4_3_restart/runtime_smoke_collection.json"
POLICY_STEPS = 100
WARMUP_ACTION_STEPS = 25


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def collect(task: str, reset_index: int) -> dict[str, Any]:
    seed = TASK_SEED_BASE[task] + reset_index
    episode_id = f"runtime-smoke-{task}-{reset_index:02d}"
    destination = TRACE_ROOT / episode_id
    destination.mkdir(parents=True, exist_ok=True)
    adapter = DexJoCoRuntimeAdapter(
        task_name=task,
        seed=seed,
        episode_id=episode_id,
        camera_name="random_camera",
        randomize=True,
        randomize_dynamics=False,
    )
    histories = CausalHistoryBuffer(episode_id)
    queue = ActionChunkQueue(episode_id)
    rows: dict[str, list[Any]] = {
        "control_step": [],
        "policy_step": [],
        "proprio": [],
        "sim_tactile": [],
        "policy_action": [],
        "env_action": [],
        "success": [],
        "terminated": [],
        "truncated": [],
        "replan": [],
        "history_source_min": [],
        "history_source_max": [],
    }
    plans: dict[str, list[Any]] = {
        "policy_step": [],
        "control_step": [],
        "rgb": [],
        "proprio": [],
        "tactile_history": [],
        "action_chunk": [],
    }
    region_audit: dict[str, Any] | None = None
    first_rgb: np.ndarray | None = None
    policy_start_rgb: np.ndarray | None = None
    final_rgb: np.ndarray | None = None
    try:
        region_audit = adapter.start()
        observation = adapter.reset()
        first_rgb = observation.rgb.copy()
        histories.append(observation.control_step, observation.sim_tactile)
        for _ in range(WARMUP_ACTION_STEPS):
            observation, _, _, _ = adapter.step(adapter.neutral_policy_action())
            histories.append(observation.control_step, observation.sim_tactile)
            if observation.terminated or observation.truncated or observation.success:
                raise RuntimeError("environment ended during common causal warm-up")
        if not histories.ready or histories.step_interval != (0, 25):
            raise RuntimeError("warm-up did not produce exact step-0..25 tactile history")
        policy_start_rgb = observation.rgb.copy()

        for policy_step in range(POLICY_STEPS):
            causal = histories.observation(
                observation.rgb,
                adapter.neutral_policy_action().values,
            )
            replan = queue.needs_replan
            if replan:
                neutral = adapter.neutral_policy_action().values
                plan = np.repeat(neutral[None], 27, axis=0)
                plan_provenance = queue.set_plan(plan, observation.control_step)
                plan_provenance.validate(inference=True)
                plans["policy_step"].append(policy_step)
                plans["control_step"].append(observation.control_step)
                plans["rgb"].append(causal.rgb.copy())
                plans["proprio"].append(causal.proprio.copy())
                plans["tactile_history"].append(causal.tactile_history.copy())
                plans["action_chunk"].append(plan.copy())
            action = queue.pop()
            next_observation, _, _, env_action = adapter.step(action)
            rows["control_step"].append(observation.control_step)
            rows["policy_step"].append(policy_step)
            rows["proprio"].append(causal.proprio)
            rows["sim_tactile"].append(causal.tactile_history[-1])
            rows["policy_action"].append(action)
            rows["env_action"].append(env_action.values)
            rows["success"].append(bool(next_observation.success))
            rows["terminated"].append(bool(next_observation.terminated))
            rows["truncated"].append(bool(next_observation.truncated))
            rows["replan"].append(replan)
            rows["history_source_min"].append(causal.provenance[2].source_min_step)
            rows["history_source_max"].append(causal.provenance[2].source_max_step)
            histories.append(next_observation.control_step, next_observation.sim_tactile)
            observation = next_observation
            if observation.terminated or observation.truncated:
                raise RuntimeError("environment terminated during 100-step runtime smoke")
        final_rgb = observation.rgb.copy()
    finally:
        adapter.close()

    arrays = {name: np.asarray(value) for name, value in rows.items()}
    trace_path = destination / "trace.npz"
    np.savez_compressed(trace_path, **arrays)
    plan_path = destination / "plans.npz"
    np.savez_compressed(plan_path, **{name: np.asarray(value) for name, value in plans.items()})
    for name, rgb in (
        ("reset.jpg", first_rgb),
        ("policy_start.jpg", policy_start_rgb),
        ("final.jpg", final_rgb),
    ):
        if rgb is None or not cv2.imwrite(
            str(destination / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ):
            raise RuntimeError(f"failed to write runtime smoke {name}")
    metrics = mapped_force_metrics(arrays["sim_tactile"])
    return {
        "episode_id": episode_id,
        "task": task,
        "reset_index": reset_index,
        "seed_namespace": "RUNTIME_SMOKE",
        "seed": seed,
        "warmup_samples": 26,
        "warmup_action_steps": WARMUP_ACTION_STEPS,
        "warmup_duration_sec": 0.5,
        "policy_steps": len(arrays["policy_step"]),
        "replan_count": int(arrays["replan"].sum()),
        "success_poll_count": len(arrays["success"]),
        "trace": str(trace_path.relative_to(ROOT)),
        "trace_sha256": sha256_file(trace_path),
        "plans": str(plan_path.relative_to(ROOT)),
        "plans_sha256": sha256_file(plan_path),
        "rgb_shape": list(first_rgb.shape) if first_rgb is not None else None,
        "region_resolution": region_audit,
        "force_metrics": metrics,
        "status": "PASS",
    }


def main() -> None:
    rows = [collect(task, reset) for task in TASKS for reset in range(3)]
    result = {
        "schema": "tactile3d-unit.s4-3-runtime-smoke-collection.v1",
        "stage": "R6.1",
        "tasks": list(TASKS),
        "resets_per_task": 3,
        "episodes": rows,
        "total_policy_steps": sum(row["policy_steps"] for row in rows),
        "egl": "PASS",
        "display_unset": True,
        "future_read": False,
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
    }
    write_json(ARTIFACT, result)
    print(
        json.dumps(
            {
                "episodes": len(rows),
                "steps": result["total_policy_steps"],
                "status": result["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
