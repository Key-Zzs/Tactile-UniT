#!/usr/bin/env python3
"""Audit native DexJoCo success contracts for the S4.3-PD tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEXJOCO_ENV_ROOT = ROOT / "third_party/dexjoco/dexjoco/dexjoco/sim/envs"
DEFAULT_ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_pd"
TASKS = ("pinch_tongs", "hammer_nail", "click_mouse")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def contracts() -> dict[str, dict[str, Any]]:
    common_action = {
        "policy": "22D [TCP xyz 3, TCP rotvec 3, Allegro joint targets 16]",
        "environment": "23D [TCP xyz 3, TCP quaternion wxyz 4, Allegro joint targets 16]",
        "adapter": "gr00t.simulation.dexjoco_adapter.policy_action_to_env_action",
    }
    return {
        "pinch_tongs": {
            "source_file": "third_party/dexjoco/dexjoco/dexjoco/sim/envs/panda_pinch_tongs_env.py",
            "initial_state": "Panda and Allegro home; tongs free body on randomized table/object xy; tongs joint initialized open.",
            "controlled_robot": "single Panda arm with right Allegro hand",
            "objects": ["tongs free body", "tongs joint_0", "table"],
            "target": "lift tongs at least 0.10 m above table and complete three open-close pinches",
            "success_predicate": "tongs_pos.z >= table_z + 0.10 AND pinch_count >= 3 continuously for 30 control steps",
            "termination_predicate": "native success OR env_step >= 1000",
            "reward": "1.0 exactly on native success, otherwise 0.0",
            "action": common_action,
            "task_state": ["_pinch_count", "_pinch_in_progress", "_success_counter"],
            "sites_bodies_joints_geoms": [
                "attachment_site",
                "table_top",
                "tongs_root",
                "joint_0",
                "tongs_pos",
                "tongs_joint_0_pos",
            ],
            "negative_state": "fresh reset",
            "known_positive_state": "tongs physical pose above _lift_z plus native pinch_count=3 held for 30 predicate updates",
        },
        "hammer_nail": {
            "source_file": "third_party/dexjoco/dexjoco/dexjoco/sim/envs/panda_hammer_nail_env.py",
            "initial_state": "Panda and Allegro home; randomized hammer xy/yaw, nail xy, and table height; nail depth zero.",
            "controlled_robot": "single Panda arm with right Allegro hand",
            "objects": ["hammer free body", "nail mocap body", "wood", "table"],
            "target": "strike the nail with sufficient downward hammer-face velocity until insertion depth reaches 0.04 m",
            "success_predicate": "_nail_depth >= _success_depth (default 0.04 m)",
            "termination_predicate": "native success OR env_step >= 1000",
            "reward": "1.0 exactly on native success, otherwise 0.0",
            "action": common_action,
            "task_state": ["_nail_depth", "_vz_buf", "hammer_hit"],
            "sites_bodies_joints_geoms": [
                "attachment_site",
                "hammer_joint",
                "hammer_body",
                "face",
                "nail",
                "nail_head",
                "nail_shaft",
                "wood",
            ],
            "negative_state": "fresh reset with nail depth 0.0 m",
            "known_positive_state": "native set_nail_depth(_success_depth) state constructor",
        },
        "click_mouse": {
            "source_file": "third_party/dexjoco/dexjoco/dexjoco/sim/envs/panda_click_mouse_env.py",
            "initial_state": "Panda home; task-specific Allegro default pose; randomized mouse xy/yaw, mousepad offset, monitor xy, and table height.",
            "controlled_robot": "single Panda arm with right Allegro hand",
            "objects": ["mouse free body", "left mouse button joint", "mousepad", "display", "table"],
            "target": "place mouse on its pad and depress the left button enough to turn the display blue",
            "success_predicate": "mouse remains inside mousepad AND display_blue for 10 consecutive control steps",
            "termination_predicate": "native success OR env_step >= 1200",
            "reward": "1.0 exactly on native success, otherwise 0.0",
            "action": common_action,
            "task_state": ["_display_blue", "_mouse_joint0_init", "_success_trigger_count"],
            "sites_bodies_joints_geoms": [
                "attachment_site",
                "mouse_root",
                "mouse_joint0_pos",
                "mouse_pos",
                "mousepad",
                "display_root",
            ],
            "negative_state": "fresh reset",
            "known_positive_state": "mouse physical xy moved onto pad with native display-blue click latch active for 10 predicate updates",
        },
    }


class _PhysicsOnlyRenderer:
    """Avoid GL use while leaving reset, physics, reward, and success untouched."""

    def __init__(self, *_args: Any, **_kwargs: Any):
        pass

    def render(self, *_args: Any, **_kwargs: Any) -> np.ndarray:
        return np.zeros((640, 640, 3), dtype=np.uint8)

    def close(self) -> None:
        pass


def _patch_renderers() -> None:
    from dexjoco.sim.envs import (
        panda_click_mouse_env,
        panda_hammer_nail_env,
        panda_pinch_tongs_env,
    )

    for module in (
        panda_click_mouse_env,
        panda_hammer_nail_env,
        panda_pinch_tongs_env,
    ):
        module.MujocoRenderer = _PhysicsOnlyRenderer


def physics_crosscheck() -> dict[str, Any]:
    if os.environ.get("DISPLAY"):
        raise RuntimeError("DISPLAY must be unset for the physics-only cross-check")
    import mujoco
    from dexjoco.tasks import CONFIG_MAPPING

    _patch_renderers()
    results: dict[str, Any] = {}
    for index, task in enumerate(TASKS):
        env = CONFIG_MAPPING[task]().get_environment(
            policy_mode=True,
            render_mode="rgb_array",
            randomize=False,
            randomize_dynamics=False,
            seed=930000 + index,
        )
        try:
            _observation, info = env.reset()
            raw = env.unwrapped
            negative = bool(raw._compute_success())
            if task == "pinch_tongs":
                qpos = raw._data.jnt("tongs_root").qpos
                qpos[2] = raw._lift_z + 0.02
                raw._pinch_count = 3
                mujoco.mj_forward(raw._model, raw._data)
                timeline = [bool(raw._compute_success()) for _ in range(30)]
            elif task == "hammer_nail":
                raw.set_nail_depth(raw._success_depth)
                timeline = [bool(raw._compute_success())]
            else:
                pad_center = raw._data.geom_xpos[raw._mousepad_geom_id].copy()
                raw._data.jnt("mouse_root").qpos[:2] = pad_center[:2]
                mujoco.mj_forward(raw._model, raw._data)
                raw._set_display_blue()
                timeline = [bool(raw._compute_success()) for _ in range(10)]
            positive = bool(timeline[-1])
            results[task] = {
                "reset_info_succeed": bool(info.get("succeed", False)),
                "obvious_negative": negative,
                "known_positive": positive,
                "positive_timeline_steps": len(timeline),
                "task_semantics_modified": False,
                "renderer_substitution": "PHYSICS_ONLY_AUDIT; NOT_USED_FOR_ACQUISITION",
                "status": "PASS" if not negative and positive else "FAIL",
            }
        finally:
            env.close()
    return results


def visualize(value: dict[str, Any], destination: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for axis, task in zip(axes, TASKS):
        item = value["tasks"][task]
        axis.axis("off")
        axis.text(
            0.5,
            0.72,
            task,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            transform=axis.transAxes,
        )
        axis.text(
            0.5,
            0.42,
            item["success_predicate"],
            ha="center",
            va="center",
            wrap=True,
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.6", "facecolor": "#e8f5e9"},
            transform=axis.transAxes,
        )
        axis.annotate(
            "native success → reward 1 → terminate",
            xy=(0.5, 0.17),
            ha="center",
            fontsize=9,
            transform=axis.transAxes,
        )
    fig.suptitle("S4.3-PD native success criteria")
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def audit(*, run_physics: bool) -> dict[str, Any]:
    task_values = contracts()
    for task, item in task_values.items():
        source = ROOT / item["source_file"]
        item["source_sha256"] = sha256_file(source)
    checks = physics_crosscheck() if run_physics else None
    passed = checks is None or all(item["status"] == "PASS" for item in checks.values())
    value = {
        "schema": "tactile3d-unit.s4-3-pd-task-success-contract.v1",
        "stage": "PD2",
        "priority": "native DexJoCo task methods and state machines",
        "tasks": task_values,
        "physics_crosscheck": checks,
        "native_success_valid": passed if checks is not None else "NOT_RUN",
        "task_semantics_modified": False,
        "status": "PASS" if passed and checks is not None else "NOT_RUN",
    }
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--physics-check", action="store_true")
    parser.add_argument("--skip-visualization", action="store_true")
    parser.add_argument("--visualize-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = args.artifact_root / "task_success_contract.json"
    plot = args.artifact_root / "plots/02_native_success_criteria.png"
    if args.visualize_existing:
        value = json.loads(artifact.read_text(encoding="utf-8"))
        visualize(value, plot)
        print(json.dumps({"stage": "PD2", "visualization": "PASS"}, sort_keys=True))
        return
    value = audit(run_physics=args.physics_check)
    write_json(artifact, value)
    if not args.skip_visualization:
        visualize(value, plot)
    print(json.dumps({"stage": "PD2", "status": value["status"]}, sort_keys=True))
    if value["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
