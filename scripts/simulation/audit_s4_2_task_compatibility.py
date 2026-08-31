#!/usr/bin/env python3
"""Audit real DexJoCo tasks against the frozen S4.2 single-arm contract."""

from __future__ import annotations

import json
import os


TASKS = (
    "pinch_tongs",
    "hammer_nail",
    "click_mouse",
    "pick_bucket",
    "fold_glasses",
    "water_plant",
)

REQUIRED_HAND_BODIES = (
    "allegro_palm",
    "ff_base",
    "ff_proximal",
    "ff_medial",
    "ff_distal",
    "ff_tip",
    "mf_base",
    "mf_proximal",
    "mf_medial",
    "mf_distal",
    "mf_tip",
    "rf_base",
    "rf_proximal",
    "rf_medial",
    "rf_distal",
    "rf_tip",
    "th_base",
    "th_proximal",
    "th_medial",
    "th_distal",
    "th_tip",
)


def main() -> None:
    if os.environ.get("DISPLAY"):
        raise SystemExit("DISPLAY must be unset for the headless task audit")

    from dexjoco.tasks import CONFIG_MAPPING

    results = []
    for task in TASKS:
        env = None
        row: dict[str, object] = {"task": task}
        try:
            env = CONFIG_MAPPING[task]().get_environment(
                policy_mode=True,
                render_mode="rgb_array",
                randomize=False,
                randomize_dynamics=False,
                seed=42300 + len(results),
            )
            observation, _ = env.reset()
            raw = env.unwrapped
            body_names = tuple(raw.model.body(index).name for index in range(raw.model.nbody))
            missing_hand_bodies = sorted(set(REQUIRED_HAND_BODIES) - set(body_names))
            action_shape = tuple(env.action_space.shape)
            state_shape = tuple(observation["state"].shape)
            compatible = (
                action_shape == (23,)
                and len(state_shape) == 1
                and state_shape[0] >= 23
                and not missing_hand_bodies
                and abs(float(raw.control_dt) - 0.02) <= 1e-12
            )
            row.update(
                {
                    "action_shape": action_shape,
                    "body_names": body_names,
                    "compatible": compatible,
                    "control_dt_sec": float(raw.control_dt),
                    "missing_hand_bodies": missing_hand_bodies,
                    "physics_dt_sec": float(raw.physics_dt),
                    "state_shape": state_shape,
                    "status": "PASS" if compatible else "FAIL",
                }
            )
        except Exception as error:  # pragma: no cover - runtime audit evidence
            row.update({"compatible": False, "error": repr(error), "status": "FAIL"})
        finally:
            if env is not None:
                env.close()
        results.append(row)

    output = {
        "schema": "tactile3d-unit.s4-2-task-compatibility-audit.v1",
        "contract": {
            "action_environment_dim": 23,
            "action_policy_dim": 22,
            "control_dt_sec": 0.02,
            "hand": "right Allegro",
            "tactile_regions": 5,
        },
        "results": results,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    if sum(bool(row["compatible"]) for row in results) < 2:
        raise SystemExit("fewer than two compatible single-arm DexJoCo tasks")


if __name__ == "__main__":
    main()
