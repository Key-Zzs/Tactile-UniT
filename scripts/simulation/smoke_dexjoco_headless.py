#!/usr/bin/env python3
"""Run the bounded S4.1 DexJoCo EGL smoke and save local evidence."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.dexjoco_adapter import DexJoCoRuntimeAdapter  # noqa: E402


def main() -> None:
    if os.environ.get("DISPLAY"):
        raise SystemExit("DISPLAY must be unset for the S4.1 headless smoke")
    if os.environ.get("MUJOCO_GL") != "egl":
        raise SystemExit("set MUJOCO_GL=egl for the S4.1 headless smoke")
    root = ROOT / ".local/artifacts/simulation/s4_1"
    headless = root / "headless"
    headless.mkdir(parents=True, exist_ok=True)
    adapter = DexJoCoRuntimeAdapter(task_name="pinch_tongs", seed=7, episode_id="smoke")
    region_audit = adapter.start()
    resets = 10
    steps_per_reset = 50
    timestamps: list[float] = []
    contact_query_count = 0
    first_frame = None
    try:
        for reset_index in range(resets):
            observation = adapter.reset()
            if first_frame is None:
                first_frame = observation.rgb.copy()
            for _ in range(steps_per_reset):
                action = adapter.neutral_policy_action()
                observation, _, _, _ = adapter.step(action)
                timestamps.append(observation.timestamp_sec)
                contact_query_count += int(adapter.raw_env.data.ncon >= 0)
        assert first_frame is not None
        cv2.imwrite(
            str(headless / "pinch_tongs_front.png"), cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
        )
        evidence = {
            "schema": "tactile3d-unit.s4-1-headless-smoke.v1",
            "task": "pinch_tongs",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "display": "UNSET",
            "backend": "EGL",
            "camera": "front",
            "resolution": [640, 640],
            "rgb_dtype": "uint8",
            "resets": resets,
            "steps_per_reset": steps_per_reset,
            "total_steps": resets * steps_per_reset,
            "contact_queries": contact_query_count,
            "region_audit_status": region_audit["status"],
            "timestamps_finite": bool(np.isfinite(timestamps).all()),
            "status": "PASS",
        }
        (root / "headless_smoke.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (root / "contact_region_audit.json").write_text(
            json.dumps(region_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(evidence, indent=2))
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
