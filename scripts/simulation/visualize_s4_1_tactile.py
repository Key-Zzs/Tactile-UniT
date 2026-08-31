#!/usr/bin/env python3
"""Create an offline RGB/contact visualization from a logged debug episode."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_1"
EPISODE = ARTIFACTS / "debug_dataset/debug-000"
OUTPUT = ARTIFACTS / "visualization/pinch_tongs_tactile.png"
REGIONS = ("palm", "index", "middle", "ring", "thumb")


def main() -> None:
    values = np.load(EPISODE / "steps.npz")
    tactile = values["sim_tactile"].reshape(-1, len(REGIONS), 6)
    normal = tactile[:, :, 1]
    tangent = tactile[:, :, 2]
    frame_index = int(np.argmax(normal.sum(axis=1)))
    frame = cv2.imread(str(EPISODE / values["rgb_reference"][frame_index]))
    if frame is None:
        raise SystemExit("logged RGB frame is unreadable")
    canvas = np.full((760, 1280, 3), 248, dtype=np.uint8)
    canvas[40:680, 20:660] = cv2.resize(frame, (640, 640))
    cv2.putText(canvas, f"pinch_tongs step {frame_index}", (20, 30), 0, 0.7, (20, 20, 20), 2)
    max_force = max(float(normal[frame_index].max()), 1e-6)
    for index, region in enumerate(REGIONS):
        y = 80 + index * 70
        width = int(430 * normal[frame_index, index] / max_force)
        cv2.putText(canvas, region, (690, y + 20), 0, 0.55, (30, 30, 30), 1)
        cv2.rectangle(canvas, (790, y), (790 + width, y + 25), (40, 90, 220), -1)
        cv2.putText(
            canvas,
            f"Fn={normal[frame_index,index]:.2f} N Ft={tangent[frame_index,index]:.2f} N",
            (790, y + 48),
            0,
            0.48,
            (30, 30, 30),
            1,
        )
    x0, y0, width, height = 690, 500, 550, 190
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (60, 60, 60), 1)
    series = normal.sum(axis=1)
    scale = max(float(series.max()), 1e-6)
    points = np.asarray(
        [
            [x0 + int(index * width / (len(series) - 1)), y0 + height - int(value * height / scale)]
            for index, value in enumerate(series)
        ],
        dtype=np.int32,
    )
    cv2.polylines(canvas, [points], False, (220, 80, 40), 2)
    cv2.putText(canvas, "total normal force timeline", (x0, y0 - 12), 0, 0.55, (30, 30, 30), 1)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(OUTPUT), canvas):
        raise OSError(f"failed to write {OUTPUT}")
    metadata = {
        "episode": "debug-000",
        "selected_step": frame_index,
        "selection": "maximum total normal force",
        "output": "visualization/pinch_tongs_tactile.png",
        "status": "PASS",
    }
    (OUTPUT.parent / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
