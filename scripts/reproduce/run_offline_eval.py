#!/usr/bin/env python3
"""Run the official UniT offline evaluator with an isolated result directory.

The task list and default protocol are read from ``examples/run_eval_loss.sh``;
the wrapper only redirects the official evaluator's output and does not alter
policy, preprocessing, split, or metric computation.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "examples/run_eval_loss.sh"


def official_tasks() -> list[str]:
    text = RECIPE.read_text(encoding="utf-8")
    match = re.search(r"GR1_DATASETS=\(\n(?P<body>.*?)\n\)", text, flags=re.DOTALL)
    if match is None:
        raise RuntimeError(f"Could not parse GR1_DATASETS from {RECIPE}")
    tasks = re.findall(r"^\s*(gr1_unified\.[^\s#]+)\s*$", match.group("body"), flags=re.MULTILINE)
    if len(tasks) != 24 or len(set(tasks)) != 24:
        raise RuntimeError(f"Expected 24 unique official tasks, found {len(tasks)}")
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-config", default="fourier_gr1_arms_waist_gausNorm_crop_cam_ego_joints_only")
    parser.add_argument("--data-split", default="[-2:]")
    parser.add_argument("--trajs", type=int, default=2)
    parser.add_argument("--cuda-visible-devices", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    tasks = official_tasks()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_paths = [str(Path(args.dataset_root) / task) for task in tasks]
    command = [
        sys.executable,
        "-u",
        "scripts/eval_policy_unit.py",
        "--dataset-path",
        *dataset_paths,
        "--model_path",
        args.checkpoint,
        "--data-config",
        args.data_config,
        "--embodiment_tag",
        "gr1",
        "--trajs",
        str(args.trajs),
        "--data_split",
        args.data_split,
        "--save_results_path",
        str(output_dir),
        "--plot_state",
    ]
    print("Official recipe:", RECIPE)
    print("Tasks:", len(tasks))
    print("Trajectories per task:", args.trajs)
    print("Data split:", args.data_split)
    print("Output:", output_dir)
    print("Executing official evaluator:", " ".join(command))
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
