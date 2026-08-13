#!/usr/bin/env python3
"""Validate the GR1 LeRobot data contract through UniT's active loader.

This checker follows ``examples/run_gr1_full.sh`` and uses
``LeRobotSingleDatasetWithGoalImage`` plus the configured UniT transforms. It
never starts training or evaluation. The vendor stats format is not directly
compatible with UniT's loader; if the loader needs to rebuild its derived stats,
the write is redirected to ``.local/cache/`` so the shared dataset remains
read-only.
"""

from __future__ import annotations

import argparse
import builtins
import json
import os
import re
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = ROOT / ".local/artifacts/reproduction/phase2"
DEFAULT_LOG = ROOT / ".local/logs/reproduction/phase2/data_contract.log"


class Reporter:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.checks: dict[str, dict[str, Any]] = {}

    def log(self, message: str) -> None:
        print(message)
        self.lines.append(message)

    def record(self, name: str, passed: bool, **details: Any) -> None:
        self.checks[name] = {"status": "PASS" if passed else "FAIL", **details}
        self.log(f"{name:<34} {'PASS' if passed else 'FAIL'}")


def parse_recipe() -> tuple[list[str], str, Path]:
    """Parse the current full-GR1 recipe instead of duplicating its contract."""
    recipe = ROOT / "examples/run_gr1_full.sh"
    text = recipe.read_text(encoding="utf-8")
    tasks_block = re.search(r"GR1_DATASETS=\(\n(?P<tasks>.*?)\n\)", text, flags=re.DOTALL)
    config_match = re.search(r"^GR1_JOINTS_CONFIG=([^\n\s]+)$", text, flags=re.MULTILINE)
    model_match = re.search(r'^PRETRAIN_BASE_CONFIG="\$PROJECT_ROOT/(?P<path>[^"]+)"$', text, flags=re.MULTILINE)
    if tasks_block is None or config_match is None or model_match is None:
        raise RuntimeError(f"Could not parse active GR1 settings from {recipe}")
    tasks = re.findall(r"^\s*(gr1_unified\.[^\s#]+)\s*$", tasks_block.group("tasks"), flags=re.MULTILINE)
    if len(tasks) != 24 or len(set(tasks)) != 24:
        raise RuntimeError(f"Expected 24 unique tasks in {recipe}, found {len(tasks)}")
    model_config = ROOT / model_match.group("path")
    return tasks, config_match.group(1), model_config


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def numeric_summary(value: Any) -> dict[str, Any]:
    array = tensor_to_numpy(value)
    numeric = np.issubdtype(array.dtype, np.number)
    if not numeric:
        return {"shape": list(array.shape), "dtype": str(array.dtype), "numeric": False}
    finite = np.isfinite(array)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "numeric": True,
        "nan_count": int(np.isnan(array).sum()),
        "inf_count": int(np.isinf(array).sum()),
        "finite": bool(finite.all()),
        "min": float(array[finite].min()) if finite.any() else None,
        "max": float(array[finite].max()) if finite.any() else None,
    }


@contextmanager
def redirect_loader_stats_writes(dataset_root: Path, cache_root: Path) -> Iterator[None]:
    """Redirect only UniT's fallback ``meta/stats.json`` writes into .local."""
    original_open = builtins.open
    dataset_root = dataset_root.resolve()
    cache_root = cache_root.resolve()

    def local_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, os.PathLike)) and any(flag in mode for flag in ("w", "a", "x", "+")):
            candidate = Path(file).resolve()
            try:
                relative = candidate.relative_to(dataset_root)
            except ValueError:
                relative = None
            if relative is not None and candidate.name == "stats.json" and candidate.parent.name == "meta":
                redirected = cache_root / relative
                redirected.parent.mkdir(parents=True, exist_ok=True)
                return original_open(redirected, mode, *args, **kwargs)
        if isinstance(file, (str, os.PathLike)) and "r" in mode:
            candidate = Path(file).resolve()
            try:
                relative = candidate.relative_to(dataset_root)
            except ValueError:
                relative = None
            if relative is not None and candidate.name == "stats.json" and candidate.parent.name == "meta":
                redirected = cache_root / relative
                if redirected.is_file():
                    return original_open(redirected, mode, *args, **kwargs)
        return original_open(file, mode, *args, **kwargs)

    builtins.open = local_open
    try:
        yield
    finally:
        builtins.open = original_open


def sample_indices(length: int, mode: str) -> list[int]:
    if length <= 0:
        return []
    if mode == "quick":
        return [length // 2]
    return sorted(set((0, length // 2, length - 1)))


def write_task_sample(artifact_dir: Path, task: str, report: dict[str, Any]) -> None:
    samples_dir = artifact_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    (samples_dir / f"{task}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_task(
    task: str,
    dataset_path: Path,
    config_name: str,
    model_config: dict[str, Any],
    mode: str,
    cache_root: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    """Open one task with the official loader and transform representative steps."""
    import torch

    from gr00t.data.dataset import LeRobotSingleDatasetWithGoalImage
    from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING, EmbodimentTag
    from gr00t.experiment.data_config_unit import load_data_config

    np.random.seed(0)
    torch.manual_seed(0)
    unit_cfg = model_config["unit_cfg"]
    backbone_cfg = model_config["backbone_cfg"]
    data_config = load_data_config(
        config_name,
        eagle_path=backbone_cfg["eagle_path"],
        use_bridge=bool(unit_cfg["use_bridge"]),
        num_bridge_tokens=unit_cfg["num_bridge_tokens"] if unit_cfg["use_bridge"] else None,
        enable_imagenet_preprocessing=True,
    )
    expected_horizon = len(data_config.action_indices)
    expected_embodiment = EMBODIMENT_TAG_MAPPING[EmbodimentTag("gr1").value]
    task_report: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "loader_opened": False,
        "samples": [],
        "episode_lengths": {},
        "errors": [],
        "expected_action_horizon": expected_horizon,
        "model_action_horizon": model_config["action_head_cfg"]["action_horizon"],
        "expected_embodiment_id": expected_embodiment,
    }
    with redirect_loader_stats_writes(dataset_path, cache_root / task):
        dataset = LeRobotSingleDatasetWithGoalImage(
            dataset_path=dataset_path,
            modality_configs=data_config.modality_config(),
            transforms=data_config.transform(),
            embodiment_tag="gr1",
            video_backend="decord",
        )
        task_report["loader_opened"] = True
        lengths = np.asarray(dataset.trajectory_lengths)
        task_report["episode_lengths"] = {
            "count": int(len(lengths)),
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": float(lengths.mean()),
        }
        for index in sample_indices(len(dataset), mode):
            trajectory_id, base_index = dataset.all_steps[index]
            raw = dataset.get_step_data(trajectory_id, base_index)
            raw_video = numeric_summary(raw["video.ego_view"])
            language = raw["annotation.human.coarse_action"]
            transformed = dataset.transforms(raw)
            state = numeric_summary(transformed["state"])
            action = numeric_summary(transformed["action"])
            observation = numeric_summary(transformed["imagenet_obs_images"])
            goal = numeric_summary(transformed["imagenet_goal_images"])
            language_ok = isinstance(language, list) and bool(language) and all(isinstance(item, str) and item for item in language)
            embodiment = int(np.asarray(transformed["embodiment_id"]).item())
            sample_report = {
                "dataset_index": int(index),
                "episode_index": int(trajectory_id),
                "frame_index": int(base_index),
                "raw_video": raw_video,
                "state": state,
                "action": action,
                "observation_image": observation,
                "goal_image": goal,
                "goal_images_present": "goal_images" in transformed,
                "language_ok": language_ok,
                "language": language,
                "embodiment_id": embodiment,
                "action_horizon": int(np.asarray(transformed["action"]).shape[0]),
            }
            task_report["samples"].append(sample_report)
    write_task_sample(artifact_dir, task, task_report)
    return task_report


def aggregate_task_reports(
    task_reports: dict[str, dict[str, Any]],
    expected_model_horizon: int,
    reporter: Reporter,
) -> bool:
    samples = [sample for report in task_reports.values() for sample in report["samples"]]
    task_count_ok = len(task_reports) == 24 and all(report["loader_opened"] for report in task_reports.values())
    rgb_ok = bool(samples) and all(sample["observation_image"]["finite"] and sample["observation_image"]["shape"][1:] == [3, 224, 224] for sample in samples)
    goal_ok = bool(samples) and all(sample["goal_image"]["finite"] and sample["goal_image"]["shape"] == [1, 3, 224, 224] and sample["goal_images_present"] for sample in samples)
    state_ok = bool(samples) and all(sample["state"]["finite"] and sample["state"]["shape"] == [1, 128] for sample in samples)
    action_ok = bool(samples) and all(sample["action"]["finite"] and sample["action"]["shape"][1] == 128 for sample in samples)
    horizon_ok = bool(samples) and all(sample["action_horizon"] == expected_model_horizon for sample in samples)
    language_ok = bool(samples) and all(sample["language_ok"] for sample in samples)
    embodiment_ok = bool(samples) and all(sample["embodiment_id"] == report["expected_embodiment_id"] for report in task_reports.values() for sample in report["samples"])
    finite_ok = bool(samples) and all(
        sample[modality]["nan_count"] == 0 and sample[modality]["inf_count"] == 0
        for sample in samples
        for modality in ("state", "action", "observation_image", "goal_image")
    )
    reporter.record("All 24 required tasks discoverable", task_count_ok, tasks=len(task_reports))
    reporter.record("Official UniT loader opens all tasks", task_count_ok)
    reporter.record("Official transforms produce valid samples", rgb_ok and goal_ok and state_ok and action_ok)
    reporter.record("RGB observation contract", rgb_ok)
    reporter.record("Goal image contract", goal_ok)
    reporter.record("State contract", state_ok)
    reporter.record("Action contract", action_ok)
    reporter.record("Action horizon matches model", horizon_ok, expected=expected_model_horizon)
    reporter.record("Language contract", language_ok)
    reporter.record("GR1 embodiment tag", embodiment_ok)
    reporter.record("No sampled NaN / Inf", finite_ok)
    return all(
        (task_count_ok, rgb_ok, goal_ok, state_ok, action_ok, horizon_ok, language_ok, embodiment_ok, finite_ok)
    )


def configure_hf_cache(hf_home: Path) -> None:
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HF_HUB_OFFLINE"] = "1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=os.environ.get("GR1_DATASET_DIR"), help="LeRobot root containing all gr1_unified.* tasks.")
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--hf-home", help="Shared Hugging Face root; defaults to $HF_HOME or $UNIT_STORAGE/huggingface.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()
    if not args.dataset_root:
        parser.error("--dataset-root or GR1_DATASET_DIR is required")
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if args.hf_home:
        hf_home = Path(args.hf_home).expanduser().resolve()
    elif os.environ.get("UNIT_STORAGE"):
        hf_home = Path(os.environ["UNIT_STORAGE"]).expanduser().resolve() / "huggingface"
    else:
        hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser().resolve()
    configure_hf_cache(hf_home)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    reporter = Reporter()
    reporter.log("Tactile3D-UniT GR1 data-contract validation")
    reporter.log(f"Mode: {args.mode}")
    reporter.log(f"Dataset root: {dataset_root}")

    task_reports: dict[str, dict[str, Any]] = {}
    try:
        tasks, config_name, model_config_path = parse_recipe()
        model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
        expected_model_horizon = int(model_config["action_head_cfg"]["action_horizon"])
        reporter.log(f"Data config: {config_name}")
        reporter.log(f"Model action horizon: {expected_model_horizon}")
        for task in tasks:
            try:
                task_reports[task] = run_task(
                    task,
                    dataset_root / task,
                    config_name,
                    model_config,
                    args.mode,
                    ROOT / ".local/cache/gr1_loader_stats",
                    args.artifact_dir,
                )
                reporter.log(f"{task:<66} PASS")
            except Exception as error:
                task_reports[task] = {"loader_opened": False, "samples": [], "errors": [f"{type(error).__name__}: {error}"]}
                reporter.log(f"{task:<66} FAIL: {type(error).__name__}: {error}")
        passed = aggregate_task_reports(task_reports, expected_model_horizon, reporter)
    except Exception as error:
        passed = False
        reporter.record("Checker execution", False, error=f"{type(error).__name__}: {error}")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "dataset_root": str(dataset_root),
        "hf_home": str(hf_home),
        "checks": reporter.checks,
        "tasks": task_reports,
        "overall": "PASS" if passed else "FAIL",
    }
    (args.artifact_dir / "contract_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text("\n".join(reporter.lines) + "\n", encoding="utf-8")
    reporter.log(f"Summary: {args.artifact_dir / 'contract_summary.json'}")
    reporter.log(f"Log: {args.log}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
