#!/usr/bin/env python3
"""Validate the local assets required for the UniT S0 GR1 recipe.

The checker is intentionally offline. It never downloads models or datasets and
does not load the full VLA checkpoint. Results are written under ``.local/`` so
machine-specific paths and validation artifacts cannot enter Git.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / ".local/artifacts/reproduction/s0_assets.json"
DEFAULT_LOG = ROOT / ".local/logs/reproduction/s0_assets.log"
QWEN_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DINO_ID = "facebook/dinov2-large"


class Reporter:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.checks: dict[str, dict[str, Any]] = {}

    def log(self, message: str) -> None:
        print(message)
        self.lines.append(message)

    def record(self, name: str, passed: bool, **details: Any) -> None:
        self.checks[name] = {"status": "PASS" if passed else "FAIL", **details}
        self.log(f"{name:<32} {'PASS' if passed else 'FAIL'}")


def required_gr1_datasets() -> list[str]:
    """Read the task list from the active S0 recipe, not a duplicate list."""
    recipe = ROOT / "examples/run_gr1_full.sh"
    text = recipe.read_text(encoding="utf-8")
    match = re.search(r"GR1_DATASETS=\(\n(?P<tasks>.*?)\n\)", text, flags=re.DOTALL)
    if match is None:
        raise RuntimeError(f"Could not parse GR1_DATASETS from {recipe}")
    tasks = re.findall(r"^\s*(gr1_unified\.[^\s#]+)\s*$", match.group("tasks"), flags=re.MULTILINE)
    if len(tasks) != 24 or len(set(tasks)) != 24:
        raise RuntimeError(f"Expected 24 unique GR1 tasks in {recipe}, found {len(tasks)}")
    return tasks


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def check_gr1_task(path: Path) -> dict[str, Any]:
    required_meta = (
        "info.json",
        "episodes.jsonl",
        "tasks.jsonl",
        "modality.json",
        "metadata.json",
        "embodiment.json",
        "stats.json",
    )
    missing_meta = [name for name in required_meta if not (path / "meta" / name).is_file()]
    details: dict[str, Any] = {
        "path": str(path),
        "missing_meta": missing_meta,
        "episodes": 0,
        "parquet_missing": [],
        "video_missing": [],
        "parquet_files": 0,
        "video_files": 0,
        "errors": [],
    }
    if missing_meta:
        return details

    try:
        info = read_json(path / "meta/info.json")
        episodes = [json.loads(line) for line in (path / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines() if line]
        tasks = [json.loads(line) for line in (path / "meta/tasks.jsonl").read_text(encoding="utf-8").splitlines() if line]
        chunk_size = int(info["chunks_size"])
        data_pattern = str(info["data_path"])
        video_pattern = str(info["video_path"])
        video_keys = [key for key, value in info["features"].items() if value.get("dtype") == "video"]
        details["episodes"] = len(episodes)
        details["tasks"] = len(tasks)
        details["expected_episodes"] = int(info.get("total_episodes", -1))
        details["video_keys"] = video_keys
        if not episodes or not tasks or not video_keys:
            details["errors"].append("empty episodes, tasks, or video metadata")
            return details
        if details["expected_episodes"] != len(episodes):
            details["errors"].append("episode metadata count does not match info.json")

        parquet_paths: list[Path] = []
        video_paths: list[Path] = []
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            chunk_index = episode_index // chunk_size
            parquet = path / data_pattern.format(
                episode_chunk=chunk_index,
                episode_index=episode_index,
            )
            parquet_paths.append(parquet)
            if not parquet.is_file() or parquet.stat().st_size == 0:
                details["parquet_missing"].append(str(parquet.relative_to(path)))
            for video_key in video_keys:
                video = path / video_pattern.format(
                    episode_chunk=chunk_index,
                    episode_index=episode_index,
                    video_key=video_key,
                )
                video_paths.append(video)
                if not video.is_file() or video.stat().st_size == 0:
                    details["video_missing"].append(str(video.relative_to(path)))
        details["parquet_files"] = len(parquet_paths)
        details["video_files"] = len(video_paths)

        # Read parquet metadata at both ends of the episode list. This is cheap
        # and catches unreadable parquet files without scanning their contents.
        import pyarrow.parquet as pq

        for parquet in (parquet_paths[0], parquet_paths[-1]):
            pq.ParquetFile(parquet).metadata
    except Exception as error:  # report every task, not only the first failure
        details["errors"].append(f"{type(error).__name__}: {error}")
    return details


def check_gr1_assets(dataset_root: Path, reporter: Reporter) -> bool:
    tasks = required_gr1_datasets()
    task_reports = {task: check_gr1_task(dataset_root / task) for task in tasks}
    passed = all(
        report["path"]
        and not report["missing_meta"]
        and not report["parquet_missing"]
        and not report["video_missing"]
        and not report["errors"]
        for report in task_reports.values()
    )
    reporter.record(
        "GR1 required datasets",
        passed,
        dataset_root=str(dataset_root),
        required_count=len(tasks),
        discovered_count=sum((dataset_root / task).is_dir() for task in tasks),
        tasks=task_reports,
    )
    return passed


def check_safetensors_index(directory: Path) -> tuple[bool, dict[str, Any]]:
    details: dict[str, Any] = {"directory": str(directory), "missing": [], "headers": {}}
    config = directory / "config.json"
    index = directory / "model.safetensors.index.json"
    for path in (config, index):
        if not path.is_file():
            details["missing"].append(path.name)
    if details["missing"]:
        return False, details
    try:
        weight_map = read_json(index)["weight_map"]
        shard_names = sorted(set(weight_map.values()))
        for shard_name in shard_names:
            shard = directory / shard_name
            if not shard.is_file() or shard.stat().st_size == 0:
                details["missing"].append(shard_name)
                continue
            from safetensors import safe_open

            # ``keys`` parses the safetensors header only; it never materializes
            # tensors or loads a model.
            with safe_open(shard, framework="pt", device="cpu") as handle:
                details["headers"][shard_name] = len(handle.keys())
        details["indexed_shards"] = shard_names
        details["weight_count"] = len(weight_map)
    except Exception as error:
        details["error"] = f"{type(error).__name__}: {error}"
    return not details["missing"] and "error" not in details, details


def check_checkpoint(storage: Path, reporter: Reporter) -> bool:
    checkpoint = storage / "checkpoints/VLA-UniT-checkpoints/VLA-UniT-3B-fulldata"
    checkpoint_ok, checkpoint_details = check_safetensors_index(checkpoint)
    tokenizer_ok, tokenizer_details = check_safetensors_index(checkpoint / "tokenizer")
    reporter.record("VLA fulldata checkpoint", checkpoint_ok, **checkpoint_details)
    reporter.record("UniT nested tokenizer", tokenizer_ok, **tokenizer_details)
    return checkpoint_ok and tokenizer_ok


def check_model_caches(hf_home: Path, reporter: Reporter) -> bool:
    # ``HF_HOME`` contains a ``hub/`` child. ``cache_dir`` expects that hub
    # directory itself, while Transformers also consults these environment
    # variables during processor/model resolution.
    hub_cache = hf_home / "hub"
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_HUB_OFFLINE"] = "1"
    details: dict[str, Any] = {"hf_home": str(hf_home)}
    qwen_ok = False
    dino_ok = False
    try:
        from transformers import AutoConfig, AutoProcessor

        qwen_config = AutoConfig.from_pretrained(QWEN_ID, local_files_only=True, cache_dir=hub_cache)
        qwen_processor = AutoProcessor.from_pretrained(QWEN_ID, local_files_only=True, cache_dir=hub_cache)
        qwen_ok = True
        details["qwen"] = {
            "config": qwen_config.__class__.__name__,
            "processor": qwen_processor.__class__.__name__,
        }
    except Exception as error:
        details["qwen_error"] = f"{type(error).__name__}: {error}"
    reporter.record("Qwen local cache", qwen_ok, **details)

    dino_details: dict[str, Any] = {"hf_home": str(hf_home)}
    try:
        from transformers import Dinov2Model

        model = Dinov2Model.from_pretrained(DINO_ID, local_files_only=True, cache_dir=hub_cache)
        dino_details["model"] = model.__class__.__name__
        dino_details["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        del model
        dino_ok = True
    except Exception as error:
        dino_details["error"] = f"{type(error).__name__}: {error}"
    reporter.record("DINO local cache", dino_ok, **dino_details)
    return qwen_ok and dino_ok


def check_simulation_assets(reporter: Reporter) -> bool:
    details: dict[str, Any] = {}
    try:
        import mujoco
        import robocasa
        import robosuite

        details["versions"] = {
            "mujoco": mujoco.__version__,
            "robosuite": robosuite.__version__,
            "robocasa": robocasa.__version__,
        }
    except Exception as error:
        details["import_error"] = f"{type(error).__name__}: {error}"
    assets = ROOT / "third_party/robocasa-gr1-tabletop-tasks/robocasa/models/assets"
    expected_dirs = ("fixtures", "objects", "scenes", "textures")
    details["assets"] = str(assets)
    details["missing_asset_dirs"] = [name for name in expected_dirs if not (assets / name).is_dir()]
    patch = ROOT / "third_party/robocasa-gr1-tabletop-tasks/robocasa/models/objects/kitchen_object_utils.py"
    details["required_patch_present"] = patch.is_file() and "basket_4/model.xml" in patch.read_text(encoding="utf-8")
    passed = "import_error" not in details and not details["missing_asset_dirs"] and details["required_patch_present"]
    reporter.record("MuJoCo / robosuite / RoboCasa", passed, **details)
    return passed


def write_results(reporter: Reporter, output: Path, log: Path, storage: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "unit_storage": str(storage),
        "checks": reporter.checks,
        "overall": "PASS" if all(item["status"] == "PASS" for item in reporter.checks.values()) else "FAIL",
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.write_text("\n".join(reporter.lines) + "\n", encoding="utf-8")
    reporter.log(f"Results JSON: {output}")
    reporter.log(f"Log: {log}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-storage", default=os.environ.get("UNIT_STORAGE"), help="Shared S0 storage root (or set UNIT_STORAGE).")
    parser.add_argument("--hf-home", help="Hugging Face cache root; defaults to <unit-storage>/huggingface.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()
    if not args.unit_storage:
        parser.error("--unit-storage or UNIT_STORAGE is required")
    storage = Path(args.unit_storage).expanduser().resolve()
    hf_home = Path(args.hf_home).expanduser().resolve() if args.hf_home else storage / "huggingface"
    reporter = Reporter()
    reporter.log("Tactile3D-UniT S0 asset validation")
    reporter.log(f"Storage root: {storage}")

    try:
        if not storage.is_dir():
            raise RuntimeError(f"Storage root is not accessible: {storage}")
        check_gr1_assets(storage / "datasets/LeRobot", reporter)
        check_checkpoint(storage, reporter)
        check_model_caches(hf_home, reporter)
        check_simulation_assets(reporter)
    except Exception as error:
        reporter.record("Checker execution", False, error=f"{type(error).__name__}: {error}")
    finally:
        write_results(reporter, args.output, args.log, storage)
    return 0 if all(item["status"] == "PASS" for item in reporter.checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
