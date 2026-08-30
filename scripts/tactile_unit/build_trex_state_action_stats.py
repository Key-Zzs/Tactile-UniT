#!/usr/bin/env python3
"""Build exact train-only T-Rex state/action mean-std statistics for S3.1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.paired_contract import (  # noqa: E402
    discover_dataset_revision,
    load_info,
    sha256_file,
    sha256_json,
)


DEFAULT_SPLIT = ROOT / ".local/artifacts/tactile_teacher/s1_0/split_manifest.json"
DEFAULT_OUTPUT = ROOT / ".local/artifacts/tactile_unit/s3_1/state_action_normalization.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = load_info(args.dataset_root)
    split = json.loads(args.split_manifest.read_text())
    train_episodes = set(map(int, split["episode_ids"]["train"]))
    accumulators = {
        "observation.state": {
            "sum": np.zeros(58, dtype=np.float64),
            "sum_square": np.zeros(58, dtype=np.float64),
        },
        "action": {
            "sum": np.zeros(58, dtype=np.float64),
            "sum_square": np.zeros(58, dtype=np.float64),
        },
    }
    frame_count = 0
    files = sorted((args.dataset_root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError("T-Rex data parquet files are missing")
    for path in files:
        table = pq.read_table(path, columns=["episode_index", "observation.state", "action"])
        episodes = np.asarray(table["episode_index"])
        mask = np.fromiter((int(value) in train_episodes for value in episodes), dtype=bool, count=len(episodes))
        count = int(mask.sum())
        if not count:
            continue
        frame_count += count
        for key in accumulators:
            values = np.asarray(table[key].to_pylist(), dtype=np.float64)[mask]
            if values.shape != (count, 58) or not np.isfinite(values).all():
                raise ValueError(f"invalid train-only {key} values in {path.name}")
            accumulators[key]["sum"] += values.sum(axis=0, dtype=np.float64)
            accumulators[key]["sum_square"] += np.square(values).sum(axis=0, dtype=np.float64)
    episode_table = pq.read_table(
        next((args.dataset_root / "meta" / "episodes").rglob("*.parquet")),
        columns=["episode_index", "length"],
    )
    episode_ids = np.asarray(episode_table["episode_index"])
    episode_lengths = np.asarray(episode_table["length"])
    expected_frames = int(
        episode_lengths[
            np.fromiter(
                (int(value) in train_episodes for value in episode_ids),
                dtype=bool,
                count=len(episode_ids),
            )
        ].sum()
    )
    if frame_count != expected_frames:
        raise ValueError(f"train frame count mismatch: {frame_count} != {expected_frames}")
    result: dict[str, Any] = {
        "schema": "tactile3d-unit.s3-1-state-action-normalization.v1",
        "dataset_revision": discover_dataset_revision(args.dataset_root),
        "fit_split": "frozen S1 train episodes only",
        "s1_split_manifest_sha256": sha256_file(args.split_manifest),
        "train_episode_count": len(train_episodes),
        "train_frame_count": frame_count,
        "mode": "mean_std",
        "ddof": 0,
        "features": {
            "observation.state": info["features"]["observation.state"]["names"],
            "action": info["features"]["action"]["names"],
        },
    }
    for key, accumulator in accumulators.items():
        mean = accumulator["sum"] / frame_count
        variance = np.maximum(accumulator["sum_square"] / frame_count - np.square(mean), 0.0)
        result[key] = {"mean": mean.tolist(), "std": np.sqrt(variance).tolist()}
    result["canonical_sha256"] = sha256_json(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "train_episode_count": len(train_episodes),
                "train_frame_count": frame_count,
                "canonical_sha256": result["canonical_sha256"],
                "output_file_sha256": sha256_file(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
