#!/usr/bin/env python3
"""Prove the raw replay episodes are the source of the official LeRobot rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
RAW_CACHE = ROOT / ".local/cache/simulation/s4_3_pi1/raw_pinch_tongs_arrays.npz"
DATASET = ROOT / ".local/external/s4_3_pi0/datasets/DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets/pinch_tongs"
ARTIFACT = ROOT / ".local/artifacts/simulation/s4_3_pi1/official_dataset_alignment.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_list(table, name: str, width: int) -> np.ndarray:
    column = table[name].combine_chunks()
    values = column.values.to_numpy(zero_copy_only=False)
    return values.reshape(len(column), width)


def main() -> None:
    with np.load(RAW_CACHE, allow_pickle=False) as source:
        raw = {name: source[name] for name in source.files}
    parquet_files = sorted((DATASET / "data").rglob("*.parquet"))
    table = pa.concat_tables([pq.read_table(path) for path in parquet_files])
    action = fixed_list(table, "action", 22)
    state = fixed_list(table, "observation.state", 23)
    timestamp = table["timestamp"].combine_chunks().to_numpy()
    episode_index = table["episode_index"].combine_chunks().to_numpy()
    frame_index = table["frame_index"].combine_chunks().to_numpy()
    index = table["index"].combine_chunks().to_numpy()

    frames = len(table)
    lengths = np.bincount(episode_index, minlength=100)
    expected_frame = np.concatenate([np.arange(length) for length in lengths])
    raw_starts = raw["episode_start"]
    raw_lengths = raw["episode_length"]
    leading_static_trim = raw_lengths - lengths
    selected = np.concatenate(
        [
            np.arange(start + trim, start + trim + length)
            for start, trim, length in zip(raw_starts, leading_static_trim, lengths, strict=True)
        ]
    )
    computed_trim = []
    for start, raw_length in zip(raw_starts, raw_lengths, strict=True):
        # The official converter computes its static prefix on the raw quaternion
        # action before selecting action_rotvec as the policy target.
        episode_action = raw["action"][start : start + raw_length]
        changing = np.flatnonzero(np.any(episode_action[:-1] != episode_action[1:], axis=1))
        if not len(changing):
            raise RuntimeError("raw episode is entirely static")
        first = int(changing[0])
        if np.all(episode_action[first] == 0):
            first += 1
        computed_trim.append(first)
    computed_trim = np.asarray(computed_trim, dtype=np.int64)
    gates = {
        "episodes_100": len(np.unique(episode_index)) == 100,
        "frames_40065": frames == 40065,
        "raw_episodes_cover_official_frames": bool(np.all(raw_lengths >= lengths)),
        "official_converter_static_trim_reproduced": np.array_equal(computed_trim, leading_static_trim),
        "episode_order_contiguous": np.array_equal(episode_index, np.repeat(np.arange(100), lengths)),
        "frame_order_contiguous": np.array_equal(frame_index, expected_frame),
        "global_index_identity": np.array_equal(index, np.arange(frames)),
        "action_byte_numeric_parity": np.array_equal(raw["action_rotvec"][selected].astype(np.float32), action),
        "state_byte_numeric_parity": np.array_equal(raw["state"][selected, :23].astype(np.float32), state),
        "timestamp_converter_parity": np.array_equal((frame_index / 30.0).astype(np.float32), timestamp),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    result = {
        "schema": "tactile3d-unit.s4-3-pi1-official-dataset-alignment.v1",
        "status": status,
        "raw_repository": "DexJoCo/DexJoCo-Datasets-Raw",
        "raw_revision": "125df5c1019e97503929ef1a0ad8f90373436afa",
        "lerobot_repository": "DexJoCo/DexJoCo-Datasets-LeRobot",
        "lerobot_revision": "5a57c54e55dc5858dd9fb949c5f67c0c9716e6b3",
        "episodes": int(len(lengths)),
        "frames": frames,
        "fps_from_info": json.loads((DATASET / "meta/info.json").read_text())["fps"],
        "raw_episode_names_in_converter_sort_order": raw["episode_name"].tolist(),
        "episode_lengths": lengths.tolist(),
        "raw_episode_lengths": raw_lengths.tolist(),
        "raw_leading_static_frames_excluded_by_official_conversion": leading_static_trim.tolist(),
        "raw_leading_static_frames_excluded_total": int(leading_static_trim.sum()),
        "source_mapping": "LeRobot episode i equals lexicographically sorted raw episode i after the official converter's leading-static-action trim",
        "action_source": "raw replay.zarr data/action_rotvec",
        "state_source": "raw replay.zarr data/state[:, :23]",
        "timestamp_source": "LeRobot converter-generated frame_index / 30 Hz after leading-static trim",
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "official_parquet_sha256": {path.relative_to(DATASET).as_posix(): sha256_file(path) for path in parquet_files},
        "raw_cache_sha256": sha256_file(RAW_CACHE),
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "episodes": len(lengths), "frames": frames, "gates": result["gates"]}))
    if status != "PASS":
        raise SystemExit("S4_3_PI1A_OFFICIAL_DATA_MUTATION_FAIL")


if __name__ == "__main__":
    main()
