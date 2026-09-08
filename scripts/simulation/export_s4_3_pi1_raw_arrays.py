#!/usr/bin/env python3
"""Export the exact official raw pinch_tongs Zarr arrays into one portable cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import zarr


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = ROOT / ".local/external/simulation/s4_3_pi1/raw/DexJoCo-Datasets-Raw/dexjoco_raw_datasets/pinch_tongs"
DEFAULT_CACHE = ROOT / ".local/cache/simulation/s4_3_pi1/raw_pinch_tongs_arrays.npz"
DEFAULT_MANIFEST = ROOT / ".local/artifacts/simulation/s4_3_pi1/raw_dataset_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    episodes = sorted(path.parent for path in args.raw_root.glob("*/replay.zarr"))
    if len(episodes) != 100:
        raise RuntimeError(f"expected exactly 100 downloaded raw episodes, got {len(episodes)}")
    names: list[str] = []
    lengths: list[int] = []
    actions: list[np.ndarray] = []
    actions_rotvec: list[np.ndarray] = []
    states: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    for episode in episodes:
        source = zarr.open(str(episode / "replay.zarr"), mode="r")["data"]
        arrays = {name: np.asarray(source[name]) for name in ("action", "action_rotvec", "state", "timestamp")}
        for name in ("action", "action_rotvec", "state"):
            if arrays[name].ndim == 3 and arrays[name].shape[1] == 1:
                arrays[name] = arrays[name][:, 0]
        if arrays["timestamp"].ndim == 2 and arrays["timestamp"].shape[1] == 1:
            arrays["timestamp"] = arrays["timestamp"][:, 0]
        length = len(arrays["action"])
        if any(len(value) != length for value in arrays.values()):
            raise RuntimeError(f"raw array length mismatch: {episode.name}")
        if arrays["action"].shape[1:] != (23,) or arrays["action_rotvec"].shape[1:] != (22,):
            raise RuntimeError(f"raw action contract mismatch: {episode.name}")
        if arrays["state"].shape[1] < 31:
            raise RuntimeError(f"raw state lacks replay fields: {episode.name} {arrays['state'].shape}")
        names.append(episode.name)
        lengths.append(length)
        actions.append(arrays["action"])
        actions_rotvec.append(arrays["action_rotvec"])
        states.append(arrays["state"])
        timestamps.append(arrays["timestamp"])

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.cache,
        episode_name=np.asarray(names),
        episode_length=np.asarray(lengths, dtype=np.int64),
        episode_start=np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.int64),
        action=np.concatenate(actions).astype(np.float64),
        action_rotvec=np.concatenate(actions_rotvec).astype(np.float64),
        state=np.concatenate(states).astype(np.float64),
        timestamp=np.concatenate(timestamps).astype(np.float64),
    )
    manifest = {
        "schema": "tactile3d-unit.s4-3-pi1-raw-dataset-manifest.v1",
        "status": "PASS",
        "repository": "DexJoCo/DexJoCo-Datasets-Raw",
        "revision": "125df5c1019e97503929ef1a0ad8f90373436afa",
        "subtree": "dexjoco_raw_datasets/pinch_tongs/**",
        "episodes": len(episodes),
        "frames": int(sum(lengths)),
        "episode_names": names,
        "episode_lengths": lengths,
        "array_shapes": {
            "action": [int(sum(lengths)), 23],
            "action_rotvec": [int(sum(lengths)), 22],
            "state": list(np.concatenate(states).shape),
            "timestamp": [int(sum(lengths))],
        },
        "cache": "$PI1_DATA_ROOT/raw_pinch_tongs_arrays.npz",
        "cache_sha256": sha256_file(args.cache),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "episodes": len(episodes), "frames": sum(lengths), "cache_sha256": manifest["cache_sha256"]}))


if __name__ == "__main__":
    main()
