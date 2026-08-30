#!/usr/bin/env python3
"""Authoritative all-file completeness and decode audit for T-Rex head_left."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.paired_contract import (  # noqa: E402
    VIDEO_KEY,
    audit_video_inventory,
    decode_rgb_frame,
    decoded_frame_stats,
    discover_dataset_revision,
    ffprobe_video,
    load_episode_video_pointers,
    load_info,
    sha256_json,
    validate_video_probe,
)


DEFAULT_OUTPUT = ROOT / ".local/artifacts/tactile_unit/s3_1/video_completeness_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--deep-subset-count", type=int, default=32)
    return parser.parse_args()


def select_deep_subset(paths: list[str], count: int) -> set[str]:
    ranked = sorted(paths, key=lambda value: hashlib.sha256(value.encode("utf-8")).digest())
    return set(ranked[: min(count, len(ranked))])


def audit_one(
    dataset_root: Path,
    relative_path: str,
    feature: dict[str, Any],
    deep: bool,
) -> dict[str, Any]:
    path = dataset_root / relative_path
    result: dict[str, Any] = {"path": relative_path, "structural_failures": [], "decode_failures": []}
    try:
        probe = ffprobe_video(path)
        result["probe"] = probe
        result["structural_failures"] = validate_video_probe(probe, feature)
    except Exception as exc:
        result["structural_failures"] = [f"{type(exc).__name__}: {exc}"]
        return result
    duration = result["probe"]["duration"]
    timestamps = {"deterministic": 0.5 * duration}
    if deep:
        timestamps.update(
            {
                "first": 0.0,
                "middle": 0.5 * duration,
                "late": max(0.0, duration - 1.0 / float(feature["info"]["video.fps"])),
            }
        )
    decoded: dict[str, Any] = {}
    arrays = {}
    for name, timestamp in timestamps.items():
        try:
            frame = decode_rgb_frame(path, timestamp)
            arrays[name] = frame
            stats = decoded_frame_stats(frame)
            decoded[name] = {"timestamp": timestamp, **stats}
            expected_shape = list(feature["shape"])
            if stats["shape"] != expected_shape:
                result["decode_failures"].append(
                    f"{name}: shape={stats['shape']}, expected {expected_shape}"
                )
            if stats["all_black"]:
                result["decode_failures"].append(f"{name}: all-black frame")
        except Exception as exc:
            result["decode_failures"].append(f"{name}: {type(exc).__name__}: {exc}")
    if deep and all(name in arrays for name in ("first", "middle", "late")):
        hashes = {hashlib.sha256(arrays[name].tobytes()).hexdigest() for name in ("first", "middle", "late")}
        if len(hashes) == 1:
            result["decode_failures"].append("first/middle/late frames are all identical")
    result["decoded"] = decoded
    return result


def main() -> None:
    args = parse_args()
    start = time.monotonic()
    info = load_info(args.dataset_root)
    feature = info["features"][VIDEO_KEY]
    pointers = load_episode_video_pointers(args.dataset_root)
    inventory = audit_video_inventory(args.dataset_root, pointers)
    paths = inventory.pop("expected_relative_paths")
    preflight_pass = not (
        inventory["missing_referenced"] or inventory["zero_size_referenced"]
    )
    results: list[dict[str, Any]] = []
    if preflight_pass:
        deep_subset = select_deep_subset(paths, args.deep_subset_count)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    audit_one, args.dataset_root, relative, feature, relative in deep_subset
                ): relative
                for relative in paths
            }
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda row: row["path"])
    container_failures = [
        {"path": row["path"], "failures": row["structural_failures"]}
        for row in results
        if row["structural_failures"]
    ]
    decode_failures = [
        {"path": row["path"], "failures": row["decode_failures"]}
        for row in results
        if row["decode_failures"]
    ]
    passed = (
        preflight_pass
        and len(results) == inventory["unique_referenced_mp4s"]
        and not container_failures
        and not decode_failures
    )
    summary: dict[str, Any] = {
        "schema": "tactile3d-unit.s3-1-video-completeness.v1",
        "video_key": VIDEO_KEY,
        "dataset_revision": discover_dataset_revision(args.dataset_root),
        "metadata": {
            "codebase_version": info["codebase_version"],
            "resolution": feature["shape"],
            "fps": feature["info"]["video.fps"],
            "codec": feature["info"]["video.codec"],
            "pixel_format": feature["info"]["video.pix_fmt"],
            "video_path_template": info["video_path"],
        },
        "inventory": inventory,
        "all_file_probe_count": len(results),
        "container_failure_count": len(container_failures),
        "container_failures": container_failures,
        "decode_failure_count": len(decode_failures),
        "decode_failures": decode_failures,
        "deep_decode_subset_count": sum("first" in row.get("decoded", {}) for row in results),
        "status": "PASS" if passed else "FAIL",
        "declaration": "T-Rex head_left stream: COMPLETE FOR S3.1" if passed else None,
        "elapsed_seconds": time.monotonic() - start,
    }
    summary["summary_sha256"] = sha256_json(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
