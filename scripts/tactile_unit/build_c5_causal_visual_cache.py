#!/usr/bin/env python3
"""Build the deduplicated frozen-frame feature bank for Track C5."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.paired_contract import (  # noqa: E402
    CANONICAL_FPS,
    discover_dataset_revision,
    load_episode_video_pointers,
    preprocess_trex_rgb,
    sha256_file,
    sha256_json,
)
from gr00t.tactile_unit.vac_latent_dataset import atomic_json, write_npy_atomic  # noqa: E402
from scripts.tactile_unit.continuous_contact_bridge_common import load_frozen_vision  # noqa: E402
from scripts.tactile_unit.vac_runtime_common import resolve_device, set_seed  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/tactile_unit/c5_causal_visual_planned_action.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, default=Path(os.environ["TREX_DATASET_DIR"]) if os.environ.get("TREX_DATASET_DIR") else None)
    parser.add_argument("--unit-checkpoint", type=Path, default=Path(os.environ["UNIT_FULLDATA_CKPT"]) if os.environ.get("UNIT_FULLDATA_CKPT") else None)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--splits", nargs="+", choices=("train", "validation", "test"), default=("train", "validation"))
    parser.add_argument("--allow-locked-test", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--cluster-frames", type=int, default=96)
    parser.add_argument("--cluster-span-seconds", type=float, default=4.0)
    parser.add_argument("--max-new-frames", type=int)
    return parser.parse_args()


def create_or_validate_array(path: Path, dtype: np.dtype, shape: tuple[int, ...], fill: int | float | bool | None = None) -> np.memmap:
    if path.is_file():
        value = np.load(path, mmap_mode="r+", allow_pickle=False)
        if value.dtype != np.dtype(dtype) or value.shape != shape:
            raise RuntimeError(f"existing cache array identity mismatch: {path.name}")
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    if fill is not None:
        value[...] = fill
        value.flush()
    return value


def frame_index_contract(episode: np.ndarray, anchor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys = sorted({(int(ep), int(t) + offset) for ep, t in zip(episode, anchor) for offset in range(-7, 1)})
    if any(frame < 0 for _, frame in keys):
        raise RuntimeError("history crosses episode start")
    key_array = np.asarray(keys, dtype=np.int32)
    lookup = {key: index for index, key in enumerate(keys)}
    history = np.asarray([[lookup[(int(ep), int(t) + offset)] for offset in range(-7, 1)] for ep, t in zip(episode, anchor)], dtype=np.int32)
    current = history[:, -1].copy()
    if not np.array_equal(key_array[current, 0], episode.astype(np.int32)) or not np.array_equal(key_array[current, 1], anchor.astype(np.int32)):
        raise RuntimeError("current frame cache mapping is not exactly I_t")
    if np.any(key_array[history, 1] > anchor[:, None]):
        raise RuntimeError("CAUSAL_LEAKAGE_FAIL: future frame in history mapping")
    expected = anchor[:, None] + np.arange(-7, 1, dtype=np.int32)[None]
    if not np.array_equal(key_array[history, 1], expected):
        raise RuntimeError("history mapping is not exactly I_t-7:t")
    return key_array[:, 0], key_array[:, 1], current, history


def target_clusters(values: Iterable[float], maximum: int, maximum_span: float) -> list[list[float]]:
    ordered = sorted(set(map(float, values)))
    groups: list[list[float]] = []
    for value in ordered:
        if not groups or len(groups[-1]) >= maximum or value - groups[-1][0] > maximum_span:
            groups.append([value])
        else:
            groups[-1].append(value)
    return groups


def decode_cluster(path: Path, targets: list[float]) -> dict[float, tuple[np.ndarray, float]]:
    import av

    chosen: dict[float, tuple[float, np.ndarray, float]] = {}
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        start = max(0.0, targets[0] - 2.0 / CANONICAL_FPS)
        if stream.time_base is not None:
            container.seek(int(start / float(stream.time_base)), stream=stream, any_frame=False, backward=True)
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            stamp = float(frame.time)
            position = int(np.searchsorted(targets, stamp))
            for index in (position - 1, position):
                if 0 <= index < len(targets):
                    target = targets[index]
                    distance = abs(stamp - target)
                    if target not in chosen or distance < chosen[target][0]:
                        chosen[target] = (distance, frame.to_ndarray(format="rgb24"), stamp)
            if stamp > targets[-1] + 2.0 / CANONICAL_FPS:
                break
    if len(chosen) != len(targets):
        raise RuntimeError(f"decoder missed requested frames in {path.name}")
    if any(distance > 1.0 / CANONICAL_FPS + 1e-4 for distance, _, _ in chosen.values()):
        raise RuntimeError("packed-video nearest-frame alignment exceeds one frame")
    return {target: (array, stamp) for target, (_, array, stamp) in chosen.items()}


@torch.inference_mode()
def frozen_frame_features(vision: torch.nn.Module, images: np.ndarray, device: torch.device) -> np.ndarray:
    pixels = torch.from_numpy(images).to(device, dtype=vision.dtype)
    patches = vision.vision_branch.vision_model(pixels).reshape(len(pixels), -1, vision.vision_branch.hidden_size)
    projected = vision.vision_branch.m_former.embeddings.projection(patches)
    compressed = vision.vq_down_resampler(projected)
    side = int(round(compressed.shape[1] ** 0.5))
    if side * side != compressed.shape[1]:
        raise RuntimeError("DINO patch grid is not square")
    grid = compressed.reshape(len(compressed), side, side, 32).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(grid, (2, 4)).permute(0, 2, 3, 1).reshape(len(grid), 8, 32)
    return pooled.float().cpu().numpy()


@torch.inference_mode()
def verify_frozen_boundary_repeat(vision: torch.nn.Module, images: np.ndarray, device: torch.device) -> None:
    first = frozen_frame_features(vision, images, device)
    second = frozen_frame_features(vision, images, device)
    if not np.array_equal(first, second):
        raise RuntimeError("frozen single-frame DINO boundary is non-deterministic")
    if first.shape != (len(images), 8, 32) or not np.isfinite(first).all():
        raise RuntimeError("frozen single-frame DINO boundary has invalid output")


def selection_is_frozen(artifact_root: Path) -> None:
    for name in ("c5_contract.json", "planned_action_contract.json", "causal_visual_selection.json", "uncertainty_selection.json", "runtime_router_contract.json"):
        path = artifact_root / name
        if not path.is_file() or json.loads(path.read_text()).get("test_loaded") is not False:
            raise RuntimeError("locked test cache requires all validation selections frozen")
        digest_candidates = (artifact_root / f"{name}.sha256", artifact_root / f"{path.stem}.sha256")
        digest_path = next((candidate for candidate in digest_candidates if candidate.is_file()), None)
        if digest_path is None or sha256_file(path) != digest_path.read_text().split()[0]:
            raise RuntimeError(f"locked test cache requires valid hash for {name}")


def prepare_split(cache_root: Path, c1_root: Path, c3dp_root: Path, exact_root: Path, split: str, expected_count: int) -> dict[str, Any]:
    source = c1_root / split
    episode = np.load(source / "episode_id.npy", mmap_mode="r", allow_pickle=False)
    anchor = np.load(source / "t.npy", mmap_mode="r", allow_pickle=False)
    pair_id = np.load(source / "pair_id.npy", mmap_mode="r", allow_pickle=False)
    if len(pair_id) != expected_count:
        raise RuntimeError(f"frozen {split} row count changed")
    if not np.array_equal(pair_id, np.load(c3dp_root / split / "pair_id.npy", mmap_mode="r", allow_pickle=False)):
        raise RuntimeError("C3DP pair identity differs from C1")
    if not np.array_equal(pair_id, np.load(exact_root / split / "pair_id.npy", mmap_mode="r", allow_pickle=False)):
        raise RuntimeError("exact Action pair identity differs from C1")
    frame_episode, frame_index, current, history = frame_index_contract(episode, anchor)
    destination = cache_root / split
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in {
        "frame_episode_id": frame_episode,
        "frame_index": frame_index,
        "current_feature_index": current,
        "history_feature_index": history,
    }.items():
        path = destination / f"{name}.npy"
        if path.is_file() and not np.array_equal(np.load(path, allow_pickle=False), value):
            raise RuntimeError(f"existing {split} {name} changed")
        if not path.is_file():
            write_npy_atomic(path, value)
    features = create_or_validate_array(destination / "frame_features.npy", np.float32, (len(frame_index), 8, 32), fill=np.nan)
    complete = create_or_validate_array(destination / "frame_complete.npy", np.bool_, (len(frame_index),), fill=False)
    return {
        "episode": episode, "anchor": anchor, "pair_id": pair_id,
        "frame_episode": frame_episode, "frame_index": frame_index,
        "current": current, "history": history, "features": features, "complete": complete,
    }


def extract_split(args: argparse.Namespace, split: str, arrays: dict[str, Any], pointers: dict[int, Any], vision: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    incomplete = np.flatnonzero(~np.asarray(arrays["complete"]))
    if args.max_new_frames is not None:
        incomplete = incomplete[:args.max_new_frames]
    grouped: dict[str, list[int]] = defaultdict(list)
    target_by_row: dict[int, float] = {}
    timestamp_owner: dict[tuple[str, int], tuple[int, int]] = {}
    for row in incomplete:
        episode, frame = int(arrays["frame_episode"][row]), int(arrays["frame_index"][row])
        pointer = pointers[episode]
        if frame >= pointer.length:
            raise RuntimeError("causal frame exceeds episode length")
        target = pointer.from_timestamp + frame / CANONICAL_FPS
        if not pointer.from_timestamp <= target < pointer.to_timestamp:
            raise RuntimeError("causal frame outside packed-video interval")
        alias_key = (pointer.relative_path, round(target * 1_000_000_000))
        owner = timestamp_owner.setdefault(alias_key, (episode, frame))
        if owner != (episode, frame):
            raise RuntimeError("packed-video timestamp alias between causal frames")
        grouped[pointer.relative_path].append(int(row))
        target_by_row[int(row)] = target
    processed = 0
    boundary_checked = bool(len(incomplete) == 0 and np.asarray(arrays["complete"]).all())
    started = time.time()
    for file_number, (relative_path, rows) in enumerate(sorted(grouped.items()), start=1):
        by_target = {target_by_row[row]: row for row in rows}
        if len(by_target) != len(rows):
            raise RuntimeError("packed-video target alias within file")
        for targets in target_clusters(by_target, args.cluster_frames, args.cluster_span_seconds):
            decoded = decode_cluster(args.dataset_root / relative_path, targets)
            for batch_start in range(0, len(targets), args.batch_size):
                batch_targets = targets[batch_start:batch_start + args.batch_size]
                images = np.stack([preprocess_trex_rgb(decoded[target][0]) for target in batch_targets])
                if not boundary_checked:
                    verify_frozen_boundary_repeat(vision, images[:min(2, len(images))], device)
                    boundary_checked = True
                features = frozen_frame_features(vision, images, device)
                destination = np.asarray([by_target[target] for target in batch_targets], dtype=np.int64)
                arrays["features"][destination] = features
                arrays["complete"][destination] = True
                processed += len(destination)
            del decoded
        arrays["features"].flush(); arrays["complete"].flush()
        if file_number % 16 == 0:
            elapsed = max(time.time() - started, 1e-6)
            print(json.dumps({"split": split, "files": file_number, "files_total": len(grouped), "new_frames": processed, "frames_per_second": processed / elapsed}), flush=True)
            gc.collect()
            try:
                ctypes.CDLL(None).malloc_trim(0)
            except (AttributeError, OSError):
                pass
    completed = int(np.count_nonzero(arrays["complete"]))
    total = len(arrays["complete"])
    if completed and not np.isfinite(np.asarray(arrays["features"])[np.asarray(arrays["complete"])]).all():
        raise RuntimeError("completed causal visual features contain non-finite values")
    return {"new_frames": processed, "complete_frames": completed, "total_unique_frames": total, "complete": completed == total, "video_files_touched": len(grouped), "frozen_boundary_repeat_exact": boundary_checked}


def split_manifest(cache_root: Path, split: str, arrays: dict[str, Any], extraction: dict[str, Any]) -> dict[str, Any]:
    root = cache_root / split
    manifest = {
        "schema": "tactile3d-unit.vac-c5-causal-visual-split.v1",
        "split": split,
        "rows": len(arrays["pair_id"]),
        "unique_frames": len(arrays["frame_index"]),
        "supports": {"current": [0], "history": list(range(-7, 1))},
        "maximum_frame_offset": 0,
        "minimum_frame_offset": -7,
        "cross_episode": False,
        "future_frame": False,
        "timestamp_alignment": {
            "method": "packed-video nearest decoded frame",
            "maximum_allowed_error_seconds": 1.0 / CANONICAL_FPS + 1e-4,
            "enforced_during_every_new_frame_decode": True,
        },
        "feature_shape": [8, 32],
        "frozen_backbone": True,
        "feature_normalization": "train-only channel mean/std applied lazily after frozen feature extraction",
        "frame_deduplicated": True,
        "source_references": {
            "pair_id": f".local/cache/tactile_unit/vac_c1/{split}/pair_id.npy",
            "episode_id": f".local/cache/tactile_unit/vac_c1/{split}/episode_id.npy",
            "t": f".local/cache/tactile_unit/vac_c1/{split}/t.npy",
            "u_a_oracle_plan": f".local/cache/tactile_unit/vac_c3msccr/{split}/u_a_correct.npy",
            "u_c_target": f".local/cache/tactile_unit/vac_c3dp/{split}/u_c.npy",
        },
        "raw_video_duplicated": False,
        "complete": extraction["complete"],
        "arrays": {},
    }
    if extraction["complete"]:
        for name in ("frame_episode_id", "frame_index", "current_feature_index", "history_feature_index", "frame_features", "frame_complete"):
            path = root / f"{name}.npy"
            manifest["arrays"][name] = {"shape": list(np.load(path, mmap_mode="r", allow_pickle=False).shape), "sha256": sha256_file(path)}
    manifest["canonical_sha256"] = sha256_json(manifest)
    atomic_json(root / "manifest.json", manifest)
    return manifest


def fit_train_visual_normalization(cache_root: Path) -> dict[str, Any]:
    """Fit the only C5 visual transform on unique frozen train frames."""
    features = np.load(cache_root / "train/frame_features.npy", mmap_mode="r", allow_pickle=False)
    complete = np.load(cache_root / "train/frame_complete.npy", mmap_mode="r", allow_pickle=False)
    if not bool(np.asarray(complete).all()):
        raise RuntimeError("train-only visual normalization requires a complete train cache")
    total = np.zeros(32, dtype=np.float64)
    squared = np.zeros(32, dtype=np.float64)
    count = 0
    for start in range(0, len(features), 4096):
        value = np.asarray(features[start:start + 4096], dtype=np.float64).reshape(-1, 32)
        total += value.sum(0); squared += np.square(value).sum(0); count += len(value)
    mean = total / count
    variance = np.maximum(squared / count - np.square(mean), 1e-12)
    value = {
        "schema": "tactile3d-unit.vac-c5-visual-normalization.v1",
        "fit_split": "frozen C1 train rows / deduplicated causal train frames only",
        "fit_unique_frames": len(features), "fit_tokens": count,
        "mean": mean.astype(np.float32).tolist(),
        "std": np.sqrt(variance).astype(np.float32).tolist(),
        "validation_or_test_used_for_fit": False, "test_loaded": False,
    }
    value["canonical_sha256"] = sha256_json(value)
    atomic_json(cache_root / "visual_feature_normalization.json", value)
    return value


def main() -> None:
    args = parse_args()
    spec = json.loads(args.config.read_text())
    if args.dataset_root is None or args.unit_checkpoint is None:
        raise RuntimeError("TREX_DATASET_DIR and UNIT_FULLDATA_CKPT (or flags) are required")
    cache_root = args.cache_root or ROOT / spec["runtime"]["cache_root"]
    artifact_root = args.artifact_root or ROOT / spec["runtime"]["artifact_root"]
    cache_root.mkdir(parents=True, exist_ok=True); artifact_root.mkdir(parents=True, exist_ok=True)
    if "test" in args.splits:
        if not args.allow_locked_test:
            raise RuntimeError("test cache is locked until all validation artifacts are frozen")
        selection_is_frozen(artifact_root)
    if discover_dataset_revision(args.dataset_root) != "bf0eb24c4b8bdd95752b553f0fc50e46a22f1cc8":
        raise RuntimeError("T-Rex dataset revision mismatch")
    set_seed(int(spec["seed"]))
    device, lock_handle, gpu = resolve_device(args.device)
    gpu.update({"preferred_physical": 1, "fallback": gpu.get("actual_physical") != 1 if gpu.get("actual_physical") is not None else True})
    try:
        vision_spec = {"frozen_identity": {"original_unit_tokenizer_files_sha256": spec["accepted"]["original_unit_tokenizer_files_sha256"]}}
        vision, vision_identity = load_frozen_vision(args.unit_checkpoint, vision_spec, device)
        if vision_identity["trainable_parameters"] != 0:
            raise RuntimeError("Original UniT Vision backbone is not frozen")
        pointers = {item.episode_id: item for item in load_episode_video_pointers(args.dataset_root)}
        manifests, extraction = {}, {}
        for split in args.splits:
            arrays = prepare_split(
                cache_root, ROOT / spec["runtime"]["c1_cache_root"],
                ROOT / spec["runtime"]["c3dp_cache_root"], ROOT / spec["runtime"]["exact_action_cache_root"],
                split, int(spec["counts"][split]),
            )
            extraction[split] = extract_split(args, split, arrays, pointers, vision, device)
            manifests[split] = split_manifest(cache_root, split, arrays, extraction[split])
        if "train" in manifests and manifests["train"]["complete"]:
            visual_normalization = fit_train_visual_normalization(cache_root)
        else:
            normalization_path = cache_root / "visual_feature_normalization.json"
            if not normalization_path.is_file():
                raise RuntimeError("causal visual cache requires frozen train-only normalization")
            visual_normalization = json.loads(normalization_path.read_text())
        if visual_normalization.get("validation_or_test_used_for_fit") is not False:
            raise RuntimeError("causal visual normalization used validation/test")
        summary = {
            "schema": "tactile3d-unit.vac-c5-causal-visual-cache.v1",
            "gpu": gpu,
            "vision_identity": vision_identity,
            "vision_preprocessing": spec["accepted"]["vision_preprocessing_identity"],
            "frozen_single_frame_boundary": spec["visual"]["frozen_frame_boundary"],
            "visual_normalization": visual_normalization,
            "splits": {name: {"rows": value["rows"], "unique_frames": value["unique_frames"], "complete": value["complete"], "canonical_sha256": value["canonical_sha256"]} for name, value in manifests.items()},
            "extraction": extraction,
            "test": "DEFERRED_LOCKED" if "test" not in args.splits else "CACHED_AFTER_FREEZE",
            "test_loaded": "test" in args.splits,
        }
        output = artifact_root / ("locked_causal_visual_cache_manifest.json" if "test" in args.splits else "causal_visual_cache_manifest.json")
        atomic_json(output, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    main()
