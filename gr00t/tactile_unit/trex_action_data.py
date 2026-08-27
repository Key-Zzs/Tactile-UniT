"""Numeric-only T-Rex action windows for S3.3.

This loader reuses the frozen S1/S3.1 episode split and train-only mean/std
statistics.  It never opens the 103-GiB RGB streams and never creates a new
train/validation/test assignment.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import numpy as np

from gr00t.tactile_unit.paired_contract import (
    ACTION_HORIZON,
    RAW_ACTION_DIM,
    RAW_STATE_DIM,
    TOKENIZER_DIM,
    load_info,
    validate_episode_splits,
)


SEGMENTS: dict[str, slice] = {
    "left_arm": slice(0, 7),
    "left_hand": slice(7, 29),
    "right_arm": slice(29, 36),
    "right_hand": slice(36, 58),
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class EpisodeActionPointer:
    episode_id: int
    length: int
    data_chunk_index: int
    data_file_index: int
    dataset_from_index: int
    dataset_to_index: int
    primitive: str

    @property
    def eligible_windows(self) -> int:
        # Inclusive a_t:t+15: the last legal anchor is length - 16.
        return max(0, self.length - ACTION_HORIZON + 1)


@dataclass(frozen=True)
class ActionWindow:
    split: str
    episode_id: int
    anchor_frame: int
    data_chunk_index: int
    data_file_index: int
    primitive: str


def load_episode_action_pointers(dataset_root: Path) -> list[EpisodeActionPointer]:
    import pyarrow.parquet as pq

    columns = [
        "episode_index",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
        "motor_primitive",
    ]
    paths = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError("T-Rex episode metadata parquet is missing")
    result: list[EpisodeActionPointer] = []
    for path in paths:
        values = pq.read_table(path, columns=columns).to_pydict()
        for row in range(len(values["episode_index"])):
            pointer = EpisodeActionPointer(
                episode_id=int(values["episode_index"][row]),
                length=int(values["length"][row]),
                data_chunk_index=int(values["data/chunk_index"][row]),
                data_file_index=int(values["data/file_index"][row]),
                dataset_from_index=int(values["dataset_from_index"][row]),
                dataset_to_index=int(values["dataset_to_index"][row]),
                primitive=str(values["motor_primitive"][row] or "unknown"),
            )
            if pointer.dataset_to_index - pointer.dataset_from_index != pointer.length:
                raise ValueError(f"episode {pointer.episode_id} interval length mismatch")
            result.append(pointer)
    result.sort(key=lambda item: item.episode_id)
    if [item.episode_id for item in result] != list(range(len(result))):
        raise ValueError("T-Rex episode IDs must be unique and contiguous")
    return result


def load_frozen_split(path: Path) -> dict[str, list[int]]:
    payload = json.loads(path.read_text())
    split = {name: list(map(int, payload["episode_ids"][name])) for name in ("train", "val", "test")}
    validate_episode_splits(split)
    return split


def deterministic_windows(
    pointers: Iterable[EpisodeActionPointer],
    split: Mapping[str, Iterable[int]],
    *,
    limits: Mapping[str, int | None],
) -> dict[str, list[ActionWindow]]:
    """Uniformly subsample legal anchors without altering episode membership."""

    pointer_by_id = {item.episode_id: item for item in pointers}
    result: dict[str, list[ActionWindow]] = {}
    for split_name in ("train", "val", "test"):
        selected_pointers = [pointer_by_id[int(value)] for value in split[split_name]]
        counts = np.asarray([item.eligible_windows for item in selected_pointers], dtype=np.int64)
        cumulative = np.cumsum(counts)
        total = int(cumulative[-1]) if len(cumulative) else 0
        requested = limits.get(split_name)
        count = total if requested is None else min(total, int(requested))
        if count <= 0:
            raise ValueError(f"no legal action windows for {split_name}")
        if count == total:
            global_offsets = np.arange(total, dtype=np.int64)
        else:
            # Midpoint bins avoid endpoint bias and cannot duplicate while count <= total.
            global_offsets = np.floor((np.arange(count) + 0.5) * total / count).astype(np.int64)
        episode_positions = np.searchsorted(cumulative, global_offsets, side="right")
        starts = np.concatenate((np.asarray([0], dtype=np.int64), cumulative[:-1]))
        windows: list[ActionWindow] = []
        for offset, position in zip(global_offsets.tolist(), episode_positions.tolist()):
            pointer = selected_pointers[position]
            anchor = int(offset - starts[position])
            if not 0 <= anchor <= pointer.length - ACTION_HORIZON:
                raise AssertionError("t:t+15 window sampler has an off-by-one error")
            windows.append(
                ActionWindow(
                    split=split_name,
                    episode_id=pointer.episode_id,
                    anchor_frame=anchor,
                    data_chunk_index=pointer.data_chunk_index,
                    data_file_index=pointer.data_file_index,
                    primitive=pointer.primitive,
                )
            )
        result[split_name] = windows
    return result


def _data_relative_path(info: Mapping[str, Any], chunk: int, file_index: int) -> str:
    template = info.get("data_path")
    if not isinstance(template, str):
        raise ValueError("T-Rex info.json has no data_path template")
    value = PurePosixPath(
        template.format(chunk_index=int(chunk), file_index=int(file_index))
    )
    if value.is_absolute() or ".." in value.parts:
        raise ValueError("unsafe T-Rex data_path template")
    return value.as_posix()


def build_action_window_cache(
    *,
    dataset_root: Path,
    split_manifest: Path,
    normalization: Path,
    output_root: Path,
    limits: Mapping[str, int | None],
) -> dict[str, Any]:
    """Build compact raw numeric action caches; no image/video access occurs."""

    import pyarrow.parquet as pq

    info = load_info(dataset_root)
    split = load_frozen_split(split_manifest)
    pointers = load_episode_action_pointers(dataset_root)
    windows = deterministic_windows(pointers, split, limits=limits)
    normalizer = json.loads(normalization.read_text())
    if normalizer.get("fit_split") != "frozen S1 train episodes only":
        raise ValueError("S3.3 requires the accepted train-only normalizer")
    if normalizer.get("mode") != "mean_std":
        raise ValueError("S3.3 requires the accepted mean/std normalization")

    output_root.mkdir(parents=True, exist_ok=True)
    primitive_names = sorted({window.primitive for values in windows.values() for window in values})
    primitive_to_id = {name: index for index, name in enumerate(primitive_names)}
    manifests: dict[str, Any] = {}

    for split_name, split_windows in windows.items():
        split_root = output_root / split_name
        split_root.mkdir(parents=True, exist_ok=True)
        count = len(split_windows)
        state_out = np.lib.format.open_memmap(
            split_root / "state_raw.npy", mode="w+", dtype=np.float32, shape=(count, RAW_STATE_DIM)
        )
        action_out = np.lib.format.open_memmap(
            split_root / "action_raw.npy",
            mode="w+",
            dtype=np.float32,
            shape=(count, ACTION_HORIZON, RAW_ACTION_DIM),
        )
        episode_out = np.lib.format.open_memmap(
            split_root / "episode_id.npy", mode="w+", dtype=np.int32, shape=(count,)
        )
        anchor_out = np.lib.format.open_memmap(
            split_root / "anchor_frame.npy", mode="w+", dtype=np.int32, shape=(count,)
        )
        primitive_out = np.lib.format.open_memmap(
            split_root / "primitive_id.npy", mode="w+", dtype=np.int16, shape=(count,)
        )
        grouped: dict[tuple[int, int], list[tuple[int, ActionWindow]]] = defaultdict(list)
        for output_index, window in enumerate(split_windows):
            grouped[(window.data_chunk_index, window.data_file_index)].append(
                (output_index, window)
            )
        for (chunk, file_index), requests in sorted(grouped.items()):
            relative = _data_relative_path(info, chunk, file_index)
            path = dataset_root / relative
            table = pq.read_table(
                path,
                columns=["episode_index", "frame_index", "observation.state", "action"],
            )
            episode_column = np.asarray(table["episode_index"], dtype=np.int64)
            frame_column = np.asarray(table["frame_index"], dtype=np.int64)
            states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            if states.shape != (len(table), RAW_STATE_DIM) or actions.shape != (
                len(table),
                RAW_ACTION_DIM,
            ):
                raise ValueError(f"invalid state/action array in {relative}")
            ranges: dict[int, tuple[int, int]] = {}
            changes = np.flatnonzero(np.diff(episode_column)) + 1
            starts = np.concatenate((np.asarray([0]), changes))
            ends = np.concatenate((changes, np.asarray([len(table)])))
            for start, end in zip(starts.tolist(), ends.tolist()):
                episode_id = int(episode_column[start])
                if not np.array_equal(frame_column[start:end], np.arange(end - start)):
                    raise ValueError(f"episode {episode_id} frame_index is not contiguous")
                ranges[episode_id] = (start, end)
            for output_index, window in requests:
                start, end = ranges[window.episode_id]
                action_start = start + window.anchor_frame
                action_end = action_start + ACTION_HORIZON
                if action_end > end:
                    raise AssertionError("a_t:t+15 crossed an episode boundary")
                state_out[output_index] = states[action_start]
                action_out[output_index] = actions[action_start:action_end]
                episode_out[output_index] = window.episode_id
                anchor_out[output_index] = window.anchor_frame
                primitive_out[output_index] = primitive_to_id[window.primitive]
        for value in (state_out, action_out, episode_out, anchor_out, primitive_out):
            value.flush()
        del state_out, action_out, episode_out, anchor_out, primitive_out
        sample_identity = [
            [window.episode_id, window.anchor_frame] for window in split_windows
        ]
        manifests[split_name] = {
            "episodes": len(set(window.episode_id for window in split_windows)),
            "windows": count,
            "available_windows": sum(
                pointers[episode_id].eligible_windows for episode_id in split[split_name]
            ),
            "sample_identity_sha256": hashlib.sha256(
                _canonical_json(sample_identity)
            ).hexdigest(),
            "primitive_counts": dict(sorted(Counter(window.primitive for window in split_windows).items())),
        }

    cache_manifest: dict[str, Any] = {
        "schema": "tactile3d-unit.s3-3-action-window-cache.v1",
        "dataset": "T-Rex",
        "dataset_revision": normalizer.get("dataset_revision"),
        "source": "accepted S1 episode split and S3.1 train-only normalization",
        "split_manifest_sha256": _sha256_file(split_manifest),
        "normalization_sha256": _sha256_file(normalization),
        "raw_state_shape": [RAW_STATE_DIM],
        "raw_action_shape": [ACTION_HORIZON, RAW_ACTION_DIM],
        "canonical_state_shape": [TOKENIZER_DIM],
        "canonical_action_shape": [ACTION_HORIZON, TOKENIZER_DIM],
        "action_interval": "a_t:t+15 (end exclusive t+16)",
        "ordering": ["left arm 7", "left hand 22", "right arm 7", "right hand 22"],
        "primitive_names": primitive_names,
        "splits": manifests,
        "leakage": validate_episode_splits(split),
        "video_decodes": 0,
    }
    cache_manifest["canonical_sha256"] = hashlib.sha256(
        _canonical_json(cache_manifest)
    ).hexdigest()
    (output_root / "manifest.json").write_text(
        json.dumps(cache_manifest, indent=2, sort_keys=True) + "\n"
    )
    return cache_manifest


class TReXActionCache:
    def __init__(self, root: Path, split: str, normalization: Path):
        self.root = Path(root)
        self.split = split
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("schema") != "tactile3d-unit.s3-3-action-window-cache.v1":
            raise ValueError("unsupported T-Rex action cache")
        split_root = self.root / split
        self.state_raw = np.load(split_root / "state_raw.npy", mmap_mode="r")
        self.action_raw = np.load(split_root / "action_raw.npy", mmap_mode="r")
        self.episode_id = np.load(split_root / "episode_id.npy", mmap_mode="r")
        self.anchor_frame = np.load(split_root / "anchor_frame.npy", mmap_mode="r")
        self.primitive_id = np.load(split_root / "primitive_id.npy", mmap_mode="r")
        stats = json.loads(Path(normalization).read_text())
        if _sha256_file(Path(normalization)) != self.manifest["normalization_sha256"]:
            raise ValueError("normalization file does not match the action cache")
        self.state_mean = np.asarray(stats["observation.state"]["mean"], dtype=np.float32)
        self.state_std = np.asarray(stats["observation.state"]["std"], dtype=np.float32)
        self.action_mean = np.asarray(stats["action"]["mean"], dtype=np.float32)
        self.action_std = np.asarray(stats["action"]["std"], dtype=np.float32)
        if self.state_raw.shape != (len(self.action_raw), RAW_STATE_DIM):
            raise ValueError("cached raw state shape mismatch")
        if self.action_raw.shape[1:] != (ACTION_HORIZON, RAW_ACTION_DIM):
            raise ValueError("cached raw action shape mismatch")

    def __len__(self) -> int:
        return len(self.state_raw)

    @staticmethod
    def _normalize(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        result = np.zeros_like(value, dtype=np.float32)
        mask = std != 0
        result[..., mask] = (value[..., mask] - mean[mask]) / std[mask]
        result[..., ~mask] = value[..., ~mask]
        return result

    def batch(self, indices: np.ndarray | list[int]) -> dict[str, np.ndarray]:
        indices = np.asarray(indices, dtype=np.int64)
        state_raw = np.asarray(self.state_raw[indices], dtype=np.float32)
        action_raw = np.asarray(self.action_raw[indices], dtype=np.float32)
        state58 = self._normalize(state_raw, self.state_mean, self.state_std)
        action58 = self._normalize(action_raw, self.action_mean, self.action_std)
        state = np.zeros((len(indices), TOKENIZER_DIM), dtype=np.float32)
        action = np.zeros((len(indices), ACTION_HORIZON, TOKENIZER_DIM), dtype=np.float32)
        state[:, :RAW_STATE_DIM] = state58
        action[:, :, :RAW_ACTION_DIM] = action58
        state_mask = np.zeros((len(indices), TOKENIZER_DIM), dtype=bool)
        action_mask = np.zeros((len(indices), ACTION_HORIZON, TOKENIZER_DIM), dtype=bool)
        state_mask[:, :RAW_STATE_DIM] = True
        action_mask[:, :, :RAW_ACTION_DIM] = True
        return {
            "state": state,
            "action": action,
            "state_mask": state_mask,
            "action_mask": action_mask,
            "state_raw": state_raw,
            "action_raw": action_raw,
            "episode_id": np.asarray(self.episode_id[indices]),
            "anchor_frame": np.asarray(self.anchor_frame[indices]),
            "primitive_id": np.asarray(self.primitive_id[indices]),
        }

    def inverse_action(self, normalized58: np.ndarray) -> np.ndarray:
        normalized58 = np.asarray(normalized58, dtype=np.float32)
        result = np.empty_like(normalized58)
        mask = self.action_std != 0
        result[..., mask] = normalized58[..., mask] * self.action_std[mask] + self.action_mean[mask]
        result[..., ~mask] = normalized58[..., ~mask]
        return result


def different_episode_indices(
    cache: TReXActionCache, indices: np.ndarray | list[int]
) -> np.ndarray:
    """Choose a deterministic globally different episode for every row."""

    indices = np.asarray(indices, dtype=np.int64)
    if len(cache) < 2 or len(np.unique(cache.episode_id)) < 2:
        raise ValueError("different-episode control needs at least two episodes")
    result = np.empty_like(indices)
    offset = max(1, len(cache) // 2)
    for output_index, source_index in enumerate(indices.tolist()):
        candidate = (source_index + offset) % len(cache)
        while int(cache.episode_id[candidate]) == int(cache.episode_id[source_index]):
            candidate = (candidate + 1) % len(cache)
        result[output_index] = candidate
    return result


def action_activity(action58: np.ndarray) -> dict[str, np.ndarray]:
    """Derived action labels.  Input may be raw or normalized [N,16,58]."""

    value = np.asarray(action58, dtype=np.float32)
    if value.ndim != 3 or value.shape[1:] != (ACTION_HORIZON, RAW_ACTION_DIM):
        raise ValueError("action_activity expects [N,16,58]")
    delta = np.diff(value, axis=1)
    segment_magnitude = {
        name: np.sqrt(np.mean(np.square(delta[..., segment]), axis=(1, 2)))
        for name, segment in SEGMENTS.items()
    }
    left = np.sqrt(np.mean(np.square(delta[..., :29]), axis=(1, 2)))
    right = np.sqrt(np.mean(np.square(delta[..., 29:]), axis=(1, 2)))
    arm = np.sqrt(
        np.mean(
            np.concatenate((np.square(delta[..., :7]), np.square(delta[..., 29:36])), axis=2),
            axis=(1, 2),
        )
    )
    hand = np.sqrt(
        np.mean(
            np.concatenate((np.square(delta[..., 7:29]), np.square(delta[..., 36:])), axis=2),
            axis=(1, 2),
        )
    )
    magnitude = np.sqrt(np.mean(np.square(delta), axis=(1, 2)))
    trend = np.mean(value[:, -1] - value[:, 0], axis=1)
    return {
        "magnitude": magnitude,
        "trend": trend,
        "left_activity": left,
        "right_activity": right,
        "arm_activity": arm,
        "hand_activity": hand,
        "active_side": (right > left).astype(np.int64),
        "arm_vs_hand": (hand > arm).astype(np.int64),
        **{f"{name}_magnitude": values for name, values in segment_magnitude.items()},
    }


def train_distribution(cache: TReXActionCache, *, sample_limit: int | None = None) -> dict[str, Any]:
    count = len(cache) if sample_limit is None else min(len(cache), int(sample_limit))
    indices = np.floor((np.arange(count) + 0.5) * len(cache) / count).astype(np.int64)
    batch = cache.batch(indices)
    activity = action_activity(batch["action"][:, :, :RAW_ACTION_DIM])
    magnitude = activity["magnitude"]
    # Separate near-static and dynamic modes with deterministic one-dimensional
    # two-means fitted on train only.  Unlike a percentile cutoff, this reports
    # the observed static fraction instead of forcing a chosen class balance.
    centers = np.asarray(np.quantile(magnitude, [0.25, 0.75]), dtype=np.float64)
    for _ in range(100):
        assignment = np.abs(magnitude[:, None] - centers[None, :]).argmin(axis=1)
        updated = np.asarray(
            [
                magnitude[assignment == cluster].mean()
                if np.any(assignment == cluster)
                else centers[cluster]
                for cluster in range(2)
            ],
            dtype=np.float64,
        )
        if np.allclose(updated, centers, rtol=0.0, atol=1e-12):
            break
        centers = updated
    centers.sort()
    threshold = float(centers.mean())
    dynamic = magnitude > threshold
    return {
        "sample_count": count,
        "dynamic_threshold_normalized_rms_delta": threshold,
        "dynamic_threshold_fit": {
            "method": "deterministic train-only 1D two-means",
            "static_center": float(centers[0]),
            "dynamic_center": float(centers[1]),
        },
        "static_fraction": float((~dynamic).mean()),
        "dynamic_fraction": float(dynamic.mean()),
        "action_magnitude": {
            "mean": float(magnitude.mean()),
            "median": float(np.median(magnitude)),
            "p90": float(np.quantile(magnitude, 0.9)),
        },
        "segments": {
            name: {
                "mean": float(activity[f"{name}_magnitude"].mean()),
                "median": float(np.median(activity[f"{name}_magnitude"])),
            }
            for name in SEGMENTS
        },
        "left_more_active_fraction": float(
            (activity["left_activity"] > activity["right_activity"]).mean()
        ),
        "right_more_active_fraction": float(activity["active_side"].mean()),
        "primitive_counts": dict(
            sorted(Counter(map(int, batch["primitive_id"].tolist())).items())
        ),
        "derivation": "train split only; dynamics are normalized action RMS temporal deltas",
    }
