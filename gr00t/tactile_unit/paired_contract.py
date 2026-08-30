"""S3.1 helpers for the paired T-Rex vision/action/contact data contract.

The functions in this module deliberately keep dataset roots out of serialized
identities.  Runtime callers pass a local root, while manifests store only
dataset-relative paths and immutable content/provenance hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np


VIDEO_KEY = "observation.images.head_left"
CANONICAL_FPS = 30.0
CANONICAL_HORIZON = 16
RAW_STATE_DIM = 58
RAW_ACTION_DIM = 58
TOKENIZER_DIM = 128
ACTION_HORIZON = 16
TREX_EMBODIMENT_TAG = "new_embodiment"
TREX_EMBODIMENT_ID = 31


@dataclass(frozen=True)
class EpisodeVideoPointer:
    episode_id: int
    length: int
    data_chunk_index: int
    data_file_index: int
    dataset_from_index: int
    dataset_to_index: int
    video_chunk_index: int
    video_file_index: int
    from_timestamp: float
    to_timestamp: float
    relative_path: str
    motor_primitive: str | None
    object_name: str | None
    target: str | None

    @property
    def duration(self) -> float:
        return self.to_timestamp - self.from_timestamp


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, block_size: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing T-Rex metadata: {info_path}")
    info = json.loads(info_path.read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"S3.1 requires LeRobot v3 metadata, got {info.get('codebase_version')!r}")
    return info


def discover_dataset_revision(dataset_root: Path) -> str | None:
    trees = dataset_root / ".cache" / "huggingface" / "trees"
    revisions = sorted(path.stem for path in trees.glob("*.json")) if trees.is_dir() else []
    return revisions[0] if len(revisions) == 1 else None


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"metadata produced an unsafe video path: {value!r}")
    return path.as_posix()


def resolve_video_path(
    info: dict[str, Any], video_key: str, chunk_index: int, file_index: int
) -> str:
    template = info.get("video_path")
    if not isinstance(template, str):
        raise ValueError("meta/info.json does not define video_path")
    return _safe_relative_path(
        template.format(
            video_key=video_key,
            chunk_index=int(chunk_index),
            file_index=int(file_index),
        )
    )


def load_episode_video_pointers(
    dataset_root: Path, video_key: str = VIDEO_KEY
) -> list[EpisodeVideoPointer]:
    import pyarrow.parquet as pq

    info = load_info(dataset_root)
    prefix = f"videos/{video_key}"
    columns = [
        "episode_index",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
        f"{prefix}/chunk_index",
        f"{prefix}/file_index",
        f"{prefix}/from_timestamp",
        f"{prefix}/to_timestamp",
        "motor_primitive",
        "object",
        "target",
    ]
    parquet_paths = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError("no meta/episodes parquet files found")
    rows: list[EpisodeVideoPointer] = []
    for parquet_path in parquet_paths:
        values = pq.read_table(parquet_path, columns=columns).to_pydict()
        for row in range(len(values["episode_index"])):
            video_chunk = int(values[f"{prefix}/chunk_index"][row])
            video_file = int(values[f"{prefix}/file_index"][row])
            pointer = EpisodeVideoPointer(
                episode_id=int(values["episode_index"][row]),
                length=int(values["length"][row]),
                data_chunk_index=int(values["data/chunk_index"][row]),
                data_file_index=int(values["data/file_index"][row]),
                dataset_from_index=int(values["dataset_from_index"][row]),
                dataset_to_index=int(values["dataset_to_index"][row]),
                video_chunk_index=video_chunk,
                video_file_index=video_file,
                from_timestamp=float(values[f"{prefix}/from_timestamp"][row]),
                to_timestamp=float(values[f"{prefix}/to_timestamp"][row]),
                relative_path=resolve_video_path(info, video_key, video_chunk, video_file),
                motor_primitive=values["motor_primitive"][row],
                object_name=values["object"][row],
                target=values["target"][row],
            )
            if pointer.dataset_to_index - pointer.dataset_from_index != pointer.length:
                raise ValueError(f"episode {pointer.episode_id} data interval length mismatch")
            rows.append(pointer)
    rows.sort(key=lambda item: item.episode_id)
    expected_ids = list(range(len(rows)))
    actual_ids = [item.episode_id for item in rows]
    if actual_ids != expected_ids:
        raise ValueError("episode metadata is not a unique contiguous episode index")
    if len(rows) != int(info["total_episodes"]):
        raise ValueError("episode metadata row count disagrees with meta/info.json")
    return rows


def episode_frame_timestamp(
    pointer: EpisodeVideoPointer, frame_index: int, fps: float = CANONICAL_FPS
) -> float:
    if frame_index < 0 or frame_index >= pointer.length:
        raise IndexError(
            f"frame {frame_index} outside episode {pointer.episode_id} length {pointer.length}"
        )
    return pointer.from_timestamp + frame_index / fps


def validate_transition_anchor(
    pointer: EpisodeVideoPointer, anchor_frame: int, horizon: int = CANONICAL_HORIZON
) -> dict[str, Any]:
    if anchor_frame < 15:
        raise ValueError("current Teacher window t-15:t is incomplete")
    if anchor_frame + horizon >= pointer.length:
        raise ValueError("future anchor t+16 is outside the episode")
    current = episode_frame_timestamp(pointer, anchor_frame)
    future = episode_frame_timestamp(pointer, anchor_frame + horizon)
    action_last = episode_frame_timestamp(pointer, anchor_frame + horizon - 1)
    if not (pointer.from_timestamp <= current < action_last < future < pointer.to_timestamp):
        raise ValueError("canonical transition timestamps are outside the packed-video interval")
    return {
        "current_packed_timestamp": current,
        "action_last_packed_timestamp": action_last,
        "future_packed_timestamp": future,
        "anchor_delta_sec": future - current,
    }


def audit_video_inventory(
    dataset_root: Path, pointers: Iterable[EpisodeVideoPointer], video_key: str = VIDEO_KEY
) -> dict[str, Any]:
    pointer_list = list(pointers)
    references = [item.relative_path for item in pointer_list]
    counts = Counter(references)
    expected = set(references)
    stream_root = dataset_root / "videos" / video_key
    actual = {
        path.relative_to(dataset_root).as_posix() for path in stream_root.rglob("*.mp4")
    }
    present = expected & actual
    zero = sorted(path for path in present if (dataset_root / path).stat().st_size == 0)
    return {
        "episodes_scanned": len(pointer_list),
        "unique_referenced_mp4s": len(expected),
        "local_mp4_count": len(actual),
        "missing_referenced": sorted(expected - actual),
        "extra_unreferenced": sorted(actual - expected),
        "zero_size_referenced": zero,
        "duplicate_path_reference_count": sum(value > 1 for value in counts.values()),
        "max_episode_references_per_path": max(counts.values(), default=0),
        "expected_relative_paths": sorted(expected),
        "local_disk_bytes": sum((dataset_root / path).stat().st_size for path in actual),
    }


def _fraction(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    return float(Fraction(value))


def ffprobe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,avg_frame_rate,r_frame_rate,duration,nb_frames",
        "-show_entries",
        "format=duration,format_name",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"ffprobe exited {completed.returncode}")
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError("no video stream")
    stream = streams[0]
    duration = float(stream.get("duration") or payload.get("format", {}).get("duration") or 0)
    return {
        "codec": stream.get("codec_name"),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "pix_fmt": stream.get("pix_fmt"),
        "avg_fps": _fraction(stream.get("avg_frame_rate")),
        "real_fps": _fraction(stream.get("r_frame_rate")),
        "duration": duration,
        "nb_frames": int(stream["nb_frames"]) if str(stream.get("nb_frames", "")).isdigit() else None,
        "format": payload.get("format", {}).get("format_name"),
    }


def validate_video_probe(probe: dict[str, Any], feature: dict[str, Any]) -> list[str]:
    video_info = feature["info"]
    failures = []
    expected = {
        "codec": video_info["video.codec"],
        "width": int(video_info["video.width"]),
        "height": int(video_info["video.height"]),
        "pix_fmt": video_info["video.pix_fmt"],
        "fps": float(video_info["video.fps"]),
    }
    for key in ("codec", "width", "height", "pix_fmt"):
        if probe[key] != expected[key]:
            failures.append(f"{key}={probe[key]!r}, expected {expected[key]!r}")
    if not math.isclose(probe["avg_fps"], expected["fps"], rel_tol=0.0, abs_tol=1e-3):
        failures.append(f"avg_fps={probe['avg_fps']}, expected {expected['fps']}")
    if probe["duration"] <= 0:
        failures.append("duration is not positive")
    return failures


def decode_rgb_frame_nearest(path: Path, timestamp: float) -> tuple[np.ndarray, float]:
    import av

    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        target = max(0.0, float(timestamp))
        if target > 0 and stream.time_base is not None:
            container.seek(
                int(target / float(stream.time_base)),
                stream=stream,
                any_frame=False,
                backward=True,
            )
        chosen = None
        chosen_time = None
        best_distance = float("inf")
        for frame in container.decode(stream):
            frame_time = float(frame.time) if frame.time is not None else None
            if frame_time is None:
                chosen = frame
                chosen_time = target
                break
            distance = abs(frame_time - target)
            if distance < best_distance:
                chosen = frame
                chosen_time = frame_time
                best_distance = distance
            if frame_time > target and distance > best_distance:
                break
        if chosen is None:
            raise RuntimeError("decoder returned no frame")
        array = chosen.to_ndarray(format="rgb24")
    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise RuntimeError(f"unexpected decoded frame {array.shape} {array.dtype}")
    if not np.isfinite(array).all():
        raise RuntimeError("decoded frame contains non-finite values")
    return array, float(chosen_time)


def decode_rgb_frame(path: Path, timestamp: float) -> np.ndarray:
    return decode_rgb_frame_nearest(path, timestamp)[0]


def decoded_frame_stats(frame: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(frame.shape),
        "dtype": str(frame.dtype),
        "min": int(frame.min()),
        "max": int(frame.max()),
        "mean": float(frame.mean()),
        "std": float(frame.std()),
        "all_black": bool(np.max(frame) == 0),
        "sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
    }


def pad_trex_state_action(
    state: np.ndarray,
    action: np.ndarray,
    *,
    target_dim: int = TOKENIZER_DIM,
) -> dict[str, np.ndarray]:
    state = np.asarray(state, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)
    if state.shape != (RAW_STATE_DIM,):
        raise ValueError(f"expected raw state [{RAW_STATE_DIM}], got {state.shape}")
    if action.shape != (ACTION_HORIZON, RAW_ACTION_DIM):
        raise ValueError(
            f"expected action [{ACTION_HORIZON},{RAW_ACTION_DIM}], got {action.shape}"
        )
    if target_dim < RAW_STATE_DIM or target_dim < RAW_ACTION_DIM:
        raise ValueError("target tokenizer dimension is too small")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("state/action values must be finite")
    padded_state = np.zeros((target_dim,), dtype=np.float32)
    padded_action = np.zeros((ACTION_HORIZON, target_dim), dtype=np.float32)
    state_mask = np.zeros((target_dim,), dtype=bool)
    action_mask = np.zeros((ACTION_HORIZON, target_dim), dtype=bool)
    padded_state[:RAW_STATE_DIM] = state
    padded_action[:, :RAW_ACTION_DIM] = action
    state_mask[:RAW_STATE_DIM] = True
    action_mask[:, :RAW_ACTION_DIM] = True
    return {
        "state": padded_state,
        "state_mask": state_mask,
        "action": padded_action,
        "action_mask": action_mask,
    }


def normalize_and_pad_trex_state_action(
    state: np.ndarray,
    action: np.ndarray,
    statistics: dict[str, Any],
    *,
    target_dim: int = TOKENIZER_DIM,
) -> dict[str, np.ndarray]:
    """Mirror UniT ``mean_std`` normalization, then apply the 128-D mask contract."""

    state = np.asarray(state, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)
    if state.shape != (RAW_STATE_DIM,) or action.shape != (ACTION_HORIZON, RAW_ACTION_DIM):
        raise ValueError("raw T-Rex state/action shape mismatch")

    def normalize(value: np.ndarray, entry: dict[str, Any]) -> np.ndarray:
        mean = np.asarray(entry["mean"], dtype=np.float32)
        std = np.asarray(entry["std"], dtype=np.float32)
        if mean.shape != (value.shape[-1],) or std.shape != mean.shape:
            raise ValueError("T-Rex normalization statistic shape mismatch")
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 0):
            raise ValueError("invalid T-Rex normalization statistics")
        result = np.zeros_like(value, dtype=np.float32)
        mask = std != 0
        result[..., mask] = (value[..., mask] - mean[mask]) / std[mask]
        # Exact behavior of gr00t.data.transform.state_action.Normalizer(mean_std).
        result[..., ~mask] = value[..., ~mask]
        return result

    normalized_state = normalize(state, statistics["observation.state"])
    normalized_action = normalize(action, statistics["action"])
    result = pad_trex_state_action(normalized_state, normalized_action, target_dim=target_dim)
    result["normalized_state_58"] = normalized_state
    result["normalized_action_58"] = normalized_action
    return result


def preprocess_trex_rgb(frame: np.ndarray) -> np.ndarray:
    """Apply the deterministic UniT/DINO evaluation family to one RGB frame.

    This mirrors ``VideoCrop(scale=.95) -> VideoResize(224) -> ImageNet`` in
    evaluation mode: center crop, bilinear antialiased resize, CHW float32 and
    ImageNet normalization.
    """

    import torch
    from torchvision.transforms.v2 import functional as tvf
    from torchvision.transforms import InterpolationMode

    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
        raise ValueError("vision input must be uint8 HWC RGB")
    height, width = frame.shape[:2]
    crop_height, crop_width = int(height * 0.95), int(width * 0.95)
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    tensor = torch.from_numpy(frame.copy()).permute(2, 0, 1)
    tensor = tvf.crop(tensor, top, left, crop_height, crop_width)
    tensor = tvf.resize(
        tensor,
        [224, 224],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    ).float().div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
    result = ((tensor - mean) / std).numpy()
    if result.shape != (3, 224, 224) or result.dtype != np.float32:
        raise AssertionError("unexpected processed vision contract")
    if not np.isfinite(result).all():
        raise ValueError("processed vision contains non-finite values")
    return result


def cache_identity(
    *, teacher_sha256: str, encoder_sha256: str, transition_manifest_sha256: str, split_sha256: str
) -> dict[str, Any]:
    value = {
        "schema": "tactile3d-unit.s3-1-contact-cache-identity.v1",
        "s1_teacher_sha256": teacher_sha256,
        "s2_encoder_sha256": encoder_sha256,
        "s2_transition_manifest_sha256": transition_manifest_sha256,
        "s1_split_manifest_sha256": split_sha256,
        "transition_horizon_frames": CANONICAL_HORIZON,
    }
    value["identity_sha256"] = sha256_json(value)
    return value


def validate_cache_identity(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError("S3.1 contact cache identity mismatch")


def validate_episode_splits(splits: dict[str, Iterable[int]]) -> dict[str, int]:
    required = ("train", "val", "test")
    if any(name not in splits for name in required):
        raise ValueError("canonical train/val/test episode splits are required")
    values = {name: list(map(int, splits[name])) for name in required}
    for name, episode_ids in values.items():
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError(f"duplicate episode within {name} split")
    sets = {name: set(episode_ids) for name, episode_ids in values.items()}
    leakage = {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }
    if any(leakage.values()):
        raise ValueError(f"episode split leakage: {leakage}")
    return leakage


def pointer_public_record(pointer: EpisodeVideoPointer) -> dict[str, Any]:
    return asdict(pointer)


def pair_id(split: str, episode_id: int, anchor_frame: int) -> str:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"invalid canonical split {split!r}")
    return f"s2-k16:{split}:e{episode_id:04d}:t{anchor_frame:06d}"


def data_relative_path(info: dict[str, Any], pointer: EpisodeVideoPointer) -> str:
    template = info.get("data_path")
    if not isinstance(template, str):
        raise ValueError("meta/info.json does not define data_path")
    return _safe_relative_path(
        template.format(
            chunk_index=pointer.data_chunk_index,
            file_index=pointer.data_file_index,
        )
    )


def make_pair_record(
    *,
    split: str,
    source_index: int,
    pointer: EpisodeVideoPointer,
    anchor_frame: int,
    anchor_time: float,
    info: dict[str, Any],
    contact_transition: int,
    dynamic: bool,
    force_trend_class: int | None = None,
) -> dict[str, Any]:
    timing = validate_transition_anchor(pointer, anchor_frame)
    expected_anchor_time = anchor_frame / float(info["fps"])
    if not math.isclose(anchor_time, expected_anchor_time, rel_tol=0.0, abs_tol=2e-5):
        raise ValueError(
            f"S2 anchor time mismatch for episode {pointer.episode_id} frame {anchor_frame}"
        )
    data_path = data_relative_path(info, pointer)
    dataset_anchor = pointer.dataset_from_index + anchor_frame
    value = {
        "pair_id": pair_id(split, pointer.episode_id, anchor_frame),
        "source": {"milestone": "S2", "split": split, "row_index": int(source_index)},
        "episode_id": pointer.episode_id,
        "anchor": {
            "frame": int(anchor_frame),
            "episode_timestamp": float(anchor_time),
            "packed_video_timestamp": timing["current_packed_timestamp"],
        },
        "vision": {
            "stream": VIDEO_KEY,
            "relative_path": pointer.relative_path,
            "current": {
                "episode_frame": int(anchor_frame),
                "episode_timestamp": float(anchor_time),
                "packed_timestamp": timing["current_packed_timestamp"],
            },
            "future": {
                "episode_frame": int(anchor_frame + CANONICAL_HORIZON),
                "episode_timestamp": float(anchor_time + CANONICAL_HORIZON / float(info["fps"])),
                "packed_timestamp": timing["future_packed_timestamp"],
            },
        },
        "state": {
            "relative_path": data_path,
            "episode_frame": int(anchor_frame),
            "dataset_index": int(dataset_anchor),
            "raw_shape": [RAW_STATE_DIM],
        },
        "action": {
            "relative_path": data_path,
            "episode_frames_inclusive": [int(anchor_frame), int(anchor_frame + 15)],
            "dataset_indices_inclusive": [int(dataset_anchor), int(dataset_anchor + 15)],
            "raw_shape": [ACTION_HORIZON, RAW_ACTION_DIM],
        },
        "contact": {
            "current_teacher_window_inclusive": [int(anchor_frame - 15), int(anchor_frame)],
            "future_teacher_window_inclusive": [int(anchor_frame + 1), int(anchor_frame + 16)],
            "h_current_shape": [256],
            "h_future_shape": [256],
            "z_c_shape": [8, 32],
            "continuous_pre_adaptor": True,
            "transition_class": int(contact_transition),
            "dynamic": bool(dynamic),
            "force_trend_class": None if force_trend_class is None else int(force_trend_class),
            "cache_split": split,
            "cache_row_index": int(source_index),
        },
        "metadata": {
            "motor_primitive": pointer.motor_primitive,
            "object": pointer.object_name,
            "target": pointer.target,
        },
        "modality_mask": {"V": 1, "A": 1, "C": 1},
    }
    return value


def load_transition_arrays(cache_root: Path, split: str) -> dict[str, np.ndarray]:
    names = (
        "episode_id",
        "anchor_frame",
        "anchor_time",
        "future_anchor_time",
        "current",
        "future",
        "contact_transition",
        "dynamic",
        "force_trend_class",
    )
    arrays = {
        name: np.load(cache_root / split / f"{name}.npy", mmap_mode="r") for name in names
    }
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"unaligned S2 transition arrays for {split}")
    return arrays


class TReXPairedDataset:
    """Lazy, indexed V+A+C adapter over the frozen S2 pair identities."""

    def __init__(
        self,
        dataset_root: Path,
        transition_cache: Path,
        *,
        split: str,
        contact_codes: Path | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.info = load_info(self.dataset_root)
        self.pointers = {row.episode_id: row for row in load_episode_video_pointers(self.dataset_root)}
        self.split = split
        self.arrays = load_transition_arrays(Path(transition_cache), split)
        self.contact_codes = (
            np.load(contact_codes, mmap_mode="r") if contact_codes is not None else None
        )
        if self.contact_codes is not None:
            if self.contact_codes.shape != (len(self), 8, 32):
                raise ValueError("contact code cache shape mismatch")

    def __len__(self) -> int:
        return len(self.arrays["episode_id"])

    def record(self, index: int) -> dict[str, Any]:
        episode = int(self.arrays["episode_id"][index])
        anchor = int(self.arrays["anchor_frame"][index])
        return make_pair_record(
            split=self.split,
            source_index=index,
            pointer=self.pointers[episode],
            anchor_frame=anchor,
            anchor_time=float(self.arrays["anchor_time"][index]),
            info=self.info,
            contact_transition=int(self.arrays["contact_transition"][index]),
            dynamic=bool(self.arrays["dynamic"][index]),
            force_trend_class=int(self.arrays["force_trend_class"][index]),
        )

    def load_state_action(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        import pyarrow.parquet as pq

        record = self.record(index)
        path = self.dataset_root / record["state"]["relative_path"]
        episode = record["episode_id"]
        table = pq.read_table(
            path,
            columns=["episode_index", "frame_index", "observation.state", "action"],
            filters=[("episode_index", "=", episode)],
        )
        table = table.sort_by([("frame_index", "ascending")])
        frames = np.asarray(table["frame_index"])
        expected = np.arange(len(frames))
        if not np.array_equal(frames, expected):
            raise ValueError(f"episode {episode} parquet frame indices are not contiguous")
        anchor = record["anchor"]["frame"]
        state = np.asarray(table["observation.state"][anchor].as_py(), dtype=np.float32)
        action = np.asarray(
            table["action"].slice(anchor, ACTION_HORIZON).to_pylist(), dtype=np.float32
        )
        return state, action

    def decode_vision(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        record = self.record(index)
        path = self.dataset_root / record["vision"]["relative_path"]
        current = decode_rgb_frame(path, record["vision"]["current"]["packed_timestamp"])
        future = decode_rgb_frame(path, record["vision"]["future"]["packed_timestamp"])
        return current, future

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.record(index)
        state, action = self.load_state_action(index)
        current_rgb, future_rgb = self.decode_vision(index)
        result = {
            "pair_id": record["pair_id"],
            "record": record,
            "vision": {"current": current_rgb, "future": future_rgb},
            "processed_vision": {
                "obs": preprocess_trex_rgb(current_rgb),
                "goal": preprocess_trex_rgb(future_rgb),
            },
            "state": state,
            "action": action,
            "contact": {
                "h_current": np.asarray(self.arrays["current"][index]),
                "h_future": np.asarray(self.arrays["future"][index]),
                "z_c": None if self.contact_codes is None else np.asarray(self.contact_codes[index]),
            },
            "modality_mask": {"V": 1, "A": 1, "C": 1},
        }
        return result
