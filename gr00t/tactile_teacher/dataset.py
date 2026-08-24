"""Read-only adapter for numeric T-Rex LeRobot v3 parquet episodes."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import TACTILE_KEY, TactileDataContract


@dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    length: int
    chunk_index: int
    file_index: int
    dataset_from_index: int
    dataset_to_index: int
    motor_primitive: str
    object_label: str
    target: str | None


@dataclass(frozen=True)
class EpisodeData:
    record: EpisodeRecord
    timestamps: np.ndarray
    frame_indices: np.ndarray
    global_indices: np.ndarray
    task_indices: np.ndarray
    wrench: np.ndarray

    def validate(self) -> None:
        n = self.record.length
        arrays = (
            self.timestamps,
            self.frame_indices,
            self.global_indices,
            self.task_indices,
            self.wrench,
        )
        if any(len(x) != n for x in arrays):
            raise ValueError(f"episode {self.record.episode_index} length mismatch")
        if self.wrench.shape != (n, 60):
            raise ValueError(f"bad wrench shape {self.wrench.shape}")
        if not np.array_equal(self.frame_indices, np.arange(n, dtype=np.int64)):
            raise ValueError(f"episode {self.record.episode_index} frame_index is not contiguous")
        if not np.all(np.diff(self.timestamps) > 0):
            raise ValueError(f"episode {self.record.episode_index} timestamps are not increasing")
        if not np.all(np.isfinite(self.wrench)):
            raise ValueError(f"episode {self.record.episode_index} wrench contains NaN/Inf")


def _fixed_list_to_numpy(column: pa.ChunkedArray, width: int, dtype: np.dtype) -> np.ndarray:
    array = column.combine_chunks()
    if not pa.types.is_list(array.type) and not pa.types.is_fixed_size_list(array.type):
        raise TypeError(f"expected list array, got {array.type}")
    lengths = np.asarray(array.value_lengths())
    if len(lengths) and not np.all(lengths == width):
        bad = np.unique(lengths[lengths != width])[:10]
        raise ValueError(f"expected list width {width}, got {bad.tolist()}")
    flat = np.asarray(array.values.to_numpy(zero_copy_only=False), dtype=dtype)
    return flat.reshape(len(array), width)


class TactileEpisodeStore:
    """Episode-addressable view over the 48 public numeric parquet files."""

    FRAME_COLUMNS = (
        TACTILE_KEY,
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    )

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        dataset_revision: str = "unknown",
        cache_files: int = 2,
    ) -> None:
        self.root = Path(dataset_root)
        self.contract = TactileDataContract.from_root(self.root, dataset_revision)
        self.cache_files = max(1, int(cache_files))
        self._cache: OrderedDict[tuple[int, int], dict[str, np.ndarray]] = OrderedDict()
        episode_files = sorted((self.root / "meta" / "episodes").rglob("*.parquet"))
        columns = [
            "episode_index",
            "length",
            "data/chunk_index",
            "data/file_index",
            "dataset_from_index",
            "dataset_to_index",
            "motor_primitive",
            "object",
            "target",
        ]
        metadata = pd.concat([pd.read_parquet(path, columns=columns) for path in episode_files])
        metadata = metadata.sort_values("episode_index").reset_index(drop=True)
        self.records = tuple(
            EpisodeRecord(
                episode_index=int(row["episode_index"]),
                length=int(row["length"]),
                chunk_index=int(row["data/chunk_index"]),
                file_index=int(row["data/file_index"]),
                dataset_from_index=int(row["dataset_from_index"]),
                dataset_to_index=int(row["dataset_to_index"]),
                motor_primitive=str(row["motor_primitive"]),
                object_label=str(row["object"]),
                target=None if pd.isna(row["target"]) else str(row["target"]),
            )
            for row in metadata.to_dict("records")
        )
        self._record_by_id = {r.episode_index: r for r in self.records}
        if len(self._record_by_id) != self.contract.total_episodes:
            raise ValueError(
                f"metadata has {len(self._record_by_id)} unique episodes, "
                f"expected {self.contract.total_episodes}"
            )
        if sum(r.length for r in self.records) != self.contract.total_frames:
            raise ValueError("episode lengths do not sum to total_frames")

    @property
    def episode_ids(self) -> tuple[int, ...]:
        return tuple(r.episode_index for r in self.records)

    def episode_to_primitive(self) -> dict[int, str]:
        return {r.episode_index: r.motor_primitive for r in self.records}

    def _file_path(self, key: tuple[int, int]) -> Path:
        chunk, file_index = key
        return self.root / "data" / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.parquet"

    def _load_file(self, key: tuple[int, int]) -> dict[str, np.ndarray]:
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        path = self._file_path(key)
        if not path.is_file():
            raise FileNotFoundError(path)
        table = pq.read_table(path, columns=list(self.FRAME_COLUMNS))
        value = {
            "wrench": _fixed_list_to_numpy(table[TACTILE_KEY], 60, np.float32),
            "timestamp": np.asarray(table["timestamp"].to_numpy(), dtype=np.float64),
            "frame_index": np.asarray(table["frame_index"].to_numpy(), dtype=np.int64),
            "episode_index": np.asarray(table["episode_index"].to_numpy(), dtype=np.int64),
            "index": np.asarray(table["index"].to_numpy(), dtype=np.int64),
            "task_index": np.asarray(table["task_index"].to_numpy(), dtype=np.int64),
        }
        self._cache[key] = value
        while len(self._cache) > self.cache_files:
            self._cache.popitem(last=False)
        return value

    def get_episode(self, episode_index: int, *, validate: bool = True) -> EpisodeData:
        record = self._record_by_id[int(episode_index)]
        arrays = self._load_file((record.chunk_index, record.file_index))
        start = int(np.searchsorted(arrays["index"], record.dataset_from_index, side="left"))
        stop = int(np.searchsorted(arrays["index"], record.dataset_to_index, side="left"))
        if stop - start != record.length:
            raise ValueError(
                f"episode {episode_index} slice length {stop-start} != metadata {record.length}"
            )
        if not np.all(arrays["episode_index"][start:stop] == record.episode_index):
            raise ValueError(f"episode {episode_index} parquet slice contains other episode ids")
        episode = EpisodeData(
            record=record,
            timestamps=arrays["timestamp"][start:stop].copy(),
            frame_indices=arrays["frame_index"][start:stop].copy(),
            global_indices=arrays["index"][start:stop].copy(),
            task_indices=arrays["task_index"][start:stop].copy(),
            wrench=arrays["wrench"][start:stop].copy(),
        )
        if validate:
            episode.validate()
        return episode

    def iter_episodes(
        self, episode_ids: Sequence[int] | None = None, *, validate: bool = True
    ) -> Iterator[EpisodeData]:
        ids = self.episode_ids if episode_ids is None else tuple(int(x) for x in episode_ids)
        # Grouping by source file keeps the two-file cache effective without changing identity.
        ids = tuple(
            sorted(
                ids,
                key=lambda x: (
                    self._record_by_id[x].chunk_index,
                    self._record_by_id[x].file_index,
                    x,
                ),
            )
        )
        for episode_id in ids:
            yield self.get_episode(episode_id, validate=validate)
