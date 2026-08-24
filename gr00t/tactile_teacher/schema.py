"""Canonical T-Rex wrench schema and public data-contract validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

SIDES = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
WRENCH_COMPONENTS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
FEATURE_ORDER = tuple(
    f"{side}.{finger}.{component}"
    for side in SIDES
    for finger in FINGERS
    for component in WRENCH_COMPONENTS
)

TACTILE_KEY = "observation.tactile_force"
REQUIRED_FRAME_COLUMNS = (
    TACTILE_KEY,
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)
REQUIRED_EPISODE_COLUMNS = (
    "episode_index",
    "length",
    "data/chunk_index",
    "data/file_index",
    "dataset_from_index",
    "dataset_to_index",
    "motor_primitive",
    "object",
    "target",
)


@dataclass(frozen=True)
class TactileDataContract:
    """Resolved public dataset contract with no private path serialization."""

    schema_version: str
    dataset_revision: str
    fps: float
    total_episodes: int
    total_frames: int
    tactile_dim: int
    feature_order: tuple[str, ...] = FEATURE_ORDER
    wrench_units: str = "UNSPECIFIED"
    wrench_frame: str = "UNSPECIFIED"
    image_modalities_deferred: bool = True

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_revision": self.dataset_revision,
            "fps": self.fps,
            "total_episodes": self.total_episodes,
            "total_frames": self.total_frames,
            "tactile_key": TACTILE_KEY,
            "tactile_dim": self.tactile_dim,
            "feature_order": list(self.feature_order),
            "wrench_units": self.wrench_units,
            "wrench_frame": self.wrench_frame,
            "image_modalities_deferred": self.image_modalities_deferred,
        }

    @classmethod
    def from_root(
        cls, dataset_root: str | Path, dataset_revision: str = "unknown"
    ) -> "TactileDataContract":
        root = Path(dataset_root)
        info_path = root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"missing LeRobot metadata: {info_path}")
        info = json.loads(info_path.read_text())
        features = info.get("features", {})
        missing = sorted(set(REQUIRED_FRAME_COLUMNS) - set(features))
        if missing:
            raise ValueError(f"missing required frame features: {missing}")
        tactile = features[TACTILE_KEY]
        shape = tactile.get("shape")
        if shape != [len(FEATURE_ORDER)]:
            raise ValueError(f"{TACTILE_KEY} shape must be [60], got {shape}")
        if float(info.get("fps", 0)) <= 0:
            raise ValueError(f"invalid dataset fps: {info.get('fps')}")

        data_files = sorted((root / "data").rglob("*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"no numeric parquet files under {root / 'data'}")
        frame_schema = set(pq.ParquetFile(data_files[0]).schema_arrow.names)
        parquet_missing = sorted(set(REQUIRED_FRAME_COLUMNS) - frame_schema)
        if parquet_missing:
            raise ValueError(f"data parquet missing columns: {parquet_missing}")

        episode_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
        if not episode_files:
            raise FileNotFoundError("missing meta/episodes parquet")
        episode_schema = set(pq.ParquetFile(episode_files[0]).schema_arrow.names)
        episode_missing = sorted(set(REQUIRED_EPISODE_COLUMNS) - episode_schema)
        if episode_missing:
            raise ValueError(f"episode metadata missing columns: {episode_missing}")

        return cls(
            schema_version=str(info.get("codebase_version", "unknown")),
            dataset_revision=dataset_revision,
            fps=float(info["fps"]),
            total_episodes=int(info["total_episodes"]),
            total_frames=int(info["total_frames"]),
            tactile_dim=int(shape[0]),
        )
