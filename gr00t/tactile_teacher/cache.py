"""Memory-mapped access to the canonical S1 wrench-window cache."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class WrenchWindowDataset(Dataset):
    """Read one deterministic episode-level split without loading it into RAM."""

    def __init__(self, cache_dir: str | Path, split: str):
        self.cache_dir = Path(cache_dir)
        self.split = split
        manifest_path = self.cache_dir / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text())
        if split not in self.manifest["splits"]:
            raise ValueError(f"unknown cache split: {split}")
        split_dir = self.cache_dir / split
        self.history = np.load(split_dir / "history.npy", mmap_mode="r")
        self.future = np.load(split_dir / "future.npy", mmap_mode="r")
        self.episode_id = np.load(split_dir / "episode_id.npy", mmap_mode="r")
        self.primitive_id = np.load(split_dir / "primitive_id.npy", mmap_mode="r")
        self.object_id = np.load(split_dir / "object_id.npy", mmap_mode="r")
        expected = int(self.manifest["splits"][split]["windows"])
        lengths = {
            len(self.history),
            len(self.future),
            len(self.episode_id),
            len(self.primitive_id),
            len(self.object_id),
        }
        if lengths != {expected}:
            raise ValueError(f"cache arrays do not align with manifest: {sorted(lengths)}")

    def __len__(self) -> int:
        return len(self.history)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # Copy read-only mmap slices so torch never writes through the cache.
        return {
            "history": torch.from_numpy(np.array(self.history[index], copy=True)),
            "future": torch.from_numpy(np.array(self.future[index], copy=True)),
            "episode_id": torch.tensor(int(self.episode_id[index]), dtype=torch.long),
            "primitive_id": torch.tensor(int(self.primitive_id[index]), dtype=torch.long),
            "object_id": torch.tensor(int(self.object_id[index]), dtype=torch.long),
        }


def load_split_arrays(cache_dir: str | Path, split: str) -> dict[str, np.ndarray]:
    """Return read-only mmap arrays for vectorized evaluation."""

    split_dir = Path(cache_dir) / split
    return {
        name: np.load(split_dir / f"{name}.npy", mmap_mode="r")
        for name in ("history", "future", "episode_id", "primitive_id", "object_id")
    }
