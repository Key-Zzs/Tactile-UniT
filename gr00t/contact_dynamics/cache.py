"""Memory-mapped access to frozen-Teacher S2 transition pairs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


ARRAY_NAMES = (
    "current",
    "future",
    "episode_id",
    "task_id",
    "anchor_frame",
    "anchor_time",
    "future_anchor_time",
    "primitive_id",
    "object_id",
    "current_force",
    "future_force",
    "current_finger_force",
    "future_finger_force",
    "contact_transition",
    "force_trend_class",
    "finger_change",
    "dynamic",
)


class ContactTransitionDataset(Dataset):
    def __init__(self, cache_dir: str | Path, split: str) -> None:
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.manifest = json.loads((self.cache_dir / "manifest.json").read_text())
        if split not in self.manifest["splits"]:
            raise ValueError(f"unknown transition split: {split}")
        self.arrays = load_transition_arrays(cache_dir, split)
        expected = int(self.manifest["splits"][split]["pairs"])
        if {len(value) for value in self.arrays.values()} != {expected}:
            raise ValueError("transition arrays do not align with manifest")

    def __len__(self) -> int:
        return len(self.arrays["current"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        result = {}
        for name, array in self.arrays.items():
            value = np.array(array[index], copy=True)
            result[name] = torch.from_numpy(value) if value.ndim else torch.tensor(value.item())
        return result


def load_transition_arrays(cache_dir: str | Path, split: str) -> dict[str, np.ndarray]:
    split_dir = Path(cache_dir) / split
    return {
        name: np.load(split_dir / f"{name}.npy", mmap_mode="r")
        for name in ARRAY_NAMES
    }
