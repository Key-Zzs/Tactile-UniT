"""Immutable cache and deterministic training helpers for S4.3 ACT policies."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from gr00t.simulation.s4_3_act import PolicyNormalization

TASKS = ("pinch_tongs", "hammer_nail", "click_mouse")
VARIANTS = ("P0", "P1", "P2", "P3")
TRAINING_SEEDS = (0, 1, 2)
CACHE_ARRAYS = {
    "vision": ((8, 32), np.float32),
    "proprio": ((22,), np.float32),
    "tactile_history": ((26, 30), np.float32),
    "contact_state": ((256,), np.float32),
    "action": ((27, 22), np.float32),
    "contact_target": ((8, 32), np.float32),
    "episode_index": ((), np.int16),
    "t": ((), np.int32),
    "pair_id": ((), np.dtype("S80")),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def valid_anchors(length: int) -> range:
    """Return t with exact T[t-25:t], a[t:t+26], and S4.2 t->t+27 target."""

    # The action-only PD capacity includes t=length-27. The frozen S4.2
    # transition target additionally needs tactile at t+27, so R8 excludes it.
    return range(25, int(length) - 27)


def set_deterministic(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def create_array(
    path: Path,
    rows: int,
    tail: tuple[int, ...],
    dtype: np.dtype,
    *,
    fill: int | float | bool | None = None,
) -> np.memmap:
    shape = (rows, *tail)
    if path.is_file():
        value = np.load(path, mmap_mode="r+", allow_pickle=False)
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise RuntimeError(f"cache array identity changed: {path}")
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    if fill is not None:
        value[...] = fill
        value.flush()
    return value


def cache_paths(root: Path, task: str, split: str) -> dict[str, Path]:
    directory = root / task / split
    return {name: directory / f"{name}.npy" for name in CACHE_ARRAYS}


def load_normalization(cache_root: Path, task: str) -> PolicyNormalization:
    payload = read_json(cache_root / task / "normalization.json")
    if payload["fit_split"] != "POLICY_EXPERT_TRAIN" or payload["task"] != task:
        raise RuntimeError("policy normalization provenance mismatch")
    return PolicyNormalization.from_json(payload["statistics"])


class PolicyCacheDataset(Dataset[dict[str, torch.Tensor]]):
    """Read-only mmap-backed S4.3 policy cache."""

    def __init__(self, cache_root: Path, task: str, split: str, variant: str) -> None:
        if task not in TASKS or variant not in VARIANTS or split not in {"train", "dev"}:
            raise ValueError("unknown policy cache scope")
        manifest = read_json(cache_root / "manifest.json")
        if manifest.get("status") != "PASS" or manifest.get("immutable") is not True:
            raise RuntimeError("S4.3 policy cache is not frozen")
        scoped = manifest["tasks"][task][split]
        self.rows = int(scoped["rows"])
        self.variant = variant
        paths = cache_paths(cache_root, task, split)
        required = ["vision", "proprio", "action"]
        if variant == "P1":
            required.append("tactile_history")
        if variant in {"P2", "P3"}:
            required.append("contact_state")
        if variant == "P3":
            required.append("contact_target")
        self.arrays = {
            name: np.load(paths[name], mmap_mode="r", allow_pickle=False) for name in required
        }
        if any(len(value) != self.rows for value in self.arrays.values()):
            raise RuntimeError("policy cache row count mismatch")

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # Copy makes tensors writable and prevents collate warnings from read-only mmaps.
        return {
            name: torch.from_numpy(np.array(value[index], dtype=np.float32, copy=True))
            for name, value in self.arrays.items()
        }


def infinite_batches(loader: Any) -> Iterator[Mapping[str, torch.Tensor]]:
    while True:
        yield from loader
