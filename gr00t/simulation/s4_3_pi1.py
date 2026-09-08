"""Contracts and small utilities for the S4.3-PI1 tactile pi0.5 study."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

import numpy as np


class TactileUnitMode(str, Enum):
    """The three frozen policy interventions used by PI1."""

    NONE = "NONE"
    CONTACT_STATE_TOKENS = "CONTACT_STATE_TOKENS"
    CONTACT_STATE_TOKENS_PHYSICAL_AUX = "CONTACT_STATE_TOKENS_PHYSICAL_AUX"


HISTORY_STEPS = 26
TACTILE_DIM = 30
CONTACT_STATE_DIM = 256
CONTACT_TOKENS = 8
CONTACT_TOKEN_WIDTH = 32
PHYSICAL_TARGET_HORIZON = 27
HISTORY_BOOTSTRAP = "LEFT_REPEAT_FIRST"


def left_repeat_history(samples: np.ndarray, anchor: int) -> tuple[np.ndarray, int]:
    """Return the frozen causal 26-sample history ending at ``anchor``."""

    values = np.asarray(samples, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != TACTILE_DIM:
        raise ValueError(f"tactile samples must be [N,{TACTILE_DIM}], got {values.shape}")
    if not 0 <= anchor < len(values):
        raise IndexError(anchor)
    start = anchor - HISTORY_STEPS + 1
    bootstrap = max(0, -start)
    observed = values[max(start, 0) : anchor + 1]
    if bootstrap:
        observed = np.concatenate(
            [np.repeat(values[:1], bootstrap, axis=0), observed], axis=0
        )
    if observed.shape != (HISTORY_STEPS, TACTILE_DIM):
        raise RuntimeError(f"history construction produced {observed.shape}")
    return observed, bootstrap


@dataclass
class OnlineTactileHistory:
    """Online form of the exact LEFT_REPEAT_FIRST training bootstrap."""

    _values: deque[np.ndarray]

    def __init__(self) -> None:
        self._values = deque(maxlen=HISTORY_STEPS)

    def reset(self, first: np.ndarray) -> np.ndarray:
        value = np.asarray(first, dtype=np.float32)
        if value.shape != (TACTILE_DIM,) or not np.isfinite(value).all():
            raise ValueError("first tactile sample must be finite [30]")
        self._values.clear()
        self._values.extend(value.copy() for _ in range(HISTORY_STEPS))
        return self.value()

    def append(self, value: np.ndarray) -> np.ndarray:
        item = np.asarray(value, dtype=np.float32)
        if item.shape != (TACTILE_DIM,) or not np.isfinite(item).all():
            raise ValueError("tactile sample must be finite [30]")
        if not self._values:
            return self.reset(item)
        self._values.append(item.copy())
        return self.value()

    def value(self) -> np.ndarray:
        if len(self._values) != HISTORY_STEPS:
            raise RuntimeError("history has not been initialized")
        return np.stack(tuple(self._values), axis=0)


def sha256_paths(paths: Iterable[Path]) -> str:
    """Canonical aggregate hash for a collection of files."""

    import hashlib

    digest = hashlib.sha256()
    for path in sorted((Path(p) for p in paths), key=lambda value: value.as_posix()):
        digest.update(path.as_posix().encode("utf-8"))
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()
