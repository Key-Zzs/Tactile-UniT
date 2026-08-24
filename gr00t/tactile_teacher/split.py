"""Deterministic, episode-level T-Rex train/validation/test splitting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


def _stable_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


@dataclass(frozen=True)
class EpisodeSplit:
    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]
    seed: int
    stratify_key: str = "motor_primitive"

    def validate(self, universe: Iterable[int]) -> None:
        groups = [set(self.train), set(self.val), set(self.test)]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("episode leakage detected between split partitions")
        expected = set(int(x) for x in universe)
        actual = groups[0] | groups[1] | groups[2]
        if actual != expected:
            missing = sorted(expected - actual)[:10]
            extra = sorted(actual - expected)[:10]
            raise ValueError(f"split does not cover universe; missing={missing}, extra={extra}")

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.to_dict(include_ids=True), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self, include_ids: bool = True) -> dict:
        out = {
            "rule": "episode-level stratified by motor_primitive",
            "seed": self.seed,
            "stratify_key": self.stratify_key,
            "counts": {
                "train": len(self.train),
                "val": len(self.val),
                "test": len(self.test),
            },
        }
        if include_ids:
            out["episode_ids"] = {
                "train": list(self.train),
                "val": list(self.val),
                "test": list(self.test),
            }
        return out


def build_episode_split(
    episode_to_stratum: Mapping[int, str],
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> EpisodeSplit:
    """Split every stratum independently while keeping every episode intact."""

    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("ratios must be positive and sum to less than one")
    strata: dict[str, list[int]] = {}
    for episode_id, label in episode_to_stratum.items():
        strata.setdefault(str(label), []).append(int(episode_id))

    partitions: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for label in sorted(strata):
        ids = np.asarray(sorted(strata[label]), dtype=np.int64)
        rng = np.random.default_rng(_stable_seed(seed, label))
        rng.shuffle(ids)
        n = len(ids)
        if n < 3:
            raise ValueError(f"stratum {label!r} has fewer than three episodes")
        n_val = max(1, int(round(n * val_ratio)))
        n_test = max(1, int(round(n * (1.0 - train_ratio - val_ratio))))
        if n_val + n_test >= n:
            n_val = 1
            n_test = 1
        n_train = n - n_val - n_test
        partitions["train"].extend(ids[:n_train].tolist())
        partitions["val"].extend(ids[n_train : n_train + n_val].tolist())
        partitions["test"].extend(ids[n_train + n_val :].tolist())

    split = EpisodeSplit(
        train=tuple(sorted(partitions["train"])),
        val=tuple(sorted(partitions["val"])),
        test=tuple(sorted(partitions["test"])),
        seed=seed,
    )
    split.validate(episode_to_stratum)
    return split
