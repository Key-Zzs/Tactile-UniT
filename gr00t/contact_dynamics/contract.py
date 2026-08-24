"""Frame-exact S2 contact-transition pairing contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransitionPairContract:
    """Define two frozen-Teacher windows separated by ``horizon_frames``.

    An anchor ``t`` names the final raw sample of the current Teacher window.
    The future Teacher window ends at ``t + k``.  With the canonical
    ``history_steps=16`` and ``k=16`` the windows are exactly ``[t-15,t]`` and
    ``[t+1,t+16]`` and therefore share no raw wrench sample.
    """

    history_steps: int = 16
    horizon_frames: int = 16
    sampling_rate_hz: float = 30.0

    def __post_init__(self) -> None:
        if self.history_steps < 2:
            raise ValueError("history_steps must be at least 2")
        if self.horizon_frames < 1:
            raise ValueError("horizon_frames must be positive")
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")

    @property
    def history_physical_span_sec(self) -> float:
        return (self.history_steps - 1) / self.sampling_rate_hz

    @property
    def anchor_delta_sec(self) -> float:
        return self.horizon_frames / self.sampling_rate_hz

    @property
    def overlap_samples(self) -> int:
        return max(0, self.history_steps - self.horizon_frames)

    @property
    def overlap_fraction(self) -> float:
        return self.overlap_samples / self.history_steps

    def current_indices(self, anchor_frame: int) -> np.ndarray:
        anchor = int(anchor_frame)
        return np.arange(anchor - self.history_steps + 1, anchor + 1, dtype=np.int64)

    def future_indices(self, anchor_frame: int) -> np.ndarray:
        future_anchor = int(anchor_frame) + self.horizon_frames
        return np.arange(
            future_anchor - self.history_steps + 1,
            future_anchor + 1,
            dtype=np.int64,
        )

    def validate_anchor(self, anchor_frame: int, episode_length: int) -> None:
        current = self.current_indices(anchor_frame)
        future = self.future_indices(anchor_frame)
        if current[0] < 0 or future[-1] >= int(episode_length):
            raise ValueError("anchor does not have two complete Teacher windows")
        actual = np.intersect1d(current, future, assume_unique=True)
        if len(actual) != self.overlap_samples:
            raise AssertionError("computed window overlap disagrees with contract")

    def to_dict(self) -> dict:
        return {
            "history_steps": self.history_steps,
            "history_physical_span_sec": self.history_physical_span_sec,
            "horizon_frames": self.horizon_frames,
            "anchor_delta_sec": self.anchor_delta_sec,
            "sampling_rate_hz": self.sampling_rate_hz,
            "current_window_relative": [-(self.history_steps - 1), 0],
            "future_window_relative": [
                self.horizon_frames - self.history_steps + 1,
                self.horizon_frames,
            ],
            "overlap_samples": self.overlap_samples,
            "overlap_fraction": self.overlap_fraction,
        }


def evenly_spaced_anchors(
    episode_length: int,
    count: int,
    *,
    history_steps: int = 16,
    maximum_horizon_frames: int = 24,
) -> np.ndarray:
    """Select deterministic integer anchors valid for every requested horizon."""

    first = history_steps - 1
    last = int(episode_length) - 1 - int(maximum_horizon_frames)
    if count < 1 or last < first:
        raise ValueError("episode is too short or requested count is invalid")
    candidates = np.arange(first, last + 1, dtype=np.int64)
    if len(candidates) < count:
        raise ValueError("episode has fewer valid anchors than requested")
    positions = np.linspace(0, len(candidates) - 1, count)
    anchors = candidates[np.rint(positions).astype(np.int64)]
    if len(np.unique(anchors)) != count:
        raise AssertionError("anchor selection produced duplicates")
    return anchors


def validate_episode_splits(split_episode_ids: dict[str, np.ndarray | list[int]]) -> dict[str, int]:
    """Reject episode leakage and return explicit pairwise overlap counts."""

    sets = {
        name: set(np.asarray(values, dtype=np.int64).tolist())
        for name, values in split_episode_ids.items()
    }
    required = {"train", "val", "test"}
    if set(sets) != required:
        raise ValueError(f"expected split names {sorted(required)}, got {sorted(sets)}")
    overlap = {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }
    if any(overlap.values()):
        raise ValueError(f"episode leakage detected: {overlap}")
    return overlap
