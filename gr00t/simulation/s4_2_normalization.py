"""Train-only normalization for the frozen 5-region S4.2 tactile schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

REGIONS = 5
FEATURES = 6


@dataclass(frozen=True)
class TactileNormalization:
    candidate: str
    force_mean: np.ndarray
    force_std: np.ndarray
    cop_mean: np.ndarray
    cop_std: np.ndarray
    force_scale_newton: float = 1.0

    def __post_init__(self) -> None:
        if self.candidate not in {"N0", "N1"}:
            raise ValueError(f"unknown normalization candidate {self.candidate!r}")
        if np.asarray(self.force_mean).shape != (REGIONS, 2):
            raise ValueError("force statistics must have shape [5,2]")
        if np.asarray(self.force_std).shape != (REGIONS, 2):
            raise ValueError("force statistics must have shape [5,2]")
        if np.asarray(self.cop_mean).shape != (REGIONS, 3):
            raise ValueError("CoP statistics must have shape [5,3]")
        if np.asarray(self.cop_std).shape != (REGIONS, 3):
            raise ValueError("CoP statistics must have shape [5,3]")

    def transform(self, tactile: np.ndarray) -> np.ndarray:
        values = np.asarray(tactile, dtype=np.float32)
        if values.shape[-1] != REGIONS * FEATURES:
            raise ValueError("tactile input must end in 30 features")
        matrix = values.reshape(*values.shape[:-1], REGIONS, FEATURES).copy()
        occupancy = matrix[..., 0] > 0.5
        force = matrix[..., 1:3]
        if self.candidate == "N1":
            force = np.log1p(np.maximum(force, 0.0) / self.force_scale_newton)
        matrix[..., 1:3] = (force - self.force_mean) / self.force_std
        normalized_cop = (matrix[..., 3:6] - self.cop_mean) / self.cop_std
        matrix[..., 3:6] = np.where(occupancy[..., None], normalized_cop, 0.0)
        matrix[..., 0] = occupancy.astype(np.float32)
        return matrix.reshape(values.shape)

    def to_json(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "force_mean": self.force_mean.tolist(),
            "force_std": self.force_std.tolist(),
            "cop_mean": self.cop_mean.tolist(),
            "cop_std": self.cop_std.tolist(),
            "force_scale_newton": self.force_scale_newton,
            "fit_split": "train",
            "empty_cop_after_transform": "exact_zero",
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "TactileNormalization":
        return cls(
            candidate=value["candidate"],
            force_mean=np.asarray(value["force_mean"], dtype=np.float32),
            force_std=np.asarray(value["force_std"], dtype=np.float32),
            cop_mean=np.asarray(value["cop_mean"], dtype=np.float32),
            cop_std=np.asarray(value["cop_std"], dtype=np.float32),
            force_scale_newton=float(value.get("force_scale_newton", 1.0)),
        )


def fit_tactile_normalization(tactile: np.ndarray, candidate: str) -> TactileNormalization:
    values = np.asarray(tactile, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != REGIONS * FEATURES:
        raise ValueError("normalization fit requires raw [N,30] tactile")
    if not np.isfinite(values).all():
        raise ValueError("normalization fit data must be finite")
    matrix = values.reshape(-1, REGIONS, FEATURES)
    occupancy = matrix[..., 0] > 0.5
    force = np.maximum(matrix[..., 1:3], 0.0)
    if candidate == "N1":
        force = np.log1p(force)
    elif candidate != "N0":
        raise ValueError(f"unknown normalization candidate {candidate!r}")
    force_mean = force.mean(axis=0)
    force_std = np.maximum(force.std(axis=0), 1e-6)
    cop_mean = np.zeros((REGIONS, 3), dtype=np.float64)
    cop_std = np.ones((REGIONS, 3), dtype=np.float64)
    for region in range(REGIONS):
        occupied = matrix[occupancy[:, region], region, 3:6]
        if len(occupied):
            cop_mean[region] = occupied.mean(axis=0)
            cop_std[region] = np.maximum(occupied.std(axis=0), 1e-6)
    return TactileNormalization(
        candidate=candidate,
        force_mean=force_mean.astype(np.float32),
        force_std=force_std.astype(np.float32),
        cop_mean=cop_mean.astype(np.float32),
        cop_std=cop_std.astype(np.float32),
    )
