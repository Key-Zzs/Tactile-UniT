"""Train-only robust statistics for the 60-D wrench input."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobustFeatureStats:
    q01: np.ndarray
    q99: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    count: int
    quantile_sample_count: int
    eps: float = 1e-6

    def __post_init__(self) -> None:
        arrays = (self.q01, self.q99, self.mean, self.std)
        if any(np.asarray(x).shape != (60,) for x in arrays):
            raise ValueError("all feature-stat arrays must have shape (60,)")
        if np.any(np.asarray(self.q99) <= np.asarray(self.q01)):
            raise ValueError("q99 must be greater than q01 for every feature")
        if self.count <= 0 or self.quantile_sample_count <= 0:
            raise ValueError("statistics require positive sample counts")

    @property
    def scale(self) -> np.ndarray:
        return np.maximum(self.q99 - self.q01, self.eps)

    def normalize(self, values: np.ndarray, clip: float = 1.5) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        normalized = 2.0 * (values - self.q01) / self.scale - 1.0
        return np.clip(normalized, -clip, clip).astype(np.float32, copy=False)

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return ((values + 1.0) * 0.5 * self.scale + self.q01).astype(np.float32, copy=False)

    def to_dict(self) -> dict:
        return {
            "method": "per-feature train-only q01/q99 mapped to [-1,1]",
            "q01": self.q01.tolist(),
            "q99": self.q99.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "count": self.count,
            "quantile_sample_count": self.quantile_sample_count,
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "RobustFeatureStats":
        return cls(
            q01=np.asarray(value["q01"], dtype=np.float32),
            q99=np.asarray(value["q99"], dtype=np.float32),
            mean=np.asarray(value["mean"], dtype=np.float64),
            std=np.asarray(value["std"], dtype=np.float64),
            count=int(value["count"]),
            quantile_sample_count=int(value["quantile_sample_count"]),
            eps=float(value.get("eps", 1e-6)),
        )


class RunningFeatureStats:
    """Numerically stable streaming moments plus deterministic quantile samples."""

    def __init__(self, dim: int = 60) -> None:
        self.dim = dim
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)
        self._quantile_samples: list[np.ndarray] = []

    def update(self, values: np.ndarray, quantile_sample: np.ndarray | None = None) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.dim:
            raise ValueError(f"expected [N,{self.dim}], got {values.shape}")
        if len(values):
            batch_count = len(values)
            batch_mean = values.mean(axis=0)
            batch_m2 = ((values - batch_mean) ** 2).sum(axis=0)
            delta = batch_mean - self.mean
            total = self.count + batch_count
            self.mean += delta * batch_count / total
            self.m2 += batch_m2 + delta**2 * self.count * batch_count / total
            self.count = total
        if quantile_sample is not None and len(quantile_sample):
            sample = np.asarray(quantile_sample, dtype=np.float32)
            if sample.ndim != 2 or sample.shape[1] != self.dim:
                raise ValueError(f"bad quantile sample shape: {sample.shape}")
            self._quantile_samples.append(sample)

    def finalize(self) -> RobustFeatureStats:
        if self.count < 2 or not self._quantile_samples:
            raise ValueError("insufficient data for robust feature statistics")
        sample = np.concatenate(self._quantile_samples, axis=0)
        q01, q99 = np.quantile(sample, [0.01, 0.99], axis=0)
        # Constant/near-constant channels remain well-defined without inventing data.
        too_narrow = (q99 - q01) < 1e-6
        if np.any(too_narrow):
            q01[too_narrow] = self.mean[too_narrow] - 0.5
            q99[too_narrow] = self.mean[too_narrow] + 0.5
        return RobustFeatureStats(
            q01=q01.astype(np.float32),
            q99=q99.astype(np.float32),
            mean=self.mean.copy(),
            std=np.sqrt(self.m2 / max(self.count - 1, 1)),
            count=self.count,
            quantile_sample_count=len(sample),
        )
