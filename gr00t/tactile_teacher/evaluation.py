"""Temporal corruptions and metrics for the canonical S1 benchmark."""

from __future__ import annotations

import numpy as np
import torch


def temporal_variant(
    history: torch.Tensor, variant: str, *, seed: int = 42
) -> torch.Tensor:
    """Apply a deterministic temporal ablation without changing feature values."""

    if variant == "full_history":
        return history
    if variant == "last_frame":
        return history[:, -1:].expand_as(history)
    if variant == "reversed_history":
        return history.flip(1)
    if variant == "shuffled_history":
        generator = torch.Generator(device=history.device).manual_seed(seed)
        order = torch.rand(history.shape[:2], device=history.device, generator=generator)
        indices = order.argsort(dim=1).unsqueeze(-1).expand_as(history)
        return history.gather(1, indices)
    raise ValueError(f"unknown temporal variant: {variant}")


def corrupt_history(
    history: torch.Tensor,
    corruption: str,
    severity: float,
    *,
    seed: int = 42,
) -> torch.Tensor:
    """Apply deterministic sensor or resampling corruption in normalized units."""

    if severity < 0:
        raise ValueError("severity must be non-negative")
    if severity == 0:
        return history
    generator = torch.Generator(device=history.device).manual_seed(seed)
    if corruption == "gaussian_noise":
        noise = torch.randn(
            history.shape, device=history.device, dtype=history.dtype, generator=generator
        )
        return history + severity * noise
    if corruption == "bias":
        bias = torch.randn(
            (len(history), 1, history.shape[2]),
            device=history.device,
            dtype=history.dtype,
            generator=generator,
        )
        return history + severity * bias
    if corruption == "frame_dropout":
        mask = torch.rand(
            history.shape[:2], device=history.device, generator=generator
        ) < severity
        mask[:, 0] = False
        result = history.clone()
        for index in range(1, history.shape[1]):
            result[:, index] = torch.where(
                mask[:, index, None], result[:, index - 1], result[:, index]
            )
        return result
    if corruption == "timestamp_jitter":
        batch, steps, features = history.shape
        offsets = severity * torch.randn(
            (batch, steps), device=history.device, dtype=history.dtype, generator=generator
        )
        query = torch.arange(steps, device=history.device, dtype=history.dtype)[None] + offsets
        query = query.clamp(0, steps - 1)
        left = query.floor().long()
        right = (left + 1).clamp(max=steps - 1)
        weight = (query - left).unsqueeze(-1)
        gather_shape = (batch, steps, features)
        left_value = history.gather(1, left.unsqueeze(-1).expand(gather_shape))
        right_value = history.gather(1, right.unsqueeze(-1).expand(gather_shape))
        return left_value + weight * (right_value - left_value)
    raise ValueError(f"unknown corruption: {corruption}")


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - target
    squared_error = np.square(error).sum()
    total_variance = np.square(target - target.mean()).sum()
    return {
        "mse": float(np.square(error).mean()),
        "mae": float(np.abs(error).mean()),
        "r2": float(1.0 - squared_error / max(total_variance, 1e-12)),
    }


def classification_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    labels = np.unique(target)
    f1_values = []
    for label in labels:
        true_positive = np.sum((target == label) & (prediction == label))
        false_positive = np.sum((target != label) & (prediction == label))
        false_negative = np.sum((target == label) & (prediction != label))
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return {
        "accuracy": float(np.mean(target == prediction)),
        "macro_f1": float(np.mean(f1_values)),
    }


def collapse_diagnostics(latent: np.ndarray, *, seed: int = 42) -> dict:
    latent = np.asarray(latent, dtype=np.float64)
    variance = latent.var(axis=0)
    covariance = np.cov(latent, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance).clip(min=0)[::-1]
    probabilities = eigenvalues / max(eigenvalues.sum(), 1e-12)
    positive = probabilities > 0
    effective_rank = np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
    norms = np.linalg.norm(latent, axis=1)
    rng = np.random.default_rng(seed)
    count = min(8192, len(latent))
    left = rng.integers(0, len(latent), size=count)
    right = rng.integers(0, len(latent), size=count)
    distances = np.linalg.norm(latent[left] - latent[right], axis=1)
    return {
        "per_dimension_variance": {
            "min": float(variance.min()),
            "median": float(np.median(variance)),
            "max": float(variance.max()),
            "near_zero_fraction": float(np.mean(variance < 1e-8)),
        },
        "effective_rank": float(effective_rank),
        "top_eigenvalue_fraction": float(probabilities[0]),
        "eigenvalues": eigenvalues.astype(np.float32).tolist(),
        "norm": {
            "mean": float(norms.mean()),
            "std": float(norms.std()),
            "min": float(norms.min()),
            "max": float(norms.max()),
        },
        "pairwise_distance": {
            "mean": float(distances.mean()),
            "std": float(distances.std()),
            "p01": float(np.quantile(distances, 0.01)),
            "p99": float(np.quantile(distances, 0.99)),
        },
    }
