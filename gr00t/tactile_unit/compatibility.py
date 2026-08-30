"""Pure helpers for the frozen shared-RQ compatibility audit."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Iterable

import numpy as np
import torch


def parameter_digest(module: torch.nn.Module, prefixes: Iterable[str] | None = None) -> str:
    """Hash named state tensors, including names, shapes, dtypes, and values."""

    allowed = tuple(prefixes) if prefixes is not None else None
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if allowed is not None and not name.startswith(allowed):
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def deterministic_contact_subset(
    episode_id: np.ndarray,
    anchor_frame: np.ndarray,
    dynamic: np.ndarray,
    transition_class: np.ndarray,
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    """Select a deterministic, proportionally stratified contact subset."""

    episode_id = np.asarray(episode_id)
    anchor_frame = np.asarray(anchor_frame)
    dynamic = np.asarray(dynamic, dtype=bool)
    transition_class = np.asarray(transition_class, dtype=np.int64)
    size = len(episode_id)
    if not all(len(value) == size for value in (anchor_frame, dynamic, transition_class)):
        raise ValueError("contact subset metadata must have aligned rows")
    if count < 1 or count > size:
        raise ValueError("subset count must be in [1, number of rows]")

    strata: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (is_dynamic, transition) in enumerate(zip(dynamic, transition_class)):
        strata[(int(is_dynamic), int(transition))].append(index)

    exact = {key: count * len(rows) / size for key, rows in strata.items()}
    allocation = {key: min(len(strata[key]), int(math.floor(value))) for key, value in exact.items()}
    remaining = count - sum(allocation.values())
    ranked = sorted(
        strata,
        key=lambda key: (-(exact[key] - math.floor(exact[key])), key),
    )
    while remaining:
        progressed = False
        for key in ranked:
            if allocation[key] < len(strata[key]):
                allocation[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise AssertionError("unable to allocate deterministic subset")

    selected: list[int] = []
    for key in sorted(strata):
        rows = strata[key]

        def row_hash(index: int) -> bytes:
            identity = (
                f"{seed}:{int(episode_id[index])}:{int(anchor_frame[index])}:{index}"
            )
            return hashlib.sha256(identity.encode("ascii")).digest()

        selected.extend(sorted(rows, key=lambda index: (row_hash(index), index))[: allocation[key]])
    result = np.asarray(sorted(selected), dtype=np.int64)
    if len(result) != count or len(np.unique(result)) != count:
        raise AssertionError("deterministic subset is not unique and complete")
    return result


@torch.inference_mode()
def quantize_with_stage_diagnostics(
    rq: torch.nn.Module, value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float]]]:
    """Run a frozen RVQ directly and expose residual statistics by stage."""

    if value.ndim != 3:
        raise ValueError("shared RQ input must have shape [B,Q,D]")
    residual = value
    quantized = torch.zeros_like(value)
    indices = []
    rows: list[dict[str, float]] = []
    for stage, layer in enumerate(rq.layers):
        before = torch.linalg.vector_norm(residual, dim=-1)
        stage_value, stage_indices, _ = layer(residual)
        quantized = quantized + stage_value
        residual = residual - stage_value
        after = torch.linalg.vector_norm(residual, dim=-1)
        rows.append(
            {
                "stage": int(stage),
                "residual_norm_before_mean": float(before.mean().item()),
                "residual_norm_after_mean": float(after.mean().item()),
                "residual_energy_after": float(residual.square().mean().item()),
            }
        )
        indices.append(stage_indices)
    return quantized, torch.stack(indices, dim=-1), rows


def quantization_metrics(value: np.ndarray, quantized: np.ndarray) -> dict[str, float]:
    value = np.asarray(value, dtype=np.float64)
    quantized = np.asarray(quantized, dtype=np.float64)
    if value.shape != quantized.shape or value.ndim != 3:
        raise ValueError("quantization metrics require equal [N,Q,D] arrays")
    error = np.square(value - quantized)
    absolute = float(error.mean())
    energy = float(np.square(value).mean())
    value_norm = np.linalg.norm(value, axis=-1)
    quantized_norm = np.linalg.norm(quantized, axis=-1)
    cosine = np.sum(value * quantized, axis=-1) / np.maximum(
        value_norm * quantized_norm, 1e-12
    )
    per_sample_error = error.mean(axis=(1, 2))
    per_sample_energy = np.square(value).mean(axis=(1, 2))
    per_sample_relative = per_sample_error / np.maximum(per_sample_energy, 1e-12)
    return {
        "absolute_mse": absolute,
        "input_energy": energy,
        "relative_distortion": float(absolute / max(energy, 1e-12)),
        "pre_post_cosine_mean": float(cosine.mean()),
        "pre_post_cosine_std": float(cosine.std()),
        "per_sample_relative_p05": float(np.quantile(per_sample_relative, 0.05)),
        "per_sample_relative_median": float(np.median(per_sample_relative)),
        "per_sample_relative_p95": float(np.quantile(per_sample_relative, 0.95)),
    }


def code_frequency(codes: np.ndarray, codebook_size: int) -> np.ndarray:
    values = np.asarray(codes, dtype=np.int64).ravel()
    if values.size == 0 or values.min() < 0 or values.max() >= codebook_size:
        raise ValueError("VQ indices are empty or outside the codebook")
    counts = np.bincount(values, minlength=codebook_size).astype(np.float64)
    return counts / counts.sum()


def codebook_usage(codes: np.ndarray, codebook_size: int) -> dict[str, float | int]:
    frequency = code_frequency(codes, codebook_size)
    positive = frequency[frequency > 0]
    entropy = float(-np.sum(positive * np.log(positive)))
    return {
        "active_codes": int(np.count_nonzero(frequency)),
        "codebook_size": int(codebook_size),
        "active_ratio": float(np.count_nonzero(frequency) / codebook_size),
        "entropy": entropy,
        "perplexity": float(np.exp(entropy)),
        "top1_frequency": float(np.max(frequency)),
        "top5_frequency": float(np.sort(frequency)[-5:].sum()),
    }


def active_set_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left_set = set(np.flatnonzero(np.asarray(left) > 0).tolist())
    right_set = set(np.flatnonzero(np.asarray(right) > 0).tolist())
    union = left_set | right_set
    return float(len(left_set & right_set) / len(union)) if union else 1.0


def jensen_shannon_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or np.any(left < 0) or np.any(right < 0):
        raise ValueError("JS inputs must be aligned nonnegative distributions")
    left = left / max(left.sum(), 1e-12)
    right = right / max(right.sum(), 1e-12)
    midpoint = 0.5 * (left + right)

    def kl(value: np.ndarray) -> float:
        mask = value > 0
        return float(np.sum(value[mask] * np.log(value[mask] / midpoint[mask])))

    return float(0.5 * (kl(left) + kl(right)))


def effective_rank(values: np.ndarray) -> tuple[float, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("effective rank requires a [N,D] matrix with N >= 2")
    eigenvalues = np.linalg.eigvalsh(np.cov(values, rowvar=False)).clip(min=0)[::-1]
    probability = eigenvalues / max(float(eigenvalues.sum()), 1e-12)
    positive = probability > 0
    rank = float(np.exp(-np.sum(probability[positive] * np.log(probability[positive]))))
    return rank, eigenvalues
