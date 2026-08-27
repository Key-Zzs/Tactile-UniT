"""Independent continuous Vision/Action/Contact shared-space mappings.

Each public encoder consumes exactly one native ``[B,8,32]`` representation.
Cross-modal tensors are accepted only by the training objective, never by an
encoder or retrieval-candidate construction path.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


MODALITIES = ("vision", "action", "contact")
SHORT_MODALITY = {"V": "vision", "A": "action", "C": "contact"}
VAC_SHAPE = (8, 32)


class IndependentEncodingError(ValueError):
    """Raised when a shared representation violates the one-modality API."""


def _validate_native(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 3 or tuple(value.shape[1:]) != VAC_SHAPE:
        raise IndependentEncodingError(f"{name} must be a tensor with shape [B,8,32]")
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise IndependentEncodingError(f"{name} must be finite floating point")


class IdentityProjector(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        _validate_native(value, "native transition")
        return value


class ResidualTokenProjector(nn.Module):
    """Small shared-over-query token projector with identity initialization."""

    def __init__(self, *, hidden_dim: int | None = None):
        super().__init__()
        self.norm = nn.LayerNorm(32)
        if hidden_dim is None:
            self.mapping = nn.Linear(32, 32)
            nn.init.zeros_(self.mapping.weight)
            nn.init.zeros_(self.mapping.bias)
        else:
            self.mapping = nn.Sequential(
                nn.Linear(32, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 32)
            )
            nn.init.zeros_(self.mapping[-1].weight)
            nn.init.zeros_(self.mapping[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        _validate_native(value, "native transition")
        return value + self.mapping(self.norm(value))


class IndependentSlotResampler(nn.Module):
    """Attend eight shared physical slots to one modality and no counterpart."""

    def __init__(self, heads: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.memory_norm = nn.LayerNorm(32)
        self.query_norm = nn.LayerNorm(32)
        self.attention = nn.MultiheadAttention(32, heads, dropout=0.0, batch_first=True)
        self.output_norm = nn.LayerNorm(32)
        self.output = nn.Sequential(
            nn.Linear(32, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 32)
        )

    def forward(self, value: torch.Tensor, shared_slots: torch.Tensor) -> torch.Tensor:
        _validate_native(value, "native transition")
        if shared_slots.shape != VAC_SHAPE:
            raise IndependentEncodingError("shared slots must have shape [8,32]")
        query = shared_slots.unsqueeze(0).expand(len(value), -1, -1)
        attended, _ = self.attention(
            self.query_norm(query), self.memory_norm(value), self.memory_norm(value), need_weights=False
        )
        mixed = query + attended
        return mixed + self.output(self.output_norm(mixed))


class RecoveryHead(nn.Module):
    def __init__(self, hidden_dim: int = 128, *, query_mixing: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(32),
            nn.Linear(32, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 32),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.query_mixing = nn.Linear(8 * 32, 8 * 32) if query_mixing else None
        if self.query_mixing is not None:
            nn.init.zeros_(self.query_mixing.weight)
            nn.init.zeros_(self.query_mixing.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        _validate_native(value, "shared transition")
        result = value + self.net(value)
        if self.query_mixing is not None:
            result = result + self.query_mixing(value.flatten(1)).view(-1, 8, 32)
        return result


class ContinuousVACSharedSpace(nn.Module):
    """Three independently callable mappings and training-only recovery heads."""

    def __init__(self, candidate: str):
        super().__init__()
        if candidate not in {"C0", "C1-linear", "C1-mlp", "C2-slot"}:
            raise ValueError(f"unknown C2 candidate {candidate!r}")
        self.candidate = candidate
        if candidate == "C0":
            factory = lambda: IdentityProjector()
            self.shared_slots = None
        elif candidate == "C1-linear":
            factory = lambda: ResidualTokenProjector()
            self.shared_slots = None
        elif candidate == "C1-mlp":
            factory = lambda: ResidualTokenProjector(hidden_dim=128)
            self.shared_slots = None
        else:
            factory = lambda: IndependentSlotResampler(heads=4, hidden_dim=64)
            self.shared_slots = nn.Parameter(torch.randn(*VAC_SHAPE) * 0.02)
        self.projectors = nn.ModuleDict({name: factory() for name in MODALITIES})
        recovery_factory = (
            (lambda: IdentityProjector())
            if candidate == "C0"
            else (lambda: RecoveryHead(query_mixing=candidate == "C2-slot"))
        )
        self.recovery = nn.ModuleDict({name: recovery_factory() for name in MODALITIES})

    def encode(self, modality: str, native: torch.Tensor) -> torch.Tensor:
        """Encode one modality; there is intentionally no counterpart argument."""

        modality = SHORT_MODALITY.get(modality, modality)
        if modality not in MODALITIES:
            raise IndependentEncodingError(f"unknown modality {modality!r}")
        _validate_native(native, f"z_{modality[0]}")
        projector = self.projectors[modality]
        if isinstance(projector, IndependentSlotResampler):
            assert self.shared_slots is not None
            result = projector(native, self.shared_slots)
        else:
            result = projector(native)
        _validate_native(result, f"u_{modality[0]}")
        return result

    def recover(self, modality: str, shared: torch.Tensor) -> torch.Tensor:
        modality = SHORT_MODALITY.get(modality, modality)
        if modality not in MODALITIES:
            raise IndependentEncodingError(f"unknown modality {modality!r}")
        return self.recovery[modality](shared)

    def forward(self, modality: str, native: torch.Tensor) -> torch.Tensor:
        return self.encode(modality, native)

    def parameter_summary(self) -> dict[str, int]:
        total = sum(value.numel() for value in self.parameters())
        trainable = sum(value.numel() for value in self.parameters() if value.requires_grad)
        shared = 0 if self.shared_slots is None else self.shared_slots.numel()
        return {"total": total, "trainable": trainable, "shared_slots": shared}


def flatten_normalize(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value.flatten(1), dim=-1, eps=1e-8)


def different_episode_info_nce(
    query: torch.Tensor,
    candidate: torch.Tensor,
    episode_id: torch.Tensor,
    *,
    temperature: float,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Symmetric-ready InfoNCE with only different-episode negatives."""

    _validate_native(query, "query")
    _validate_native(candidate, "candidate")
    if query.shape != candidate.shape or episode_id.shape != (len(query),):
        raise ValueError("InfoNCE batch geometry mismatch")
    logits = flatten_normalize(query) @ flatten_normalize(candidate).T / float(temperature)
    same_episode = episode_id[:, None] == episode_id[None, :]
    diagonal = torch.eye(len(query), dtype=torch.bool, device=query.device)
    allowed = ~same_episode | diagonal
    logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
    loss = -F.log_softmax(logits, dim=1).diagonal()
    if sample_weight is not None:
        weight = sample_weight.to(loss, non_blocking=True)
        return torch.sum(loss * weight) / torch.clamp(weight.sum(), min=1.0)
    return loss.mean()


def relational_preservation(native: torch.Tensor, shared: torch.Tensor, maximum: int = 128) -> torch.Tensor:
    count = min(len(native), maximum)
    if count < 2:
        return native.new_zeros(())
    native_flat = flatten_normalize(native[:count]).detach()
    shared_flat = flatten_normalize(shared[:count])
    mask = ~torch.eye(count, dtype=torch.bool, device=native.device)
    return F.mse_loss((shared_flat @ shared_flat.T)[mask], (native_flat @ native_flat.T)[mask])


def variance_floor(shared: torch.Tensor, floor: float = 0.1) -> torch.Tensor:
    flattened = shared.flatten(1)
    standard_deviation = torch.sqrt(flattened.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(float(floor) - standard_deviation).mean()


@dataclass(frozen=True)
class VACLossWeights:
    alignment: float = 1.0
    native: float = 5.0
    relational: float = 0.25
    variance: float = 0.05


def continuous_vac_loss(
    model: ContinuousVACSharedSpace,
    native: Mapping[str, torch.Tensor],
    episode_id: torch.Tensor,
    dynamic: torch.Tensor,
    *,
    temperature: float,
    dynamic_weight: float,
    weights: VACLossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if set(native) != set(MODALITIES):
        raise ValueError("loss requires exactly vision, action, and contact")
    shared = {name: model.encode(name, native[name]) for name in MODALITIES}
    sample_weight = torch.where(
        dynamic.bool(),
        torch.full_like(dynamic, float(dynamic_weight), dtype=torch.float32),
        torch.ones_like(dynamic, dtype=torch.float32),
    )
    pair_losses = []
    for left, right in (("vision", "action"), ("vision", "contact"), ("action", "contact")):
        pair_losses.append(
            different_episode_info_nce(
                shared[left], shared[right], episode_id,
                temperature=temperature, sample_weight=sample_weight,
            )
        )
        pair_losses.append(
            different_episode_info_nce(
                shared[right], shared[left], episode_id,
                temperature=temperature, sample_weight=sample_weight,
            )
        )
    alignment = torch.stack(pair_losses).mean()
    native_loss = torch.stack(
        [F.mse_loss(model.recover(name, shared[name]), native[name].detach()) for name in MODALITIES]
    ).mean()
    relational = torch.stack(
        [relational_preservation(native[name], shared[name]) for name in MODALITIES]
    ).mean()
    variance = torch.stack([variance_floor(shared[name]) for name in MODALITIES]).mean()
    total = (
        weights.alignment * alignment
        + weights.native * native_loss
        + weights.relational * relational
        + weights.variance * variance
    )
    return total, {
        "total": total.detach(),
        "alignment": alignment.detach(),
        "native": native_loss.detach(),
        "relational": relational.detach(),
        "variance": variance.detach(),
    }


def numpy_flatten_normalize(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(len(value), -1)
    return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)


def different_episode_permutation(episode_id: np.ndarray, seed: int = 0) -> np.ndarray:
    episode = np.asarray(episode_id, dtype=np.int64)
    if len(np.unique(episode)) < 2:
        raise ValueError("different-episode control requires at least two episodes")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(episode))
    result = np.empty(len(episode), dtype=np.int64)
    pools: dict[int, np.ndarray] = {
        int(current): order[episode[order] != current] for current in np.unique(episode)
    }
    offsets: dict[int, int] = {key: 0 for key in pools}
    for index, current in enumerate(episode):
        key = int(current)
        pool = pools[key]
        result[index] = pool[offsets[key] % len(pool)]
        offsets[key] += 1
    return result


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(len(left), -1)
    y = np.asarray(right, dtype=np.float64).reshape(len(right), -1)
    x -= x.mean(axis=0, keepdims=True)
    y -= y.mean(axis=0, keepdims=True)
    numerator = np.square(np.linalg.norm(x.T @ y, ord="fro"))
    denominator = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    return float(numerator / max(float(denominator), 1e-12))


def retrieval_metrics(left: np.ndarray, right: np.ndarray, chunk: int = 512) -> dict[str, Any]:
    query = numpy_flatten_normalize(left).astype(np.float32)
    candidate = numpy_flatten_normalize(right).astype(np.float32)
    if len(query) != len(candidate):
        raise ValueError("paired retrieval requires equal query/candidate counts")
    ranks = np.empty(len(query), dtype=np.int64)
    for start in range(0, len(query), chunk):
        stop = min(start + chunk, len(query))
        similarity = query[start:stop] @ candidate.T
        positive = similarity[np.arange(stop - start), np.arange(start, stop)]
        ranks[start:stop] = 1 + np.sum(similarity > positive[:, None], axis=1)
    count = len(ranks)
    return {
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "chance": {
            "recall_at_1": 1.0 / count,
            "recall_at_5": min(5, count) / count,
            "recall_at_10": min(10, count) / count,
        },
    }


def bootstrap_mean_ci(value: np.ndarray, *, samples: int, seed: int) -> list[float]:
    data = np.asarray(value, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 256):
        stop = min(start + 256, samples)
        indices = rng.integers(0, len(data), size=(stop - start, len(data)))
        means[start:stop] = data[indices].mean(axis=1)
    return [float(item) for item in np.quantile(means, [0.025, 0.975])]


def pairwise_alignment_metrics(
    left: np.ndarray,
    right: np.ndarray,
    episode_id: np.ndarray,
    *,
    bootstrap_samples: int = 5000,
    seed: int = 0,
    retrieval_chunk: int = 512,
) -> dict[str, Any]:
    left_norm = numpy_flatten_normalize(left)
    right_norm = numpy_flatten_normalize(right)
    negative = different_episode_permutation(episode_id, seed)
    paired = np.sum(left_norm * right_norm, axis=1)
    shuffled = np.sum(left_norm * right_norm[negative], axis=1)
    margin = paired - shuffled
    return {
        "paired_cosine": float(paired.mean()),
        "different_episode_shuffled_cosine": float(shuffled.mean()),
        "paired_minus_shuffled_margin": float(margin.mean()),
        "margin_bootstrap_ci95": bootstrap_mean_ci(margin, samples=bootstrap_samples, seed=seed + 1),
        "linear_cka": linear_cka(left, right),
        "retrieval": {
            "forward": retrieval_metrics(left, right, retrieval_chunk),
            "reverse": retrieval_metrics(right, left, retrieval_chunk),
        },
    }


def effective_rank(value: np.ndarray) -> float:
    flat = np.asarray(value, dtype=np.float64).reshape(len(value), -1)
    flat -= flat.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(flat, full_matrices=False, compute_uv=False)
    probability = np.square(singular)
    probability /= max(float(probability.sum()), 1e-12)
    entropy = -np.sum(probability * np.log(np.maximum(probability, 1e-12)))
    return float(np.exp(entropy))


def geometry_diagnostics(value: np.ndarray) -> dict[str, Any]:
    tokens = np.asarray(value, dtype=np.float64)
    flat = tokens.reshape(len(tokens), -1)
    token_norm = np.linalg.norm(tokens, axis=-1)
    normalized = tokens / np.maximum(token_norm[..., None], 1e-12)
    query_cosine = np.einsum("bqd,bkd->bqk", normalized, normalized)
    off_diagonal = ~np.eye(8, dtype=bool)
    sample = flat[: min(4096, len(flat))]
    sample_norm = sample / np.maximum(np.linalg.norm(sample, axis=1, keepdims=True), 1e-12)
    return {
        "per_dimension_variance": {
            "minimum": float(flat.var(axis=0).min()),
            "mean": float(flat.var(axis=0).mean()),
            "near_zero_fraction": float(np.mean(flat.var(axis=0) < 1e-6)),
        },
        "effective_rank": effective_rank(value),
        "query_diversity": {
            "mean_cosine_distance": float((1.0 - query_cosine[:, off_diagonal]).mean()),
            "collapsed_pair_fraction": float(np.mean(query_cosine[:, off_diagonal] > 0.999)),
        },
        "off_diagonal_cosine": float(query_cosine[:, off_diagonal].mean()),
        "token_norm": {
            "minimum": float(token_norm.min()),
            "mean": float(token_norm.mean()),
            "maximum": float(token_norm.max()),
        },
        "pairwise_distance": {
            "mean_cosine_distance": float((1.0 - sample_norm @ sample_norm.T)[~np.eye(len(sample), dtype=bool)].mean())
            if len(sample) > 1 else 0.0
        },
    }


def state_dict_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def save_checkpoint(
    path: Path,
    model: ContinuousVACSharedSpace,
    metadata: Mapping[str, Any],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema": "tactile3d-unit.vac-c2-checkpoint.v1",
            "candidate": model.candidate,
            "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "state_dict_sha256": state_dict_digest(model),
            "metadata": dict(metadata),
        },
        temporary,
    )
    temporary.replace(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(
    path: Path, map_location: str | torch.device = "cpu"
) -> tuple[ContinuousVACSharedSpace, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema") != "tactile3d-unit.vac-c2-checkpoint.v1":
        raise ValueError("unsupported C2 checkpoint schema")
    model = ContinuousVACSharedSpace(str(payload["candidate"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    if state_dict_digest(model) != payload.get("state_dict_sha256"):
        raise ValueError("C2 checkpoint state digest mismatch")
    return model, dict(payload.get("metadata", {}))
