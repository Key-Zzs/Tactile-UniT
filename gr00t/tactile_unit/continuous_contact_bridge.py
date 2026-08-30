"""Small continuous Vision/Contact bridge candidates for Track C0.

These modules never instantiate a codebook and never update the frozen Vision,
S1, or S2 components. They consume native float32 ``[B,8,32]`` transition
representations. The causal gate depends only on current visual features and
``h_t^c``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_tokens(value: torch.Tensor, name: str) -> None:
    if value.ndim != 3 or tuple(value.shape[1:]) != (8, 32):
        raise ValueError(f"{name} must have shape [B,8,32]")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")


class SharedQueryProjector(nn.Module):
    """LayerNorm+Linear or a small shared residual MLP over query tokens."""

    def __init__(self, kind: Literal["linear", "residual_mlp"] = "residual_mlp") -> None:
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.net = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 32))
        elif kind == "residual_mlp":
            self.net = nn.Sequential(
                nn.LayerNorm(32),
                nn.Linear(32, 64),
                nn.GELU(),
                nn.Linear(64, 32),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        else:
            raise ValueError(f"unknown projector kind {kind!r}")

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _check_tokens(values, "values")
        mapped = self.net(values)
        return values + mapped if self.kind == "residual_mlp" else mapped


class TwoTowerContinuousProjector(nn.Module):
    """B1: independent small Vision and Contact projectors."""

    def __init__(self, kind: Literal["linear", "residual_mlp"] = "residual_mlp") -> None:
        super().__init__()
        self.vision = SharedQueryProjector(kind)
        self.contact = SharedQueryProjector(kind)

    def forward(
        self, vision: torch.Tensor, contact: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _check_tokens(vision, "vision")
        _check_tokens(contact, "contact")
        if len(vision) != len(contact):
            raise ValueError("Vision and Contact batch sizes differ")
        return self.vision(vision), self.contact(contact)


class _CrossAttentionDirection(nn.Module):
    def __init__(self, heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(32)
        self.context_norm = nn.LayerNorm(32)
        self.attention = nn.MultiheadAttention(32, heads, dropout=dropout, batch_first=True)
        self.output_norm = nn.LayerNorm(32)
        self.ffn = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        normalized_context = self.context_norm(context)
        attended, _ = self.attention(
            self.query_norm(query), normalized_context, normalized_context, need_weights=False
        )
        value = query + attended
        return value + self.ffn(self.output_norm(value))


class TokenSetCrossAttentionBridge(nn.Module):
    """B2: bidirectional token-set attention without positional correspondence."""

    def __init__(self, heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.vision_from_contact = _CrossAttentionDirection(heads, dropout)
        self.contact_from_vision = _CrossAttentionDirection(heads, dropout)

    def forward(
        self, vision: torch.Tensor, contact: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _check_tokens(vision, "vision")
        _check_tokens(contact, "contact")
        if len(vision) != len(contact):
            raise ValueError("Vision and Contact batch sizes differ")
        return (
            self.vision_from_contact(vision, contact),
            self.contact_from_vision(contact, vision),
        )


class CausalContactGate(nn.Module):
    """B3 gate using only current Vision and current Contact state."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(32 + 256),
            nn.Linear(32 + 256, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        current_vision: torch.Tensor,
        current_contact: torch.Tensor | None,
        contact_available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _check_tokens(current_vision, "current_vision")
        batch = len(current_vision)
        if current_contact is None:
            return current_vision.new_zeros((batch, 1, 1))
        if current_contact.ndim != 2 or tuple(current_contact.shape) != (batch, 256):
            raise ValueError("current_contact must have shape [B,256]")
        pooled = current_vision.mean(dim=1)
        gate = torch.sigmoid(self.network(torch.cat([pooled, current_contact], dim=-1)))
        if contact_available is not None:
            mask = contact_available.to(device=gate.device, dtype=gate.dtype)
            if mask.ndim == 1:
                mask = mask[:, None]
            if tuple(mask.shape) != (batch, 1):
                raise ValueError("contact_available must have shape [B] or [B,1]")
            gate = gate * mask
        return gate[:, :, None]

    def residual_fuse(
        self,
        vision: torch.Tensor,
        contact_residual: torch.Tensor | None,
        current_contact: torch.Tensor | None,
        contact_available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse contact with exact Vision fallback when Contact is unavailable."""

        _check_tokens(vision, "vision")
        gate = self(vision, current_contact, contact_available)
        if contact_residual is None:
            return vision
        _check_tokens(contact_residual, "contact_residual")
        if contact_residual.shape != vision.shape:
            raise ValueError("contact_residual shape differs from Vision")
        return vision + gate * contact_residual


@dataclass(frozen=True)
class BridgeLosses:
    total: torch.Tensor
    contrastive: torch.Tensor
    prediction: torch.Tensor
    relational: torch.Tensor


def pooled_tokens(values: torch.Tensor) -> torch.Tensor:
    _check_tokens(values, "values")
    return F.normalize(values.flatten(1), dim=-1)


def paired_info_nce(
    vision: torch.Tensor, contact: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """Symmetric in-batch paired InfoNCE over whole token sets."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    left, right = pooled_tokens(vision), pooled_tokens(contact)
    logits = left @ right.T / temperature
    labels = torch.arange(len(left), device=left.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def relational_preservation(projected: torch.Tensor, native: torch.Tensor) -> torch.Tensor:
    """Preserve within-modality pairwise geometry without forcing distributions equal."""

    projected_similarity = pooled_tokens(projected) @ pooled_tokens(projected).T
    native_similarity = pooled_tokens(native) @ pooled_tokens(native).T
    return F.mse_loss(projected_similarity, native_similarity)


def bridge_objective(
    projected_vision: torch.Tensor,
    projected_contact: torch.Tensor,
    native_vision: torch.Tensor,
    native_contact: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    temperature: float = 0.07,
    prediction_weight: float = 1.0,
    relational_weight: float = 0.1,
) -> BridgeLosses:
    """C0 paired, bidirectional-prediction, and relational objective."""

    contrastive = paired_info_nce(projected_vision, projected_contact, temperature)
    per_row = 0.5 * (
        (projected_vision - native_contact).square().mean(dim=(1, 2))
        + (projected_contact - native_vision).square().mean(dim=(1, 2))
    )
    if weights is not None:
        if weights.ndim != 1 or len(weights) != len(per_row):
            raise ValueError("weights must have shape [B]")
        prediction = (per_row * weights).sum() / weights.sum().clamp_min(1e-12)
    else:
        prediction = per_row.mean()
    relational = 0.5 * (
        relational_preservation(projected_vision, native_vision)
        + relational_preservation(projected_contact, native_contact)
    )
    total = contrastive + prediction_weight * prediction + relational_weight * relational
    return BridgeLosses(total, contrastive, prediction, relational)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
