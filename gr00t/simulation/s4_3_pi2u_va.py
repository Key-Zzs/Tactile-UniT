"""Contact-free continuous Vision/Action bridge for S4.3-PI2U BVA.

The public model API accepts one native transition tensor at a time.  This
module intentionally has no Contact modality, Contact parameters, masks, or
targets; paired Vision/Action tensors meet only inside the training loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


VA_SHAPE = (8, 32)
VA_MODALITIES = ("vision", "action")


def _check(value: torch.Tensor, name: str) -> None:
    if value.ndim != 3 or tuple(value.shape[1:]) != VA_SHAPE:
        raise ValueError(f"{name} must have shape [B,8,32]")
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite floating point")


class VAProjector(nn.Module):
    """The selected S4.2 B3 slot projector, restricted to one modality."""

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(32, 4, dropout=0.0, batch_first=True)
        self.mapping = nn.Sequential(nn.Linear(32, width), nn.GELU(), nn.Linear(width, 32))
        self.norm = nn.LayerNorm(32)

    def forward(self, native: torch.Tensor) -> torch.Tensor:
        _check(native, "native VA transition")
        attended, _ = self.attention(native, native, native, need_weights=False)
        value = native + attended
        return self.norm(value + self.mapping(value))


class VAOnlyBridge(nn.Module):
    """Two independent projectors and training-only native recovery heads."""

    modalities = VA_MODALITIES

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.projectors = nn.ModuleDict({name: VAProjector(width) for name in self.modalities})
        self.recovery = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(32, width), nn.GELU(), nn.Linear(width, 32))
                for name in self.modalities
            }
        )

    def encode(self, modality: str, native: torch.Tensor) -> torch.Tensor:
        if modality not in self.modalities:
            raise ValueError(f"VA-only bridge rejects modality {modality!r}")
        result = self.projectors[modality](native)
        _check(result, f"u_{modality[0]}")
        return result

    def recover(self, modality: str, shared: torch.Tensor) -> torch.Tensor:
        if modality not in self.modalities:
            raise ValueError(f"VA-only bridge rejects modality {modality!r}")
        _check(shared, "shared VA transition")
        return shared + self.recovery[modality](shared)


def _normalized(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value.flatten(1), dim=-1, eps=1e-8)


def different_episode_info_nce(
    query: torch.Tensor,
    candidate: torch.Tensor,
    episode_id: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """InfoNCE whose negatives are selected only by episode identity."""

    _check(query, "query")
    _check(candidate, "candidate")
    logits = _normalized(query) @ _normalized(candidate).T / float(temperature)
    same_episode = episode_id[:, None] == episode_id[None, :]
    diagonal = torch.eye(len(query), dtype=torch.bool, device=query.device)
    logits = logits.masked_fill(same_episode & ~diagonal, torch.finfo(logits.dtype).min)
    return -F.log_softmax(logits, dim=1).diagonal().mean()


def relational_preservation(native: torch.Tensor, shared: torch.Tensor, maximum: int = 128) -> torch.Tensor:
    count = min(len(native), maximum)
    if count < 2:
        return native.new_zeros(())
    source = _normalized(native[:count]).detach()
    target = _normalized(shared[:count])
    mask = ~torch.eye(count, dtype=torch.bool, device=native.device)
    return F.mse_loss((target @ target.T)[mask], (source @ source.T)[mask])


def variance_floor(shared: torch.Tensor, floor: float = 0.1) -> torch.Tensor:
    standard_deviation = torch.sqrt(shared.flatten(1).var(dim=0, unbiased=False) + 1e-4)
    return F.relu(float(floor) - standard_deviation).mean()


@dataclass(frozen=True)
class VALossWeights:
    alignment: float = 1.0
    native: float = 5.0
    relational: float = 0.25
    variance: float = 0.05


def va_bridge_loss(
    model: VAOnlyBridge,
    vision: torch.Tensor,
    action: torch.Tensor,
    episode_id: torch.Tensor,
    *,
    temperature: float = 0.10,
    weights: VALossWeights = VALossWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Symmetric V/A alignment plus the frozen S4.2 anti-collapse terms."""

    shared_v = model.encode("vision", vision)
    shared_a = model.encode("action", action)
    alignment = 0.5 * (
        different_episode_info_nce(shared_v, shared_a, episode_id, temperature=temperature)
        + different_episode_info_nce(shared_a, shared_v, episode_id, temperature=temperature)
    )
    native = 0.5 * (
        F.mse_loss(model.recover("vision", shared_v), vision.detach())
        + F.mse_loss(model.recover("action", shared_a), action.detach())
    )
    relational = 0.5 * (
        relational_preservation(vision, shared_v) + relational_preservation(action, shared_a)
    )
    variance = 0.5 * (variance_floor(shared_v) + variance_floor(shared_a))
    total = (
        weights.alignment * alignment
        + weights.native * native
        + weights.relational * relational
        + weights.variance * variance
    )
    return total, {
        "total": total.detach(),
        "alignment": alignment.detach(),
        "native": native.detach(),
        "relational": relational.detach(),
        "variance": variance.detach(),
    }
