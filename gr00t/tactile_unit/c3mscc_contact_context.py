"""Current-Contact-conditioned shared Contact prediction for Track C3-MS-CC.

The predictor accepts only frozen shared Vision/Action tokens and the current
causal Contact context.  Target Contact tokens, native Contact latents, future
Contact state, pair identity, labels, and the Contact-private residual are not
part of its interface.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .c3dp_shared_private import TargetSlotBlock
from .continuous_vac_shared_space import VAC_SHAPE, state_dict_digest


LEGAL_SOURCES = {"AH", "VAH"}
SOURCE_COMPONENTS = {
    "AH": ("u_a", "h_current"),
    "VAH": ("u_v", "u_a", "h_current"),
}
FORBIDDEN_INPUTS = {
    "u_c", "z_c", "h_future", "r_c_priv", "pair_id", "contact_transition",
    "force_trend_class", "primitive_id", "object_id", "source_index",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ContactContextProjector(nn.Module):
    def __init__(self, tokens: int = 8, width: int = 32):
        super().__init__()
        if tokens not in {4, 8} or width != 32:
            raise ValueError("C3-MS-CC H tokenization exceeds the frozen contract")
        self.tokens = int(tokens)
        self.width = int(width)
        self.network = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, tokens * width), nn.GELU()
        )

    def forward(self, h_current: torch.Tensor) -> torch.Tensor:
        if h_current.ndim != 2 or h_current.shape[1] != 256:
            raise ValueError("h_current must have shape [B,256]")
        return self.network(h_current).view(-1, self.tokens, self.width)


class ContactContextPredictor(nn.Module):
    """Small source-structured target-slot fusion predictor."""

    def __init__(
        self,
        source: str,
        *,
        h_tokens: int = 8,
        blocks: int = 2,
        heads: int = 4,
        mlp_width: int = 64,
    ):
        super().__init__()
        if source not in LEGAL_SOURCES:
            raise ValueError(f"unsupported C3-MS-CC source {source!r}")
        if blocks not in {1, 2} or heads > 4 or mlp_width > 128:
            raise ValueError("C3-MS-CC architecture exceeds preregistered bounds")
        self.source = source
        self.h_tokens = int(h_tokens)
        self.block_count = int(blocks)
        self.heads = int(heads)
        self.mlp_width = int(mlp_width)
        self.h_projector = ContactContextProjector(h_tokens, 32)
        self.source_embedding = nn.Parameter(torch.randn(3, 32) * 0.02)
        self.target_slots = nn.Parameter(torch.randn(*VAC_SHAPE) * 0.02)
        self.blocks = nn.ModuleList(
            [TargetSlotBlock(32, heads, mlp_width) for _ in range(blocks)]
        )
        self.output_norm = nn.LayerNorm(32)

    def source_tokens(
        self,
        u_a: torch.Tensor,
        h_current: torch.Tensor,
        u_v: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if u_a.ndim != 3 or u_a.shape[1:] != VAC_SHAPE:
            raise ValueError("u_a must have shape [B,8,32]")
        if len(h_current) != len(u_a):
            raise ValueError("unaligned Action and Contact context")
        action = u_a + self.source_embedding[1].view(1, 1, 32)
        context = self.h_projector(h_current) + self.source_embedding[2].view(1, 1, 32)
        if self.source == "AH":
            if u_v is not None:
                raise ValueError("AH predictor forbids Vision input")
            return torch.cat([action, context], dim=1)
        if u_v is None or u_v.ndim != 3 or u_v.shape[1:] != VAC_SHAPE:
            raise ValueError("VAH predictor requires u_v [B,8,32]")
        if len(u_v) != len(u_a):
            raise ValueError("unaligned Vision and Action")
        vision = u_v + self.source_embedding[0].view(1, 1, 32)
        return torch.cat([vision, action, context], dim=1)

    def forward(
        self,
        u_a: torch.Tensor,
        h_current: torch.Tensor,
        u_v: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory = self.source_tokens(u_a, h_current, u_v)
        value = self.target_slots.unsqueeze(0).expand(len(memory), -1, -1)
        for block in self.blocks:
            value = block(value, memory)
        value = self.output_norm(value)
        if value.shape[1:] != VAC_SHAPE:
            raise RuntimeError("STRUCTURAL_FAIL: invalid Contact prediction shape")
        return value

    def parameter_summary(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


@dataclass(frozen=True)
class C3MSCCLossWeights:
    shared: float = 1.0
    cosine: float = 0.25
    relational: float = 0.1
    physics: float = 0.25
    delta: float = 0.1
    covariance: float = 0.05
    order: float = 0.05


def per_sample_mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.square(left - right).flatten(1).mean(1)


def relational_loss(prediction: torch.Tensor, target: torch.Tensor, maximum: int = 128) -> torch.Tensor:
    count = min(len(prediction), maximum)
    if count < 2:
        return prediction.new_zeros(())
    left = F.normalize(prediction[:count].flatten(1), dim=1, eps=1e-8)
    right = F.normalize(target[:count].detach().flatten(1), dim=1, eps=1e-8)
    mask = ~torch.eye(count, dtype=torch.bool, device=prediction.device)
    return F.mse_loss((left @ left.T)[mask], (right @ right.T)[mask])


def covariance_loss(
        prediction: torch.Tensor, target: torch.Tensor, variance_floor: float = 0.1
) -> torch.Tensor:
    left = prediction.flatten(1)
    right = target.detach().flatten(1)
    left = left - left.mean(0)
    right = right - right.mean(0)
    left_std = torch.sqrt(left.var(0, unbiased=False) + 1e-4)
    variance = F.relu(float(variance_floor) - left_std).mean()
    denominator = max(len(left) - 1, 1)
    left_cov = left.T @ left / denominator
    right_cov = right.T @ right / denominator
    covariance = F.mse_loss(left_cov, right_cov)
    return variance + covariance


def contact_prediction_loss(
    predictor: ContactContextPredictor,
    shared_space: nn.Module,
    decoder: nn.Module,
    *,
    u_a: torch.Tensor,
    h_current: torch.Tensor,
    u_c: torch.Tensor,
    dynamic: torch.Tensor,
    u_v: torch.Tensor | None,
    invalid_u_a: tuple[torch.Tensor, ...] = (),
    enhanced: bool,
    dynamic_weight: float,
    order_margin: float,
    variance_floor: float,
    weights: C3MSCCLossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute label-free shared, relational, physics, covariance and order losses."""

    target = u_c.detach()
    prediction = predictor(u_a, h_current, u_v)
    sample_weight = torch.where(
        dynamic.bool(),
        torch.full_like(dynamic, float(dynamic_weight), dtype=torch.float32),
        torch.ones_like(dynamic, dtype=torch.float32),
    )
    shared_rows = per_sample_mse(prediction, target)
    shared = torch.sum(shared_rows * sample_weight) / sample_weight.sum().clamp_min(1.0)
    cosine_rows = 1.0 - F.cosine_similarity(prediction.flatten(1), target.flatten(1), dim=1)
    cosine = torch.sum(cosine_rows * sample_weight) / sample_weight.sum().clamp_min(1.0)
    relational = relational_loss(prediction, target)
    zero = prediction.new_zeros(())
    physics = delta = covariance = order = zero
    if enhanced:
        predicted_native = shared_space.recover("contact", prediction)
        with torch.no_grad():
            oracle_native = shared_space.recover("contact", target)
            oracle_future = decoder(oracle_native, h_current)
        predicted_future = decoder(predicted_native, h_current)
        physics = F.mse_loss(predicted_future, oracle_future)
        delta = F.mse_loss(
            predicted_future - h_current, (oracle_future - h_current).detach()
        )
        covariance = covariance_loss(prediction, target, variance_floor)
        if invalid_u_a:
            dynamic_mask = dynamic.bool()
            if dynamic_mask.any():
                correct_error = shared_rows[dynamic_mask]
                rankings = []
                for invalid in invalid_u_a:
                    invalid_prediction = predictor(invalid, h_current, u_v)
                    invalid_error = per_sample_mse(invalid_prediction, target)[dynamic_mask]
                    rankings.append(F.relu(float(order_margin) + correct_error - invalid_error).mean())
                order = torch.stack(rankings).mean()
    total = (
        weights.shared * shared
        + weights.cosine * cosine
        + weights.relational * relational
        + (weights.physics * physics + weights.delta * delta
           + weights.covariance * covariance + weights.order * order if enhanced else zero)
    )
    terms = {
        "shared": shared.detach(), "cosine": cosine.detach(),
        "relational": relational.detach(), "physics": physics.detach(),
        "delta": delta.detach(), "covariance": covariance.detach(),
        "order": order.detach(), "total": total.detach(),
    }
    return total, terms


def save_checkpoint(
    path: Path, predictor: ContactContextPredictor, metadata: Mapping[str, Any]
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema": "tactile3d-unit.vac-c3mscc-predictor.v1",
            "source": predictor.source,
            "h_tokens": predictor.h_tokens,
            "blocks": predictor.block_count,
            "heads": predictor.heads,
            "mlp_width": predictor.mlp_width,
            "state_dict": {name: value.detach().cpu() for name, value in predictor.state_dict().items()},
            "state_dict_sha256": state_dict_digest(predictor),
            "metadata": dict(metadata),
        },
        temporary,
    )
    temporary.replace(path)
    return sha256_file(path)


def load_checkpoint(
    path: Path, device: str | torch.device = "cpu"
) -> tuple[ContactContextPredictor, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "tactile3d-unit.vac-c3mscc-predictor.v1":
        raise ValueError("unsupported C3-MS-CC checkpoint")
    model = ContactContextPredictor(
        str(payload["source"]), h_tokens=int(payload["h_tokens"]),
        blocks=int(payload["blocks"]), heads=int(payload["heads"]),
        mlp_width=int(payload["mlp_width"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    if state_dict_digest(model) != payload.get("state_dict_sha256"):
        raise ValueError("C3-MS-CC predictor state digest mismatch")
    return model.to(device), dict(payload.get("metadata", {}))
