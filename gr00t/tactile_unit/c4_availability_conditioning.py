"""Missing-modality Contact fallback and explicit availability routing for Track C4.

The learned fallback accepts shared Action tokens, and optionally shared Vision
tokens.  Current/future Contact, native Contact latents, labels, pair identity,
and the Contact-private residual are intentionally absent from its interface.
Availability is supplied as explicit metadata; tensor contents are never used to
infer whether a modality is present.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .c3dp_shared_private import TargetSlotBlock
from .c3mscc_contact_context import covariance_loss, per_sample_mse, relational_loss
from .continuous_vac_shared_space import VAC_SHAPE, state_dict_digest


LEGAL_FALLBACK_SOURCES = {"A", "VA"}
FALLBACK_SOURCE_COMPONENTS = {"A": ("u_a",), "VA": ("u_v", "u_a")}
FORBIDDEN_FALLBACK_INPUTS = {
    "h_current", "h_future", "u_c", "z_c", "r_c_priv", "pair_id",
    "contact_transition", "force_trend_class", "primitive_id", "object_id",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class AvailabilityMode(str, Enum):
    FULL_AH = "FULL_AH"
    FALLBACK_VA = "FALLBACK_VA"
    FALLBACK_A = "FALLBACK_A"
    ABSTAIN_NO_ACTION = "ABSTAIN_NO_ACTION"


@dataclass(frozen=True)
class ModalityAvailability:
    vision_available: bool
    action_available: bool
    contact_context_available: bool

    def __post_init__(self) -> None:
        for name in (
            "vision_available", "action_available", "contact_context_available"
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an explicit bool")


@dataclass(frozen=True)
class ContactPredictionResult:
    prediction_available: bool
    mode: AvailabilityMode
    u_hat_c: torch.Tensor | None
    uncertainty: float | torch.Tensor | None
    vision_available: bool
    action_available: bool
    contact_context_available: bool
    rank_warning: bool


def route_availability(availability: ModalityAvailability) -> AvailabilityMode:
    """Apply the exhaustive deterministic router truth table."""

    if not availability.action_available:
        return AvailabilityMode.ABSTAIN_NO_ACTION
    if availability.contact_context_available:
        return AvailabilityMode.FULL_AH
    if availability.vision_available:
        return AvailabilityMode.FALLBACK_VA
    return AvailabilityMode.FALLBACK_A


class ContactFallbackPredictor(nn.Module):
    """Source-typed target-slot predictor with strict A/VA source isolation."""

    def __init__(
        self,
        source: str,
        *,
        blocks: int = 2,
        heads: int = 4,
        mlp_width: int = 64,
    ):
        super().__init__()
        if source not in LEGAL_FALLBACK_SOURCES:
            raise ValueError(f"unsupported C4 fallback source {source!r}")
        if blocks not in {1, 2} or heads > 4 or mlp_width > 128:
            raise ValueError("C4 fallback architecture exceeds preregistered bounds")
        self.source = source
        self.block_count = int(blocks)
        self.heads = int(heads)
        self.mlp_width = int(mlp_width)
        self.source_embedding = nn.Parameter(torch.randn(2, 32) * 0.02)
        self.target_slots = nn.Parameter(torch.randn(*VAC_SHAPE) * 0.02)
        self.blocks = nn.ModuleList(
            [TargetSlotBlock(32, heads, mlp_width) for _ in range(blocks)]
        )
        self.output_norm = nn.LayerNorm(32)

    @staticmethod
    def _validate_tokens(value: torch.Tensor, name: str) -> None:
        if value.ndim != 3 or value.shape[1:] != VAC_SHAPE:
            raise ValueError(f"{name} must have shape [B,8,32]")

    def source_tokens(
        self, u_a: torch.Tensor, u_v: torch.Tensor | None = None
    ) -> torch.Tensor:
        self._validate_tokens(u_a, "u_a")
        action = u_a + self.source_embedding[1].view(1, 1, 32)
        if self.source == "A":
            if u_v is not None:
                raise ValueError("A fallback forbids Vision input")
            return action
        if u_v is None:
            raise ValueError("VA fallback requires explicit u_v")
        self._validate_tokens(u_v, "u_v")
        if len(u_v) != len(u_a):
            raise ValueError("unaligned Vision and Action")
        vision = u_v + self.source_embedding[0].view(1, 1, 32)
        return torch.cat((vision, action), dim=1)

    def forward(
        self, u_a: torch.Tensor, u_v: torch.Tensor | None = None
    ) -> torch.Tensor:
        memory = self.source_tokens(u_a, u_v)
        value = self.target_slots.unsqueeze(0).expand(len(memory), -1, -1)
        for block in self.blocks:
            value = block(value, memory)
        value = self.output_norm(value)
        if value.shape[1:] != VAC_SHAPE:
            raise RuntimeError("STRUCTURAL_FAIL: invalid C4 Contact fallback shape")
        return value

    def parameter_summary(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters()
            if parameter.requires_grad
        )
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


@dataclass(frozen=True)
class C4FallbackLossWeights:
    shared: float = 1.0
    cosine: float = 0.25
    relational: float = 0.1
    physics: float = 0.25
    covariance: float = 0.05
    order: float = 0.05


def fallback_prediction_loss(
    predictor: ContactFallbackPredictor,
    shared_space: nn.Module,
    decoder: nn.Module,
    *,
    u_a: torch.Tensor,
    u_c: torch.Tensor,
    dynamic: torch.Tensor,
    u_v: torch.Tensor | None = None,
    teacher_h_current: torch.Tensor | None = None,
    invalid_u_a: tuple[torch.Tensor, ...] = (),
    enhanced: bool,
    dynamic_weight: float,
    order_margin: float,
    variance_floor: float,
    weights: C4FallbackLossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Label-free loss; ``teacher_h_current`` never enters the predictor."""

    target = u_c.detach()
    prediction = predictor(u_a, u_v)
    sample_weight = torch.where(
        dynamic.bool(),
        torch.full_like(dynamic, float(dynamic_weight), dtype=torch.float32),
        torch.ones_like(dynamic, dtype=torch.float32),
    )
    shared_rows = per_sample_mse(prediction, target)
    shared = torch.sum(shared_rows * sample_weight) / sample_weight.sum().clamp_min(1.0)
    cosine_rows = 1.0 - F.cosine_similarity(
        prediction.flatten(1), target.flatten(1), dim=1
    )
    cosine = torch.sum(cosine_rows * sample_weight) / sample_weight.sum().clamp_min(1.0)
    relational = relational_loss(prediction, target)
    zero = prediction.new_zeros(())
    physics = covariance = order = zero
    if enhanced:
        if teacher_h_current is None:
            raise ValueError("enhanced loss requires teacher-side h_current")
        if teacher_h_current.requires_grad:
            raise ValueError("teacher-side h_current must be stop-gradient")
        predicted_native = shared_space.recover("contact", prediction)
        with torch.no_grad():
            oracle_native = shared_space.recover("contact", target)
            oracle_future = decoder(oracle_native, teacher_h_current)
        predicted_future = decoder(predicted_native, teacher_h_current.detach())
        physics = F.mse_loss(predicted_future, oracle_future)
        covariance = covariance_loss(prediction, target, variance_floor)
        dynamic_mask = dynamic.bool()
        if invalid_u_a and dynamic_mask.any():
            rankings = []
            for invalid in invalid_u_a:
                invalid_prediction = predictor(invalid, u_v)
                invalid_error = per_sample_mse(invalid_prediction, target)[dynamic_mask]
                rankings.append(
                    F.relu(
                        float(order_margin)
                        + shared_rows[dynamic_mask]
                        - invalid_error
                    ).mean()
                )
            order = torch.stack(rankings).mean()
    total = (
        weights.shared * shared
        + weights.cosine * cosine
        + weights.relational * relational
        + (weights.physics * physics + weights.covariance * covariance
           + weights.order * order if enhanced else zero)
    )
    terms = {
        "shared": shared.detach(), "cosine": cosine.detach(),
        "relational": relational.detach(), "physics": physics.detach(),
        "covariance": covariance.detach(), "order": order.detach(),
        "total": total.detach(),
    }
    return total, terms


def save_fallback_checkpoint(
    path: Path, predictor: ContactFallbackPredictor, metadata: Mapping[str, Any]
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema": "tactile3d-unit.vac-c4-fallback.v1",
            "source": predictor.source,
            "blocks": predictor.block_count,
            "heads": predictor.heads,
            "mlp_width": predictor.mlp_width,
            "state_dict": {
                name: value.detach().cpu()
                for name, value in predictor.state_dict().items()
            },
            "state_dict_sha256": state_dict_digest(predictor),
            "metadata": dict(metadata),
        },
        temporary,
    )
    temporary.replace(path)
    return sha256_file(path)


def load_fallback_checkpoint(
    path: Path, device: str | torch.device = "cpu"
) -> tuple[ContactFallbackPredictor, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "tactile3d-unit.vac-c4-fallback.v1":
        raise ValueError("unsupported C4 fallback checkpoint")
    model = ContactFallbackPredictor(
        str(payload["source"]), blocks=int(payload["blocks"]),
        heads=int(payload["heads"]), mlp_width=int(payload["mlp_width"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    if state_dict_digest(model) != payload.get("state_dict_sha256"):
        raise ValueError("C4 fallback state digest mismatch")
    return model.to(device), dict(payload.get("metadata", {}))


class AvailabilityRouter:
    """Typed inference adapter around frozen full and fallback predictors."""

    def __init__(
        self,
        full_predictor: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        va_predictor: ContactFallbackPredictor,
        a_predictor: ContactFallbackPredictor,
        uncertainty: Callable[[AvailabilityMode, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        *,
        rank_warning: bool = True,
    ):
        if va_predictor.source != "VA" or a_predictor.source != "A":
            raise ValueError("router requires source-isolated VA and A fallbacks")
        self.full_predictor = full_predictor
        self.va_predictor = va_predictor
        self.a_predictor = a_predictor
        self.uncertainty_estimator = uncertainty
        self.rank_warning = bool(rank_warning)

    def predict(
        self,
        availability: ModalityAvailability,
        *,
        u_a: torch.Tensor | None = None,
        h_current: torch.Tensor | None = None,
        u_v: torch.Tensor | None = None,
    ) -> ContactPredictionResult:
        mode = route_availability(availability)
        common = dict(
            vision_available=availability.vision_available,
            action_available=availability.action_available,
            contact_context_available=availability.contact_context_available,
            rank_warning=self.rank_warning,
        )
        if mode is AvailabilityMode.ABSTAIN_NO_ACTION:
            return ContactPredictionResult(False, mode, None, None, **common)
        if u_a is None:
            raise ValueError("available Action requires u_a")
        if mode is AvailabilityMode.FULL_AH:
            if h_current is None:
                raise ValueError("available Contact context requires h_current")
            prediction = self.full_predictor(u_a, h_current)
            source = torch.cat((u_a, h_current[:, None, :32]), dim=1)
        elif mode is AvailabilityMode.FALLBACK_VA:
            if u_v is None:
                raise ValueError("available Vision requires u_v")
            prediction = self.va_predictor(u_a, u_v)
            source = torch.cat((u_v, u_a), dim=1)
        else:
            prediction = self.a_predictor(u_a)
            source = u_a
        uncertainty = None
        if self.uncertainty_estimator is not None:
            uncertainty = self.uncertainty_estimator(mode, prediction, source)
        return ContactPredictionResult(True, mode, prediction, uncertainty, **common)
