"""Causal current/history visual encoders and Contact fallback models for C5."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .c3dp_shared_private import TargetSlotBlock
from .c3mscc_contact_context import covariance_loss, per_sample_mse, relational_loss
from .c4_availability_conditioning import ContactFallbackPredictor, sha256_file
from .continuous_vac_shared_space import VAC_SHAPE, state_dict_digest


class VisualSupport(str, Enum):
    CURRENT_FRAME = "CURRENT_FRAME"
    CAUSAL_HISTORY_8 = "CAUSAL_HISTORY_8"
    NONE = "NONE"


SUPPORT_OFFSETS = {
    VisualSupport.CURRENT_FRAME: (0,),
    VisualSupport.CAUSAL_HISTORY_8: (-7, -6, -5, -4, -3, -2, -1, 0),
}


@dataclass(frozen=True)
class CausalFrameSelection:
    support: VisualSupport
    episode_id: int
    anchor_t: int
    frame_indices: tuple[int, ...]

    @classmethod
    def create(cls, support: VisualSupport | str, episode_id: int, anchor_t: int, episode_length: int) -> "CausalFrameSelection":
        support = VisualSupport(support)
        if support is VisualSupport.NONE:
            raise ValueError("NONE has no visual frame selection")
        frames = tuple(anchor_t + offset for offset in SUPPORT_OFFSETS[support])
        if frames != tuple(range(anchor_t - 7, anchor_t + 1)) and support is VisualSupport.CAUSAL_HISTORY_8:
            raise RuntimeError("history must be exactly I_t-7:t")
        if frames != (anchor_t,) and support is VisualSupport.CURRENT_FRAME:
            raise RuntimeError("current support must be exactly I_t")
        if min(frames) < 0 or max(frames) >= episode_length:
            raise ValueError("causal frames cross the episode boundary")
        if max(frames) > anchor_t:
            raise RuntimeError("CAUSAL_LEAKAGE_FAIL: future visual frame")
        return cls(support, int(episode_id), int(anchor_t), frames)


class CausalVisualEncoder(nn.Module):
    """Small trainable slotwise temporal aggregator over frozen 8x32 frame features."""

    def __init__(self, support: VisualSupport | str, *, layers: int = 1, heads: int = 4, mlp_width: int = 64):
        super().__init__()
        self.support = VisualSupport(support)
        if self.support is VisualSupport.NONE:
            raise ValueError("visual encoder requires current or history support")
        if layers not in {1, 2} or heads > 4 or mlp_width > 128:
            raise ValueError("C5 temporal aggregator exceeds bounded capacity")
        self.layers, self.heads, self.mlp_width = int(layers), int(heads), int(mlp_width)
        self.expected_frames = len(SUPPORT_OFFSETS[self.support])
        self.temporal_position = nn.Parameter(torch.zeros(self.expected_frames, 32))
        if self.expected_frames == 1:
            self.temporal = nn.Identity()
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=32, nhead=heads, dim_feedforward=mlp_width,
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=layers)
        self.target_slots = nn.Parameter(torch.randn(*VAC_SHAPE) * 0.02)
        self.resampler = TargetSlotBlock(32, heads, mlp_width)
        self.output_norm = nn.LayerNorm(32)

    def forward(self, frozen_frame_features: torch.Tensor) -> torch.Tensor:
        if frozen_frame_features.ndim != 4 or frozen_frame_features.shape[1:] != (self.expected_frames, 8, 32):
            raise ValueError(f"frozen frame features must be [B,{self.expected_frames},8,32]")
        if frozen_frame_features.requires_grad:
            raise ValueError("frozen visual features must be stop-gradient")
        batch = len(frozen_frame_features)
        temporal = frozen_frame_features.permute(0, 2, 1, 3).reshape(batch * 8, self.expected_frames, 32)
        temporal = self.temporal(temporal + self.temporal_position[None])
        memory = temporal[:, -1].reshape(batch, 8, 32)
        targets = self.target_slots.unsqueeze(0).expand(batch, -1, -1)
        result = self.output_norm(self.resampler(targets, memory))
        if result.shape != (batch, 8, 32):
            raise RuntimeError("STRUCTURAL_FAIL: causal visual output shape")
        return result


class DirectCausalContactPredictor(nn.Module):
    """Eight Contact target slots attending only causal visual and planned Action."""

    def __init__(self, *, blocks: int = 2, heads: int = 4, mlp_width: int = 64, visual_head: bool = False):
        super().__init__()
        if blocks not in {1, 2} or heads > 4 or mlp_width > 128:
            raise ValueError("C5 direct fallback exceeds bounded capacity")
        self.block_count, self.heads, self.mlp_width = int(blocks), int(heads), int(mlp_width)
        self.visual_head_enabled = bool(visual_head)
        self.source_embedding = nn.Parameter(torch.randn(2, 32) * 0.02)
        self.target_slots = nn.Parameter(torch.randn(*VAC_SHAPE) * 0.02)
        self.blocks = nn.ModuleList([TargetSlotBlock(32, heads, mlp_width) for _ in range(blocks)])
        self.output_norm = nn.LayerNorm(32)
        self.visual_head = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 32)) if visual_head else None

    def forward(self, c_v: torch.Tensor, u_a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        for value, name in ((c_v, "c_v"), (u_a, "u_a")):
            if value.ndim != 3 or value.shape[1:] != VAC_SHAPE:
                raise ValueError(f"{name} must have shape [B,8,32]")
        if len(c_v) != len(u_a):
            raise ValueError("unaligned causal Vision and planned Action")
        memory = torch.cat((c_v + self.source_embedding[0], u_a + self.source_embedding[1]), dim=1)
        value = self.target_slots.unsqueeze(0).expand(len(memory), -1, -1)
        for block in self.blocks:
            value = block(value, memory)
        prediction = self.output_norm(value)
        visual_prediction = self.visual_head(c_v) if self.visual_head is not None else None
        return prediction, visual_prediction


class CausalVisionSubstituter(nn.Module):
    """Predict offline u_v from causal tokens; true u_v is never accepted as input."""

    def __init__(self, *, blocks: int = 1, heads: int = 4, mlp_width: int = 64):
        super().__init__()
        self.target_slots = nn.Parameter(torch.randn(*VAC_SHAPE) * 0.02)
        self.blocks = nn.ModuleList([TargetSlotBlock(32, heads, mlp_width) for _ in range(blocks)])
        self.norm = nn.LayerNorm(32)

    def forward(self, c_v: torch.Tensor) -> torch.Tensor:
        if c_v.ndim != 3 or c_v.shape[1:] != VAC_SHAPE:
            raise ValueError("c_v must have shape [B,8,32]")
        value = self.target_slots.unsqueeze(0).expand(len(c_v), -1, -1)
        for block in self.blocks:
            value = block(value, c_v)
        return self.norm(value)


class ModularCausalContactPredictor(nn.Module):
    """Causal Vision substitution followed by a frozen C4 offline F_VA upper-bound model."""

    def __init__(self, substituter: CausalVisionSubstituter, frozen_f_va: ContactFallbackPredictor):
        super().__init__()
        if frozen_f_va.source != "VA":
            raise ValueError("modular fallback requires the accepted VA predictor")
        self.substituter = substituter
        self.frozen_f_va = frozen_f_va.eval().requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen_f_va.eval()
        return self

    def forward(self, c_v: torch.Tensor, u_a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u_hat_v = self.substituter(c_v)
        return self.frozen_f_va(u_a, u_hat_v), u_hat_v


@dataclass(frozen=True)
class C5LossWeights:
    shared: float = 1.0
    cosine: float = 0.25
    relational: float = 0.1
    physics: float = 0.25
    covariance: float = 0.05
    visual: float = 0.1
    order: float = 0.05


def causal_fallback_loss(
    predictor: nn.Module,
    shared_space: nn.Module,
    decoder: nn.Module,
    *,
    c_v: torch.Tensor,
    u_a: torch.Tensor,
    u_c: torch.Tensor,
    dynamic: torch.Tensor,
    teacher_h_current: torch.Tensor | None,
    teacher_u_v: torch.Tensor | None,
    invalid_u_a: tuple[torch.Tensor, ...],
    enhanced: bool,
    dynamic_weight: float,
    order_margin: float,
    variance_floor: float,
    weights: C5LossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Label-free C5 objective; teacher H/u_v are target-side only."""

    target = u_c.detach()
    prediction, visual_prediction = predictor(c_v, u_a)
    row_weight = torch.where(dynamic.bool(), torch.full_like(dynamic, dynamic_weight, dtype=torch.float32), torch.ones_like(dynamic, dtype=torch.float32))
    rows = per_sample_mse(prediction, target)
    shared = torch.sum(rows * row_weight) / row_weight.sum().clamp_min(1.0)
    cosine_rows = 1.0 - F.cosine_similarity(prediction.flatten(1), target.flatten(1), dim=1)
    cosine = torch.sum(cosine_rows * row_weight) / row_weight.sum().clamp_min(1.0)
    relational = relational_loss(prediction, target)
    zero = prediction.new_zeros(())
    physics = covariance = visual = order = zero
    if visual_prediction is not None:
        if teacher_u_v is None:
            raise ValueError("visual distillation requires teacher_u_v target")
        visual = F.mse_loss(visual_prediction, teacher_u_v.detach())
    if enhanced:
        if teacher_h_current is None or teacher_h_current.requires_grad:
            raise ValueError("physics requires stop-gradient teacher_h_current")
        recovered = shared_space.recover("contact", prediction)
        with torch.no_grad():
            oracle_future = decoder(shared_space.recover("contact", target), teacher_h_current)
        physics = F.mse_loss(decoder(recovered, teacher_h_current.detach()), oracle_future)
        covariance = covariance_loss(prediction, target, variance_floor)
        mask = dynamic.bool()
        if invalid_u_a and mask.any():
            terms = []
            for invalid in invalid_u_a:
                invalid_prediction = predictor(c_v, invalid)[0]
                invalid_error = per_sample_mse(invalid_prediction, target)[mask]
                terms.append(F.relu(order_margin + rows[mask] - invalid_error).mean())
            order = torch.stack(terms).mean()
    total = (weights.shared * shared + weights.cosine * cosine + weights.relational * relational +
             weights.physics * physics + weights.covariance * covariance + weights.visual * visual + weights.order * order)
    return total, {name: value.detach() for name, value in {
        "shared": shared, "cosine": cosine, "relational": relational, "physics": physics,
        "covariance": covariance, "visual": visual, "order": order, "total": total,
    }.items()}


def save_causal_checkpoint(path: Path, visual: CausalVisualEncoder, predictor: nn.Module, metadata: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    family = "direct" if isinstance(predictor, DirectCausalContactPredictor) else "modular"
    trainable_predictor = predictor.substituter if family == "modular" else predictor
    payload = {
        "schema": "tactile3d-unit.vac-c5-causal-fallback.v1",
        "family": family,
        "support": visual.support.value,
        "visual": {"layers": visual.layers, "heads": visual.heads, "mlp_width": visual.mlp_width, "state_dict": {k: v.detach().cpu() for k, v in visual.state_dict().items()}, "state_dict_sha256": state_dict_digest(visual)},
        "predictor_state_dict": {k: v.detach().cpu() for k, v in trainable_predictor.state_dict().items()},
        "predictor_state_dict_sha256": state_dict_digest(trainable_predictor),
        "metadata": dict(metadata),
    }
    if family == "direct":
        payload["predictor"] = {"blocks": predictor.block_count, "heads": predictor.heads, "mlp_width": predictor.mlp_width, "visual_head": predictor.visual_head_enabled}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return sha256_file(path)


def load_causal_checkpoint(
    path: Path,
    device: str | torch.device = "cpu",
    *,
    frozen_f_va: ContactFallbackPredictor | None = None,
) -> tuple[CausalVisualEncoder, nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "tactile3d-unit.vac-c5-causal-fallback.v1":
        raise ValueError("unsupported C5 causal fallback checkpoint")
    visual_spec = payload["visual"]
    visual = CausalVisualEncoder(
        payload["support"], layers=int(visual_spec["layers"]),
        heads=int(visual_spec["heads"]), mlp_width=int(visual_spec["mlp_width"]),
    )
    visual.load_state_dict(visual_spec["state_dict"], strict=True)
    if state_dict_digest(visual) != visual_spec["state_dict_sha256"]:
        raise ValueError("C5 causal visual state digest mismatch")
    if payload["family"] == "direct":
        spec = payload["predictor"]
        predictor: nn.Module = DirectCausalContactPredictor(
            blocks=int(spec["blocks"]), heads=int(spec["heads"]),
            mlp_width=int(spec["mlp_width"]), visual_head=bool(spec["visual_head"]),
        )
        predictor.load_state_dict(payload["predictor_state_dict"], strict=True)
        digest_model = predictor
    elif payload["family"] == "modular":
        if frozen_f_va is None:
            raise ValueError("loading modular C5 fallback requires frozen_f_va")
        substituter = CausalVisionSubstituter()
        substituter.load_state_dict(payload["predictor_state_dict"], strict=True)
        predictor = ModularCausalContactPredictor(substituter, frozen_f_va)
        digest_model = substituter
    else:
        raise ValueError("unsupported C5 causal fallback family")
    if state_dict_digest(digest_model) != payload["predictor_state_dict_sha256"]:
        raise ValueError("C5 causal predictor state digest mismatch")
    return visual.to(device), predictor.to(device), dict(payload.get("metadata", {}))
