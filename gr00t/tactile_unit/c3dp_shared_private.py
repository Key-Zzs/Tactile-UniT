"""Frozen dual-path Contact and shared cross-modal prediction for Track C3-DP.

The canonical predictor accepts exactly one source shared representation plus
source/target modality identities.  It never accepts target samples, pair IDs,
native target latents, or the Contact-private residual.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .continuous_vac_shared_space import (
    VAC_SHAPE,
    ContinuousVACSharedSpace,
    _validate_native,
    geometry_diagnostics,
    state_dict_digest,
)

ACCEPTED_C2R_CHECKPOINT_SHA256 = "21dccb8fc7fbe6de2598c18e718bd65f226e220e44352ab3d43246e7f9abdf89"
MODALITY_TO_ID = {"vision": 0, "action": 1, "contact": 2, "V": 0, "A": 1, "C": 2}
ID_TO_MODALITY = {0: "vision", 1: "action", 2: "contact"}
ORDERED_DIRECTIONS = (
    ("vision", "action"),
    ("action", "vision"),
    ("vision", "contact"),
    ("contact", "vision"),
    ("action", "contact"),
    ("contact", "action"),
)
SHORT_DIRECTION = {
    ("vision", "action"): "V->A",
    ("action", "vision"): "A->V",
    ("vision", "contact"): "V->C",
    ("contact", "vision"): "C->V",
    ("action", "contact"): "A->C",
    ("contact", "action"): "C->A",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_c2r_checkpoint(path: Path) -> str:
    if not Path(path).is_file() or sha256_file(path) != ACCEPTED_C2R_CHECKPOINT_SHA256:
        raise RuntimeError("C3DP_SHARED_SPACE_CHECKPOINT_INVALID")
    return ACCEPTED_C2R_CHECKPOINT_SHA256


def freeze_shared_space(model: ContinuousVACSharedSpace) -> dict[str, Any]:
    model.eval().requires_grad_(False)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if trainable:
        raise RuntimeError("STRUCTURAL_FAIL: C2-R shared parameters remain trainable")
    return {
        "state_dict_sha256": state_dict_digest(model),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_count": 0,
        "training": bool(model.training),
    }


@dataclass(frozen=True)
class ContactDualPath:
    shared: torch.Tensor
    shared_native: torch.Tensor
    private_residual: torch.Tensor


def decompose_contact(
    shared_space: ContinuousVACSharedSpace, native_contact: torch.Tensor
) -> ContactDualPath:
    """Compute ``z_c = R_c(P_c(z_c)) + r_c_priv`` by definition."""

    _validate_native(native_contact, "z_c")
    shared = shared_space.encode("contact", native_contact)
    shared_native = shared_space.recover("contact", shared)
    private_residual = native_contact - shared_native
    for name, value in (
        ("u_c", shared),
        ("z_c_shared", shared_native),
        ("r_c_priv", private_residual),
    ):
        _validate_native(value, name)
    if (
        shared.data_ptr() == private_residual.data_ptr()
        or shared_native.data_ptr() == private_residual.data_ptr()
    ):
        raise RuntimeError("STRUCTURAL_FAIL: shared/private storage alias")
    return ContactDualPath(shared, shared_native, private_residual)


class TargetSlotBlock(nn.Module):
    def __init__(self, width: int = 32, heads: int = 4, mlp_width: int = 64):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Sequential(
            nn.Linear(width, mlp_width), nn.GELU(), nn.Linear(mlp_width, width)
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        value = query + attended
        return value + self.output(self.output_norm(value))


class SharedCrossModalPredictor(nn.Module):
    """One source/target-conditioned predictor routing all six ordered pairs."""

    def __init__(
        self,
        candidate: str,
        *,
        hidden_dim: int = 128,
        attention_layers: int = 2,
        heads: int = 4,
    ):
        super().__init__()
        if candidate not in {"P0", "P1", "P2"}:
            raise ValueError(f"unknown C3-DP candidate {candidate!r}")
        if hidden_dim > 128 or attention_layers not in {1, 2} or heads > 4:
            raise ValueError("C3-DP candidate exceeds bounded architecture")
        self.candidate = candidate
        self.hidden_dim = int(hidden_dim)
        self.attention_layers = int(attention_layers)
        self.heads = int(heads)
        self.source_embedding = nn.Embedding(3, 32)
        self.target_embedding = nn.Embedding(3, 32)
        nn.init.normal_(self.source_embedding.weight, std=0.02)
        nn.init.normal_(self.target_embedding.weight, std=0.02)
        if candidate == "P0":
            self.norm = nn.LayerNorm(32)
            self.mapping = nn.Linear(32, 32)
            nn.init.zeros_(self.mapping.weight)
            nn.init.zeros_(self.mapping.bias)
        elif candidate == "P1":
            self.norm = nn.LayerNorm(32)
            self.mapping = nn.Sequential(
                nn.Linear(32, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 32)
            )
            nn.init.zeros_(self.mapping[-1].weight)
            nn.init.zeros_(self.mapping[-1].bias)
        else:
            self.target_slots = nn.Parameter(torch.randn(*VAC_SHAPE) * 0.02)
            self.blocks = nn.ModuleList(
                [TargetSlotBlock(32, heads, min(hidden_dim, 64)) for _ in range(attention_layers)]
            )

    @staticmethod
    def _modality_id(value: str | int) -> int:
        result = MODALITY_TO_ID.get(value, value)
        if not isinstance(result, int) or result not in ID_TO_MODALITY:
            raise ValueError(f"unknown modality identity {value!r}")
        return result

    def forward(
        self,
        source_shared_tokens: torch.Tensor,
        source_modality_id: str | int,
        target_modality_id: str | int,
    ) -> torch.Tensor:
        _validate_native(source_shared_tokens, "source_shared_tokens")
        source_id = self._modality_id(source_modality_id)
        target_id = self._modality_id(target_modality_id)
        if source_id == target_id:
            raise ValueError("canonical C3-DP prediction requires target != source")
        source_condition = self.source_embedding.weight[source_id].view(1, 1, 32)
        target_condition = self.target_embedding.weight[target_id].view(1, 1, 32)
        if self.candidate in {"P0", "P1"}:
            conditioned = self.norm(source_shared_tokens) + source_condition + target_condition
            result = source_shared_tokens + self.mapping(conditioned)
        else:
            memory = source_shared_tokens + source_condition
            result = self.target_slots.unsqueeze(0).expand(len(memory), -1, -1) + target_condition
            for block in self.blocks:
                result = block(result, memory)
        _validate_native(result, "predicted_target_shared_tokens")
        return result

    def parameter_summary(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


@dataclass(frozen=True)
class C3DPLossWeights:
    shared: float = 1.0
    shared_native: float = 1.0
    relational: float = 0.1
    variance: float = 0.02
    cosine: float = 0.25


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.sum(value * weight) / torch.clamp(weight.sum(), min=1.0)


def prediction_relational_loss(
    prediction: torch.Tensor, target: torch.Tensor, maximum: int = 128
) -> torch.Tensor:
    count = min(len(prediction), maximum)
    if count < 2:
        return prediction.new_zeros(())
    predicted = F.normalize(prediction[:count].flatten(1), dim=-1, eps=1e-8)
    oracle = F.normalize(target[:count].detach().flatten(1), dim=-1, eps=1e-8)
    diagonal = ~torch.eye(count, dtype=torch.bool, device=prediction.device)
    return F.mse_loss((predicted @ predicted.T)[diagonal], (oracle @ oracle.T)[diagonal])


def prediction_variance_floor(prediction: torch.Tensor, floor: float = 0.1) -> torch.Tensor:
    deviation = torch.sqrt(prediction.flatten(1).var(0, unbiased=False) + 1e-4)
    return F.relu(float(floor) - deviation).mean()


def cross_modal_prediction_loss(
    predictor: SharedCrossModalPredictor,
    shared_space: ContinuousVACSharedSpace,
    shared: Mapping[str, torch.Tensor],
    dynamic: torch.Tensor,
    *,
    dynamic_weight: float,
    weights: C3DPLossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if set(shared) != {"vision", "action", "contact"}:
        raise ValueError("C3-DP loss requires exactly V/A/C shared tensors")
    if dynamic.ndim != 1 or any(len(value) != len(dynamic) for value in shared.values()):
        raise ValueError("unaligned C3-DP batch")
    sample_weight = torch.where(
        dynamic.bool(),
        torch.full_like(dynamic, float(dynamic_weight), dtype=torch.float32),
        torch.ones_like(dynamic, dtype=torch.float32),
    )
    terms: dict[str, list[torch.Tensor]] = {
        "shared": [],
        "shared_native": [],
        "relational": [],
        "variance": [],
        "cosine": [],
    }
    per_direction: dict[str, torch.Tensor] = {}
    for source, target in ORDERED_DIRECTIONS:
        prediction = predictor(shared[source], source, target)
        oracle = shared[target].detach()
        per_sample = torch.square(prediction - oracle).flatten(1).mean(1)
        shared_loss = _weighted_mean(per_sample, sample_weight)
        cosine_loss = _weighted_mean(
            1.0 - F.cosine_similarity(prediction.flatten(1), oracle.flatten(1), dim=1),
            sample_weight,
        )
        predicted_native = shared_space.recover(target, prediction)
        with torch.no_grad():
            oracle_native = shared_space.recover(target, oracle)
        native_loss = _weighted_mean(
            torch.square(predicted_native - oracle_native).flatten(1).mean(1), sample_weight
        )
        relational = prediction_relational_loss(prediction, oracle)
        variance = prediction_variance_floor(prediction)
        terms["shared"].append(shared_loss)
        terms["shared_native"].append(native_loss)
        terms["relational"].append(relational)
        terms["variance"].append(variance)
        terms["cosine"].append(cosine_loss)
        per_direction[SHORT_DIRECTION[(source, target)]] = shared_loss.detach()
    averaged = {name: torch.stack(value).mean() for name, value in terms.items()}
    total = (
        weights.shared * averaged["shared"]
        + weights.shared_native * averaged["shared_native"]
        + weights.relational * averaged["relational"]
        + weights.variance * averaged["variance"]
        + weights.cosine * averaged["cosine"]
    )
    detached = {name: value.detach() for name, value in averaged.items()}
    detached.update({f"direction/{name}": value for name, value in per_direction.items()})
    detached["total"] = total.detach()
    return total, detached


def predictor_state_digest(model: nn.Module) -> str:
    return state_dict_digest(model)


def save_predictor_checkpoint(
    path: Path,
    predictor: SharedCrossModalPredictor,
    metadata: Mapping[str, Any],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema": "tactile3d-unit.vac-c3dp-predictor.v1",
            "candidate": predictor.candidate,
            "hidden_dim": predictor.hidden_dim,
            "attention_layers": predictor.attention_layers,
            "heads": predictor.heads,
            "state_dict": {
                name: value.detach().cpu() for name, value in predictor.state_dict().items()
            },
            "state_dict_sha256": predictor_state_digest(predictor),
            "frozen_c2r_checkpoint_sha256": ACCEPTED_C2R_CHECKPOINT_SHA256,
            "metadata": dict(metadata),
        },
        temporary,
    )
    temporary.replace(path)
    return sha256_file(path)


def load_predictor_checkpoint(
    path: Path, map_location: str | torch.device = "cpu"
) -> tuple[SharedCrossModalPredictor, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema") != "tactile3d-unit.vac-c3dp-predictor.v1":
        raise ValueError("unsupported C3-DP predictor checkpoint")
    if payload.get("frozen_c2r_checkpoint_sha256") != ACCEPTED_C2R_CHECKPOINT_SHA256:
        raise RuntimeError("C3DP_SHARED_SPACE_CHECKPOINT_INVALID")
    predictor = SharedCrossModalPredictor(
        str(payload["candidate"]),
        hidden_dim=int(payload["hidden_dim"]),
        attention_layers=int(payload["attention_layers"]),
        heads=int(payload["heads"]),
    )
    predictor.load_state_dict(payload["state_dict"], strict=True)
    if predictor_state_digest(predictor) != payload.get("state_dict_sha256"):
        raise ValueError("C3-DP predictor state digest mismatch")
    return predictor, dict(payload.get("metadata", {}))


def dual_path_numpy_audit(native: np.ndarray, shared_native: np.ndarray) -> dict[str, float]:
    residual = np.asarray(native, dtype=np.float32) - np.asarray(shared_native, dtype=np.float32)
    reconstructed = np.asarray(shared_native, dtype=np.float32) + residual
    difference = reconstructed - np.asarray(native, dtype=np.float32)
    return {
        "max_abs_error": float(np.max(np.abs(difference))),
        "mse": float(np.mean(np.square(difference, dtype=np.float64))),
    }


def private_geometry(
    native: np.ndarray,
    shared: np.ndarray,
    shared_native: np.ndarray,
    private: np.ndarray,
) -> dict[str, Any]:
    from .continuous_vac_shared_space import linear_cka

    native_energy = float(np.mean(np.square(np.asarray(native, dtype=np.float64))))
    private_energy = float(np.mean(np.square(np.asarray(private, dtype=np.float64))))
    result = geometry_diagnostics(private)
    result.update(
        {
            "norm_mean": float(np.linalg.norm(private, axis=-1).mean()),
            "variance": float(np.var(private, dtype=np.float64)),
            "energy_fraction_of_native": private_energy / max(native_energy, 1e-12),
            "cka_with_native_z_c": linear_cka(private, native),
            "cka_with_shared_u_c": linear_cka(private, shared),
            "cka_with_shared_native": linear_cka(private, shared_native),
        }
    )
    return result


def output_geometry_gate(value: np.ndarray) -> tuple[dict[str, Any], bool]:
    geometry = geometry_diagnostics(value)
    collapsed = bool(
        geometry["per_dimension_variance"]["near_zero_fraction"] >= 0.5
        or geometry["query_diversity"]["collapsed_pair_fraction"] >= 0.5
    )
    return geometry, not collapsed
