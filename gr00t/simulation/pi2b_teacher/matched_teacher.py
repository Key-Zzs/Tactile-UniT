"""Strictly matched VA/VAC teachers with an additive Contact treatment.

The VA loss block is computed identically for both teachers.  The VAC teacher
adds Contact parameters and Contact losses without renormalizing the VA block.
All public encoders consume one native modality only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gr00t.tactile_unit.continuous_vac_shared_space import (
    IndependentSlotResampler,
    RecoveryHead,
    different_episode_info_nce,
    relational_preservation,
    variance_floor,
)

NATIVE_FIELDS = {"vision": "z_v", "action": "z_a", "contact": "z_c"}
COMMON_MODALITIES = ("vision", "action")
ALL_MODALITIES = ("vision", "action", "contact")


def _component_seed(base_seed: int, component: str) -> int:
    payload = f"pi2b-teacher\0{base_seed}\0{component}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31)


def _construct_with_seed(base_seed: int, component: str, factory):
    # Components are always constructed on CPU and moved afterwards.  Keeping
    # CUDA out of initialization also lets the matching audit remain CPU-only.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_component_seed(base_seed, component))
        return factory()


class MatchedTeacher(nn.Module):
    """Independent slot encoders with byte-matched common VA components."""

    def __init__(self, modalities: Iterable[str], *, initialization_seed: int = 42) -> None:
        super().__init__()
        modalities = tuple(modalities)
        if modalities not in (COMMON_MODALITIES, ALL_MODALITIES):
            raise ValueError("modalities must be exactly ('vision','action') or VA plus contact")
        self.modalities = modalities
        self.initialization_seed = int(initialization_seed)
        self.shared_slots = nn.Parameter(
            _construct_with_seed(
                initialization_seed,
                "shared_slots",
                lambda: torch.randn(8, 32) * 0.02,
            )
        )
        self.projectors = nn.ModuleDict(
            {
                name: _construct_with_seed(
                    initialization_seed,
                    f"projector:{name}",
                    lambda: IndependentSlotResampler(heads=4, hidden_dim=64),
                )
                for name in modalities
            }
        )
        self.recovery = nn.ModuleDict(
            {
                name: _construct_with_seed(
                    initialization_seed,
                    f"recovery:{name}",
                    lambda: RecoveryHead(hidden_dim=128, query_mixing=True),
                )
                for name in modalities
            }
        )

    def encode(self, modality: str, native: torch.Tensor) -> torch.Tensor:
        if modality not in self.modalities:
            raise ValueError(f"teacher has no {modality!r} path")
        return self.projectors[modality](native, self.shared_slots)

    def recover(self, modality: str, shared: torch.Tensor) -> torch.Tensor:
        if modality not in self.modalities:
            raise ValueError(f"teacher has no {modality!r} recovery head")
        return self.recovery[modality](shared)

    def forward(self, native: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if tuple(native) != self.modalities:
            raise ValueError(f"expected ordered fields {self.modalities}, got {tuple(native)}")
        return {name: self.encode(name, native[name]) for name in self.modalities}


@dataclass(frozen=True)
class MatchLossWeights:
    alignment: float = 1.0
    native: float = 5.0
    relational: float = 0.25
    variance: float = 0.05
    contact_pair: float = 1.0
    contact_native: float = 5.0
    contact_relational: float = 0.25
    contact_variance: float = 0.05


def _symmetric_alignment(
    left: torch.Tensor,
    right: torch.Tensor,
    episode_id: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return torch.stack(
        (
            different_episode_info_nce(left, right, episode_id, temperature=temperature),
            different_episode_info_nce(right, left, episode_id, temperature=temperature),
        )
    ).mean()


def matched_teacher_loss(
    model: MatchedTeacher,
    native: Mapping[str, torch.Tensor],
    episode_id: torch.Tensor,
    *,
    temperature: float,
    weights: MatchLossWeights,
    contact_enabled: bool | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return identical VA loss plus an explicit additive Contact block."""

    use_contact = "contact" in model.modalities if contact_enabled is None else contact_enabled
    if use_contact and "contact" not in model.modalities:
        raise ValueError("Contact treatment requested from a VA-only teacher")
    active_modalities = ALL_MODALITIES if use_contact else COMMON_MODALITIES
    expected = set(active_modalities)
    if set(native) != expected:
        raise ValueError(f"native fields {set(native)} do not match active fields {expected}")
    shared = {name: model.encode(name, native[name]) for name in active_modalities}
    va_alignment = _symmetric_alignment(
        shared["vision"], shared["action"], episode_id, temperature
    )
    va_native = torch.stack(
        [
            F.mse_loss(model.recover(name, shared[name]), native[name].detach())
            for name in COMMON_MODALITIES
        ]
    ).mean()
    va_relational = torch.stack(
        [relational_preservation(native[name], shared[name]) for name in COMMON_MODALITIES]
    ).mean()
    va_variance = torch.stack([variance_floor(shared[name]) for name in COMMON_MODALITIES]).mean()
    va_total = (
        weights.alignment * va_alignment
        + weights.native * va_native
        + weights.relational * va_relational
        + weights.variance * va_variance
    )
    zero = va_total.new_zeros(())
    c_alignment = c_native = c_relational = c_variance = zero
    if use_contact:
        c_alignment = torch.stack(
            [
                _symmetric_alignment(shared[other], shared["contact"], episode_id, temperature)
                for other in COMMON_MODALITIES
            ]
        ).mean()
        c_native = F.mse_loss(
            model.recover("contact", shared["contact"]), native["contact"].detach()
        )
        c_relational = relational_preservation(native["contact"], shared["contact"])
        c_variance = variance_floor(shared["contact"])
    contact_total = (
        weights.contact_pair * c_alignment
        + weights.contact_native * c_native
        + weights.contact_relational * c_relational
        + weights.contact_variance * c_variance
    )
    total = va_total + contact_total
    values = {
        "total": total,
        "va_total": va_total,
        "va_alignment": va_alignment,
        "va_native": va_native,
        "va_relational": va_relational,
        "va_variance": va_variance,
        "contact_total": contact_total,
        "contact_alignment": c_alignment,
        "contact_native": c_native,
        "contact_relational": c_relational,
        "contact_variance": c_variance,
    }
    return total, values


def build_matched_pair(initialization_seed: int = 42) -> tuple[MatchedTeacher, MatchedTeacher]:
    return (
        MatchedTeacher(COMMON_MODALITIES, initialization_seed=initialization_seed),
        MatchedTeacher(ALL_MODALITIES, initialization_seed=initialization_seed),
    )


def clip_active_grad_norm_(parameters: Iterable[torch.nn.Parameter], max_norm: float) -> torch.Tensor:
    """Apply one global norm over parameters with a nonzero active gradient.

    Filtering exact-zero gradients makes the C-disabled VAC graph identical to
    the VA graph while retaining one global VA+C clip when Contact is active.
    """

    active = [
        parameter
        for parameter in parameters
        if parameter.grad is not None and bool(torch.count_nonzero(parameter.grad.detach()))
    ]
    if not active:
        return torch.zeros((), dtype=torch.float32)
    norms = torch.stack(
        [torch.linalg.vector_norm(parameter.grad.detach().float(), ord=2) for parameter in active]
    )
    total = torch.linalg.vector_norm(norms, ord=2)
    coefficient = torch.clamp(float(max_norm) / (total + 1e-6), max=1.0)
    for parameter in active:
        parameter.grad.mul_(coefficient.to(device=parameter.grad.device, dtype=parameter.grad.dtype))
    return total


class StrictFieldStore:
    """NPZ field whitelist that records and rejects undeclared accesses."""

    def __init__(self, path: Path, allowed_fields: Iterable[str]) -> None:
        self.path = Path(path)
        self.allowed_fields = frozenset(allowed_fields)
        self.accessed_fields: list[str] = []

    def load(self) -> dict[str, np.ndarray]:
        with np.load(self.path, allow_pickle=False) as source:
            missing = self.allowed_fields - set(source.files)
            if missing:
                raise KeyError(f"missing required fields: {sorted(missing)}")
            values = {}
            for field in sorted(self.allowed_fields):
                if field not in self.allowed_fields:
                    raise PermissionError(f"field {field!r} is not whitelisted")
                self.accessed_fields.append(field)
                values[field] = source[field]
        return values

    def read(self, field: str) -> np.ndarray:
        if field not in self.allowed_fields:
            raise PermissionError(f"field {field!r} is not whitelisted")
        with np.load(self.path, allow_pickle=False) as source:
            self.accessed_fields.append(field)
            return source[field]


def common_state(model: MatchedTeacher) -> dict[str, torch.Tensor]:
    prefixes = ("shared_slots", "projectors.vision.", "projectors.action.", "recovery.vision.", "recovery.action.")
    return {
        name: value
        for name, value in model.state_dict().items()
        if name == "shared_slots" or name.startswith(prefixes[1:])
    }
