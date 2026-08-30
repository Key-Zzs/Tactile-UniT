"""Causal API types for the continuous Contact bridge.

The accepted Contact transition latent is computed from a future tactile
window. This module makes that teacher-only role explicit and keeps it out of
runtime observations by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import numpy as np
import torch

TensorLike = np.ndarray | torch.Tensor


class ContactMode(str, Enum):
    """Execution modes with different oracle-field policies."""

    OFFLINE_TRAINING = "offline_training"
    OFFLINE_EVALUATION = "offline_evaluation"
    ORACLE_EVALUATION = "oracle_evaluation"
    INFERENCE = "inference"


def _validate_tensor(value: TensorLike, trailing_shape: tuple[int, ...], name: str) -> None:
    if not isinstance(value, (np.ndarray, torch.Tensor)):
        raise TypeError(f"{name} must be a numpy array or torch tensor")
    shape = tuple(int(item) for item in value.shape)
    if value.ndim != len(trailing_shape) + 1 or shape[1:] != trailing_shape:
        suffix = ",".join(map(str, trailing_shape))
        raise ValueError(f"{name} must have shape [B,{suffix}]")
    if value.dtype not in (np.dtype("float32"), torch.float32):
        raise TypeError(f"{name} must be float32")
    finite = (
        bool(torch.isfinite(value).all())
        if isinstance(value, torch.Tensor)
        else bool(np.isfinite(value).all())
    )
    if not finite:
        raise ValueError(f"{name} contains non-finite values")


@dataclass(frozen=True)
class CurrentContactContext:
    """Current causal Contact state, available during inference."""

    h_t_c: TensorLike
    inference_available: bool = True

    def __post_init__(self) -> None:
        _validate_tensor(self.h_t_c, (256,), "h_t_c")
        if not self.inference_available:
            raise ValueError("CurrentContactContext must be inference-available")


@dataclass(frozen=True)
class ContactTransitionTarget:
    """Future-derived continuous Contact transition teacher."""

    z_c: TensorLike
    teacher_only: bool = True
    horizon_frames: int = 16

    def __post_init__(self) -> None:
        _validate_tensor(self.z_c, (8, 32), "z_c")
        if not self.teacher_only or self.horizon_frames != 16:
            raise ValueError("Contact transition target must be teacher-only at k=16")


@dataclass(frozen=True)
class VisionTransitionTarget:
    """Future-derived Original UniT Vision transition teacher."""

    z_v: TensorLike
    teacher_only: bool = True
    horizon_frames: int = 16

    def __post_init__(self) -> None:
        _validate_tensor(self.z_v, (8, 32), "z_v")
        if not self.teacher_only or self.horizon_frames != 16:
            raise ValueError("Vision transition target must be teacher-only at k=16")


@dataclass(frozen=True)
class PredictedContactTransition:
    """Causally predicted Contact transition, legal at inference time."""

    z_hat_c: TensorLike
    inference_generated: bool = True

    def __post_init__(self) -> None:
        _validate_tensor(self.z_hat_c, (8, 32), "z_hat_c")
        if not self.inference_generated:
            raise ValueError("PredictedContactTransition must be inference-generated")


class FutureContactLeakageError(ValueError):
    """Raised when future-derived information enters an online API."""


@dataclass(frozen=True)
class ContactBridgeBatch:
    """Typed batch that enforces offline-teacher/runtime separation."""

    current_contact: CurrentContactContext | None
    contact_target: ContactTransitionTarget | None = None
    vision_target: VisionTransitionTarget | None = None
    predicted_contact: PredictedContactTransition | None = None

    def validate_for(self, mode: ContactMode | str) -> None:
        mode = ContactMode(mode)
        if mode is ContactMode.INFERENCE and (
            self.contact_target is not None or self.vision_target is not None
        ):
            raise FutureContactLeakageError(
                "inference batch contains a future-derived transition teacher"
            )


_ORACLE_FIELD_NAMES = frozenset(
    {
        "i_t+16",
        "i_future",
        "future_image",
        "goal_image",
        "h_t+16",
        "h_future",
        "future_contact",
        "z_c",
        "z_c_target",
        "contact_transition_target",
        "vision_transition_target",
    }
)


def reject_future_oracles(
    payload: Mapping[str, Any], *, mode: ContactMode | str = ContactMode.INFERENCE
) -> None:
    """Reject known future/oracle fields in nested runtime mappings.

    Offline training/evaluation and explicit oracle evaluation may carry true
    transition teachers. Inference is the mode that rejects them.
    """

    if ContactMode(mode) is not ContactMode.INFERENCE:
        return

    def visit(value: Any, prefix: str) -> None:
        if not isinstance(value, Mapping):
            return
        for raw_key, child in value.items():
            key = str(raw_key).lower()
            path = f"{prefix}.{key}" if prefix else key
            if key in _ORACLE_FIELD_NAMES:
                raise FutureContactLeakageError(
                    f"future-derived field {path!r} is illegal outside oracle evaluation"
                )
            visit(child, path)

    visit(payload, "")


def runtime_contact_batch(
    h_t_c: TensorLike | None, z_hat_c: TensorLike | None = None
) -> ContactBridgeBatch:
    """Build an inference-safe batch from current and predicted values only."""

    batch = ContactBridgeBatch(
        current_contact=None if h_t_c is None else CurrentContactContext(h_t_c),
        predicted_contact=None if z_hat_c is None else PredictedContactTransition(z_hat_c),
    )
    batch.validate_for(ContactMode.INFERENCE)
    return batch
