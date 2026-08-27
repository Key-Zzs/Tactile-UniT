"""Typed continuous Vision/Action/Contact transition contracts.

The offline teachers in this module all describe the same physical
``t -> t+16`` interval.  They are deliberately separated from the causal
online context so a future policy cannot accidentally receive a
future-derived representation as an observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .causal_contact_contract import CurrentContactContext, TensorLike


CANONICAL_VAC_HORIZON = 16
TRANSITION_SHAPE = (8, 32)
STATE_SHAPE = (128,)
ACTION_SHAPE = (16, 128)
CONTACT_CONTEXT_SHAPE = (256,)


class VACContractError(ValueError):
    """Raised when an integrated transition violates the public contract."""


class FutureOracleLeakageError(VACContractError):
    """Raised when offline/future information enters an online observation."""


def _shape(value: TensorLike) -> tuple[int, ...]:
    return tuple(int(item) for item in value.shape)


def _validate_float_batch(value: TensorLike, trailing: tuple[int, ...], name: str) -> int:
    if not isinstance(value, (np.ndarray, torch.Tensor)):
        raise TypeError(f"{name} must be a numpy array or torch tensor")
    if value.dtype not in (np.dtype("float32"), torch.float32):
        raise TypeError(f"{name} must be float32")
    if value.ndim != len(trailing) + 1 or _shape(value)[1:] != trailing:
        expected = ",".join(str(item) for item in trailing)
        raise VACContractError(f"{name} must have shape [B,{expected}]")
    finite = (
        bool(torch.isfinite(value).all())
        if isinstance(value, torch.Tensor)
        else bool(np.isfinite(value).all())
    )
    if not finite:
        raise VACContractError(f"{name} contains non-finite values")
    return int(value.shape[0])


def _validate_vector(value: Sequence[int] | np.ndarray, count: int, name: str) -> None:
    if len(value) != count:
        raise VACContractError(f"{name} must contain exactly B values")


@dataclass(frozen=True)
class TransitionAnchor:
    """Identity and time anchor for one canonical T-Rex transition."""

    pair_id: str
    episode_id: int
    t: int
    t_future: int

    def __post_init__(self) -> None:
        if not self.pair_id:
            raise VACContractError("pair_id must be non-empty")
        if self.episode_id < 0 or self.t < 15:
            raise VACContractError("episode_id must be non-negative and t must include t-15:t")
        if self.t_future != self.t + CANONICAL_VAC_HORIZON:
            raise VACContractError("canonical transition must be exactly t -> t+16")


@dataclass(frozen=True)
class ModalityAvailability:
    """Explicit availability mask; missing is distinct from an all-zero value."""

    vision: bool
    action: bool
    contact: bool


@dataclass(frozen=True)
class ActionTransitionTarget:
    """Demonstration/planned Action teacher, never a current sensory input."""

    z_a: TensorLike
    teacher_only: bool = True
    horizon_frames: int = CANONICAL_VAC_HORIZON

    def __post_init__(self) -> None:
        _validate_float_batch(self.z_a, TRANSITION_SHAPE, "z_a")
        if not self.teacher_only or self.horizon_frames != CANONICAL_VAC_HORIZON:
            raise VACContractError("Action transition target must be teacher-only at k=16")


@dataclass(frozen=True)
class PredictedOrPlannedActionTransition:
    """A policy-owned action transition that is legal after planning."""

    z_hat_a: TensorLike
    policy_generated: bool = True

    def __post_init__(self) -> None:
        _validate_float_batch(self.z_hat_a, TRANSITION_SHAPE, "z_hat_a")
        if not self.policy_generated:
            raise VACContractError("online Action transition must be policy-generated")


@dataclass(frozen=True)
class OfflineVACTransitionTeachers:
    """A same-order batch of native continuous V/A/C transition teachers."""

    pair_id: Sequence[str]
    episode_id: Sequence[int] | np.ndarray
    t: Sequence[int] | np.ndarray
    t_future: Sequence[int] | np.ndarray
    z_v: TensorLike
    z_a: TensorLike
    z_c: TensorLike
    h_t_c: TensorLike
    state: TensorLike
    action: TensorLike
    modality_masks: Mapping[str, Sequence[bool] | np.ndarray]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        count = _validate_float_batch(self.z_v, TRANSITION_SHAPE, "z_v")
        for name, value, trailing in (
            ("z_a", self.z_a, TRANSITION_SHAPE),
            ("z_c", self.z_c, TRANSITION_SHAPE),
            ("h_t_c", self.h_t_c, CONTACT_CONTEXT_SHAPE),
            ("state", self.state, STATE_SHAPE),
            ("action", self.action, ACTION_SHAPE),
        ):
            if _validate_float_batch(value, trailing, name) != count:
                raise VACContractError("all modalities must have the same batch length")
        for name, value in (
            ("pair_id", self.pair_id),
            ("episode_id", self.episode_id),
            ("t", self.t),
            ("t_future", self.t_future),
        ):
            _validate_vector(value, count, name)
        if len(set(map(str, self.pair_id))) != count:
            raise VACContractError("pair_id values must be unique within an acceptance batch")
        for current, future in zip(self.t, self.t_future):
            if int(future) != int(current) + CANONICAL_VAC_HORIZON:
                raise VACContractError("every row must describe exactly t -> t+16")
        if set(self.modality_masks) != {"vision", "action", "contact"}:
            raise VACContractError("modality_masks must contain vision, action, and contact")
        for name, mask in self.modality_masks.items():
            _validate_vector(mask, count, f"{name} modality mask")
            if np.asarray(mask).dtype != np.dtype("bool"):
                raise TypeError(f"{name} modality mask must be boolean")
        if not self.provenance:
            raise VACContractError("offline teachers require checkpoint/cache provenance")

    @property
    def batch_size(self) -> int:
        return len(self.pair_id)


_ORACLE_FIELD_NAMES = frozenset(
    {
        "i_t+16",
        "i_future",
        "future_image",
        "future_visual_observation",
        "goal_image",
        "h_t+16",
        "h_t+16^c",
        "h_future",
        "future_contact",
        "z_v",
        "z_v_target",
        "vision_transition_target",
        "z_c",
        "z_c_target",
        "contact_transition_target",
        "z_a",
        "z_a_target",
        "action_transition_target",
        "demonstration_action_transition",
        "offline_transition_teachers",
    }
)


def reject_online_oracles(payload: Any, *, oracle_eval: bool = False) -> None:
    """Recursively reject future/teacher fields in an online observation.

    Explicit oracle evaluation is the only bypass.  Both mappings and nested
    lists/tuples are traversed so wrapping a teacher does not evade the guard.
    """

    if oracle_eval:
        return

    def visit(value: Any, prefix: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key).lower()
                path = f"{prefix}.{key}" if prefix else key
                if key in _ORACLE_FIELD_NAMES:
                    raise FutureOracleLeakageError(
                        f"offline/future field {path!r} requires oracle_eval=True"
                    )
                visit(child, path)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{prefix}[{index}]")

    visit(payload, "")


@dataclass(frozen=True)
class OnlineCausalContext:
    """Observation-side values legally available at deployment time."""

    current_visual_observation: Any | None
    robot_state: TensorLike | None
    current_tactile_history: Any | None = None
    current_contact: CurrentContactContext | None = None
    predicted_or_planned_action: PredictedOrPlannedActionTransition | None = None
    modality_available: ModalityAvailability = field(
        default_factory=lambda: ModalityAvailability(False, False, False)
    )
    task_metadata: Mapping[str, Any] = field(default_factory=dict)
    observation_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.robot_state is not None:
            _validate_float_batch(self.robot_state, STATE_SHAPE, "robot_state")
        if self.current_contact is not None and not isinstance(
            self.current_contact, CurrentContactContext
        ):
            raise TypeError("current_contact must be a CurrentContactContext")
        if self.predicted_or_planned_action is not None and not isinstance(
            self.predicted_or_planned_action, PredictedOrPlannedActionTransition
        ):
            raise FutureOracleLeakageError(
                "online Action value must be PredictedOrPlannedActionTransition, not a teacher"
            )
        reject_online_oracles(self.current_visual_observation)
        reject_online_oracles(self.current_tactile_history)
        reject_online_oracles(self.task_metadata)
        reject_online_oracles(self.observation_metadata)


def validate_integrated_manifest_row(row: Mapping[str, Any]) -> TransitionAnchor:
    """Validate exact Vision/Action/Contact timing for one S3.1 manifest row."""

    anchor = int(row["anchor"]["frame"])
    episode = int(row["episode_id"])
    vision_current = int(row["vision"]["current"]["episode_frame"])
    vision_future = int(row["vision"]["future"]["episode_frame"])
    action_start, action_end = map(int, row["action"]["episode_frames_inclusive"])
    current_start, current_end = map(
        int, row["contact"]["current_teacher_window_inclusive"]
    )
    future_start, future_end = map(
        int, row["contact"]["future_teacher_window_inclusive"]
    )
    expected = {
        "vision_current": anchor,
        "vision_future": anchor + 16,
        "action_start": anchor,
        "action_end": anchor + 15,
        "current_start": anchor - 15,
        "current_end": anchor,
        "future_start": anchor + 1,
        "future_end": anchor + 16,
    }
    actual = {
        "vision_current": vision_current,
        "vision_future": vision_future,
        "action_start": action_start,
        "action_end": action_end,
        "current_start": current_start,
        "current_end": current_end,
        "future_start": future_start,
        "future_end": future_end,
    }
    if actual != expected:
        raise VACContractError(f"pair {row.get('pair_id')!r} timing mismatch: {actual}")
    if current_end >= future_start and current_end != future_start - 1:
        raise VACContractError("Contact Teacher windows overlap")
    if row.get("modality_mask") != {"A": 1, "C": 1, "V": 1}:
        raise VACContractError("canonical paired row must contain V, A, and C")
    return TransitionAnchor(str(row["pair_id"]), episode, anchor, anchor + 16)
