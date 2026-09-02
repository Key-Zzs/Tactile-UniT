"""Pure NumPy online runtime contracts for the causal S4.3 ACT benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from gr00t.simulation.s4_3_policy import TensorProvenance

HISTORY_STEPS = 26
TACTILE_DIM = 30
ACTION_STEPS = 27
ACTION_DIM = 22
REPLAN_STRIDE = 5


@dataclass(frozen=True)
class CausalObservation:
    episode_id: str
    current_control_step: int
    rgb: np.ndarray
    proprio: np.ndarray
    tactile_history: np.ndarray
    provenance: tuple[TensorProvenance, ...]

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb)
        proprio = np.asarray(self.proprio, dtype=np.float32)
        tactile = np.asarray(self.tactile_history, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError("current RGB must be uint8 HWC")
        if proprio.shape != (ACTION_DIM,) or not np.isfinite(proprio).all():
            raise ValueError("current proprio must be finite 22D")
        if tactile.shape != (HISTORY_STEPS, TACTILE_DIM) or not np.isfinite(tactile).all():
            raise ValueError("causal tactile history must be finite [26,30]")
        for item in self.provenance:
            item.validate(inference=True)
            if (
                item.episode_id != self.episode_id
                or item.current_control_step != self.current_control_step
            ):
                raise ValueError("observation provenance identity mismatch")
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "proprio", proprio)
        object.__setattr__(self, "tactile_history", tactile)


class CausalHistoryBuffer:
    """Keep exactly the latest 26 same-episode tactile observations."""

    def __init__(self, episode_id: str) -> None:
        self.episode_id = episode_id
        self._steps: list[int] = []
        self._values: list[np.ndarray] = []

    def append(self, control_step: int, tactile: np.ndarray) -> None:
        value = np.asarray(tactile, dtype=np.float32)
        if value.shape != (TACTILE_DIM,) or not np.isfinite(value).all():
            raise ValueError("runtime tactile sample must be finite 30D")
        if self._steps and control_step != self._steps[-1] + 1:
            raise ValueError("runtime tactile control steps must be consecutive")
        self._steps.append(int(control_step))
        self._values.append(value.copy())
        if len(self._steps) > HISTORY_STEPS:
            self._steps.pop(0)
            self._values.pop(0)

    @property
    def ready(self) -> bool:
        return len(self._steps) == HISTORY_STEPS

    @property
    def step_interval(self) -> tuple[int, int]:
        if not self.ready:
            raise RuntimeError("causal tactile history is not warm")
        return self._steps[0], self._steps[-1]

    def value(self) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("causal tactile history is not warm")
        return np.stack(self._values).astype(np.float32, copy=False)

    def observation(self, rgb: np.ndarray, proprio: np.ndarray) -> CausalObservation:
        start, stop = self.step_interval
        provenance = (
            TensorProvenance(self.episode_id, stop, stop, stop, "OBSERVATION"),
            TensorProvenance(self.episode_id, stop, stop, stop, "OBSERVATION"),
            TensorProvenance(self.episode_id, stop, start, stop, "OBSERVATION"),
        )
        return CausalObservation(
            episode_id=self.episode_id,
            current_control_step=stop,
            rgb=rgb,
            proprio=proprio,
            tactile_history=self.value(),
            provenance=provenance,
        )


class ActionChunkQueue:
    """Accept a 27-step plan and expose exactly the first five Actions."""

    def __init__(self, episode_id: str, stride: int = REPLAN_STRIDE) -> None:
        if stride != REPLAN_STRIDE:
            raise ValueError("canonical replan stride is exactly 5")
        self.episode_id = episode_id
        self.stride = stride
        self._actions: np.ndarray | None = None
        self._cursor = 0
        self._plan_step: int | None = None
        self.replan_count = 0

    @property
    def needs_replan(self) -> bool:
        return self._actions is None or self._cursor >= self.stride

    def set_plan(self, actions: np.ndarray, current_control_step: int) -> TensorProvenance:
        value = np.asarray(actions, dtype=np.float32)
        if value.shape != (ACTION_STEPS, ACTION_DIM) or not np.isfinite(value).all():
            raise ValueError("ACT plan must be finite [27,22]")
        self._actions = value.copy()
        self._cursor = 0
        self._plan_step = int(current_control_step)
        self.replan_count += 1
        return TensorProvenance(
            self.episode_id,
            int(current_control_step),
            int(current_control_step),
            int(current_control_step + ACTION_STEPS - 1),
            "PLAN",
        )

    def pop(self) -> np.ndarray:
        if self.needs_replan or self._actions is None:
            raise RuntimeError("Action queue requires a fresh plan")
        value = self._actions[self._cursor].copy()
        self._cursor += 1
        return value

    @property
    def executed_from_current_plan(self) -> int:
        return self._cursor


def mapped_force_metrics(tactile: np.ndarray) -> dict[str, float | int]:
    """Compute the frozen five-region force/contact diagnostics."""

    values = np.asarray(tactile, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != TACTILE_DIM:
        raise ValueError("mapped tactile trace must be [T,30]")
    matrix = values.reshape(len(values), 5, 6)
    occupied = matrix[..., 0] > 0.5
    any_contact = occupied.any(axis=1)
    free_to_contact = int(np.sum((~any_contact[:-1]) & any_contact[1:]))
    contact_to_free = int(np.sum(any_contact[:-1] & (~any_contact[1:])))
    normal = np.maximum(matrix[..., 1], 0.0).sum(axis=1)
    tangential = np.maximum(matrix[..., 2], 0.0).sum(axis=1)
    return {
        "free_to_contact": free_to_contact,
        "contact_to_free": contact_to_free,
        "peak_normal_force": float(normal.max(initial=0.0)),
        "integrated_normal_force": float(normal.sum() * 0.02),
        "peak_tangential_force": float(tangential.max(initial=0.0)),
    }


def validate_trace_provenance(items: Iterable[TensorProvenance]) -> None:
    for item in items:
        item.validate(inference=True)
