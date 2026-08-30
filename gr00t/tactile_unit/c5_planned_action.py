"""Typed, provenance-aware candidate-action boundary for Track C5."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping

import numpy as np
import torch


ACTION_HORIZON = 16
RAW_ACTION_DIM = 58
CANONICAL_ACTION_DIM = 128
TREX_EMBODIMENT_ID = 31


class PlannedActionSource(str, Enum):
    POLICY_GENERATED = "POLICY_GENERATED"
    DEMONSTRATION_TEACHER = "DEMONSTRATION_TEACHER"
    ORACLE_EVAL = "ORACLE_EVAL"


class ActionRepresentation(str, Enum):
    RAW_58 = "RAW_58"
    NORMALIZED_PADDED_128 = "NORMALIZED_PADDED_128"


@dataclass(frozen=True)
class PlannedActionChunk:
    """A numeric action chunk plus mandatory origin and legality metadata."""

    actions: torch.Tensor
    source: PlannedActionSource
    start_time: float | torch.Tensor
    representation: ActionRepresentation
    normalization_state: str
    validity_mask: torch.Tensor
    horizon: int
    embodiment: int
    planner_policy_id: str | None = None

    def __post_init__(self) -> None:
        try:
            source = PlannedActionSource(self.source)
            representation = ActionRepresentation(self.representation)
        except ValueError as error:
            raise ValueError("planned Action source/representation must be explicit") from error
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "representation", representation)
        if not isinstance(self.actions, torch.Tensor) or self.actions.ndim != 3:
            raise ValueError("actions must be a tensor [B,16,D]")
        expected_dim = RAW_ACTION_DIM if representation is ActionRepresentation.RAW_58 else CANONICAL_ACTION_DIM
        if self.actions.shape[1:] != (ACTION_HORIZON, expected_dim):
            raise ValueError(f"actions must have exact shape [B,16,{expected_dim}]; a_t+16 is forbidden")
        if not self.actions.is_floating_point() or not bool(torch.isfinite(self.actions).all()):
            raise ValueError("planned actions must be finite floating-point values")
        if self.horizon != ACTION_HORIZON:
            raise ValueError("planned Action horizon must be exactly 16 (a_t through a_t+15)")
        if self.embodiment != TREX_EMBODIMENT_ID:
            raise ValueError("planned Action embodiment must be T-Rex ID 31")
        if not self.normalization_state:
            raise ValueError("normalization_state is mandatory")
        expected_normalization = (
            "RAW_UNNORMALIZED" if representation is ActionRepresentation.RAW_58
            else "TRAIN_ONLY_STANDARDIZED_PADDED_128"
        )
        if self.normalization_state != expected_normalization:
            raise ValueError(f"normalization_state must be {expected_normalization}")
        if not isinstance(self.validity_mask, torch.Tensor) or self.validity_mask.dtype is not torch.bool:
            raise TypeError("validity_mask must be an explicit bool tensor")
        if self.validity_mask.shape != self.actions.shape[:2]:
            raise ValueError("validity_mask must have shape [B,16]")
        if not bool(self.validity_mask.all()):
            raise ValueError("C5 requires all 16 candidate-action steps to be valid")
        if isinstance(self.start_time, torch.Tensor) and self.start_time.shape not in {torch.Size([]), torch.Size([len(self.actions)])}:
            raise ValueError("start_time must be scalar or [B]")
        if isinstance(self.start_time, torch.Tensor) and not bool(torch.isfinite(self.start_time).all()):
            raise ValueError("start_time must be finite")
        if not isinstance(self.start_time, (float, int, torch.Tensor)):
            raise TypeError("start_time must be numeric")

    def assert_legal(self, *, runtime: bool, offline_training: bool = False, oracle_eval: bool = False) -> None:
        if runtime and self.source is not PlannedActionSource.POLICY_GENERATED:
            raise PermissionError("runtime accepts only POLICY_GENERATED planned Action")
        if self.source is PlannedActionSource.DEMONSTRATION_TEACHER and not offline_training:
            raise PermissionError("DEMONSTRATION_TEACHER requires offline_training=True")
        if self.source is PlannedActionSource.ORACLE_EVAL and not oracle_eval:
            raise PermissionError("ORACLE_EVAL requires oracle_eval=True")


@dataclass(frozen=True)
class PlannedActionEncoding:
    z_a: torch.Tensor
    u_a: torch.Tensor
    source: PlannedActionSource
    embodiment: int


class TrainOnlyActionNormalizer:
    """Vectorized accepted train-only mean/std transform and zero padding."""

    def __init__(self, stats: Mapping[str, object]):
        fit_split = stats.get("fit_split")
        if fit_split not in {"frozen S1 train episodes only", "frozen train split only"}:
            raise ValueError("Action normalization must be accepted train-only mean/std")
        if all(key in stats for key in ("state_mean", "state_std", "action_mean", "action_std")):
            moments = {key: stats[key] for key in ("state_mean", "state_std", "action_mean", "action_std")}
        elif stats.get("mode") == "mean_std" and "action" in stats:
            state = stats.get("observation.state", stats.get("state"))
            if not isinstance(state, Mapping) or not isinstance(stats["action"], Mapping):
                raise ValueError("accepted normalization moments are missing")
            moments = {
                "state_mean": state["mean"], "state_std": state["std"],
                "action_mean": stats["action"]["mean"], "action_std": stats["action"]["std"],
            }
        else:
            raise ValueError("accepted normalization moments are missing")
        self.state_mean = torch.as_tensor(np.asarray(moments["state_mean"]), dtype=torch.float32)
        self.state_std = torch.as_tensor(np.asarray(moments["state_std"]), dtype=torch.float32)
        self.action_mean = torch.as_tensor(np.asarray(moments["action_mean"]), dtype=torch.float32)
        self.action_std = torch.as_tensor(np.asarray(moments["action_std"]), dtype=torch.float32)
        if any(tuple(value.shape) != (RAW_ACTION_DIM,) for value in (
            self.state_mean, self.state_std, self.action_mean, self.action_std,
        )):
            raise ValueError("normalization does not use the canonical 58D ordering")
        if not all(bool(torch.isfinite(value).all()) for value in (
            self.state_mean, self.state_std, self.action_mean, self.action_std,
        )) or bool((self.state_std <= 0).any()) or bool((self.action_std <= 0).any()):
            raise ValueError("accepted normalization moments are invalid")

    @staticmethod
    def _pad(value: torch.Tensor) -> torch.Tensor:
        padding = CANONICAL_ACTION_DIM - value.shape[-1]
        return torch.nn.functional.pad(value, (0, padding))

    def transform(self, state: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if state.shape != (len(actions), RAW_ACTION_DIM) or actions.shape[1:] != (ACTION_HORIZON, RAW_ACTION_DIM):
            raise ValueError("raw state/actions must be [B,58] and [B,16,58]")
        device, dtype = actions.device, actions.dtype
        sm, ss = self.state_mean.to(device, dtype), self.state_std.to(device, dtype)
        am, ass = self.action_mean.to(device, dtype), self.action_std.to(device, dtype)
        return self._pad((state - sm) / ss), self._pad((actions - am) / ass)


def encode_planned_action(
    plan: PlannedActionChunk,
    current_state: torch.Tensor,
    action_encoder: object,
    shared_action_encoder: Callable[[torch.Tensor], torch.Tensor],
    *,
    normalizer: TrainOnlyActionNormalizer | None = None,
    runtime: bool,
    offline_training: bool = False,
    oracle_eval: bool = False,
) -> PlannedActionEncoding:
    """Encode continuously through frozen A-R and P_a; the source tag never changes numerics."""

    plan.assert_legal(runtime=runtime, offline_training=offline_training, oracle_eval=oracle_eval)
    if plan.representation is ActionRepresentation.RAW_58:
        if normalizer is None:
            raise ValueError("RAW_58 plans require the accepted train-only normalizer")
        state, actions = normalizer.transform(current_state, plan.actions)
    else:
        if current_state.shape != (len(plan.actions), CANONICAL_ACTION_DIM):
            raise ValueError("canonical current_state must have shape [B,128]")
        if plan.normalization_state != "TRAIN_ONLY_STANDARDIZED_PADDED_128":
            raise ValueError("canonical plans must declare accepted normalization provenance")
        state, actions = current_state, plan.actions
    embodiment = torch.full((len(actions),), TREX_EMBODIMENT_ID, dtype=torch.long, device=actions.device)
    encoded = action_encoder.encode(state, actions, embodiment)
    z_a = encoded[0] if isinstance(encoded, (tuple, list)) else encoded
    if z_a.shape != (len(actions), 8, 32):
        raise RuntimeError("A-R continuous pre-RQ output must be [B,8,32]")
    u_a = shared_action_encoder(z_a)
    if u_a.shape != z_a.shape:
        raise RuntimeError("P_a output must be [B,8,32]")
    return PlannedActionEncoding(z_a=z_a, u_a=u_a, source=plan.source, embodiment=plan.embodiment)
