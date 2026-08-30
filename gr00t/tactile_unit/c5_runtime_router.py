"""Final C5 runtime availability router with no future-Vision route."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .c5_causal_visual import VisualSupport
from .c5_planned_action import PlannedActionChunk, PlannedActionSource
from .c5_uncertainty import C5RuntimeMode


@dataclass(frozen=True)
class C5Availability:
    vision_available: bool
    action_available: bool
    contact_context_available: bool

    def __post_init__(self) -> None:
        for name in ("vision_available", "action_available", "contact_context_available"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an explicit bool")


def route_c5_availability(value: C5Availability) -> C5RuntimeMode:
    if not value.action_available:
        return C5RuntimeMode.ABSTAIN_NO_ACTION
    if value.contact_context_available:
        return C5RuntimeMode.FULL_AH
    if value.vision_available:
        return C5RuntimeMode.FALLBACK_CAUSAL_VA
    return C5RuntimeMode.FALLBACK_A


@dataclass(frozen=True)
class CausalContactPredictionResult:
    prediction_available: bool
    mode: C5RuntimeMode
    u_hat_c: torch.Tensor | None
    uncertainty: torch.Tensor | None
    visual_support: VisualSupport
    planned_action_source: PlannedActionSource | None
    policy_plan_domain_validated: bool
    contact_context_available: bool
    vision_available: bool
    action_available: bool
    rank_warning: bool
    oracle_mode: bool


class C5RuntimeRouter:
    def __init__(
        self,
        full_predictor: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        causal_predictor: Callable[[torch.Tensor, torch.Tensor], torch.Tensor | tuple[torch.Tensor, object]],
        a_predictor: Callable[[torch.Tensor], torch.Tensor],
        *,
        visual_support: VisualSupport,
        uncertainty: Callable[[C5RuntimeMode, torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor] | None = None,
        policy_plan_domain_validated: bool = False,
        rank_warning: bool = True,
    ):
        if visual_support not in {VisualSupport.CURRENT_FRAME, VisualSupport.CAUSAL_HISTORY_8}:
            raise ValueError("causal fallback needs a legal current/history support")
        self.full_predictor, self.causal_predictor, self.a_predictor = full_predictor, causal_predictor, a_predictor
        self.visual_support, self.uncertainty = visual_support, uncertainty
        self.policy_plan_domain_validated, self.rank_warning = bool(policy_plan_domain_validated), bool(rank_warning)

    def predict(
        self,
        availability: C5Availability,
        *,
        plan: PlannedActionChunk | None = None,
        u_a_plan: torch.Tensor | None = None,
        h_current: torch.Tensor | None = None,
        c_v: torch.Tensor | None = None,
        plan_ood_score: torch.Tensor | None = None,
        runtime: bool = True,
        oracle_eval: bool = False,
    ) -> CausalContactPredictionResult:
        mode = route_c5_availability(availability)
        common = dict(
            policy_plan_domain_validated=self.policy_plan_domain_validated,
            contact_context_available=availability.contact_context_available,
            vision_available=availability.vision_available,
            action_available=availability.action_available,
            rank_warning=self.rank_warning,
            oracle_mode=bool(oracle_eval),
        )
        if mode is C5RuntimeMode.ABSTAIN_NO_ACTION:
            return CausalContactPredictionResult(False, mode, None, None, VisualSupport.NONE, None, **common)
        if plan is None or u_a_plan is None:
            raise ValueError("available Action requires PlannedActionChunk and u_a_plan")
        plan.assert_legal(runtime=runtime, oracle_eval=oracle_eval)
        if mode is C5RuntimeMode.FULL_AH:
            if h_current is None:
                raise ValueError("FULL_AH requires current Contact context")
            if h_current.shape != (len(u_a_plan), 8 * 32):
                raise ValueError("FULL_AH Contact context must be [B,256]")
            prediction = self.full_predictor(u_a_plan, h_current)
            source = torch.cat((u_a_plan, h_current.reshape(len(h_current), 8, 32)), dim=1)
            support = VisualSupport.NONE
        elif mode is C5RuntimeMode.FALLBACK_CAUSAL_VA:
            if c_v is None:
                raise ValueError("causal visual fallback requires c_v")
            result = self.causal_predictor(c_v, u_a_plan)
            prediction = result[0] if isinstance(result, tuple) else result
            source = torch.cat((c_v, u_a_plan), dim=1)
            support = self.visual_support
        else:
            prediction = self.a_predictor(u_a_plan)
            source, support = u_a_plan, VisualSupport.NONE
        uncertainty = None if self.uncertainty is None else self.uncertainty(mode, prediction, source, plan_ood_score)
        return CausalContactPredictionResult(True, mode, prediction, uncertainty, support, plan.source, **common)

    def predict_offline_oracle_va(self, *args: object, runtime: bool = True, **kwargs: object) -> None:
        if runtime:
            raise PermissionError("OFFLINE_ORACLE_VA is not runtime-routable")
        raise NotImplementedError("offline oracle evaluation is deliberately outside the runtime router")
