import inspect

import pytest
import torch

from gr00t.tactile_unit.c5_causal_visual import VisualSupport
from gr00t.tactile_unit.c5_planned_action import ActionRepresentation, PlannedActionChunk, PlannedActionSource
from gr00t.tactile_unit.c5_runtime_router import C5Availability, C5RuntimeRouter, route_c5_availability
from gr00t.tactile_unit.c5_uncertainty import CalibratedC5Uncertainty, C5ContactUncertaintyEstimator, C5RuntimeMode


def plan(source=PlannedActionSource.POLICY_GENERATED):
    return PlannedActionChunk(torch.randn(2, 16, 128), source, 0.0, ActionRepresentation.NORMALIZED_PADDED_128, "TRAIN_ONLY_STANDARDIZED_PADDED_128", torch.ones(2, 16, dtype=torch.bool), 16, 31)


@pytest.mark.parametrize("vision,action,contact,mode", [
    (True, True, True, C5RuntimeMode.FULL_AH),
    (False, True, True, C5RuntimeMode.FULL_AH),
    (True, True, False, C5RuntimeMode.FALLBACK_CAUSAL_VA),
    (False, True, False, C5RuntimeMode.FALLBACK_A),
    (True, False, True, C5RuntimeMode.ABSTAIN_NO_ACTION),
    (False, False, False, C5RuntimeMode.ABSTAIN_NO_ACTION),
])
def test_exhaustive_router_modes(vision, action, contact, mode):
    assert route_c5_availability(C5Availability(vision, action, contact)) is mode


def router():
    return C5RuntimeRouter(lambda action, h: action, lambda vision, action: vision + action, lambda action: action, visual_support=VisualSupport.CURRENT_FRAME)


def test_no_action_abstains_without_contact_prediction():
    result = router().predict(C5Availability(True, False, True))
    assert not result.prediction_available and result.u_hat_c is None
    assert result.mode is C5RuntimeMode.ABSTAIN_NO_ACTION


def test_runtime_rejects_demo_and_oracle_plan_sources():
    availability = C5Availability(False, True, False)
    for source in (PlannedActionSource.DEMONSTRATION_TEACHER, PlannedActionSource.ORACLE_EVAL):
        with pytest.raises(PermissionError):
            router().predict(availability, plan=plan(source), u_a_plan=torch.randn(2, 8, 32))


def test_full_causal_and_a_only_routes_use_only_declared_sources():
    value = plan()
    action, context, vision = torch.randn(2, 8, 32), torch.randn(2, 256), torch.randn(2, 8, 32)
    full = router().predict(C5Availability(True, True, True), plan=value, u_a_plan=action, h_current=context)
    causal = router().predict(C5Availability(True, True, False), plan=value, u_a_plan=action, c_v=vision)
    a_only = router().predict(C5Availability(False, True, False), plan=value, u_a_plan=action)
    assert full.visual_support is VisualSupport.NONE
    assert causal.visual_support is VisualSupport.CURRENT_FRAME
    assert a_only.visual_support is VisualSupport.NONE


def test_full_uncertainty_receives_all_action_and_contact_slots():
    seen = {}
    def uncertainty(mode, prediction, source, ood):
        seen["mode"], seen["shape"] = mode, source.shape
        return torch.zeros(len(prediction))
    value = C5RuntimeRouter(
        lambda action, h: action, lambda vision, action: vision + action,
        lambda action: action, visual_support=VisualSupport.CURRENT_FRAME,
        uncertainty=uncertainty,
    )
    value.predict(
        C5Availability(False, True, True), plan=plan(),
        u_a_plan=torch.randn(2, 8, 32), h_current=torch.randn(2, 256),
    )
    assert seen == {"mode": C5RuntimeMode.FULL_AH, "shape": torch.Size([2, 16, 32])}


def test_offline_future_vision_upper_bound_is_not_runtime_routable():
    with pytest.raises(PermissionError, match="not runtime-routable"):
        router().predict_offline_oracle_va(runtime=True)


def test_uncertainty_is_small_target_free_mode_aware_model():
    model = C5ContactUncertaintyEstimator()
    assert model.parameter_count() <= 50_000
    assert "target" not in inspect.signature(model.forward).parameters
    prediction, source = torch.randn(4, 8, 32), torch.randn(4, 16, 32)
    values = model(C5RuntimeMode.FALLBACK_CAUSAL_VA, prediction, source)
    assert values.shape == (4,) and torch.isfinite(values).all()
    with pytest.raises(ValueError, match="ABSTAIN"):
        model(C5RuntimeMode.ABSTAIN_NO_ACTION, prediction, source)


def test_runtime_uncertainty_is_positive_calibrated_variance():
    base = C5ContactUncertaintyEstimator()
    calibrated = CalibratedC5Uncertainty(base, 2.5)
    prediction, source = torch.randn(4, 8, 32), torch.randn(4, 16, 32)
    value = calibrated(C5RuntimeMode.FULL_AH, prediction, source)
    assert value.shape == (4,) and torch.all(value > 0)
    assert all(not parameter.requires_grad for parameter in calibrated.parameters())
