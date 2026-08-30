import inspect
import json
from pathlib import Path

import pytest
import torch

from gr00t.tactile_unit.c4_availability_conditioning import AvailabilityMode
from gr00t.tactile_unit.c4_uncertainty import (
    ContactUncertaintyEstimator, heteroscedastic_nll,
)


ROOT = Path(__file__).resolve().parents[2]


def locked():
    path = ROOT / ".local/artifacts/tactile_unit/vac_c4/locked_test_evaluation.json"
    if not path.is_file():
        pytest.skip("local C4 artifacts unavailable")
    return json.loads(path.read_text())


def test_uncertainty_model_is_small_scalar_and_mode_aware():
    model = ContactUncertaintyEstimator()
    assert model.parameter_count() <= 50_000
    prediction = torch.randn(4, 8, 32)
    source = torch.randn(4, 16, 32)
    value = model(AvailabilityMode.FALLBACK_VA, prediction, source)
    assert value.shape == (4,)
    assert torch.all(value >= model.log_variance_min)
    assert torch.all(value <= model.log_variance_max)


def test_abstain_has_no_finite_uncertainty():
    model = ContactUncertaintyEstimator()
    with pytest.raises(ValueError, match="ABSTAIN"):
        model(AvailabilityMode.ABSTAIN_NO_ACTION, torch.randn(2, 8, 32), torch.randn(2, 8, 32))


def test_uncertainty_interface_contains_no_target_or_error():
    parameters = set(inspect.signature(ContactUncertaintyEstimator.forward).parameters)
    assert parameters == {"self", "mode", "prediction", "source"}
    assert parameters.isdisjoint({"u_c", "target", "error", "r_c_priv", "h_future"})


def test_uncertainty_nll_cannot_modify_frozen_mean_prediction():
    model = ContactUncertaintyEstimator()
    prediction = torch.randn(3, 8, 32, requires_grad=True)
    target = torch.randn(3, 8, 32)
    log_variance = model(AvailabilityMode.FALLBACK_A, prediction.detach(), prediction.detach())
    heteroscedastic_nll(log_variance, prediction, target).backward()
    assert prediction.grad is None
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_uncertainty_selection_uses_one_common_validation_scale():
    root = ROOT / ".local/artifacts/tactile_unit/vac_c4"
    if not (root / "uncertainty_selection.json").is_file():
        pytest.skip("local C4 uncertainty selection is unavailable")
    value = json.loads((root / "uncertainty_selection.json").read_text())
    assert value["test_loaded"] is False
    assert value["common_scale_across_modes"] is True
    assert value["high_error_threshold_definition"] == "selected canonical fallback validation shared-error 75th percentile"
    assert value["mean_predictors_frozen"] is True
    assert value["parameters"] <= 50_000


@pytest.mark.parametrize("mode", ["FULL_AH", "FALLBACK_VA", "FALLBACK_A"])
def test_locked_uncertainty_is_nonconstant_correlated_and_beats_constant_nll(mode):
    value = locked()["uncertainty"][mode]
    assert value["spearman"] > 0 and value["spearman_ci95"][0] > 0
    assert value["uncertainty_std"] > 0
    assert value["nll"] < value["constant_variance_nll"]
    assert value["risk_coverage"]["top20_removal_reduction"] > 0


def test_canonical_fallback_high_error_and_risk_coverage_gates_pass():
    value = locked()["uncertainty"]["FALLBACK_VA"]
    assert value["auroc"] >= 0.60
    assert value["risk_coverage"]["top20_removal_reduction"] >= 0.10


def test_fallback_uncertainty_exceeds_full_with_positive_bootstrap_ci():
    value = locked()["availability_sensitivity"]
    assert value["mean_fallback"] > value["mean_full"]
    assert value["difference_ci95"][0] > 0
    assert value["gate"] is True


def test_dynamic_and_both_boundaries_are_reported_for_every_mode():
    value = locked()["boundary_uncertainty"]
    assert set(value) == {"FULL_AH", "FALLBACK_VA", "FALLBACK_A"}
    for metrics in value.values():
        assert set(metrics) == {"static", "dynamic", "free_to_contact", "contact_to_free"}
        assert all(item > 0 for item in metrics.values())


def test_corrupt_h_diagnostics_report_error_and_uncertainty_without_detector():
    value = locked()["availability_corruption"]
    assert set(value) == {"wrong_time", "different_episode", "noisy"}
    assert all(row["shared_mse"] > 0 and row["mean_uncertainty"] > 0 for row in value.values())


def test_all_uncertainty_hard_gates_pass():
    assert all(locked()["uncertainty_gates"].values())
