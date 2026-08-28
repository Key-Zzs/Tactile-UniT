import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch

from gr00t.tactile_unit.c4_availability_conditioning import (
    AvailabilityMode, AvailabilityRouter, ContactFallbackPredictor,
    ModalityAvailability, route_availability,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/tactile_unit/c4_missing_modality_uncertainty.json"


def config():
    return json.loads(CONFIG.read_text())


def artifacts():
    root = ROOT / ".local/artifacts/tactile_unit/vac_c4"
    if not (root / "locked_test_evaluation.json").is_file():
        pytest.skip("local C4 artifacts unavailable")
    return root


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "vision,action,contact,mode",
    [
        (True, True, True, AvailabilityMode.FULL_AH),
        (False, True, True, AvailabilityMode.FULL_AH),
        (True, True, False, AvailabilityMode.FALLBACK_VA),
        (False, True, False, AvailabilityMode.FALLBACK_A),
        (True, False, True, AvailabilityMode.ABSTAIN_NO_ACTION),
        (True, False, False, AvailabilityMode.ABSTAIN_NO_ACTION),
        (False, False, True, AvailabilityMode.ABSTAIN_NO_ACTION),
        (False, False, False, AvailabilityMode.ABSTAIN_NO_ACTION),
    ],
)
def test_explicit_router_truth_table(vision, action, contact, mode):
    assert route_availability(ModalityAvailability(vision, action, contact)) is mode


def test_availability_flags_must_be_explicit_bools():
    with pytest.raises(TypeError, match="explicit bool"):
        ModalityAvailability(1, True, True)


def test_zero_tensor_content_never_changes_declared_availability():
    availability = ModalityAvailability(False, True, True)
    assert route_availability(availability) is AvailabilityMode.FULL_AH


def test_fallback_sources_and_output_shape_are_isolated():
    action = torch.randn(3, 8, 32)
    vision = torch.randn(3, 8, 32)
    a = ContactFallbackPredictor("A")
    va = ContactFallbackPredictor("VA")
    assert a(action).shape == (3, 8, 32)
    assert va(action, vision).shape == (3, 8, 32)
    with pytest.raises(ValueError, match="forbids Vision"):
        a(action, vision)
    with pytest.raises(ValueError, match="requires explicit u_v"):
        va(action)


def test_a_and_va_architectures_have_parameter_parity():
    assert ContactFallbackPredictor("A").parameter_summary()["total"] == ContactFallbackPredictor("VA").parameter_summary()["total"]


def test_fallback_interface_has_no_contact_or_private_inputs():
    parameters = set(inspect.signature(ContactFallbackPredictor.forward).parameters)
    assert parameters == {"self", "u_a", "u_v"}
    assert parameters.isdisjoint({"h_current", "h_future", "u_c", "z_c", "r_c_priv", "pair_id"})


def test_router_abstains_without_action_and_returns_no_prediction():
    a, va = ContactFallbackPredictor("A"), ContactFallbackPredictor("VA")
    router = AvailabilityRouter(lambda action, contact: action, va, a)
    result = router.predict(ModalityAvailability(True, False, True))
    assert result.mode is AvailabilityMode.ABSTAIN_NO_ACTION
    assert result.prediction_available is False
    assert result.u_hat_c is None and result.uncertainty is None


def test_full_route_ignores_vision_tensor():
    calls = []
    def full(action, context):
        calls.append((action, context))
        return action
    action, context = torch.randn(2, 8, 32), torch.randn(2, 256)
    router = AvailabilityRouter(full, ContactFallbackPredictor("VA"), ContactFallbackPredictor("A"))
    first = router.predict(ModalityAvailability(True, True, True), u_a=action, h_current=context, u_v=torch.ones_like(action))
    second = router.predict(ModalityAvailability(False, True, True), u_a=action, h_current=context)
    assert torch.equal(first.u_hat_c, second.u_hat_c)
    assert len(calls) == 2


def test_config_freezes_scope_architecture_trials_and_splits():
    value = config()
    assert value["counts"] == {"train": 65536, "validation": 8192, "test": 17504}
    assert len(value["training"]["trials"]) == 5 <= value["architecture"]["maximum_trials"] <= 6
    assert [row["id"] for row in value["training"]["trials"]] == ["T0", "T1", "T2", "T3", "T4"]
    assert value["scope"]["c5_started"] is False
    assert value["scope"]["private_residual_prediction"] is False
    assert value["scope"]["full_path_retuning"] is False


def test_full_checkpoint_exact_sha_is_preregistered_and_present():
    value = config()
    path = ROOT / value["runtime"]["full_checkpoint"]
    assert sha(path) == value["accepted"]["full_checkpoint_sha256"] == "862d5bef53dc027a34212cdd22e82f8c5e07896c53ecc275580266fa5c8b469e"


def test_all_selection_artifacts_are_validation_only_hashed_and_pretest():
    root = artifacts()
    for stem in ("fallback_selection", "uncertainty_selection"):
        path = root / f"{stem}.json"
        assert sha(path) == (root / f"{stem}.sha256").read_text().split()[0]
        value = json.loads(path.read_text())
        assert value["test_loaded"] is False
        assert value["selected_via"] == "VALIDATION ONLY"


def test_fallback_selection_uses_va_and_exactly_five_bounded_trials():
    root = artifacts()
    selection = json.loads((root / "fallback_selection.json").read_text())
    summary = json.loads((root / "fallback_training_summary.json").read_text())
    assert selection["source"] == "VA" and selection["candidate"] == "T4"
    assert summary["total_trials"] == 5 <= summary["maximum_trials"] <= 6


def test_locked_fallback_passes_semantic_force_latent_physics_temporal_and_misuse_gates():
    locked = json.loads((artifacts() / "locked_test_evaluation.json").read_text())
    assert locked["rows"] == 17504 and locked["selection_frozen_before_test"] is True
    assert all(locked["fallback_gates"].values())
    va = locked["fallbacks"]["VA"]
    assert va["semantics"]["contact_transition"]["semantic_ratio"] >= 0.50
    assert va["semantics"]["force_trend_class"]["semantic_ratio"] >= 0.65
    assert va["action_temporal"]["exact_ar_transform"] is True
    assert va["action_temporal"]["variants"]["reversed"]["difference_ci95"][0] > 0
    assert va["action_temporal"]["variants"]["shuffled"]["difference_ci95"][0] > 0
    assert va["physics"]["teacher_side_h_only"] is True


def test_vision_materially_improves_missing_h_fallback_with_positive_cis():
    value = json.loads((artifacts() / "locked_test_evaluation.json").read_text())["vision_incremental"]
    assert value["classification"] == "VISION_MATERIALLY_IMPROVES_MISSING_H_FALLBACK"
    assert value["bootstrap_ci95"]["contact_transition"][0] > 0
    assert value["bootstrap_ci95"]["force_trend_class"][0] > 0


def test_full_path_is_identity_preserved_and_numerically_reproduced():
    value = json.loads((artifacts() / "locked_test_evaluation.json").read_text())
    full = value["full_nonregression"]
    assert full["pass"] is True and full["checkpoint_identity"] is True
    assert abs(full["contact_ratio"] - 0.900316) < 0.002
    assert abs(full["force_ratio"] - 0.794947) < 0.002
    assert full["exact_action"] and full["h_context"] and full["physics"]
    assert value["identity_before"]["actual"] == value["identity_after"]["actual"]


def test_private_residual_is_never_a_fallback_source_or_target():
    source = (ROOT / "gr00t/tactile_unit/c4_availability_conditioning.py").read_text()
    assert '"r_c_priv"' in source
    assert set(inspect.signature(ContactFallbackPredictor.forward).parameters).isdisjoint({"r_c_priv", "z_c", "u_c"})


def test_locked_repeated_mean_inference_is_exact():
    value = json.loads((artifacts() / "locked_test_evaluation.json").read_text())
    assert value["repeated_evaluation_exact"] is True
    assert all(value["repeat_checks"].values())


def test_final_decision_stops_before_c5_c6_and_m3():
    value = json.loads((artifacts() / "final_decision.json").read_text())
    assert value["decision"] == "C4_READY_VA_FALLBACK"
    assert value["c5_readiness"] == "READY"
    assert value["c5"] == "NOT STARTED"
    assert value["c6_m3"] == "NOT STARTED"
    assert value["m3"] == "NOT ESTABLISHED"
