import inspect
from pathlib import Path

import numpy as np
import torch

from gr00t.tactile_unit.c3dp_shared_private import (
    ACCEPTED_C2R_CHECKPOINT_SHA256,
    C3DPLossWeights,
    SharedCrossModalPredictor,
    cross_modal_prediction_loss,
    decompose_contact,
    dual_path_numpy_audit,
    freeze_shared_space,
    private_geometry,
    sha256_file,
    verify_c2r_checkpoint,
)
from gr00t.tactile_unit.continuous_vac_shared_space import ContinuousVACSharedSpace

ROOT = Path(__file__).resolve().parents[2]


def test_c2r_checkpoint_sha_is_exact():
    path = ROOT / ".local/experiments/tactile_unit/vac_c2r/selected.pt"
    assert verify_c2r_checkpoint(path) == ACCEPTED_C2R_CHECKPOINT_SHA256
    assert sha256_file(path) == ACCEPTED_C2R_CHECKPOINT_SHA256


def test_c1_manifest_is_the_accepted_frozen_manifest():
    path = ROOT / ".local/cache/tactile_unit/vac_c1/manifest.json"
    assert sha256_file(path) == "dd2657fca1e53987fe74dba6defdf81ced2dd17b12946404136aa229dcf17040"


def test_dual_path_is_arithmetic_identity_and_has_required_shape():
    torch.manual_seed(2)
    model = ContinuousVACSharedSpace("C2-slot").eval()
    native = torch.randn(7, 8, 32)
    value = decompose_contact(model, native)
    assert value.shared.shape == (7, 8, 32)
    assert value.shared_native.shape == (7, 8, 32)
    assert value.private_residual.shape == (7, 8, 32)
    assert torch.allclose(value.shared_native + value.private_residual, native, atol=1e-6, rtol=0.0)
    audit = dual_path_numpy_audit(native.numpy(), value.shared_native.detach().numpy())
    assert audit["max_abs_error"] <= 1e-6
    assert audit["mse"] <= 1e-12


def test_shared_and_private_have_no_storage_alias():
    model = ContinuousVACSharedSpace("C2-slot").eval()
    value = decompose_contact(model, torch.randn(3, 8, 32))
    assert value.shared.data_ptr() != value.private_residual.data_ptr()
    assert value.shared_native.data_ptr() != value.private_residual.data_ptr()


def test_all_c2r_parameters_are_frozen_and_eval():
    model = ContinuousVACSharedSpace("C2-slot")
    result = freeze_shared_space(model)
    assert result["trainable_count"] == 0
    assert result["training"] is False
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_predictor_only_gradients_and_no_shared_space_mutation():
    torch.manual_seed(5)
    shared_space = ContinuousVACSharedSpace("C2-slot")
    boundary = freeze_shared_space(shared_space)
    predictor = SharedCrossModalPredictor("P1")
    shared = {name: torch.randn(12, 8, 32) for name in ("vision", "action", "contact")}
    loss, metrics = cross_modal_prediction_loss(
        predictor,
        shared_space,
        shared,
        torch.arange(12) % 2 == 0,
        dynamic_weight=2.0,
        weights=C3DPLossWeights(),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert len([name for name in metrics if name.startswith("direction/")]) == 6
    assert any(parameter.grad is not None for parameter in predictor.parameters())
    assert all(parameter.grad is None for parameter in shared_space.parameters())
    assert freeze_shared_space(shared_space)["state_dict_sha256"] == boundary["state_dict_sha256"]


def test_contact_canonical_target_is_shared_not_native_or_private():
    signature = inspect.signature(cross_modal_prediction_loss)
    assert "private_residual" not in signature.parameters
    assert "native_contact" not in signature.parameters
    source = inspect.getsource(cross_modal_prediction_loss)
    assert "oracle = shared[target].detach()" in source
    assert "private" not in source


def test_private_residual_is_never_predictor_input():
    signature = inspect.signature(SharedCrossModalPredictor.forward)
    assert list(signature.parameters) == [
        "self",
        "source_shared_tokens",
        "source_modality_id",
        "target_modality_id",
    ]
    assert "private" not in inspect.getsource(SharedCrossModalPredictor.forward)


def test_private_geometry_reports_required_fields():
    rng = np.random.default_rng(4)
    native = rng.normal(size=(20, 8, 32)).astype(np.float32)
    shared = rng.normal(size=native.shape).astype(np.float32)
    recovered = rng.normal(size=native.shape).astype(np.float32)
    private = native - recovered
    result = private_geometry(native, shared, recovered, private)
    assert result["effective_rank"] > 0
    assert result["norm_mean"] > 0
    assert result["energy_fraction_of_native"] > 0
    assert "cka_with_native_z_c" in result
    assert "cka_with_shared_u_c" in result


def test_train_and_audit_sources_never_load_locked_test():
    for relative in (
        "scripts/tactile_unit/train_c3dp_cross_prediction.py",
        "scripts/tactile_unit/audit_c3dp_dual_path.py",
    ):
        source = (ROOT / relative).read_text()
        assert 'load_split(c1_root, "test"' not in source
        assert 'load_derived_split(cache_root, "test"' not in source
        assert '"test_loaded": False' in source


def test_test_cache_requires_frozen_selection_lock():
    source = (ROOT / "scripts/tactile_unit/build_c3dp_cache.py").read_text()
    assert 'if "test" in args.splits:' in source
    assert "validate_selection_lock(config)" in source
    evaluation = (ROOT / "scripts/tactile_unit/evaluate_c3dp_cross_prediction.py").read_text()
    assert evaluation.index("selection = validate_selection_lock(config)") < evaluation.index(
        'build_split(config, "test"'
    )


def test_c4_additions_are_bounded_and_no_causal_student_was_added():
    tracked = sorted(path.name for path in (ROOT / "gr00t/tactile_unit").glob("*c4*"))
    assert tracked == ["c4_availability_conditioning.py", "c4_uncertainty.py"]
    assert list((ROOT / "gr00t/tactile_unit").glob("*c5*")) == []
    assert list((ROOT / "gr00t/tactile_unit").glob("*causal_student*")) == []
    config = (ROOT / "configs/tactile_unit/c3dp_shared_private_cross_prediction.json").read_text()
    assert '"c4_started": false' in config
    assert '"c5_started": false' in config
