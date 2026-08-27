import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gr00t.tactile_unit.c3dp_shared_private import (
    ORDERED_DIRECTIONS,
    SHORT_DIRECTION,
    SharedCrossModalPredictor,
    load_predictor_checkpoint,
    output_geometry_gate,
    save_predictor_checkpoint,
)
from gr00t.tactile_unit.continuous_vac_shared_space import retrieval_metrics
from scripts.tactile_unit.audit_c3dp_dual_path import (
    apply_ridge,
    fit_ridge,
    regression_metrics,
)
from scripts.tactile_unit.evaluate_c3dp_cross_prediction import (
    per_sample_mse,
    r2_score,
    subset_means,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "candidate,kwargs",
    [
        ("P0", {"hidden_dim": 32, "attention_layers": 1, "heads": 1}),
        ("P1", {"hidden_dim": 128, "attention_layers": 1, "heads": 1}),
        ("P2", {"hidden_dim": 64, "attention_layers": 2, "heads": 4}),
    ],
)
def test_all_bounded_candidates_output_required_shape(candidate, kwargs):
    model = SharedCrossModalPredictor(candidate, **kwargs)
    output = model(torch.randn(5, 8, 32), "vision", "contact")
    assert output.shape == (5, 8, 32)
    assert torch.isfinite(output).all()
    assert model.parameter_summary()["total"] < 100_000


def test_one_shared_predictor_routes_all_six_ordered_pairs():
    model = SharedCrossModalPredictor("P1")
    value = torch.randn(3, 8, 32)
    observed = {}
    for source, target in ORDERED_DIRECTIONS:
        observed[SHORT_DIRECTION[(source, target)]] = model(value, source, target)
    assert set(observed) == {"V->A", "A->V", "V->C", "C->V", "A->C", "C->A"}
    assert all(output.shape == (3, 8, 32) for output in observed.values())


def test_source_and_target_must_differ():
    model = SharedCrossModalPredictor("P0")
    with pytest.raises(ValueError, match="target != source"):
        model(torch.randn(2, 8, 32), "vision", "vision")


def test_target_condition_is_identity_only_not_target_sample():
    signature = inspect.signature(SharedCrossModalPredictor.forward)
    assert list(signature.parameters) == [
        "self",
        "source_shared_tokens",
        "source_modality_id",
        "target_modality_id",
    ]
    assert "actual_target" not in inspect.getsource(SharedCrossModalPredictor.forward)


def test_pair_id_cannot_be_passed_as_feature():
    model = SharedCrossModalPredictor("P0")
    with pytest.raises(TypeError):
        model(torch.randn(2, 8, 32), "vision", "action", pair_id=torch.arange(2))


def test_p2_attention_memory_is_source_only():
    source = inspect.getsource(SharedCrossModalPredictor.forward)
    assert "memory = source_shared_tokens + source_condition" in source
    assert "target_slots" in source
    assert "actual_target" not in source


def test_direction_balancing_is_exactly_six_equal_terms():
    source = (ROOT / "gr00t/tactile_unit/c3dp_shared_private.py").read_text()
    assert "for source, target in ORDERED_DIRECTIONS:" in source
    assert "averaged = {name: torch.stack(value).mean()" in source


def test_mean_shuffled_different_episode_and_wrong_time_controls_exist():
    source = (ROOT / "scripts/tactile_unit/evaluate_c3dp_cross_prediction.py").read_text()
    for control in ("mean", "shuffled_source", "different_episode", "same_episode_wrong_time"):
        assert f'"{control}"' in source


def test_prediction_mse_and_cosine_gate_formulas_are_paired():
    prediction = np.asarray([[[1.0]], [[3.0]]])
    target = np.asarray([[[2.0]], [[1.0]]])
    assert np.array_equal(per_sample_mse(prediction, target), np.asarray([1.0, 4.0]))
    source = (ROOT / "scripts/tactile_unit/evaluate_c3dp_cross_prediction.py").read_text()
    assert "improvement = controls[strongest] - error" in source
    assert "margin = cosine_true - cosine_shuffled" in source
    assert "improvement_ci[0] > 0" in source
    assert "margin_ci[0] > 0" in source


def test_retrieval_uses_independent_target_candidate_set():
    rng = np.random.default_rng(7)
    target = rng.normal(size=(12, 8, 32)).astype(np.float32)
    metrics = retrieval_metrics(target.copy(), target, chunk=5)
    assert metrics["recall_at_1"] == 1.0
    source = inspect.getsource(retrieval_metrics)
    assert "candidate = numpy_flatten_normalize(right)" in source


def test_dynamic_and_boundary_subsets_are_explicit():
    value = np.arange(6, dtype=np.float64)
    result = subset_means(
        value,
        np.asarray([0, 1, 0, 1, 0, 1], dtype=bool),
        np.asarray([0, 1, 2, 3, 0, 1]),
    )
    assert result["dynamic"]["count"] == 3
    assert result["rare_boundary"]["count"] == 3
    assert result["free_to_contact"]["count"] == 2
    assert result["contact_to_free"]["count"] == 1


def test_private_residual_ridge_is_train_fitted_and_deterministic():
    rng = np.random.default_rng(11)
    source = rng.normal(size=(50, 8, 32)).astype(np.float32)
    target = (0.25 * source).astype(np.float32)
    first = fit_ridge(source, target, 1.0)
    second = fit_ridge(source, target, 1.0)
    assert np.array_equal(first["coefficient"], second["coefficient"])
    prediction = apply_ridge(first, source)
    assert regression_metrics(prediction, target)["r2"] > 0.99


def test_r2_formula_matches_perfect_and_mean_prediction():
    target = np.arange(10, dtype=np.float64)
    assert r2_score(target, target) == pytest.approx(1.0)
    assert r2_score(target, np.full_like(target, target.mean())) == pytest.approx(0.0)


def test_predicted_output_noncollapse_gate():
    rng = np.random.default_rng(9)
    rich = rng.normal(size=(128, 8, 32)).astype(np.float32)
    constant = np.ones((128, 8, 32), dtype=np.float32)
    _, rich_pass = output_geometry_gate(rich)
    _, constant_pass = output_geometry_gate(constant)
    assert rich_pass
    assert not constant_pass


def test_predictor_checkpoint_reload_is_exact(tmp_path):
    torch.manual_seed(12)
    model = SharedCrossModalPredictor("P2", hidden_dim=64, attention_layers=2, heads=4).eval()
    value = torch.randn(4, 8, 32)
    expected = model(value, "action", "vision")
    path = tmp_path / "predictor.pt"
    digest = save_predictor_checkpoint(
        path, model, {"selection_split": "validation only", "test_loaded": False}
    )
    loaded, metadata = load_predictor_checkpoint(path)
    actual = loaded.eval()(value, "action", "vision")
    assert len(digest) == 64
    assert metadata["test_loaded"] is False
    assert torch.equal(expected, actual)


def test_evaluator_records_deterministic_locked_protocol_and_causal_boundary():
    source = (ROOT / "scripts/tactile_unit/evaluate_c3dp_cross_prediction.py").read_text()
    assert '"evaluation": "LOCKED BENCHMARK RE-EVALUATION"' in source
    assert '"first_look_untouched": False' in source
    assert '"future_teacher_exposed_as_runtime_observation": False' in source
    assert '"C5_started": False' in source


def test_no_private_paths_in_predictor_or_canonical_loss():
    source = inspect.getsource(SharedCrossModalPredictor) + inspect.getsource(
        __import__(
            "gr00t.tactile_unit.c3dp_shared_private", fromlist=["cross_modal_prediction_loss"]
        ).cross_modal_prediction_loss
    )
    assert "r_c_priv" not in source
    assert "private_residual" not in source


def test_contact_cross_semantics_fit_same_family_probes_and_retention_formula():
    source = (ROOT / "scripts/tactile_unit/evaluate_c3dp_cross_prediction.py").read_text()
    assert 'train_representations[key]["predicted"], train[label]' in source
    assert 'predicted_probe, value["prediction"], target' in source
    assert 'retention = (float(predicted["macro_f1"]) - majority_f1)' in source
    assert "contact_cross_retention_min" in source


def test_contact_shared_physics_uses_frozen_decoder_and_controls():
    source = inspect.getsource(
        __import__(
            "scripts.tactile_unit.evaluate_c3dp_cross_prediction",
            fromlist=["contact_physics"],
        ).contact_physics
    )
    assert "prediction = decoder(z, current)" in source
    assert 'value["shared_oracle_mse"]' in source
    evaluator = (ROOT / "scripts/tactile_unit/evaluate_c3dp_cross_prediction.py").read_text()
    assert 'for key in ("V->C", "A->C")' in evaluator


def test_action_target_evaluation_recovers_and_decodes_without_target_context():
    source = (ROOT / "scripts/tactile_unit/evaluate_c3dp_cross_prediction.py").read_text()
    assert 'recover_numpy(shared_space, "action", shared' in source
    assert "zero_action = torch.zeros" in source
    assert "action_model.decode(z, state_features, embodiment)" in source
    assert "no target action chunk" in source


def test_vision_target_evaluation_uses_frozen_recovery():
    source = inspect.getsource(
        __import__(
            "scripts.tactile_unit.evaluate_c3dp_cross_prediction",
            fromlist=["vision_evaluation"],
        ).vision_evaluation
    )
    assert 'recover_numpy(shared_space, "vision", shared' in source
    assert 'target = np.asarray(split.arrays["z_v"])' in source
    assert "linear_cka(recovered, target)" in source


def test_action_and_contact_source_reversal_is_evaluated():
    source = inspect.getsource(
        __import__(
            "scripts.tactile_unit.evaluate_c3dp_cross_prediction",
            fromlist=["source_perturbations"],
        ).source_perturbations
    )
    assert "action.flip(1)" in source
    assert "reversed_contact_shared" in source
    assert '"reversed_mse"' in source
    assert '"different_episode_mse"' in source


def test_frozen_selection_lock_precedes_test_and_hashes_exactly():
    artifact = ROOT / ".local/artifacts/tactile_unit/vac_c3dp"
    selection = json.loads((artifact / "selection.json").read_text())
    digest = (artifact / "selection.sha256").read_text().split()[0]
    from gr00t.tactile_unit.c3dp_shared_private import sha256_file

    assert selection["test_loaded"] is False
    assert sha256_file(artifact / "selection.json") == digest
    assert sha256_file(ROOT / selection["checkpoint"]) == selection["checkpoint_sha256"]


def test_locked_evaluation_keeps_native_and_causal_identities():
    evaluation = json.loads(
        (ROOT / ".local/artifacts/tactile_unit/vac_c3dp/locked_test_evaluation.json").read_text()
    )
    assert evaluation["integrity"]["shared_space_unchanged"] is True
    assert evaluation["integrity"]["native_unchanged"] is True
    assert evaluation["causal_boundary"]["future_teacher_exposed_as_runtime_observation"] is False
    assert evaluation["causal_boundary"]["online_legal_current_tactile"] == "h_t^c"
    assert evaluation["scope"]["C4"] == "NOT STARTED"
    assert evaluation["scope"]["C5"] == "NOT STARTED"


def test_locked_decision_is_registered_and_consistent_with_gates():
    evaluation = json.loads(
        (ROOT / ".local/artifacts/tactile_unit/vac_c3dp/locked_test_evaluation.json").read_text()
    )
    registered = {
        "C3DP_SHARED_CROSS_PREDICTION_READY",
        "C3DP_SHARED_CROSS_PREDICTION_READY_WITH_PRIVATE_WARNING",
        "C3DP_CROSS_PREDICTION_INSUFFICIENT",
        "C3DP_SHARED_SEMANTIC_LOSS",
        "STRUCTURAL_FAIL",
    }
    assert evaluation["decision"] in registered
    assert evaluation["gates"]["structural"] is True
    assert evaluation["all_six_prediction_gates"] is True
    assert evaluation["contact_semantics"]["gate"] is False
    assert evaluation["decision"] == "C3DP_SHARED_SEMANTIC_LOSS"


def test_new_tracked_sources_do_not_contain_machine_private_paths():
    relative_paths = (
        "configs/tactile_unit/c3dp_shared_private_cross_prediction.json",
        "gr00t/tactile_unit/c3dp_shared_private.py",
        "scripts/tactile_unit/c3dp_runtime.py",
        "scripts/tactile_unit/build_c3dp_cache.py",
        "scripts/tactile_unit/audit_c3dp_dual_path.py",
        "scripts/tactile_unit/train_c3dp_cross_prediction.py",
        "scripts/tactile_unit/evaluate_c3dp_cross_prediction.py",
        "scripts/tactile_unit/visualize_c3dp_cross_prediction.py",
        "docs/research/track_c_c3dp_shared_private_prediction.md",
    )
    for relative in relative_paths:
        source = (ROOT / relative).read_text()
        assert "/" + "home/" not in source
        assert "/" + "mnt/" not in source
