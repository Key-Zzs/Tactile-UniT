import inspect
import json
import subprocess
from pathlib import Path

import pytest
import torch

from gr00t.tactile_unit.c3mscc_contact_context import (
    C3MSCCLossWeights,
    ContactContextPredictor,
    FORBIDDEN_INPUTS,
    SOURCE_COMPONENTS,
    contact_prediction_loss,
    load_checkpoint,
    save_checkpoint,
    sha256_file,
)
from gr00t.tactile_unit.continuous_vac_shared_space import ContinuousVACSharedSpace, state_dict_digest
from gr00t.contact_dynamics.models import LatentTransitionDecoder
from scripts.tactile_unit.c3mscc_runtime import load_config, validate_selection_lock

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/tactile_unit/c3mscc_contact_context_prediction.json"


def config():
    return json.loads(CONFIG.read_text())


def batch(size=6):
    return {
        "u_v": torch.randn(size, 8, 32),
        "u_a": torch.randn(size, 8, 32),
        "u_c": torch.randn(size, 8, 32),
        "h_current": torch.randn(size, 256),
        "dynamic": torch.arange(size) % 2 == 0,
    }


@pytest.mark.parametrize(("source", "tokens"), [("AH", 16), ("VAH", 24)])
def test_source_tokenization_and_prediction_shapes(source, tokens):
    value = batch()
    model = ContactContextPredictor(source)
    vision = value["u_v"] if source == "VAH" else None
    assert model.source_tokens(value["u_a"], value["h_current"], vision).shape == (6, tokens, 32)
    assert model(value["u_a"], value["h_current"], vision).shape == (6, 8, 32)


def test_ah_rejects_vision_and_vah_requires_it():
    value = batch()
    with pytest.raises(ValueError):
        ContactContextPredictor("AH")(value["u_a"], value["h_current"], value["u_v"])
    with pytest.raises(ValueError):
        ContactContextPredictor("VAH")(value["u_a"], value["h_current"])


def test_predictor_interface_has_no_target_future_private_or_identity():
    signature = set(inspect.signature(ContactContextPredictor.forward).parameters)
    assert signature == {"self", "u_a", "h_current", "u_v"}
    assert not signature & FORBIDDEN_INPUTS
    for parts in SOURCE_COMPONENTS.values():
        assert not set(parts) & FORBIDDEN_INPUTS


def test_architecture_is_bounded_and_under_parameter_target():
    value = config()
    for source in ("AH", "VAH"):
        model = ContactContextPredictor(source)
        assert model.parameter_summary()["total"] < value["architecture"]["maximum_parameters"]
        assert model.block_count <= 2
        assert model.heads <= 4
        assert model.mlp_width <= 128


def test_predictor_only_gradients_with_frozen_physics_path():
    value = batch(8)
    predictor = ContactContextPredictor("VAH")
    shared = ContinuousVACSharedSpace("C2-slot").eval().requires_grad_(False)
    decoder = LatentTransitionDecoder().eval().requires_grad_(False)
    before_shared = state_dict_digest(shared)
    before_decoder = state_dict_digest(decoder)
    loss, terms = contact_prediction_loss(
        predictor, shared, decoder,
        u_a=value["u_a"], h_current=value["h_current"], u_c=value["u_c"],
        dynamic=value["dynamic"], u_v=value["u_v"],
        invalid_u_a=(value["u_a"].flip(1), value["u_a"].roll(1, 0)),
        enhanced=True, dynamic_weight=2.0, order_margin=0.02,
        variance_floor=0.1, weights=C3MSCCLossWeights(),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert set(terms) == {"shared", "cosine", "relational", "physics", "delta", "covariance", "order", "total"}
    assert any(parameter.grad is not None for parameter in predictor.parameters())
    assert all(parameter.grad is None for parameter in shared.parameters())
    assert all(parameter.grad is None for parameter in decoder.parameters())
    assert state_dict_digest(shared) == before_shared
    assert state_dict_digest(decoder) == before_decoder


def test_checkpoint_reload_is_deterministic(tmp_path):
    torch.manual_seed(2)
    model = ContactContextPredictor("AH").eval()
    value = batch(3)
    before = model(value["u_a"], value["h_current"])
    path = tmp_path / "model.pt"
    digest = save_checkpoint(path, model, {"test_loaded": False})
    loaded, metadata = load_checkpoint(path)
    after = loaded.eval()(value["u_a"], value["h_current"])
    assert sha256_file(path) == digest
    assert metadata["test_loaded"] is False
    assert torch.equal(before, after)


def test_config_freezes_trials_selection_gpu_and_scope():
    value = config()
    assert value["counts"] == {"train": 65536, "validation": 8192, "test": 17504}
    assert [row["id"] for row in value["training"]["trials"]] == ["T0", "T1", "T2", "T3"]
    assert value["training"]["maximum_trials"] == 6
    assert value["validation"]["selection_split"] == "validation only"
    assert value["validation"]["simplicity_tolerance"] == 0.01
    assert value["gpu"]["allowed_physical"] == [1, 2, 3]
    assert value["gpu"]["forbidden_physical"] == [0]
    assert "explicit user authorization" in value["gpu"]["gpu1_authorization"]
    assert value["scope"]["c4_started"] is False
    assert value["scope"]["c5_started"] is False
    assert value["scope"]["m3_established"] is False


def test_pretest_commands_never_load_locked_test():
    for relative in (
        "scripts/tactile_unit/audit_c3mscc_contract.py",
        "scripts/tactile_unit/train_c3mscc_contact_prediction.py",
    ):
        path = ROOT / relative
        if path.is_file():
            source = path.read_text()
            assert 'load_aligned_split(config, "test")' not in source


def test_all_accepted_stage_file_identities_are_exact():
    value = config()
    paths = {
        "c1_manifest_sha256": ".local/cache/tactile_unit/vac_c1/manifest.json",
        "c2_checkpoint_sha256": ".local/experiments/tactile_unit/vac_c2/selected.pt",
        "c2r_checkpoint_sha256": ".local/experiments/tactile_unit/vac_c2r/selected.pt",
        "c3dp_checkpoint_sha256": ".local/experiments/tactile_unit/vac_c3dp/selected.pt",
        "action_checkpoint_sha256": ".local/experiments/tactile_unit/s3_3_r/selected.pt",
        "s1_checkpoint_sha256": ".local/experiments/tactile_teacher/s1_teacher/best.pt",
        "s2_checkpoint_sha256": ".local/experiments/contact_dynamics/s2_models/proposed_best.pt",
    }
    if not all((ROOT / path).is_file() for path in paths.values()):
        pytest.skip("local accepted stage artifacts are unavailable")
    assert all(sha256_file(ROOT / path) == value["accepted"][name] for name, path in paths.items())


def test_selection_is_hashed_validation_only_and_pretest():
    if not (ROOT / ".local/artifacts/tactile_unit/vac_c3mscc/selection.json").is_file():
        pytest.skip("local C3-MS-CC selection artifact is unavailable")
    selection = validate_selection_lock(load_config())
    assert selection["selected_via"] == "VALIDATION ONLY"
    assert selection["selection_split"] == "validation only"
    assert selection["test_loaded"] is False
    assert selection["source"] == "VAH"
    assert selection["trial"] == "T1"


def test_selection_reducer_implements_exact_ah_else_vah_rule():
    source = (ROOT / "scripts/tactile_unit/freeze_c3mscc_selection.py").read_text()
    assert 'best_ah["best"]["validation"]["gates"]["all"]' in source
    assert '>= best_utility - tolerance' in source
    assert 'row["trial"]["source"] == "VAH"' in source
    assert "best validation-only V+A+H trial" in source


def test_locked_evaluator_validates_selection_before_test_load():
    source = (ROOT / "scripts/tactile_unit/evaluate_c3mscc_contact_prediction.py").read_text()
    assert source.index("selection = validate_selection_lock(config)") < source.index(
        'test = load_aligned_split(config, "test")'
    )


def locked():
    path = ROOT / ".local/artifacts/tactile_unit/vac_c3mscc/locked_test_evaluation.json"
    if not path.is_file():
        pytest.skip("local C3-MS-CC locked evaluation is unavailable")
    return json.loads(path.read_text())


def test_locked_protocol_row_count_and_status_are_exact():
    value = locked()
    assert value["rows"] == 17504
    assert value["first_look_untouched"] is False
    assert value["evaluation_type"] == "LOCKED BENCHMARK RE-EVALUATION AFTER C3-R0 DIAGNOSIS"
    assert value["test_loaded"] is True


@pytest.mark.parametrize("source", ["AH", "VAH"])
def test_both_required_sources_have_complete_semantic_reporting(source):
    value = locked()["sources"][source]
    contact = value["semantics"]["contact_transition"]
    force = value["semantics"]["force_trend_class"]
    assert contact["semantic_ratio"] >= 0.75
    assert force["semantic_ratio"] >= 0.75
    assert set(contact["per_class"]) == {"0", "1", "2", "3"}
    assert "future_change" in contact
    assert "free_to_contact" in contact
    assert "contact_to_free" in contact


@pytest.mark.parametrize("source", ["AH", "VAH"])
def test_shared_target_cosine_and_retrieval_gates_are_reported(source):
    value = locked()["sources"][source]["shared_target"]
    assert value["improvement_ci95"][0] > 0
    assert value["cosine_margin_ci95"][0] > 0
    assert value["retrieval"]["recall_at_10"] >= 1.5 * value["retrieval"]["chance"]["recall_at_10"]
    assert value["gate"] is True


@pytest.mark.parametrize("source", ["AH", "VAH"])
def test_h_context_controls_and_contact_evidence_are_complete(source):
    value = locked()["sources"][source]["h_context"]
    assert set(value["controls_mse"]) == {"zero", "mean", "different_episode", "wrong_time", "time_shuffled"}
    assert value["improvement_ci95"][0] > 0
    assert value["correct_contact_f1"] > max(
        item["macro_f1"] for item in value["invalid_semantics"].values()
    )
    assert value["gate"] is True


@pytest.mark.parametrize("source", ["AH", "VAH"])
def test_action_temporal_gate_fails_closed_without_exact_ar_transform(source):
    value = locked()["sources"][source]["action_temporal"]
    assert value["exact_ar_transform"] is False
    assert set(value["variants"]) == {"reversed_surrogate", "shuffled", "different_episode"}
    assert value["gate"] is False
    assert "exact frozen A-R" in value["method"]


def test_shared_physics_beats_controls_for_ah_but_not_selected_vah():
    value = locked()["sources"]
    assert value["AH"]["physics"]["gate"] is True
    assert value["VAH"]["physics"]["gate"] is False
    assert "dynamic_improvement_ci95" in value["AH"]["physics"]
    assert "native_future_mse" in value["VAH"]["physics"]


@pytest.mark.parametrize("source", ["AH", "VAH"])
def test_geometry_is_noncollapsed_and_complete(source):
    value = locked()["sources"][source]
    geometry = value["geometry"]
    assert value["noncollapse"] is True
    assert geometry["effective_rank"] > 0
    assert "per_dimension_variance" in geometry
    assert "query_diversity" in geometry
    assert "token_norm" in geometry
    assert "cka_with_oracle" in geometry


def test_frozen_component_and_private_residual_integrity_passes():
    value = locked()["frozen_components"]
    assert value["before_after_pass"] is True
    assert value["all_requires_grad_false"] is True
    assert value["d_c_matches_accepted"] is True
    assert set(value["private_residual_sha256"]) == {"train", "validation", "test"}
    assert set(value["digests"]) == {"P_v", "P_a", "P_c", "R_v", "R_a", "R_c", "D_c", "shared_space"}


def test_vision_interpretation_uses_bootstrap_and_is_ah_sufficient():
    value = locked()
    assert value["vision_classification"] == "A_PLUS_H_SUFFICIENT_VISION_OPTIONAL"
    for metric in ("contact_transition", "force_trend_class"):
        assert len(value["vision_incremental"][metric]["bootstrap_ci95"]) == 2


def test_final_decision_and_stop_scope_are_exact():
    path = ROOT / ".local/artifacts/tactile_unit/vac_c3mscc/final_decision.json"
    if not path.is_file():
        pytest.skip("local C3-MS-CC final decision is unavailable")
    final = json.loads(path.read_text())
    assert final["decision"] == "C3MSCC_ACTION_TEMPORAL_FAIL"
    assert final["c4_readiness"] == "NOT READY"
    assert final["c4"] == "NOT STARTED"
    assert final["c5"] == "NOT STARTED"
    assert final["c6"] == "NOT STARTED"
    assert final["m3"] == "NOT ESTABLISHED"


def test_required_runtime_artifacts_and_plots_exist():
    root = ROOT / ".local/artifacts/tactile_unit/vac_c3mscc"
    if not root.is_dir():
        pytest.skip("local C3-MS-CC runtime artifacts are unavailable")
    required = {
        "contract_audit.json", "trial_manifest.json", "training_summary.json",
        "selection.json", "locked_test_evaluation.json", "source_ablation.json",
        "context_ablation.json", "temporal_ablation.json", "final_decision.json",
        "HUMAN_ACCEPTANCE.md",
    }
    assert all((root / name).is_file() for name in required)
    summary = json.loads((root / "visualization_summary.json").read_text())
    assert len(summary["plots"]) >= 10
    assert all((root / "plots" / name).is_file() for name in summary["plots"])


def test_local_runtime_artifacts_are_git_ignored():
    completed = subprocess.run(
        ["git", "check-ignore", ".local/artifacts/tactile_unit/vac_c3mscc/selection.json"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert completed.returncode == 0
