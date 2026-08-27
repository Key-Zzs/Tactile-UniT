import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gr00t.tactile_unit.c3r0_conditional_sufficiency import (
    CONTACT_BOUNDARY_CLASSES,
    SOURCE_COMPONENTS,
    SmallContactCeiling,
    TrainStandardizer,
    knn_target_predictions,
    local_target_ambiguity,
    majority_neighbor_prediction,
    normalized_label_entropy,
    root_cause_decision,
    semantic_ratio,
    sha256_file,
    source_features,
)
from scripts.tactile_unit.c3r0_runtime import FREEZE_FILES, validate_test_freeze

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/tactile_unit/c3r0_conditional_sufficiency_audit.json"


def config():
    return json.loads(CONFIG_PATH.read_text())


def test_frozen_c1_c2_c2r_and_c3dp_identities_are_exact():
    value = config()
    paths = {
        "c1_manifest_sha256": ".local/cache/tactile_unit/vac_c1/manifest.json",
        "c2_checkpoint_sha256": ".local/experiments/tactile_unit/vac_c2/selected.pt",
        "c2r_checkpoint_sha256": ".local/experiments/tactile_unit/vac_c2r/selected.pt",
        "c3dp_predictor_sha256": ".local/experiments/tactile_unit/vac_c3dp/selected.pt",
    }
    assert all(sha256_file(ROOT / path) == value["accepted"][name] for name, path in paths.items())


def test_native_checkpoint_identities_are_exact():
    value = config()
    paths = {
        "action_checkpoint_sha256": ".local/experiments/tactile_unit/s3_3_r/selected.pt",
        "s1_checkpoint_sha256": ".local/experiments/tactile_teacher/s1_teacher/best.pt",
        "s2_checkpoint_sha256": ".local/experiments/contact_dynamics/s2_models/proposed_best.pt",
    }
    assert all(sha256_file(ROOT / path) == value["accepted"][name] for name, path in paths.items())


def arrays(size=6):
    rng = np.random.default_rng(3)
    return {
        "u_v": rng.normal(size=(size, 8, 32)).astype(np.float32),
        "u_a": rng.normal(size=(size, 8, 32)).astype(np.float32),
        "u_c": rng.normal(size=(size, 8, 32)).astype(np.float32),
        "z_c": rng.normal(size=(size, 8, 32)).astype(np.float32),
        "h_current": rng.normal(size=(size, 256)).astype(np.float32),
        "h_future": rng.normal(size=(size, 256)).astype(np.float32),
        "r_c_priv": rng.normal(size=(size, 8, 32)).astype(np.float32),
        "pair_id": np.arange(size),
    }


@pytest.mark.parametrize(
    ("source", "width"),
    [("V", 256), ("A", 256), ("VA", 512), ("H", 256), ("VH", 512), ("AH", 512), ("VAH", 768)],
)
def test_source_sets_have_exact_legal_width(source, width):
    assert source_features(source, arrays()).shape == (6, width)


def test_source_contract_has_no_target_private_future_or_pair_id():
    for source in ("V", "A", "VA", "H", "VH", "AH", "VAH"):
        assert not ({"u_c", "z_c", "r_c_priv", "h_future", "pair_id"} & set(SOURCE_COMPONENTS[source]))


def test_missing_legal_source_fails_closed():
    value = arrays()
    del value["u_v"]
    with pytest.raises(KeyError):
        source_features("V", value)


def test_train_only_standardizer_is_fixed_on_transform():
    train = np.arange(40, dtype=np.float32).reshape(10, 4)
    validation = np.full((3, 4), 1000.0, dtype=np.float32)
    standardizer = TrainStandardizer.fit(train)
    before = (standardizer.mean.copy(), standardizer.scale.copy())
    standardizer.transform(validation)
    assert np.array_equal(before[0], standardizer.mean)
    assert np.array_equal(before[1], standardizer.scale)
    assert np.allclose(standardizer.transform(train).mean(0), 0.0, atol=1e-6)


def test_standardizer_flattens_token_geometry():
    train = np.arange(5 * 8 * 32, dtype=np.float32).reshape(5, 8, 32)
    standardizer = TrainStandardizer.fit(train)
    assert standardizer.mean.shape == (256,)
    assert standardizer.scale.shape == (256,)
    assert standardizer.transform(train).shape == (5, 256)


def test_semantic_retention_formula():
    assert semantic_ratio(0.5, 0.2, 0.6) == pytest.approx(0.75)


def test_normalized_entropy_formula_and_range():
    labels = np.array([[0, 0, 0, 0], [0, 1, 2, 3]])
    value = normalized_label_entropy(labels, 4)
    assert value[0] == pytest.approx(0.0)
    assert value[1] == pytest.approx(1.0)
    assert np.all((value >= 0) & (value <= 1))


def test_majority_neighbor_formula_is_deterministic():
    labels = np.array([[1, 1, 3, 1], [2, 0, 2, 0]])
    assert majority_neighbor_prediction(labels, 4).tolist() == [1, 0]


def test_local_target_variance_global_normalization():
    targets = np.array([[[[0.0]], [[2.0]]], [[[1.0]], [[1.0]]]], dtype=np.float32)
    result = local_target_ambiguity(targets, global_variance=2.0)
    assert result["local_variance"]["mean"] == pytest.approx(0.5)
    assert result["local_over_global"]["mean"] == pytest.approx(0.25)


def test_knn_copy_medoid_and_mean_contracts():
    targets = np.arange(6 * 2, dtype=np.float32).reshape(6, 1, 2)
    indices = np.array([[0, 1, 4], [2, 3, 5]])
    result = knn_target_predictions(indices, targets, 3)
    assert np.array_equal(result["1nn"], targets[[0, 2]])
    assert all(any(np.array_equal(value, candidate) for candidate in targets[row]) for row, value in zip(indices, result["medoid"]))
    assert np.allclose(result["mean"][0], targets[indices[0]].mean(0))


def test_boundary_classes_are_free_to_contact_and_contact_to_free():
    assert CONTACT_BOUNDARY_CLASSES == (1, 3)


@pytest.mark.parametrize("source", ["VA", "VAH"])
@pytest.mark.parametrize("architecture", ["M0", "M1"])
def test_deterministic_ceiling_is_bounded_and_target_free(source, architecture):
    model = SmallContactCeiling(source, architecture)
    width = 512 if source == "VA" else 768
    assert model(torch.randn(2, width)).shape == (2, 8, 32)
    assert model.parameter_count() <= 100000
    signature = inspect.signature(model.forward)
    assert list(signature.parameters) == ["source_features_flat"]


def test_deterministic_ceiling_rejects_single_source_and_unknown_architecture():
    with pytest.raises(ValueError):
        SmallContactCeiling("V", "M0")
    with pytest.raises(ValueError):
        SmallContactCeiling("VA", "M2")


def test_pretest_command_never_loads_locked_test():
    source = (ROOT / "scripts/tactile_unit/audit_c3r0_source_semantics.py").read_text()
    assert 'load_aligned_split(config, "test")' not in source
    assert 'load_aligned_split(config, "train")' in source
    assert 'load_aligned_split(config, "validation")' in source


def test_locked_evaluation_must_validate_all_freeze_files_first():
    evaluation_path = ROOT / "scripts/tactile_unit/evaluate_c3r0_conditional_sufficiency.py"
    if not evaluation_path.is_file():
        pytest.skip("locked evaluation is added after pretest implementation")
    source = evaluation_path.read_text()
    assert source.index("validate_test_freeze(config)") < source.index('load_aligned_split(config, "test")')


def test_protocol_requires_all_four_freeze_artifacts():
    assert set(FREEZE_FILES) == {
        "audit_protocol.json", "probe_selection.json", "knn_protocol.json",
        "deterministic_ceiling_selection.json",
    }


@pytest.mark.parametrize(
    ("evidence", "primary", "next_stage"),
    [
        ({"structural_pass": False}, "STRUCTURAL_FAIL", "NO_NEXT_STAGE_DUE_TO_STRUCTURAL_FAIL"),
        ({"single_source_sufficient": True, "predictor_gap": True}, "PREDICTOR_OBJECTIVE_BOTTLENECK", "C3-R1_SEMANTIC_RANK_PRESERVING_PREDICTOR"),
        ({"va_sufficient": True}, "MULTISOURCE_COMPLEMENTARITY_REQUIRED", "C3-MS_MULTISOURCE_PREDICTION"),
        ({"vah_sufficient": True}, "CAUSAL_CONTACT_CONTEXT_REQUIRED", "C3-MS-CC_CAUSAL_CONTEXT_PREDICTION"),
        ({"multimodality": True}, "CONDITIONAL_MULTIMODALITY_LIKELY", "C3-DISTRIBUTIONAL_CONTACT_PREDICTION"),
        ({"direct_high_target_low": True}, "SHARED_CONTACT_TARGET_TOO_ENTANGLED", "C3-SHARED_TARGET_REFACTOR"),
    ],
)
def test_root_cause_rules_are_deterministic(evidence, primary, next_stage):
    assert root_cause_decision(evidence) == (primary, next_stage)


def test_locked_reducer_limits_predictor_nonparametric_evidence_to_v_and_a():
    source = (ROOT / "scripts/tactile_unit/evaluate_c3r0_conditional_sufficiency.py").read_text()
    fragment = source[source.index("best_nonparam_contact = max("):source.index("strongest_entropy =")]
    assert 'for source in ("V", "A")' in fragment
    assert 'for source in config["sources"]' not in fragment


def test_config_freezes_split_knn_trial_and_scope_bounds():
    value = config()
    assert value["counts"] == {"train": 65536, "validation": 8192, "test": 17504}
    assert value["knn"]["k"] == [1, 5, 10, 20]
    assert value["deterministic_ceiling"]["trials_total"] == 4
    assert value["scope"]["private_residual_target"] is False
    assert value["scope"]["c3r1_started"] is False
    assert value["scope"]["c4_started"] is False
    assert value["scope"]["c5_started"] is False
    assert value["scope"]["m3_established"] is False


def test_gpu_policy_excludes_zero_and_one():
    value = config()["gpu"]
    assert value["allowed_physical"] == [2, 3]
    assert value["forbidden_physical"] == [0, 1]


def test_existing_selection_validation_is_rejected_before_files_exist(tmp_path):
    value = config()
    value["runtime"]["artifact_root"] = str(tmp_path / "absent")
    with pytest.raises(RuntimeError):
        validate_test_freeze(value)
