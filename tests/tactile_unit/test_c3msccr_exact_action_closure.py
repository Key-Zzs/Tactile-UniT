import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from gr00t.tactile_unit.c3msccr_exact_action_closure import (
    ACTION_ORDERING,
    canonical_action_from_raw,
    deterministic_temporal_orders,
    perturb_raw_action,
    raw_action_from_canonical,
    same_split_different_indices,
)
from scripts.tactile_unit.evaluate_c3msccr_closure import frozen_reducer


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/tactile_unit/c3msccr_exact_action_closure.json"


def config():
    return json.loads(CONFIG.read_text())


def feature_stats():
    rng = np.random.default_rng(1)
    return {
        "action_mean": rng.normal(size=58).astype(np.float32).tolist(),
        "action_std": rng.uniform(0.1, 2.0, size=58).astype(np.float32).tolist(),
    }


def test_raw_round_trip_and_padding_are_exact():
    rng = np.random.default_rng(2)
    raw = rng.normal(size=(4, 16, 58)).astype(np.float32)
    canonical = canonical_action_from_raw(raw, feature_stats())
    restored = raw_action_from_canonical(canonical, feature_stats())
    assert canonical.shape == (4, 16, 128)
    assert np.array_equal(canonical[..., 58:], np.zeros((4, 16, 70), np.float32))
    assert np.allclose(restored, raw, rtol=1e-6, atol=1e-6)


def test_reverse_and_shuffle_happen_on_raw_sequence_before_features():
    raw = np.arange(5 * 16 * 58, dtype=np.float32).reshape(5, 16, 58)
    orders = deterministic_temporal_orders(5, 3544)
    different = np.asarray([1, 2, 3, 4, 0])
    variants = perturb_raw_action(raw, temporal_orders=orders, different_indices=different)
    assert np.array_equal(variants["correct"], raw)
    assert np.array_equal(variants["reversed"], raw[:, ::-1])
    assert np.array_equal(variants["shuffled"][0], raw[0, orders[0]])
    assert np.array_equal(variants["different"], raw[different])


def test_temporal_shuffle_is_deterministic_and_every_row_is_a_permutation():
    first = deterministic_temporal_orders(32, 3544)
    second = deterministic_temporal_orders(32, 3544)
    assert np.array_equal(first, second)
    assert np.all(np.sort(first, axis=1) == np.arange(16))
    assert not np.any(np.all(first == np.arange(16), axis=1))


def test_different_episode_mapping_is_same_split_and_never_same_episode():
    episode = np.repeat(np.arange(4), 5)
    mapping = same_split_different_indices(episode, 43172)
    assert mapping.min() >= 0 and mapping.max() < len(episode)
    assert np.all(episode[mapping] != episode)
    assert np.array_equal(mapping, same_split_different_indices(episode, 43172))


def test_contract_freezes_exact_action_and_scope():
    value = config()
    assert value["contract"]["raw_action_shape"] == [16, 58]
    assert value["contract"]["action_interval"] == "a_t:t+15"
    assert tuple(value["contract"]["ordering"]) == ACTION_ORDERING
    assert value["contract"]["reverse_before_feature_construction"] is True
    assert value["contract"]["refit_statistics"] is False
    assert value["contract"]["rq"] is False
    assert value["remediation"]["maximum_trials"] <= 2
    assert value["remediation"]["source"] == "AH"
    assert value["gpu"]["allowed_physical"] == [1, 2, 3]
    assert value["gpu"]["forbidden_physical"] == [0]
    assert "explicit user authorization" in value["gpu"]["gpu1_authorization"]
    assert value["scope"]["c4_started"] is False
    assert value["scope"]["vision_remediation"] is False


def test_invalid_padding_fails_closed():
    value = np.zeros((2, 16, 128), dtype=np.float32)
    value[..., 58] = 1
    with pytest.raises(ValueError, match="padding"):
        raw_action_from_canonical(value, feature_stats())


def test_frozen_reducer_preserves_exact_point_zero_one_rule():
    def trial(name, source, utility, passed):
        return {
            "trial": {"id": name, "source": source},
            "best": {"utility": utility, "validation": {"gates": {"all": passed}}},
        }

    rows = [trial("T0", "AH", 0.593, True), trial("T1", "VAH", 0.599, True)]
    selected, _ = frozen_reducer(rows, 0.01)
    assert selected["trial"]["id"] == "T0"
    rows[0]["best"]["validation"]["gates"]["all"] = False
    selected, _ = frozen_reducer(rows, 0.01)
    assert selected["trial"]["id"] == "T1"


def artifacts():
    root = ROOT / ".local/artifacts/tactile_unit/vac_c3msccr"
    if not (root / "locked_closure_evaluation.json").is_file():
        pytest.skip("local C3-MS-CC-R closure artifacts are unavailable")
    return root


def test_exact_cache_alignment_cross_split_and_correct_reproduction():
    value = config()
    for split in ("train", "validation", "test"):
        exact = ROOT / value["runtime"]["exact_cache_root"] / split
        c1 = ROOT / value["runtime"]["c1_cache_root"] / split
        pair = np.load(exact / "pair_id.npy", mmap_mode="r", allow_pickle=False)
        source = np.load(exact / "source_index.npy", mmap_mode="r", allow_pickle=False)
        different = np.load(exact / "different_source_index.npy", mmap_mode="r", allow_pickle=False)
        episode = np.load(c1 / "episode_id.npy", mmap_mode="r", allow_pickle=False)
        assert np.array_equal(pair, np.load(c1 / "pair_id.npy", mmap_mode="r", allow_pickle=False))
        assert np.array_equal(source, np.load(c1 / "source_index.npy", mmap_mode="r", allow_pickle=False))
        assert different.min() >= 0 and different.max() < len(pair)
        assert np.all(episode[different] != episode)
        expected_z = np.load(c1 / "z_a.npy", mmap_mode="r", allow_pickle=False)
        actual_z = np.load(exact / "z_a_correct.npy", mmap_mode="r", allow_pickle=False)
        assert np.max(np.abs(np.asarray(expected_z) - np.asarray(actual_z))) <= value["cache"]["correct_reproduction_atol"]


def test_ar_temporal_integrity_and_source_reducer_audits_pass():
    root = artifacts()
    temporal = json.loads((root / "ar_temporal_integrity.json").read_text())
    source = json.loads((root / "source_selection_audit.json").read_text())
    assert temporal["a_r_temporal_property_reproduced"] is True
    for name in ("reversed", "shuffled", "different"):
        assert temporal["decoder"]["dynamic"][name]["difference_ci95"][0] > 0
    assert source["t0_original_failed_gates"] == ["action_exact_ar", "action_surrogate", "physics"]
    assert source["reducer_implementation"] == "VALID"
    assert source["selection_bug"] is False
    assert source["simplicity_tolerance"] == 0.01
    assert source["test_loaded"] is False


def test_frozen_exact_validation_legitimately_triggers_only_physics_remediation():
    frozen = json.loads((artifacts() / "frozen_candidate_validation.json").read_text())
    t0 = next(row for row in frozen["trials"] if row["trial"]["id"] == "T0")
    t1 = next(row for row in frozen["trials"] if row["trial"]["id"] == "T1")
    assert frozen["test_loaded"] is False
    assert frozen["t0_failed_gates_after_exact_evidence"] == ["physics"]
    assert t0["validation"]["gates"]["action_exact_ar"] is True
    assert t1["validation"]["gates"]["action_exact_ar"] is True
    assert frozen["remediation_required"] is True


def test_remediation_is_single_ah_only_exact_objective_and_architecture_preserved():
    value = json.loads((artifacts() / "remediation_trials.json").read_text())
    trial = value["trials"][0]
    assert value["total_new_trials"] == 1 <= value["maximum_trials"] <= 2
    assert trial["source"] == "AH"
    assert trial["vision_used"] is False
    assert trial["architecture_changed"] is False
    assert trial["initialized_from"] == "frozen T0"
    assert trial["objective"] == "exact temporal ranking plus existing shared physics"
    assert value["all_validation_hard_gates"] is True
    assert value["test_loaded"] is False


def test_closure_selection_is_hashed_validation_only_and_pretest():
    root = artifacts()
    path = root / "closure_selection.json"
    selection = json.loads(path.read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == (root / "closure_selection.sha256").read_text().split()[0]
    assert selection["source"] == "AH"
    assert selection["mode"] == "BOUNDED_AH_REMEDIATION"
    assert selection["selected_via"] == "VALIDATION ONLY"
    assert selection["test_loaded"] is False
    assert selection["validation_hard_gates"]["all"] is True


def test_locked_closure_all_hard_gates_and_exact_action_evidence_pass():
    locked = json.loads((artifacts() / "locked_closure_evaluation.json").read_text())
    assert locked["evaluation"] == "LOCKED POST-HOC CLOSURE RE-EVALUATION"
    assert locked["first_look_untouched"] is False
    assert locked["rows"] == 17504
    assert locked["selection_frozen_before_test"] is True
    assert locked["repeated_evaluation_exact"] is True
    assert all(locked["hard_gates"].values())
    assert locked["decision"] == "C3MSCCR_READY_AH_MINIMAL_WITH_RANK_WARNING"
    action = locked["metrics"]["action_temporal"]
    assert action["exact_ar_transform"] is True
    assert action["variants"]["reversed"]["dynamic_difference_ci95"][0] > 0
    assert action["variants"]["shuffled"]["dynamic_difference_ci95"][0] > 0
    assert action["variants"]["different"]["dynamic_difference_ci95"][0] > 0


def test_locked_semantics_context_retrieval_physics_geometry_and_boundaries():
    locked = json.loads((artifacts() / "locked_closure_evaluation.json").read_text())
    metrics = locked["metrics"]
    assert metrics["semantics"]["contact_transition"]["semantic_ratio"] >= 0.75
    assert metrics["semantics"]["force_trend_class"]["semantic_ratio"] >= 0.75
    assert metrics["h_context"]["improvement_ci95"][0] > 0
    assert metrics["h_context"]["correct_contact_f1"] > max(
        value["macro_f1"] for value in metrics["h_context"]["invalid_semantics"].values()
    )
    shared = metrics["shared_target"]
    assert shared["improvement_ci95"][0] > 0
    assert shared["cosine_margin_ci95"][0] > 0
    assert shared["retrieval"]["recall_at_10"] >= 1.5 * shared["retrieval"]["chance"]["recall_at_10"]
    physics = metrics["physics"]
    assert physics["improvement_ci95"][0] > 0
    assert physics["dynamic_improvement_ci95"][0] > 0
    assert metrics["noncollapse"] is True
    assert metrics["geometry"]["effective_rank"] > 0
    assert np.isfinite(metrics["semantics"]["contact_transition"]["future_change"]["macro_f1"])


def test_private_residual_and_all_frozen_identities_are_unchanged():
    locked = json.loads((artifacts() / "locked_closure_evaluation.json").read_text())
    assert locked["identity_before"]["pass"] is True
    assert locked["identity_after"]["pass"] is True
    assert locked["identity_before"]["actual"] == locked["identity_after"]["actual"]
    assert locked["identity_after"]["equality"]["c3dp"] is True
    assert locked["shared_state_before"] == locked["shared_state_after"]


def test_final_decision_stops_before_c4_c5_c6_and_m3():
    value = json.loads((artifacts() / "final_decision.json").read_text())
    assert value["decision"] == "C3MSCCR_READY_AH_MINIMAL_WITH_RANK_WARNING"
    assert value["classification"] == "A_PLUS_H_CANONICAL_MINIMAL_SOURCE"
    assert value["canonical_source"] == "A+H"
    assert value["vision"] == "OPTIONAL_SHORT_HORIZON_CONTEXT"
    assert value["c4_readiness"] == "READY_WITH_RANK_WARNING"
    assert value["c4"] == "NOT STARTED"
    assert value["c5"] == "NOT STARTED"
    assert value["c6_m3"] == "NOT STARTED"
    assert value["m3"] == "NOT ESTABLISHED"
