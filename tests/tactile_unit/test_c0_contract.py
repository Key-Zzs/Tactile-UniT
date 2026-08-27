from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tactile_unit"))

from continuous_contact_bridge_common import (  # noqa: E402
    distribution_metrics,
    different_episode_permutation,
    same_episode_wrong_time_permutation,
)


def test_different_episode_negatives_never_match_episode() -> None:
    episodes = np.repeat(np.arange(8), 4)
    permutation = different_episode_permutation(episodes, seed=42)
    assert sorted(permutation.tolist()) == list(range(len(episodes)))
    assert np.all(episodes != episodes[permutation])


def test_wrong_time_negatives_stay_in_episode_and_do_not_overlap() -> None:
    episodes = np.repeat(np.arange(2), 4)
    anchors = np.tile(np.array([15, 47, 79, 111]), 2)
    permutation = same_episode_wrong_time_permutation(episodes, anchors, minimum_offset=32)
    assert np.all(permutation >= 0)
    assert np.all(episodes == episodes[permutation])
    assert np.all(np.abs(anchors - anchors[permutation]) >= 32)


def test_distribution_metrics_are_finite_for_noncollapsed_tokens() -> None:
    values = np.random.default_rng(4).normal(size=(16, 8, 32)).astype(np.float32)
    metrics = distribution_metrics(values)
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["effective_rank"] > 1
    assert metrics["global_std"] > 0


def test_c0_spec_freezes_continuous_native_contact() -> None:
    spec = json.loads(Path("configs/tactile_unit/c0_continuous_contact_bridge.json").read_text())
    contract = spec["continuous_contact_contract"]
    assert spec["track_b_base_sha"] == "7051f8140239a7e72c51aa0749bac703eb60a923"
    assert contract["interface"] == "CONTINUOUS"
    assert contract["transition_target_shape"] == [8, 32]
    assert contract["current_context_shape"] == [256]
    assert contract["dtype"] == "float32"
    assert contract["normalization"] == "native frozen S2 E_c output"
    assert not contract["whitening_allowed"]
    assert not spec["scope"]["contact_discretization"]
    assert not spec["scope"]["contact_retraining"]


def test_c0_spec_distinguishes_checkpoint_and_module_digests() -> None:
    identity = json.loads(
        Path("configs/tactile_unit/c0_continuous_contact_bridge.json").read_text()
    )["frozen_identity"]
    assert identity["s2_checkpoint_sha256"].startswith("c36c0531")
    assert identity["s2_encoder_parameter_digest"].startswith("1d751899")
    assert identity["s2_decoder_parameter_digest"].startswith("50ec1fd7")
    assert (
        len(
            {
                identity["s2_checkpoint_sha256"],
                identity["s2_encoder_parameter_digest"],
                identity["s2_decoder_parameter_digest"],
            }
        )
        == 3
    )


def test_revised_m3_is_spec_only_and_has_all_preregistered_gates() -> None:
    protocol = json.loads(
        Path("configs/tactile_unit/m3_continuous_vac_evaluation.json").read_text()
    )
    assert protocol["status"] == "SPEC_ONLY_NOT_EXECUTED"
    assert not protocol["same_codebook_required"]
    assert [row["id"] for row in protocol["preregistered_gates"]] == list(range(1, 13))
    assert protocol["dependencies"]["full_track_c"] == "not started by this protocol file"
