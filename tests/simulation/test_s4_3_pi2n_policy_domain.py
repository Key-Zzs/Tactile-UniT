from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/simulation/s4_3_pi2n_policy_domain_probe.json"
SCRIPT = ROOT / "scripts/simulation/diagnose_s4_3_pi2n_policy_domain.py"


def load_module():
    spec = importlib.util.spec_from_file_location("diagnose_s4_3_pi2n_policy_domain", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_policy_domain_protocol_is_frozen_and_group_disjoint() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["status"] == "FROZEN_BEFORE_NUMERICAL_PROBE"
    assert protocol["split"]["fit_episodes"] == 80
    assert protocol["split"]["check_episodes"] == 20
    assert protocol["probe"]["fit_families_consumed"] == 1
    assert protocol["probe"]["trainable_parameters_family_total"] <= protocol["probe"]["parameters_per_fit_max"]
    assert protocol["route_constraints"]["X_route_not_finalized_here"] is True
    assert protocol["route_constraints"]["candidate_closed_loop_results_forbidden"] is True


def test_different_episode_control_has_no_identity_group() -> None:
    module = load_module()
    episode = np.repeat(np.asarray([4, 9, 14]), [3, 5, 4])
    indices = module.different_episode_indices(episode)
    assert indices.shape == episode.shape
    assert np.all(episode != episode[indices])


def test_ridge_readout_is_fixed_and_finite() -> None:
    module = load_module()
    rng = np.random.default_rng(5)
    x = rng.normal(size=(100, 8)).astype(np.float32)
    weights = rng.normal(size=(8, 3)).astype(np.float32)
    y = x @ weights
    prediction, details = module.standardized_ridge(x[:80], y[:80], x[80:], 0.001)
    assert prediction.shape == (20, 3)
    assert np.isfinite(prediction).all()
    assert details["trainable_parameters"] == 24
    assert np.mean(np.square(prediction - y[80:])) < 1e-3


def test_diagnostic_never_reads_candidate_checkpoints_or_outcomes() -> None:
    source = SCRIPT.read_text()
    assert "B_VA27/pinch_tongs" not in source
    assert "B_VAC_V/pinch_tongs" not in source
    assert "paired_statistics" not in source
    assert '"candidate_policy_checkpoints_read": False' in source
    assert '"formal_rollout_outcomes_read": False' in source
