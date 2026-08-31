from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gr00t.simulation.dexjoco_adapter import (
    SimEnvAction,
    SimObservation,
    SimPolicyAction,
    policy_action_to_env_action,
    proprio_to_neutral_policy_action,
)
from gr00t.simulation.episode_logger import DexJoCoEpisodeLogger
from gr00t.simulation.simulated_tactile import (
    ContactSample,
    FEATURE_NAMES,
    aggregate_contact_samples,
)
from gr00t.simulation.timing import (
    SimTactileHistoryBuffer,
    TimingContract,
    transition_pair_for_anchor,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs/simulation"


def test_action_contract_dimensions_order_conversion_and_determinism():
    policy = SimPolicyAction(np.asarray([1, 2, 3, 0, 0, 0, *np.arange(16)], dtype=np.float32))
    first = policy_action_to_env_action(policy)
    second = policy_action_to_env_action(policy)
    assert isinstance(first, SimEnvAction)
    assert first.values.shape == (23,)
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_allclose(first.values[:3], [1, 2, 3])
    np.testing.assert_allclose(first.values[3:7], [1, 0, 0, 0], atol=1e-7)
    np.testing.assert_allclose(first.values[7:], np.arange(16))
    with pytest.raises(ValueError, match="22D"):
        SimPolicyAction(np.zeros(23))
    with pytest.raises(ValueError, match="finite"):
        SimPolicyAction(np.full(22, np.nan))


def test_neutral_action_preserves_current_pose_and_hand():
    state = np.concatenate([[0.1, 0.2, 0.3, 1, 0, 0, 0], np.arange(16), np.zeros(8)])
    neutral = proprio_to_neutral_policy_action(state)
    converted = policy_action_to_env_action(neutral)
    np.testing.assert_allclose(converted.values[:23], state[:23], atol=1e-6)


def test_timing_is_physical_time_based_and_future_histories_do_not_overlap():
    timing = TimingContract(physics_dt=0.002, control_dt=0.02)
    assert timing.physics_substeps == 10
    assert timing.control_hz == 50
    assert timing.history_samples == 26
    assert timing.transition_control_steps == 27
    assert timing.actual_transition_horizon_sec == pytest.approx(0.54)
    pair = transition_pair_for_anchor(25, 53, timing)
    assert pair.current_history_indices == tuple(range(26))
    assert pair.future_history_indices == tuple(range(27, 53))
    assert pair.raw_overlap is False


def test_history_buffer_resamples_exact_duration_and_rejects_future_leakage():
    buffer = SimTactileHistoryBuffer(2)
    for index in range(31):
        buffer.append(index * 0.02, np.asarray([index, index * 2], dtype=np.float32))
    times, history = buffer.history(anchor_sec=0.6, duration_sec=0.5, sample_dt=0.02)
    assert times.shape == (26,)
    assert history.shape == (26, 2)
    assert times[0] == pytest.approx(0.1) and times[-1] == pytest.approx(0.6)
    np.testing.assert_allclose(history[-1], [30, 60])
    with pytest.raises(ValueError, match="future"):
        buffer.history(anchor_sec=0.62)


def test_contact_aggregation_free_contact_multiple_cop_and_release():
    regions = ("index", "thumb")
    free, diagnostics = aggregate_contact_samples([], regions)
    assert free.shape == (len(regions) * len(FEATURE_NAMES),)
    assert np.count_nonzero(free) == 0 and diagnostics["contact_count"] == 0
    samples = [
        ContactSample("index", 2.0, 0.5, np.asarray([1.0, 0.0, 0.0])),
        ContactSample("index", 1.0, 0.25, np.asarray([0.0, 3.0, 0.0])),
        ContactSample("thumb", -1.0, 99.0, np.asarray([9.0, 9.0, 9.0])),
    ]
    tactile, diagnostics = aggregate_contact_samples(samples, regions)
    matrix = tactile.reshape(2, 6)
    np.testing.assert_allclose(matrix[0], [1, 3, 0.75, 2 / 3, 1, 0], atol=1e-6)
    np.testing.assert_array_equal(matrix[1], np.zeros(6))
    assert diagnostics["contact_count_by_region"] == {"index": 2, "thumb": 0}
    released, _ = aggregate_contact_samples([], regions)
    np.testing.assert_array_equal(released, free)


def test_scalar_force_contract_is_geom_order_invariant_and_regions_are_distinct():
    forward = ContactSample("index", max(5.0, 0), float(np.linalg.norm([3.0, 4.0])), np.zeros(3))
    reversed_order = ContactSample(
        "index", max(5.0, 0), float(np.linalg.norm([-3.0, -4.0])), np.zeros(3)
    )
    first, _ = aggregate_contact_samples([forward], ("index", "thumb"))
    second, _ = aggregate_contact_samples([reversed_order], ("index", "thumb"))
    np.testing.assert_array_equal(first, second)
    assert first.reshape(2, 6)[0, 0] == 1 and first.reshape(2, 6)[1, 0] == 0


def test_episode_logger_schema_rgb_references_and_monotonic_timestamps(tmp_path):
    logger = DexJoCoEpisodeLogger(tmp_path, "episode-1", {"task": "pinch_tongs", "seed": 1})
    policy = SimPolicyAction(np.zeros(22))
    env = policy_action_to_env_action(policy)
    for step in range(2):
        observation = SimObservation(
            timestamp_sec=(step + 1) * 0.02,
            control_step=step + 1,
            episode_id="episode-1",
            task_name="pinch_tongs",
            rgb=np.full((8, 8, 3), step * 10, dtype=np.uint8),
            proprio=np.zeros(31),
            sim_tactile=np.zeros(30),
            terminated=False,
            truncated=False,
            success=False,
        )
        logger.append(observation, policy, env, 0.0, {"contact_count": 0})
    manifest = logger.finish()
    assert manifest["steps"] == 2
    values = np.load(tmp_path / "episode-1/steps.npz")
    assert values["episode_id"].tolist() == ["episode-1", "episode-1"]
    assert values["task"].tolist() == ["pinch_tongs", "pinch_tongs"]
    assert values["seed"].tolist() == [1, 1]
    assert values["proprio"].shape == (2, 31)
    assert values["sim_tactile"].shape == (2, 30)
    assert np.all(np.diff(values["timestamp_sec"]) > 0)
    for reference in values["rgb_reference"]:
        assert (tmp_path / "episode-1" / reference).is_file()


def test_tracked_configs_freeze_runtime_schema_and_prohibit_training():
    runtime = json.loads((CONFIG_ROOT / "s4_1_dexjoco_runtime.json").read_text())
    contract = json.loads((CONFIG_ROOT / "s4_1_dexjoco_contract.json").read_text())
    tactile = json.loads((CONFIG_ROOT / "s4_1_sim_tactile_contract.json").read_text())
    regions = json.loads((CONFIG_ROOT / "s4_1_dexjoco_contact_regions.json").read_text())
    assert runtime["environment"]["name"] == "tactile-unit-dexjoco"
    assert runtime["environment"]["m3_environment"] == "unit"
    assert runtime["scope"]["policy_training"] is False
    assert runtime["scope"]["contact_representation_training"] is False
    assert (
        contract["observation"]["proprio_dim"]
        == len(contract["observation"]["proprio_names"])
        == 31
    )
    assert tactile["total_dim"] == len(regions["regions"]) * tactile["per_region_dim"] == 30
    assert tactile["normalization"].startswith("raw physical")
    assert all(not row["geom_names"] for row in regions["regions"])
    assert "numeric geom IDs are forbidden" in regions["resolution"]


def test_dexjoco_pin_m3_provenance_privacy_and_local_artifacts_are_untracked():
    stage = subprocess.run(
        ["git", "ls-files", "--stage", "third_party/dexjoco"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.split()
    assert stage[:2] == ["160000", "8d23b0fab23b17a58c4b55f3942e17013aaf8267"]
    assert (
        subprocess.run(["git", "merge-base", "--is-ancestor", "m3", "HEAD"], cwd=ROOT).returncode
        == 0
    )
    assert (
        subprocess.run(
            ["git", "ls-files", ".local"], cwd=ROOT, text=True, capture_output=True, check=True
        ).stdout
        == ""
    )
    tracked = [
        ROOT / "gr00t/simulation",
        ROOT / "scripts/simulation",
        ROOT / "configs/simulation",
        ROOT / "docs/research/s4_1_dexjoco_runtime_sim_tactile.md",
        ROOT / "tests/simulation",
    ]
    forbidden = (
        "/" + "home/",
        "/" + "mnt/",
        "Author" + "ization:",
        "Bear" + "er ",
        "github" + "_pat_",
        "HF" + "_TOKEN",
    )
    files = [path for root in tracked for path in ([root] if root.is_file() else root.rglob("*"))]
    for path in files:
        if path.is_file() and "__pycache__" not in path.parts:
            text = path.read_text(errors="ignore")
            assert not any(term in text for term in forbidden), path
