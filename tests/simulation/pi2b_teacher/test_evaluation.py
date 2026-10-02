from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gr00t.simulation.pi2b_teacher.evaluation import (
    deterministic_control_indices,
    holm_adjust,
    retrieval_metrics,
    stratified_group_bootstrap,
)

ROOT = Path(__file__).resolve().parents[3]


def test_independent_retrieval_reports_full_metrics_and_chance() -> None:
    value = np.eye(12, dtype=np.float32).reshape(12, 1, 12)
    result = retrieval_metrics(value, value)
    assert result["candidate_count"] == 12
    assert result["recall_at_1"] == 1.0
    assert result["recall_at_5"] == 1.0
    assert result["recall_at_10"] == 1.0
    assert result["mrr"] == 1.0
    assert result["median_rank"] == 1.0
    assert result["chance"]["recall_at_1"] == 1.0 / 12.0


def test_negative_controls_never_use_forbidden_identity() -> None:
    task = np.asarray(["a", "a", "a", "b", "b", "b"])
    group = np.asarray(["a0", "a1", "a2", "b0", "b1", "b2"])
    episode = np.asarray(["e0", "e1", "e2", "e3", "e4", "e5"])
    different = deterministic_control_indices(task, group, episode, same_task=False)
    same_task = deterministic_control_indices(task, group, episode, same_task=True)
    assert np.all(group[different] != group)
    assert np.all(task[same_task] == task)
    assert np.all(episode[same_task] != episode)


def test_stratified_group_bootstrap_is_paired_and_deterministic() -> None:
    values = {f"a{i}": float(i) for i in range(3)} | {f"b{i}": float(-i) for i in range(3)}
    tasks = {name: name[0] for name in values}
    first = stratified_group_bootstrap(values, tasks, samples=1000, seed=42)
    second = stratified_group_bootstrap(values, tasks, samples=1000, seed=42)
    assert first == second
    assert first["estimate"] == 0.0


def test_holm_adjustment_is_monotone() -> None:
    result = holm_adjust({"a": 0.01, "b": 0.02, "c": 0.20})
    assert result["a"] <= result["b"] <= result["c"]
    assert all(0.0 <= value <= 1.0 for value in result.values())


def test_evaluation_protocol_freezes_real_action_reversal_and_group_statistics() -> None:
    config = json.loads((ROOT / "configs/simulation/pi2b_teacher/evaluation.json").read_text())
    assert config["probe"]["alpha"] == 10.0
    assert config["statistics"]["unit"] == "source_group_with_episode_windows_intact"
    assert config["statistics"]["bootstrap_samples"] == 10000
    assert "reverse raw [27,22] action time" in config["controls"]["reversed_action"]
    source = (ROOT / "scripts/simulation/pi2b_teacher/build_confirmation_cache.py").read_text()
    assert 'action[:, ::-1].copy()' in source
    assert 'z_a_reversed' in source
    assert 'z_c_reversed' in source
