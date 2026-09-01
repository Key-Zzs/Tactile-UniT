from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.simulation.audit_s4_3_demonstrations import (
    audit_dataset,
    load_ds_groups,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]


def _write_episode(root: Path, task: str, source: str, success: bool) -> None:
    episode_id = f"{source}-p00"
    directory = root / "episodes" / episode_id
    frames = directory / "frames"
    frames.mkdir(parents=True)
    length = 4
    for index in range(length):
        (frames / f"{index:06d}.jpg").write_bytes(b"frame")
    arrays = {
        "control_step": np.arange(1, length + 1),
        "timestamp_sec": np.arange(1, length + 1) * 0.02,
        "rgb_reference": np.asarray([f"frames/{index:06d}.jpg" for index in range(length)]),
        "proprio": np.zeros((length, 31)),
        "policy_action": np.zeros((length, 22), dtype=np.float32),
        "env_action": np.zeros((length, 23), dtype=np.float32),
        "sim_tactile": np.zeros((length, 30), dtype=np.float32),
        "reward": np.asarray([0, 0, 0, int(success)], dtype=np.float32),
        "terminated": np.asarray([False, False, False, success]),
        "truncated": np.zeros(length, dtype=bool),
        "success": np.asarray([False, False, False, success]),
        "contact_count": np.asarray([0, 1, 1, 0]),
    }
    np.savez_compressed(directory / "steps.npz", **arrays)
    manifest = {
        "metadata": {
            "task": task,
            "source_trajectory_id": source,
            "split": "train",
            "source_type": "repository_owned_deterministic_script",
        },
        "checksums": {"steps_npz_sha256": sha256_file(directory / "steps.npz")},
    }
    (directory / "metadata.json").write_text(json.dumps(manifest), encoding="utf-8")


def _split(path: Path) -> None:
    dev = [
        [task, f"{task}-script-{index:02d}"]
        for task in ("pinch_tongs", "hammer_nail", "click_mouse")
        for index in range(3)
    ]
    path.write_text(json.dumps({"ds_dev_group_ids": dev}), encoding="utf-8")


def test_frozen_ds_groups_are_exactly_disjoint_33_9(tmp_path: Path) -> None:
    split = tmp_path / "split.json"
    _split(split)
    train, dev = load_ds_groups(split)
    assert len(train) == 33
    assert len(dev) == 9
    assert not train & dev


def test_zero_success_complete_source_is_hard_failure(tmp_path: Path) -> None:
    split = tmp_path / "split.json"
    _split(split)
    for task in ("pinch_tongs", "hammer_nail", "click_mouse"):
        for index in range(14):
            _write_episode(tmp_path / "dataset", task, f"{task}-script-{index:02d}", False)
    audit, manifest = audit_dataset(tmp_path / "dataset", split)
    assert not audit["integrity_failures"]
    assert audit["totals"]["episodes"] == 42
    assert audit["totals"]["successful_demonstrations"] == 0
    assert audit["decision"] == "S4_3_0_DEMONSTRATION_CONTRACT_FAIL"
    assert audit["downstream_allowed"] is False
    assert manifest["overlap"] == {
        "train_dev": 0,
        "policy_formal_validation": 0,
        "policy_TEST_V1": 0,
        "policy_TEST_V2": 0,
    }


def test_tracked_failure_decision_stops_all_dependent_policy_work() -> None:
    decision = json.loads(
        (ROOT / "configs/simulation/s4_3_0_final_decision.json").read_text()
    )
    assert decision["decision"] == "S4_3_0_DEMONSTRATION_CONTRACT_FAIL"
    assert decision["final_s4_3_2_classification"] == "STRUCTURAL_FAIL"
    assert decision["policy_training_started"] is False
    assert decision["policy_eval_started"] is False
    assert set(decision["downstream"].values()) == {"NOT_RUN_DEPENDENCY_BLOCKED"}
    assert decision["s4_3_3_readiness"] == "NOT_READY"
