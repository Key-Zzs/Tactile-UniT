from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gr00t.simulation.dexjoco_adapter import SimObservation, SimPolicyAction, policy_action_to_env_action
from gr00t.simulation.episode_logger import DexJoCoEpisodeLogger
from gr00t.simulation.s4_2_dataset import (
    FormalTestAccessError,
    build_pair_arrays,
    episode_references,
    pair_id,
    references_for_split,
)

ROOT = Path(__file__).resolve().parents[2]


def write_episode(root: Path, episode_id: str, split: str, source: str) -> None:
    metadata = {
        "episode_schema": "tactile3d-unit.dexjoco-s4-2-formal-episode.v1",
        "task": "pinch_tongs",
        "split": split,
        "seed": 1,
        "source_type": "repository_owned_deterministic_script",
        "source_trajectory_id": source,
        "randomization_id": "test",
        "randomization_parameters": {},
        "perturbation_id": 0,
    }
    logger = DexJoCoEpisodeLogger(root / "episodes", episode_id, metadata)
    policy = SimPolicyAction(np.zeros(22, dtype=np.float32))
    env_action = policy_action_to_env_action(policy)
    for index in range(60):
        tactile = np.zeros(30, dtype=np.float32)
        tactile[0] = float(index >= 30)
        tactile[1] = float(max(0, index - 29))
        observation = SimObservation(
            timestamp_sec=(index + 1) * 0.02,
            control_step=index + 1,
            episode_id=episode_id,
            task_name="pinch_tongs",
            rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            proprio=np.arange(31, dtype=np.float64),
            sim_tactile=tactile,
            terminated=False,
            truncated=False,
            success=False,
        )
        logger.append(observation, policy, env_action, 0.0, {"contact_count": 0})
    logger.finish()


def test_pair_builder_has_exact_histories_action_chunk_and_ids(tmp_path: Path) -> None:
    write_episode(tmp_path, "train-1", "train", "source-1")
    arrays = build_pair_arrays("train", tmp_path)
    assert arrays["current_history"].shape == (8, 26, 30)
    assert arrays["future_history"].shape == (8, 26, 30)
    assert arrays["teacher_future"].shape == (8, 13, 30)
    assert arrays["action_chunk"].shape == (8, 27, 22)
    assert arrays["current_state"].shape == (8, 23)
    assert arrays["anchor_step"].tolist() == list(range(25, 33))
    assert arrays["future_step"].tolist() == list(range(52, 60))
    assert arrays["pair_id"][0] == pair_id("pinch_tongs", "train-1", 25, 52, "train")
    assert not set(range(0, 26)).intersection(range(27, 53))


def test_test_split_is_guarded_except_explicit_quality_or_locked_access(tmp_path: Path) -> None:
    write_episode(tmp_path, "test-1", "test", "source-test")
    with pytest.raises(FormalTestAccessError, match="locked"):
        references_for_split("test", tmp_path)
    assert len(references_for_split("test", tmp_path, purpose="dataset_quality")) == 1
    freeze = tmp_path / "pretest_freeze.json"
    freeze.write_text(
        json.dumps(
            {"test_loaded": False, "training_complete": True, "selection_complete": True}
        )
    )
    assert len(
        references_for_split(
            "test", tmp_path, purpose="locked_test", pretest_freeze=freeze
        )
    ) == 1


def test_episode_metadata_and_checksums_are_formal_and_discoverable(tmp_path: Path) -> None:
    write_episode(tmp_path, "train-1", "train", "source-1")
    references = list(episode_references(tmp_path))
    assert len(references) == 1
    manifest = json.loads((references[0].directory / "metadata.json").read_text())
    assert manifest["schema"] == "tactile3d-unit.dexjoco-s4-2-formal-episode.v1"
    assert set(manifest["checksums"]) == {"steps_npz_sha256", "rgb_tree_sha256"}
    with np.load(references[0].directory / "steps.npz", allow_pickle=False) as values:
        assert values["split"].tolist() == ["train"] * 60
        assert values["source_trajectory_id"].tolist() == ["source-1"] * 60


def test_multitask_region_config_preserves_five_region_schema() -> None:
    config = json.loads(
        (ROOT / "configs/simulation/s4_2_dexjoco_contact_regions.json").read_text()
    )
    assert set(config["tasks"]) == {"pinch_tongs", "hammer_nail", "click_mouse"}
    assert [row["region_name"] for row in config["regions"]] == [
        "right_palm",
        "right_index",
        "right_middle",
        "right_ring",
        "right_thumb",
    ]
    assert all(not row["geom_names"] for row in config["regions"])
