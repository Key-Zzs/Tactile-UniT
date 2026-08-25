from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.tactile_unit.paired_contract import (
    EpisodeVideoPointer,
    TREX_EMBODIMENT_ID,
    TREX_EMBODIMENT_TAG,
    audit_video_inventory,
    cache_identity,
    episode_frame_timestamp,
    make_pair_record,
    normalize_and_pad_trex_state_action,
    pad_trex_state_action,
    preprocess_trex_rgb,
    resolve_video_path,
    sha256_json,
    validate_cache_identity,
    validate_episode_splits,
    validate_transition_anchor,
    validate_video_probe,
)


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "configs/tactile_unit/s3_1_paired_vac_contract.json"


def pointer(**overrides) -> EpisodeVideoPointer:
    values = {
        "episode_id": 7,
        "length": 100,
        "data_chunk_index": 0,
        "data_file_index": 2,
        "dataset_from_index": 1000,
        "dataset_to_index": 1100,
        "video_chunk_index": 0,
        "video_file_index": 5,
        "from_timestamp": 10.0,
        "to_timestamp": 10.0 + 100 / 30,
        "relative_path": "videos/observation.images.head_left/chunk-000/file-005.mp4",
        "motor_primitive": "grasp_and_lifting",
        "object_name": "cup",
        "target": None,
    }
    values.update(overrides)
    return EpisodeVideoPointer(**values)


def info() -> dict:
    return {
        "fps": 30,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }


def feature() -> dict:
    return {
        "shape": [360, 640, 3],
        "info": {
            "video.codec": "h264",
            "video.width": 640,
            "video.height": 360,
            "video.pix_fmt": "yuv420p",
            "video.fps": 30,
        },
    }


def test_metadata_video_path_resolution_uses_chunk_and_file() -> None:
    value = resolve_video_path(info(), "observation.images.head_left", 3, 19)
    assert value == "videos/observation.images.head_left/chunk-003/file-019.mp4"


def test_metadata_video_path_rejects_traversal() -> None:
    malicious = dict(info(), video_path="../{video_key}/{file_index}.mp4")
    with pytest.raises(ValueError, match="unsafe"):
        resolve_video_path(malicious, "observation.images.head_left", 0, 0)


def test_packed_video_episode_timestamp_mapping() -> None:
    item = pointer()
    assert episode_frame_timestamp(item, 0) == 10.0
    assert episode_frame_timestamp(item, 16) == pytest.approx(10.0 + 16 / 30)
    with pytest.raises(IndexError):
        episode_frame_timestamp(item, item.length)


def test_missing_video_detection(tmp_path: Path) -> None:
    result = audit_video_inventory(tmp_path, [pointer()])
    assert result["missing_referenced"] == [pointer().relative_path]
    assert result["local_mp4_count"] == 0


@pytest.mark.parametrize(
    "field,value",
    [("width", 320), ("height", 240), ("avg_fps", 29.0), ("codec", "hevc")],
)
def test_wrong_video_metadata_is_rejected(field: str, value) -> None:
    probe = {
        "codec": "h264",
        "width": 640,
        "height": 360,
        "pix_fmt": "yuv420p",
        "avg_fps": 30.0,
        "duration": 1.0,
    }
    probe[field] = value
    assert validate_video_probe(probe, feature())


def test_t_t16_and_action_t_t15_contract() -> None:
    row = make_pair_record(
        split="test",
        source_index=3,
        pointer=pointer(),
        anchor_frame=20,
        anchor_time=20 / 30,
        info=info(),
        contact_transition=2,
        dynamic=False,
    )
    assert row["vision"]["current"]["episode_frame"] == 20
    assert row["vision"]["future"]["episode_frame"] == 36
    assert row["action"]["episode_frames_inclusive"] == [20, 35]
    assert row["contact"]["current_teacher_window_inclusive"] == [5, 20]
    assert row["contact"]["future_teacher_window_inclusive"] == [21, 36]
    assert row["vision"]["relative_path"] == pointer().relative_path


def test_transition_must_stay_in_same_episode() -> None:
    with pytest.raises(ValueError, match="outside the episode"):
        validate_transition_anchor(pointer(length=36, dataset_to_index=1036, to_timestamp=11.2), 20)


def test_episode_split_leakage_is_rejected() -> None:
    assert validate_episode_splits({"train": [0, 1], "val": [2], "test": [3]}) == {
        "train_val": 0,
        "train_test": 0,
        "val_test": 0,
    }
    with pytest.raises(ValueError, match="leakage"):
        validate_episode_splits({"train": [0, 1], "val": [1, 2], "test": [3]})


def test_raw_and_padded_state_action_dimensions_and_values() -> None:
    state = np.arange(58, dtype=np.float32)
    action = np.arange(16 * 58, dtype=np.float32).reshape(16, 58)
    value = pad_trex_state_action(state, action)
    assert value["state"].shape == (128,)
    assert value["action"].shape == (16, 128)
    np.testing.assert_array_equal(value["state"][:58], state)
    np.testing.assert_array_equal(value["action"][:, :58], action)
    assert not value["state"][58:].any()
    assert not value["action"][:, 58:].any()
    assert value["state_mask"].sum() == 58
    assert value["action_mask"].sum() == 16 * 58


def test_train_statistics_normalize_before_padding_without_dimension_collision() -> None:
    state = np.arange(58, dtype=np.float32)
    action = np.arange(16 * 58, dtype=np.float32).reshape(16, 58)
    statistics = {
        "observation.state": {"mean": np.ones(58), "std": np.full(58, 2.0)},
        "action": {"mean": np.ones(58), "std": np.full(58, 4.0)},
    }
    value = normalize_and_pad_trex_state_action(state, action, statistics)
    np.testing.assert_allclose(value["state"][:58], (state - 1) / 2)
    np.testing.assert_allclose(value["action"][:, :58], (action - 1) / 4)
    assert not value["state"][58:].any()
    assert not value["action"][:, 58:].any()


def test_trex_uses_generic_new_embodiment_without_gr1_alias() -> None:
    assert EMBODIMENT_TAG_MAPPING[TREX_EMBODIMENT_TAG] == TREX_EMBODIMENT_ID == 31
    assert EMBODIMENT_TAG_MAPPING["gr1"] != TREX_EMBODIMENT_ID
    embodiment = json.loads(SPEC.read_text())["state_action"]["embodiment"]
    assert embodiment["released_tokenizer_max_num_embodiments"] == 30
    assert embodiment["released_tokenizer_requires_category_expansion"] is True


def test_visual_preprocessing_shape_dtype_and_finiteness() -> None:
    frame = np.arange(360 * 640 * 3, dtype=np.uint8).reshape(360, 640, 3)
    value = preprocess_trex_rgb(frame)
    assert value.shape == (3, 224, 224)
    assert value.dtype == np.float32
    assert np.isfinite(value).all()


def test_contact_contract_and_frozen_hashes_are_public() -> None:
    spec = json.loads(SPEC.read_text())
    contact = spec["contact"]
    assert contact["h_current_shape"] == [256]
    assert contact["h_future_shape"] == [256]
    assert contact["z_c_shape"] == [8, 32]
    assert contact["s1_teacher_checkpoint_sha256"] == "54aedbfe0d72b18822624874ef3724512357c31ea03876513c6dea75d3aae8ac"
    assert contact["s2_encoder_checkpoint_sha256"] == "c36c0531bba461875384cebf6bd91c34d43d3f84d2083c15c47ae7dee4e64fa4"


def test_manifest_hash_is_deterministic() -> None:
    value = {"rows": [{"pair_id": "a"}, {"pair_id": "b"}], "count": 2}
    assert sha256_json(value) == sha256_json(json.loads(json.dumps(value)))


def test_cache_identity_rejects_mismatch() -> None:
    expected = cache_identity(
        teacher_sha256="a" * 64,
        encoder_sha256="b" * 64,
        transition_manifest_sha256="c" * 64,
        split_sha256="d" * 64,
    )
    validate_cache_identity(dict(expected), expected)
    actual = dict(expected)
    actual["transition_horizon_frames"] = 8
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_cache_identity(actual, expected)


def test_public_spec_has_no_private_absolute_paths() -> None:
    text = SPEC.read_text()
    assert "/" + "home/" not in text
    assert "/" + "mnt/" not in text
    assert "TREX_DATASET_DIR" in text
