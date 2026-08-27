import copy
import json
from pathlib import Path

import numpy as np
import pytest

from gr00t.tactile_unit.paired_contract import sha256_json
from gr00t.tactile_unit.vac_latent_dataset import (
    CACHE_SCHEMA,
    REQUIRED_ARRAYS,
    VACLatentCacheError,
    canonical_pair_ids,
    deterministic_train_subset,
    deterministic_uniform_subset,
    load_split,
    split_manifest,
    validate_cache,
    write_npy_atomic,
)


def _make_cache(root: Path, episodes=(range(10, 13), range(20, 23), range(30, 33))):
    split_names = ("train", "validation", "test")
    manifests = {}
    for split, episode_values in zip(split_names, episodes):
        episode = np.asarray(list(episode_values), dtype=np.int32)
        count = len(episode)
        split_root = root / split
        arrays = {}
        for name, (dtype, trailing) in REQUIRED_ARRAYS.items():
            if name == "pair_id":
                value = canonical_pair_ids(split, episode, np.arange(15, 15 + count, dtype=np.int32))
            elif name == "episode_id":
                value = episode
            elif name == "t":
                value = np.arange(15, 15 + count, dtype=np.int32)
            elif name == "t_future":
                value = np.arange(31, 31 + count, dtype=np.int32)
            elif name == "source_index":
                value = np.arange(count, dtype=np.int32)
            elif name == "dynamic":
                value = np.arange(count) % 2 == 0
            else:
                value = np.zeros((count, *trailing), dtype=dtype)
            arrays[name] = value
            write_npy_atomic(split_root / f"{name}.npy", value)
        manifests[split] = split_manifest(split_root, root, count)
    manifest = {
        "schema": CACHE_SCHEMA,
        "implementation_version": "test",
        "format": "array-sharded-npy-v1",
        "dtype": "float32",
        "transition_shape": [8, 32],
        "horizon_frames": 16,
        "sample_order": "test",
        "selection_manifest_sha256": "0" * 64,
        "provenance": {"checkpoint": "frozen"},
        "splits": manifests,
    }
    manifest["canonical_sha256"] = sha256_json(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    return manifest


def test_manifest_determinism_and_train_coverage():
    size = 80
    primitive = np.arange(size) % 4
    objects = np.arange(size) % 5
    dynamic = np.arange(size) % 3 == 0
    boundary = np.where(np.arange(size) % 7 == 0, 1, 0)
    first = deterministic_train_subset(
        count=60, seed=9, primitive_id=primitive, object_id=objects,
        dynamic=dynamic, contact_transition=boundary, forced_indices=[3, 11],
    )
    second = deterministic_train_subset(
        count=60, seed=9, primitive_id=primitive, object_id=objects,
        dynamic=dynamic, contact_transition=boundary, forced_indices=[3, 11],
    )
    assert np.array_equal(first, second)
    assert {3, 11} <= set(map(int, first))
    assert np.array_equal(
        deterministic_uniform_subset(50, 12, seed=7, split="validation"),
        deterministic_uniform_subset(50, 12, seed=7, split="validation"),
    )


def test_cache_geometry_order_dtype_finite_and_cold_reload(tmp_path):
    _make_cache(tmp_path)
    first = load_split(tmp_path, "train")
    second = load_split(tmp_path, "train")
    assert first.arrays["z_v"].shape == (3, 8, 32)
    assert first.arrays["z_a"].dtype == np.float32
    assert first.arrays["z_c"].dtype == np.float32
    assert np.array_equal(first.arrays["pair_id"], second.arrays["pair_id"])
    assert np.array_equal(first.arrays["t_future"], first.arrays["t"] + 16)
    result = validate_cache(tmp_path, expected_counts={"train": 3, "validation": 3, "test": 3})
    assert result["status"] == "C1_READY"
    assert result["modality_order"] == "PASS"


def test_duplicate_pair_rejected(tmp_path):
    manifest = _make_cache(tmp_path)
    pair_id = np.asarray(np.load(tmp_path / "train" / "pair_id.npy", allow_pickle=False)).copy()
    pair_id[1] = pair_id[0]
    write_npy_atomic(tmp_path / "train" / "pair_id.npy", pair_id)
    manifest["splits"]["train"] = split_manifest(tmp_path / "train", tmp_path, len(pair_id))
    manifest["canonical_sha256"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "canonical_sha256"}
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    with pytest.raises(VACLatentCacheError, match="pair_id"):
        validate_cache(tmp_path)


def test_split_leakage_rejected(tmp_path):
    _make_cache(tmp_path, episodes=(range(10, 13), range(10, 13), range(30, 33)))
    with pytest.raises(VACLatentCacheError, match="episode split leakage"):
        validate_cache(tmp_path)


def test_corrupted_array_and_provenance_rejected(tmp_path):
    manifest = _make_cache(tmp_path)
    value = np.load(tmp_path / "train" / "z_v.npy", allow_pickle=False)
    value = np.asarray(value).copy()
    value[0, 0, 0] = 1.0
    write_npy_atomic(tmp_path / "train" / "z_v.npy", value)
    with pytest.raises(VACLatentCacheError, match="corrupted VAC array"):
        load_split(tmp_path, "train")

    _make_cache(tmp_path)
    damaged = copy.deepcopy(manifest)
    damaged["provenance"]["checkpoint"] = "changed"
    (tmp_path / "manifest.json").write_text(json.dumps(damaged))
    with pytest.raises(VACLatentCacheError, match="canonical digest"):
        load_split(tmp_path, "train")


def test_exact_960_identity_guard_rejects_incomplete_list(tmp_path):
    _make_cache(tmp_path)
    with pytest.raises(VACLatentCacheError, match="canonical 960"):
        validate_cache(tmp_path, canonical_pair_ids_960=["only-one"])
