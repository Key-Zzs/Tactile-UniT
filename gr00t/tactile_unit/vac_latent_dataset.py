"""Frozen array-sharded dataset contract for continuous VAC transition latents.

The cache deliberately stores no RGB and never deserializes Python objects.
Every row is keyed by the accepted S2 ``(split, episode, t)`` identity, and all
three native representations share one immutable row order.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .paired_contract import CANONICAL_HORIZON, pair_id, sha256_file, sha256_json


CACHE_SCHEMA = "tactile3d-unit.vac-c1-array-cache.v1"
SPLIT_SCHEMA = "tactile3d-unit.vac-c1-array-split.v1"
PUBLIC_TO_SOURCE = {"train": "train", "validation": "val", "test": "test"}
TRANSITION_SHAPE = (8, 32)


class VACLatentCacheError(ValueError):
    """Raised when identity, provenance, or geometry is unsafe."""


REQUIRED_ARRAYS: dict[str, tuple[np.dtype, tuple[int, ...]]] = {
    "pair_id": (np.dtype("U64"), ()),
    "episode_id": (np.dtype("int32"), ()),
    "t": (np.dtype("int32"), ()),
    "t_future": (np.dtype("int32"), ()),
    "source_index": (np.dtype("int32"), ()),
    "z_v": (np.dtype("float32"), TRANSITION_SHAPE),
    "z_a": (np.dtype("float32"), TRANSITION_SHAPE),
    "z_c": (np.dtype("float32"), TRANSITION_SHAPE),
    "dynamic": (np.dtype("bool"), ()),
    "contact_transition": (np.dtype("int8"), ()),
    "force_trend_class": (np.dtype("int8"), ()),
    "primitive_id": (np.dtype("int16"), ()),
    "object_id": (np.dtype("int16"), ()),
    "task_id": (np.dtype("int64"), ()),
    "state": (np.dtype("float32"), (128,)),
    "action": (np.dtype("float32"), (16, 128)),
    "h_current": (np.dtype("float32"), (256,)),
    "h_future": (np.dtype("float32"), (256,)),
    "current_force": (np.dtype("float32"), ()),
    "future_force": (np.dtype("float32"), ()),
}


def canonical_pair_ids(
    split: str, episode_id: np.ndarray, anchor_frame: np.ndarray
) -> np.ndarray:
    if split not in PUBLIC_TO_SOURCE:
        raise VACLatentCacheError(f"unknown public split {split!r}")
    source_split = PUBLIC_TO_SOURCE[split]
    return np.asarray(
        [pair_id(source_split, int(episode), int(t)) for episode, t in zip(episode_id, anchor_frame)],
        dtype="U64",
    )


def pair_id_digest(values: Sequence[str] | np.ndarray) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def _hash_score(seed: int, split: str, index: int) -> int:
    payload = f"vac-c1:{seed}:{split}:{index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def deterministic_uniform_subset(size: int, count: int, *, seed: int, split: str) -> np.ndarray:
    if count < 0 or count > size:
        raise VACLatentCacheError("subset count is outside source split")
    if count == size:
        return np.arange(size, dtype=np.int64)
    ranked = sorted(range(size), key=lambda index: (_hash_score(seed, split, index), index))
    return np.asarray(sorted(ranked[:count]), dtype=np.int64)


def deterministic_train_subset(
    *,
    count: int,
    seed: int,
    primitive_id: np.ndarray,
    object_id: np.ndarray,
    dynamic: np.ndarray,
    contact_transition: np.ndarray,
    forced_indices: Iterable[int] = (),
) -> np.ndarray:
    """Select a broad deterministic train-only subset without test feedback.

    One row per observed primitive/object/dynamic/boundary stratum is included
    first.  Dynamic and free/contact boundaries receive only a modest ranking
    preference for the remainder.  Forced rows are identity-safe reusable C0
    cache rows and never depend on validation or test results.
    """

    arrays = [np.asarray(value) for value in (primitive_id, object_id, dynamic, contact_transition)]
    size = len(arrays[0])
    if any(len(value) != size for value in arrays) or count > size:
        raise VACLatentCacheError("unaligned train selection metadata")
    forced = {int(index) for index in forced_indices}
    if any(index < 0 or index >= size for index in forced):
        raise VACLatentCacheError("forced reusable row is outside train split")

    strata: dict[tuple[int, int, bool, bool], tuple[int, int]] = {}
    for index in range(size):
        key = (
            int(primitive_id[index]),
            int(object_id[index]),
            bool(dynamic[index]),
            int(contact_transition[index]) in {1, 2},
        )
        score = _hash_score(seed, "train-stratum", index)
        if key not in strata or (score, index) < strata[key]:
            strata[key] = (score, index)
    selected = forced | {value[1] for value in strata.values()}
    if len(selected) > count:
        raise VACLatentCacheError("coverage and reusable rows exceed requested train subset")

    maximum = float(2**64 - 1)
    candidates: list[tuple[float, int]] = []
    for index in range(size):
        if index in selected:
            continue
        weight = 1.0
        if bool(dynamic[index]):
            weight *= 2.0
        if int(contact_transition[index]) in {1, 2}:
            weight *= 2.0
        candidates.append((_hash_score(seed, "train-fill", index) / maximum / weight, index))
    candidates.sort()
    selected.update(index for _, index in candidates[: count - len(selected)])
    return np.asarray(sorted(selected), dtype=np.int64)


def write_npy_atomic(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(value), allow_pickle=False)
    temporary.replace(path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def array_record(path: Path, root: Path) -> dict[str, Any]:
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "bytes": path.stat().st_size,
    }

def split_manifest(split_root: Path, cache_root: Path, count: int) -> dict[str, Any]:
    arrays = {
        name: array_record(split_root / f"{name}.npy", cache_root)
        for name in REQUIRED_ARRAYS
    }
    value: dict[str, Any] = {
        "schema": SPLIT_SCHEMA,
        "split": split_root.name,
        "count": int(count),
        "sample_order": "ascending frozen S2 source_index",
        "arrays": arrays,
    }
    value["canonical_sha256"] = sha256_json(value)
    return value


@dataclass
class VACLatentSplit:
    root: Path
    metadata: Mapping[str, Any]
    arrays: dict[str, np.ndarray]

    def __len__(self) -> int:
        return int(self.metadata["count"])

    def batch(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        index = np.asarray(indices, dtype=np.int64)
        return {name: np.asarray(value[index]) for name, value in self.arrays.items()}


def _dtype_matches(name: str, actual: np.dtype, expected: np.dtype) -> bool:
    if name == "pair_id":
        return actual.kind == "U" and actual.itemsize <= expected.itemsize
    return actual == expected


def load_split(
    cache_root: Path,
    split: str,
    *,
    verify_hashes: bool = True,
    mmap_mode: str | None = "r",
) -> VACLatentSplit:
    root = Path(cache_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise VACLatentCacheError("missing frozen VAC cache manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != CACHE_SCHEMA:
        raise VACLatentCacheError("unsupported VAC cache schema")
    if sha256_json({key: value for key, value in manifest.items() if key != "canonical_sha256"}) != manifest.get("canonical_sha256"):
        raise VACLatentCacheError("VAC root manifest canonical digest mismatch")
    if split not in manifest.get("splits", {}):
        raise VACLatentCacheError(f"split {split!r} is absent")
    metadata = manifest["splits"][split]
    split_root = root / split
    arrays: dict[str, np.ndarray] = {}
    count = int(metadata["count"])
    if set(metadata.get("arrays", {})) != set(REQUIRED_ARRAYS):
        raise VACLatentCacheError("VAC split array schema mismatch")
    for name, (dtype, trailing) in REQUIRED_ARRAYS.items():
        record = metadata["arrays"][name]
        path = root / record["path"]
        if not path.is_file():
            raise VACLatentCacheError(f"missing VAC array {name}")
        if verify_hashes and sha256_file(path) != record["sha256"]:
            raise VACLatentCacheError(f"corrupted VAC array {name}")
        value = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
        if value.shape != (count, *trailing) or not _dtype_matches(name, value.dtype, dtype):
            raise VACLatentCacheError(f"invalid {name} geometry or dtype")
        arrays[name] = value

    expected_ids = canonical_pair_ids(split, arrays["episode_id"], arrays["t"])
    if not np.array_equal(arrays["pair_id"], expected_ids):
        raise VACLatentCacheError("pair_id does not match split/episode/t identity")
    if len(np.unique(arrays["pair_id"])) != count:
        raise VACLatentCacheError("duplicate pair_id inside split")
    if not np.array_equal(arrays["t_future"], arrays["t"] + CANONICAL_HORIZON):
        raise VACLatentCacheError("VAC cache contains a non-canonical transition")
    if count > 1 and np.any(np.diff(np.asarray(arrays["source_index"], dtype=np.int64)) <= 0):
        raise VACLatentCacheError("source_index order is not strictly increasing")
    for name in ("z_v", "z_a", "z_c", "state", "action", "h_current", "h_future"):
        if not np.isfinite(arrays[name]).all():
            raise VACLatentCacheError(f"non-finite values in {name}")
    return VACLatentSplit(split_root, metadata, arrays)


def validate_cache(
    cache_root: Path,
    *,
    expected_counts: Mapping[str, int] | None = None,
    canonical_pair_ids_960: Sequence[str] | None = None,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    loaded = {
        split: load_split(cache_root, split, verify_hashes=verify_hashes)
        for split in ("train", "validation", "test")
    }
    if expected_counts is not None:
        for split, count in expected_counts.items():
            if len(loaded[split]) != int(count):
                raise VACLatentCacheError(f"unexpected {split} count")
    ids = {split: set(map(str, value.arrays["pair_id"])) for split, value in loaded.items()}
    if ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"]:
        raise VACLatentCacheError("cross-split duplicate pair_id")
    episodes = {
        split: set(map(int, np.asarray(value.arrays["episode_id"])))
        for split, value in loaded.items()
    }
    if episodes["train"] & episodes["validation"] or episodes["train"] & episodes["test"] or episodes["validation"] & episodes["test"]:
        raise VACLatentCacheError("episode split leakage")
    canonical_ok = None
    if canonical_pair_ids_960 is not None:
        wanted = list(map(str, canonical_pair_ids_960))
        canonical_ok = len(wanted) == 960 and len(set(wanted)) == 960 and set(wanted) <= ids["test"]
        if not canonical_ok:
            raise VACLatentCacheError("exact canonical 960 identity is not contained in test")
    return {
        "status": "C1_READY",
        "counts": {split: len(value) for split, value in loaded.items()},
        "pair_id_digest": {
            split: pair_id_digest(value.arrays["pair_id"]) for split, value in loaded.items()
        },
        "canonical_960": canonical_ok,
        "episode_leakage": {"train_validation": 0, "train_test": 0, "validation_test": 0},
        "modality_order": "PASS",
        "finite_float32": "PASS",
    }
