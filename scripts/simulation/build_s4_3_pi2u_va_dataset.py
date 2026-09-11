#!/usr/bin/env python3
"""Materialize clean V/A-only TRAIN and DEV caches from frozen S4.2 pairs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ".local/cache/simulation/s4_2_formal"
OUTPUT = ROOT / ".local/cache/simulation/s4_3_pi2u/va_bridge"
ARTIFACT = ROOT / ".local/artifacts/simulation/s4_3_pi2u/va_dataset_manifest.json"
ALLOWED = ("pair_id", "episode_id", "task", "source_trajectory_id", "anchor_step", "future_step", "z_v", "z_a")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(split: str) -> dict[str, object]:
    source_path = SOURCE / f"paired_{split}.npz"
    with np.load(source_path, allow_pickle=False) as source:
        arrays = {name: np.array(source[name], copy=True) for name in ALLOWED}
    if arrays["z_v"].shape != (len(arrays["pair_id"]), 8, 32):
        raise RuntimeError("Vision source shape mismatch")
    if arrays["z_a"].shape != arrays["z_v"].shape:
        raise RuntimeError("Action source shape mismatch")
    if not np.all(arrays["future_step"] - arrays["anchor_step"] == 27):
        raise RuntimeError("VA transition horizon mismatch")
    output_path = OUTPUT / f"{split}.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    with np.load(output_path, allow_pickle=False) as clean:
        if tuple(clean.files) != ALLOWED:
            raise RuntimeError(f"clean VA cache has unexpected fields: {clean.files}")
        if any(name in clean.files for name in ("z_c", "h_current", "h_future", "dynamic")):
            raise RuntimeError("Contact-derived field entered clean VA cache")
    groups = set(zip(arrays["task"].tolist(), arrays["source_trajectory_id"].tolist()))
    return {
        "pairs": len(arrays["pair_id"]),
        "groups": len(groups),
        "group_values": sorted([list(value) for value in groups]),
        "source_sha256": sha256(source_path),
        "cache": f"$PI2U_ROOT/cache/va_bridge/{split}.npz",
        "cache_sha256": sha256(output_path),
        "fields": list(ALLOWED),
    }


def main() -> None:
    train = build("train")
    validation = build("validation")
    overlap = {tuple(value) for value in train.pop("group_values")} & {
        tuple(value) for value in validation.pop("group_values")
    }
    if overlap:
        raise RuntimeError("VA source-group leakage")
    result = {
        "schema": "tactile3d-unit.s4-3-pi2u-va-dataset.v1",
        "status": "PASS",
        "source": "frozen S4.2 exact paired representation caches",
        "transition": "t -> t+27 (0.54 s)",
        "modalities": ["Vision", "Action"],
        "contact_loaded_into_output": False,
        "contact_derived_selection": False,
        "policy_outcomes_used": False,
        "splits": {"train": train, "validation": validation},
        "source_group_overlap": 0,
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
