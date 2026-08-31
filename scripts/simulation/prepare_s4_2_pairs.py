#!/usr/bin/env python3
"""Fit train-only tactile statistics and cache only train/validation S4.2 pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_2_dataset import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    build_pair_arrays,
    canonical_json_sha256,
    load_episode,
    references_for_split,
    sha256_file,
)
from gr00t.simulation.s4_2_normalization import fit_tactile_normalization  # noqa: E402

DEFAULT_CACHE = ROOT / ".local/cache/simulation/s4_2/pairs"
DEFAULT_ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_references = references_for_split("train", args.dataset)
    raw_train = np.concatenate(
        [load_episode(reference)["sim_tactile"] for reference in train_references], axis=0
    )
    candidates = {
        name: fit_tactile_normalization(raw_train, name).to_json() for name in ("N0", "N1")
    }
    normalization = {
        "schema": "tactile3d-unit.s4-2-normalization-candidates.v1",
        "fit_split": "train",
        "fit_frames": int(len(raw_train)),
        "candidates": candidates,
        "selected": None,
        "selection_split": "validation",
        "test_loaded": False,
        "selection_uses_test": False,
        "training_uses_test": False,
    }
    args.cache.mkdir(parents=True, exist_ok=True)
    cache_manifest = {
        "schema": "tactile3d-unit.s4-2-pair-cache.v1",
        "splits": {},
        "test_cached": False,
        "test_loaded": False,
    }
    for split in ("train", "validation"):
        arrays = build_pair_arrays(split, args.dataset)
        tactile = arrays["current_history"].reshape(len(arrays["pair_id"]), 26, 5, 6)
        current_contact = tactile[:, -1, :, 0].sum(axis=1) > 0
        future = arrays["future_history"].reshape(len(arrays["pair_id"]), 26, 5, 6)
        future_contact = future[:, -1, :, 0].sum(axis=1) > 0
        arrays["contact_transition"] = (
            current_contact.astype(np.int8) * 2 + future_contact.astype(np.int8)
        )
        arrays["current_total_force"] = tactile[:, -1, :, 1].sum(axis=1).astype(np.float32)
        arrays["future_total_force"] = future[:, -1, :, 1].sum(axis=1).astype(np.float32)
        arrays["force_delta_abs"] = np.abs(
            arrays["future_total_force"] - arrays["current_total_force"]
        ).astype(np.float32)
        destination = args.cache / f"{split}.npz"
        np.savez_compressed(destination, **arrays)
        cache_manifest["splits"][split] = {
            "path": str(destination.relative_to(ROOT)),
            "sha256": sha256_file(destination),
            "pairs": int(len(arrays["pair_id"])),
            "fields": sorted(arrays),
        }
    cache_manifest["canonical_sha256"] = canonical_json_sha256(cache_manifest["splits"])
    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "normalization_candidates.json").write_text(
        json.dumps(normalization, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.artifacts / "pair_cache_manifest.json").write_text(
        json.dumps(cache_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"normalization": normalization, "cache": cache_manifest}, indent=2))


if __name__ == "__main__":
    main()
