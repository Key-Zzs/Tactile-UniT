#!/usr/bin/env python3
"""Freeze and audit the exact official DexJoCo pi0.5 training configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEXJOCO = ROOT / "third_party/dexjoco"
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_pi0"
PROTOCOL_PATH = ROOT / "configs/simulation/s4_3_pi0_official_pi05_protocol.json"
OFFICIAL_NORM_PATH = DEXJOCO / "openpi/assets/pinch_tongs/local_repo/norm_stats.json"
RECOMPUTED_NORM_PATH = ROOT / ".local/cache/simulation/s4_3_pi0/assets/pinch_tongs/local_repo/norm_stats.json"
NORM_RECOMPUTE_TOLERANCE = 1e-3

SOURCE_FILES = (
    "openpi/README.md",
    "openpi/config.yaml",
    "openpi/src/openpi/training/config.py",
    "openpi/src/openpi/training/dexjoco_configs.py",
    "openpi/src/openpi/training/optimizer.py",
    "openpi/scripts/compute_norm_stats.py",
    "openpi/scripts/train.py",
    "openpi/assets/pinch_tongs/local_repo/norm_stats.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_leaves(value: Any) -> list[float]:
    if isinstance(value, dict):
        return [item for child in value.values() for item in numeric_leaves(child)]
    if isinstance(value, list):
        return [item for child in value for item in numeric_leaves(child)]
    if isinstance(value, (int, float)):
        return [float(value)]
    return []


def write_json(name: str, payload: Any) -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    target = ARTIFACT_ROOT / name
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    official_norm = json.loads(OFFICIAL_NORM_PATH.read_text(encoding="utf-8"))
    recomputed_norm = json.loads(RECOMPUTED_NORM_PATH.read_text(encoding="utf-8"))
    official_values = numeric_leaves(official_norm)
    recomputed_values = numeric_leaves(recomputed_norm)
    same_shape = len(official_values) == len(recomputed_values)
    max_abs_difference = (
        max(abs(left - right) for left, right in zip(official_values, recomputed_values, strict=True))
        if same_shape
        else None
    )
    norm_gate = same_shape and max_abs_difference is not None and max_abs_difference <= NORM_RECOMPUTE_TOLERANCE

    payload = {
        "schema": "tactile3d-unit.s4-3-pi0-official-training-config.v1",
        "stage": "PI0-5",
        "source_repository": "brave-eai/dexjoco",
        "source_commit": "8d23b0fab23b17a58c4b55f3942e17013aaf8267",
        "tracked_protocol": "configs/simulation/s4_3_pi0_official_pi05_protocol.json",
        "tracked_protocol_sha256": sha256_file(PROTOCOL_PATH),
        "source_file_sha256": {path: sha256_file(DEXJOCO / path) for path in SOURCE_FILES},
        "scope": protocol["scope"],
        "model": protocol["model"],
        "data": protocol["data"],
        "training": protocol["training"],
        "normalization_audit": {
            "workflow": "official scripts/compute_norm_stats.py pinch_tongs",
            "batches": 1252,
            "batch_size": 32,
            "frames": 40065,
            "canonical_training_stats": "third_party/dexjoco/openpi/assets/pinch_tongs/local_repo/norm_stats.json",
            "canonical_sha256": sha256_file(OFFICIAL_NORM_PATH),
            "recomputed_stats": "$REPO_ROOT/.local/cache/simulation/s4_3_pi0/assets/"
            "pinch_tongs/local_repo/norm_stats.json",
            "recomputed_sha256": sha256_file(RECOMPUTED_NORM_PATH),
            "numeric_leaf_count_match": same_shape,
            "max_absolute_difference": max_abs_difference,
            "tolerance": NORM_RECOMPUTE_TOLERANCE,
            "status": "PASS" if norm_gate else "FAIL",
        },
        "runtime_path_overrides_only": {
            "data.root": "$REPO_ROOT/.local/external/s4_3_pi0/datasets/"
            "DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets/pinch_tongs",
            "weight_loader.params_path": "$REPO_ROOT/.local/external/s4_3_pi0/models/"
            "DexJoCo-Pi05/pi05_base/params",
            "checkpoint_base_dir": "$REPO_ROOT/.local/experiments/simulation/s4_3_pi0/training",
        },
        "canonical_norm_stats_selected": True,
        "official_algorithmic_defaults_overridden": False,
        "wandb_mode": "offline (official launcher-supported infrastructure mode)",
        "sweep": False,
        "status": "PASS" if norm_gate else "FAIL",
    }
    write_json("official_training_config.json", payload)
    print(
        json.dumps({"training_config": payload["status"], "max_norm_abs_diff": max_abs_difference}, sort_keys=True)
    )
    if payload["status"] != "PASS":
        raise SystemExit("S4_3_PI0_TRAINING_CONFIG_GATE_FAIL")


if __name__ == "__main__":
    main()
