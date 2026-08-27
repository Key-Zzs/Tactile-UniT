"""Runtime-only loading and freeze helpers for C3-R0 commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from gr00t.tactile_unit.c3r0_conditional_sufficiency import sha256_file
from gr00t.tactile_unit.vac_latent_dataset import load_split
from scripts.tactile_unit.c3dp_runtime import load_derived_split

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/tactile_unit/c3r0_conditional_sufficiency_audit.json"
FREEZE_FILES = (
    "audit_protocol.json",
    "probe_selection.json",
    "knn_protocol.json",
    "deterministic_ceiling_selection.json",
)


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_aligned_split(config: Mapping[str, Any], split: str) -> dict[str, np.ndarray]:
    native = load_split(ROOT / config["runtime"]["c1_cache_root"], split, verify_hashes=True)
    derived, manifest = load_derived_split(
        ROOT / config["runtime"]["c3dp_cache_root"], split, verify_hashes=True
    )
    if len(native) != int(manifest["count"]):
        raise RuntimeError("STRUCTURAL_FAIL: C1/C3-DP row count mismatch")
    for name in ("pair_id", "episode_id", "t", "t_future", "contact_transition", "dynamic"):
        if not np.array_equal(np.asarray(native.arrays[name]), np.asarray(derived[name])):
            raise RuntimeError(f"STRUCTURAL_FAIL: C1/C3-DP {name} mismatch")
    result = {name: value for name, value in derived.items()}
    for name in (
        "z_c", "h_current", "force_trend_class", "primitive_id", "object_id",
        "state", "action", "h_future", "current_force", "future_force", "source_index",
    ):
        result[name] = native.arrays[name]
    return result


def identity_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    runtime = config["runtime"]
    paths = {
        "c1_manifest": runtime["c1_cache_root"] + "/manifest.json",
        "c2": runtime["c2_checkpoint"],
        "c2r": runtime["c2r_checkpoint"],
        "c3dp": runtime["c3dp_checkpoint"],
        "action": runtime["action_checkpoint"],
        "s1": runtime["s1_checkpoint"],
        "s2": runtime["s2_checkpoint"],
    }
    actual = {name: sha256_file(ROOT / path) for name, path in paths.items()}
    expected = {
        "c1_manifest": config["accepted"]["c1_manifest_sha256"],
        "c2": config["accepted"]["c2_checkpoint_sha256"],
        "c2r": config["accepted"]["c2r_checkpoint_sha256"],
        "c3dp": config["accepted"]["c3dp_predictor_sha256"],
        "action": config["accepted"]["action_checkpoint_sha256"],
        "s1": config["accepted"]["s1_checkpoint_sha256"],
        "s2": config["accepted"]["s2_checkpoint_sha256"],
    }
    equality = {name: actual[name] == value for name, value in expected.items()}
    return {"actual": actual, "expected": expected, "equality": equality, "pass": all(equality.values())}


def write_freeze_files(artifact_root: Path, values: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    if set(values) != set(FREEZE_FILES):
        raise ValueError("C3-R0 freeze file set changed")
    hashes: dict[str, str] = {}
    for name in FREEZE_FILES:
        value = dict(values[name])
        if value.get("test_loaded") is not False:
            raise RuntimeError("STRUCTURAL_FAIL: pretest artifact does not lock test_loaded=false")
        path = artifact_root / name
        atomic_json(path, value)
        hashes[name] = sha256_file(path)
    return hashes


def validate_test_freeze(config: Mapping[str, Any]) -> dict[str, Any]:
    artifact_root = ROOT / config["runtime"]["artifact_root"]
    selection_path = artifact_root / "selection.json"
    if not selection_path.is_file():
        raise RuntimeError("C3-R0 selection is not frozen")
    selection = json.loads(selection_path.read_text())
    if selection.get("test_loaded") is not False or selection.get("selection_split") != "validation only":
        raise RuntimeError("STRUCTURAL_FAIL: C3-R0 selection permits test leakage")
    expected = selection.get("protocol_hashes", {})
    if set(expected) != set(FREEZE_FILES):
        raise RuntimeError("STRUCTURAL_FAIL: incomplete C3-R0 protocol freeze")
    for name in FREEZE_FILES:
        path = artifact_root / name
        if not path.is_file() or sha256_file(path) != expected[name]:
            raise RuntimeError(f"STRUCTURAL_FAIL: invalid frozen protocol {name}")
        value = json.loads(path.read_text())
        if value.get("test_loaded") is not False:
            raise RuntimeError("STRUCTURAL_FAIL: test loaded before protocol freeze")
    return selection
