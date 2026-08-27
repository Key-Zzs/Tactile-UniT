"""Runtime-only loading, identity, and selection-lock helpers for C3-MS-CC."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from gr00t.tactile_unit.c3mscc_contact_context import sha256_file
from gr00t.tactile_unit.continuous_vac_shared_space import load_checkpoint, state_dict_digest
from gr00t.tactile_unit.vac_latent_dataset import load_split
from scripts.tactile_unit.c3dp_runtime import load_derived_split

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/tactile_unit/c3mscc_contact_context_prediction.json"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_aligned_split(config: Mapping[str, Any], split: str) -> dict[str, np.ndarray]:
    native = load_split(ROOT / config["runtime"]["c1_cache_root"], split, verify_hashes=True)
    shared, manifest = load_derived_split(
        ROOT / config["runtime"]["c3dp_cache_root"], split, verify_hashes=True
    )
    if len(native) != int(manifest["count"]):
        raise RuntimeError("STRUCTURAL_FAIL: C1/C3-DP row count mismatch")
    for name in ("pair_id", "episode_id", "t", "t_future", "contact_transition", "dynamic"):
        if not np.array_equal(np.asarray(native.arrays[name]), np.asarray(shared[name])):
            raise RuntimeError(f"STRUCTURAL_FAIL: C1/C3-DP {name} mismatch")
    result = {name: value for name, value in shared.items()}
    for name in (
        "h_current", "h_future", "force_trend_class", "current_force", "future_force",
        "state", "action", "primitive_id", "object_id", "source_index", "z_c",
    ):
        result[name] = native.arrays[name]
    expected = int(config["counts"][split])
    if len(result["u_c"]) != expected:
        raise RuntimeError(f"STRUCTURAL_FAIL: {split} row count changed")
    return result


def load_frozen_shared_space(config: Mapping[str, Any], device: torch.device):
    path = ROOT / config["runtime"]["c2r_checkpoint"]
    expected = config["accepted"]["c2r_checkpoint_sha256"]
    if sha256_file(path) != expected:
        raise RuntimeError("STRUCTURAL_FAIL: C2-R checkpoint changed")
    model, metadata = load_checkpoint(path, map_location=device)
    model.eval().requires_grad_(False).to(device)
    digest = state_dict_digest(model)
    if digest != config["accepted"]["shared_state_sha256"]:
        raise RuntimeError("STRUCTURAL_FAIL: C2-R state changed")
    return model, metadata, digest


def identity_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    runtime = config["runtime"]
    paths = {
        "c1": runtime["c1_cache_root"] + "/manifest.json",
        "c2": runtime["c2_checkpoint"],
        "c2r": runtime["c2r_checkpoint"],
        "c3dp": runtime["c3dp_checkpoint"],
        "action": runtime["action_checkpoint"],
        "s1": runtime["s1_checkpoint"],
        "s2": runtime["s2_checkpoint"],
    }
    expected_keys = {
        "c1": "c1_manifest_sha256", "c2": "c2_checkpoint_sha256",
        "c2r": "c2r_checkpoint_sha256", "c3dp": "c3dp_checkpoint_sha256",
        "action": "action_checkpoint_sha256", "s1": "s1_checkpoint_sha256",
        "s2": "s2_checkpoint_sha256",
    }
    actual = {name: sha256_file(ROOT / path) for name, path in paths.items()}
    expected = {
        name: str(config["accepted"][key]) for name, key in expected_keys.items()
    }
    equality = {name: actual[name] == expected[name] for name in paths}
    return {"actual": actual, "expected": expected, "equality": equality, "pass": all(equality.values())}


def validate_selection_lock(config: Mapping[str, Any]) -> dict[str, Any]:
    root = ROOT / config["runtime"]["artifact_root"]
    path = root / "selection.json"
    digest_path = root / "selection.sha256"
    if not path.is_file() or not digest_path.is_file():
        raise RuntimeError("C3-MS-CC selection is not frozen")
    if sha256_file(path) != digest_path.read_text().split()[0]:
        raise RuntimeError("STRUCTURAL_FAIL: C3-MS-CC selection hash mismatch")
    selection = json.loads(path.read_text())
    if selection.get("test_loaded") is not False or selection.get("selected_via") != "VALIDATION ONLY":
        raise RuntimeError("STRUCTURAL_FAIL: test permitted during C3-MS-CC selection")
    checkpoint = ROOT / selection["checkpoint"]
    if sha256_file(checkpoint) != selection["checkpoint_sha256"]:
        raise RuntimeError("STRUCTURAL_FAIL: selected checkpoint changed")
    return selection
