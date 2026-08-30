"""Runtime helpers shared by Track C4 commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from gr00t.tactile_unit.c3mscc_contact_context import (
    load_checkpoint as load_full_checkpoint,
)
from gr00t.tactile_unit.c4_availability_conditioning import (
    ContactFallbackPredictor, load_fallback_checkpoint, sha256_file,
)
from gr00t.tactile_unit.c4_uncertainty import load_uncertainty_checkpoint
from scripts.tactile_unit.c3mscc_runtime import (
    identity_snapshot as parent_identity_snapshot,
    load_aligned_split,
    load_frozen_shared_space,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/tactile_unit/c4_missing_modality_uncertainty.json"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_parent_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads((ROOT / config["runtime"]["parent_config"]).read_text())


def atomic_json(path: Path, value: Mapping[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    def encode(item: Any):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=encode) + "\n")
    temporary.replace(path)


def identity_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    runtime = config["runtime"]
    paths = {
        "c1": runtime["c1_cache_root"] + "/manifest.json",
        "c2": runtime["c2_checkpoint"], "c2r": runtime["c2r_checkpoint"],
        "c3dp": runtime["c3dp_checkpoint"], "full": runtime["full_checkpoint"],
        "action": runtime["action_checkpoint"], "s1": runtime["s1_checkpoint"],
        "s2": runtime["s2_checkpoint"],
    }
    keys = {
        "c1": "c1_manifest_sha256", "c2": "c2_checkpoint_sha256",
        "c2r": "c2r_checkpoint_sha256", "c3dp": "c3dp_checkpoint_sha256",
        "full": "full_checkpoint_sha256", "action": "action_checkpoint_sha256",
        "s1": "s1_checkpoint_sha256", "s2": "s2_checkpoint_sha256",
    }
    actual = {name: sha256_file(ROOT / path) for name, path in paths.items()}
    expected = {name: str(config["accepted"][keys[name]]) for name in paths}
    equality = {name: actual[name] == expected[name] for name in paths}
    return {"actual": actual, "expected": expected, "equality": equality, "pass": all(equality.values())}


def load_split(config: Mapping[str, Any], split: str) -> dict[str, np.ndarray]:
    parent = load_parent_config(config)
    value = load_aligned_split(parent, split)
    exact_root = ROOT / config["runtime"]["exact_cache_root"] / split
    value = dict(value)
    variants = ("correct", "reversed", "shuffled", "different")
    for variant in variants:
        path = exact_root / f"u_a_{variant}.npy"
        if not path.is_file():
            raise RuntimeError(f"STRUCTURAL_FAIL: exact Action cache missing {path.name}")
        value[f"u_a_{variant}"] = np.load(path, mmap_mode="r", allow_pickle=False)
    value["u_a"] = value["u_a_correct"]
    if len(value["u_c"]) != int(config["counts"][split]):
        raise RuntimeError(f"STRUCTURAL_FAIL: {split} count changed")
    return value


def load_full(config: Mapping[str, Any], device: torch.device):
    path = ROOT / config["runtime"]["full_checkpoint"]
    if sha256_file(path) != config["accepted"]["full_checkpoint_sha256"]:
        raise RuntimeError("C4_FULL_PATH_CHECKPOINT_INVALID")
    model, metadata = load_full_checkpoint(path, device)
    if model.source != "AH":
        raise RuntimeError("C4_FULL_PATH_CHECKPOINT_INVALID")
    return model.eval().requires_grad_(False), metadata


def predict_fallback(
    model: ContactFallbackPredictor,
    split: Mapping[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    *,
    u_a: np.ndarray | None = None,
    u_v: np.ndarray | None = None,
) -> np.ndarray:
    action = np.asarray(split["u_a"] if u_a is None else u_a)
    vision = None
    if model.source == "VA":
        vision = np.asarray(split["u_v"] if u_v is None else u_v)
    output = np.empty((len(action), 8, 32), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(action), batch_size):
            stop = min(start + batch_size, len(action))
            a = torch.from_numpy(np.array(action[start:stop], copy=True)).to(device)
            v = None if vision is None else torch.from_numpy(
                np.array(vision[start:stop], copy=True)
            ).to(device)
            output[start:stop] = model(a, v).float().cpu().numpy()
    return output


def validate_fallback_selection(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    root = ROOT / config["runtime"]["artifact_root"]
    path = root / "fallback_selection.json"
    digest_path = root / "fallback_selection.sha256"
    if not path.is_file() or not digest_path.is_file():
        raise RuntimeError("C4 fallback selection is not frozen")
    digest = sha256_file(path)
    if digest != digest_path.read_text().split()[0]:
        raise RuntimeError("STRUCTURAL_FAIL: C4 fallback selection hash mismatch")
    value = json.loads(path.read_text())
    if value.get("test_loaded") is not False or value.get("selected_via") != "VALIDATION ONLY":
        raise RuntimeError("STRUCTURAL_FAIL: invalid fallback selection protocol")
    if sha256_file(ROOT / value["checkpoint"]) != value["checkpoint_sha256"]:
        raise RuntimeError("STRUCTURAL_FAIL: selected fallback changed")
    return value, digest


def load_selected_fallback(config: Mapping[str, Any], device: torch.device):
    selection, digest = validate_fallback_selection(config)
    model, metadata = load_fallback_checkpoint(ROOT / selection["checkpoint"], device)
    if metadata.get("test_loaded") is not False:
        raise RuntimeError("STRUCTURAL_FAIL: fallback checkpoint saw test")
    return model.eval().requires_grad_(False), selection, digest


def validate_uncertainty_selection(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    root = ROOT / config["runtime"]["artifact_root"]
    path = root / "uncertainty_selection.json"
    digest_path = root / "uncertainty_selection.sha256"
    if not path.is_file() or not digest_path.is_file():
        raise RuntimeError("C4 uncertainty selection is not frozen")
    digest = sha256_file(path)
    if digest != digest_path.read_text().split()[0]:
        raise RuntimeError("STRUCTURAL_FAIL: uncertainty selection hash mismatch")
    value = json.loads(path.read_text())
    if value.get("test_loaded") is not False or value.get("selected_via") != "VALIDATION ONLY":
        raise RuntimeError("STRUCTURAL_FAIL: invalid uncertainty selection protocol")
    if sha256_file(ROOT / value["checkpoint"]) != value["checkpoint_sha256"]:
        raise RuntimeError("STRUCTURAL_FAIL: selected uncertainty checkpoint changed")
    return value, digest


def load_selected_uncertainty(config: Mapping[str, Any], device: torch.device):
    selection, digest = validate_uncertainty_selection(config)
    model, metadata = load_uncertainty_checkpoint(ROOT / selection["checkpoint"], device)
    if metadata.get("test_loaded") is not False:
        raise RuntimeError("STRUCTURAL_FAIL: uncertainty checkpoint saw test")
    return model.eval().requires_grad_(False), selection, digest
