"""Runtime-only data and metric helpers for Track C3-DP commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from gr00t.tactile_unit.c3dp_shared_private import sha256_file, verify_c2r_checkpoint
from gr00t.tactile_unit.continuous_vac_shared_space import (
    load_checkpoint,
    state_dict_digest,
)
from gr00t.tactile_unit.vac_latent_dataset import load_split

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/tactile_unit/c3dp_shared_private_cross_prediction.json"
DERIVED_ARRAYS = (
    "u_v",
    "u_a",
    "u_c",
    "z_v_shared",
    "z_a_shared",
    "z_c_shared",
    "r_c_priv",
    "pair_id",
    "episode_id",
    "t",
    "t_future",
    "dynamic",
    "contact_transition",
)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_npy_atomic(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(value), allow_pickle=False)
    temporary.replace(path)


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_frozen_shared_space(config: Mapping[str, Any], device: torch.device):
    checkpoint = ROOT / config["runtime"]["c2r_checkpoint"]
    digest = verify_c2r_checkpoint(checkpoint)
    model, metadata = load_checkpoint(checkpoint, device)
    model.eval().requires_grad_(False).to(device)
    return model, metadata, digest


def encode_derived_split(
    model, split, device: torch.device, batch_size: int
) -> dict[str, np.ndarray]:
    result = {
        name: np.empty((len(split), 8, 32), dtype=np.float32)
        for name in ("u_v", "u_a", "u_c", "z_v_shared", "z_a_shared", "z_c_shared", "r_c_priv")
    }
    source = {"vision": "z_v", "action": "z_a", "contact": "z_c"}
    short = {"vision": "v", "action": "a", "contact": "c"}
    with torch.inference_mode():
        for start in range(0, len(split), batch_size):
            stop = min(start + batch_size, len(split))
            for modality in ("vision", "action", "contact"):
                native = torch.from_numpy(
                    np.array(split.arrays[source[modality]][start:stop], copy=True)
                ).to(device)
                shared = model.encode(modality, native)
                recovered = model.recover(modality, shared)
                suffix = short[modality]
                result[f"u_{suffix}"][start:stop] = shared.float().cpu().numpy()
                result[f"z_{suffix}_shared"][start:stop] = recovered.float().cpu().numpy()
                if modality == "contact":
                    result["r_c_priv"][start:stop] = (native - recovered).float().cpu().numpy()
    for name in ("pair_id", "episode_id", "t", "t_future", "dynamic", "contact_transition"):
        result[name] = np.asarray(split.arrays[name])
    return result


def save_derived_split(
    root: Path,
    split_name: str,
    arrays: Mapping[str, np.ndarray],
    *,
    c1_manifest_sha256: str,
    c2r_checkpoint_sha256: str,
    shared_state_sha256: str,
) -> dict[str, Any]:
    if set(arrays) != set(DERIVED_ARRAYS):
        raise ValueError("C3-DP derived cache schema mismatch")
    split_root = Path(root) / split_name
    records = {}
    for name in DERIVED_ARRAYS:
        path = split_root / f"{name}.npy"
        write_npy_atomic(path, arrays[name])
        records[name] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "shape": list(arrays[name].shape),
            "dtype": str(arrays[name].dtype),
        }
    manifest = {
        "schema": "tactile3d-unit.vac-c3dp-derived-cache.v1",
        "split": split_name,
        "count": len(arrays["u_v"]),
        "arrays": records,
        "c1_manifest_sha256": c1_manifest_sha256,
        "c2r_checkpoint_sha256": c2r_checkpoint_sha256,
        "shared_state_sha256": shared_state_sha256,
        "test_loaded": split_name == "test",
    }
    atomic_json(split_root / "manifest.json", manifest)
    return manifest


def load_derived_split(
    root: Path,
    split_name: str,
    *,
    verify_hashes: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest_path = Path(root) / split_name / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing C3-DP derived {split_name} cache")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "tactile3d-unit.vac-c3dp-derived-cache.v1":
        raise ValueError("unsupported C3-DP derived cache")
    if set(manifest.get("arrays", {})) != set(DERIVED_ARRAYS):
        raise ValueError("C3-DP derived cache arrays changed")
    arrays = {}
    for name, record in manifest["arrays"].items():
        path = Path(root) / record["path"]
        if verify_hashes and sha256_file(path) != record["sha256"]:
            raise ValueError(f"corrupt C3-DP derived array {name}")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(value.shape) != record["shape"] or str(value.dtype) != record["dtype"]:
            raise ValueError(f"invalid C3-DP derived array {name}")
        arrays[name] = value
    count = int(manifest["count"])
    if any(len(value) != count for value in arrays.values()):
        raise ValueError("unaligned C3-DP derived arrays")
    return arrays, manifest


def build_split(config: Mapping[str, Any], split_name: str, device: torch.device, batch_size: int):
    c1_root = ROOT / config["runtime"]["c1_cache_root"]
    c3_root = ROOT / config["runtime"]["cache_root"]
    c1_manifest = c1_root / "manifest.json"
    before = sha256_file(c1_manifest)
    source = load_split(c1_root, split_name, verify_hashes=True)
    model, _, checkpoint_sha = load_frozen_shared_space(config, device)
    shared_sha = state_dict_digest(model)
    arrays = encode_derived_split(model, source, device, batch_size)
    manifest = save_derived_split(
        c3_root,
        split_name,
        arrays,
        c1_manifest_sha256=before,
        c2r_checkpoint_sha256=checkpoint_sha,
        shared_state_sha256=shared_sha,
    )
    after = sha256_file(c1_manifest)
    if before != after:
        raise RuntimeError("STRUCTURAL_FAIL: C1 manifest changed while building C3-DP cache")
    return manifest


def validate_selection_lock(config: Mapping[str, Any]) -> dict[str, Any]:
    artifact_root = ROOT / config["runtime"]["artifact_root"]
    selection_path = artifact_root / "selection.json"
    digest_path = artifact_root / "selection.sha256"
    if not selection_path.is_file() or not digest_path.is_file():
        raise RuntimeError("C3-DP selection is not frozen")
    expected = digest_path.read_text().split()[0]
    if sha256_file(selection_path) != expected:
        raise RuntimeError("C3-DP selection hash mismatch")
    selection = json.loads(selection_path.read_text())
    if (
        selection.get("test_loaded") is not False
        or selection.get("selection_split") != "validation only"
    ):
        raise RuntimeError("STRUCTURAL_FAIL: C3-DP selection permits test leakage")
    checkpoint = ROOT / selection["checkpoint"]
    if sha256_file(checkpoint) != selection["checkpoint_sha256"]:
        raise RuntimeError("C3-DP selected checkpoint hash mismatch")
    return selection


def ensure_cache_identities(config: Mapping[str, Any], *manifests: Mapping[str, Any]) -> None:
    c1 = sha256_file(ROOT / config["runtime"]["c1_cache_root"] / "manifest.json")
    expected_c2r = str(config["accepted_c2r_checkpoint_sha256"])
    for manifest in manifests:
        if (
            manifest["c1_manifest_sha256"] != c1
            or manifest["c2r_checkpoint_sha256"] != expected_c2r
        ):
            raise RuntimeError("STRUCTURAL_FAIL: derived cache provenance mismatch")


def shared_digest(model) -> str:
    return state_dict_digest(model)
