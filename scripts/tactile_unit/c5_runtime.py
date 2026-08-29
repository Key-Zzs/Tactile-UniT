"""Loading, identity, and inference helpers shared by Track C5 commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from gr00t.tactile_unit.c3mscc_contact_context import load_checkpoint as load_full_checkpoint
from gr00t.tactile_unit.c4_availability_conditioning import load_fallback_checkpoint, sha256_file
from gr00t.tactile_unit.c5_causal_visual import VisualSupport, load_causal_checkpoint
from gr00t.tactile_unit.c5_uncertainty import load_c5_uncertainty_checkpoint
from scripts.tactile_unit.c3mscc_runtime import load_aligned_split


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/tactile_unit/c5_causal_visual_planned_action.json"
C3_CONFIG = ROOT / "configs/tactile_unit/c3mscc_contact_context_prediction.json"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


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
        "offline_va": runtime["offline_va_checkpoint"], "emergency_a": runtime["emergency_a_checkpoint"],
        "c4_uncertainty": runtime["c4_uncertainty_checkpoint"], "action": runtime["action_checkpoint"],
        "s1": runtime["s1_checkpoint"], "s2": runtime["s2_checkpoint"],
    }
    keys = {
        "c1": "c1_manifest_sha256", "c2": "c2_checkpoint_sha256", "c2r": "c2r_checkpoint_sha256",
        "c3dp": "c3dp_checkpoint_sha256", "full": "full_checkpoint_sha256",
        "offline_va": "offline_va_checkpoint_sha256", "emergency_a": "emergency_a_checkpoint_sha256",
        "c4_uncertainty": "c4_uncertainty_checkpoint_sha256", "action": "action_checkpoint_sha256",
        "s1": "s1_checkpoint_sha256", "s2": "s2_checkpoint_sha256",
    }
    actual = {name: sha256_file(ROOT / path) for name, path in paths.items()}
    expected = {name: str(config["accepted"][keys[name]]) for name in paths}
    equality = {name: actual[name] == expected[name] for name in paths}
    return {"actual": actual, "expected": expected, "equality": equality, "pass": all(equality.values())}


def load_split(config: Mapping[str, Any], split: str) -> dict[str, np.ndarray]:
    if split not in config["counts"]:
        raise ValueError("unknown C5 split")
    parent = json.loads(C3_CONFIG.read_text())
    value = dict(load_aligned_split(parent, split))
    exact = ROOT / config["runtime"]["exact_action_cache_root"] / split
    for variant in ("correct", "reversed", "shuffled", "different"):
        value[f"u_a_{variant}"] = np.load(exact / f"u_a_{variant}.npy", mmap_mode="r", allow_pickle=False)
    value["u_a"] = value["u_a_correct"]
    visual = ROOT / config["runtime"]["cache_root"] / split
    manifest_path = visual / "manifest.json"
    if not manifest_path.is_file() or not json.loads(manifest_path.read_text()).get("complete"):
        raise RuntimeError(f"C5 causal visual cache is incomplete for {split}")
    value["frame_features"] = np.load(visual / "frame_features.npy", mmap_mode="r", allow_pickle=False)
    value["current_feature_index"] = np.load(visual / "current_feature_index.npy", mmap_mode="r", allow_pickle=False)
    value["history_feature_index"] = np.load(visual / "history_feature_index.npy", mmap_mode="r", allow_pickle=False)
    normalization_path = ROOT / config["runtime"]["cache_root"] / "visual_feature_normalization.json"
    if not normalization_path.is_file():
        raise RuntimeError("C5 train-only visual normalization is missing")
    normalization = json.loads(normalization_path.read_text())
    if normalization.get("validation_or_test_used_for_fit") is not False or not str(normalization.get("fit_split", "")).startswith("frozen C1 train"):
        raise RuntimeError("C5 visual normalization is not train-only")
    value["visual_feature_mean"] = np.asarray(normalization["mean"], dtype=np.float32)
    value["visual_feature_std"] = np.asarray(normalization["std"], dtype=np.float32)
    if value["visual_feature_mean"].shape != (32,) or value["visual_feature_std"].shape != (32,) or np.any(value["visual_feature_std"] <= 0):
        raise RuntimeError("C5 visual normalization geometry is invalid")
    if len(value["u_c"]) != int(config["counts"][split]):
        raise RuntimeError(f"STRUCTURAL_FAIL: {split} count changed")
    return value


def visual_batch(split: Mapping[str, np.ndarray], support: VisualSupport, rows: np.ndarray | slice) -> np.ndarray:
    bank = split["frame_features"]
    if support is VisualSupport.CURRENT_FRAME:
        indices = np.asarray(split["current_feature_index"])[rows]
        value = np.array(bank[indices], copy=True)[:, None]
        return (value - split["visual_feature_mean"]) / split["visual_feature_std"]
    if support is VisualSupport.CAUSAL_HISTORY_8:
        indices = np.asarray(split["history_feature_index"])[rows]
        value = np.array(bank[indices], copy=True)
        return (value - split["visual_feature_mean"]) / split["visual_feature_std"]
    raise ValueError("NONE has no visual batch")


@torch.inference_mode()
def predict_causal(visual, predictor, split: Mapping[str, np.ndarray], support: VisualSupport, device: torch.device, batch_size: int, *, u_a: np.ndarray | None = None, visual_override: np.ndarray | None = None) -> np.ndarray:
    action = np.asarray(split["u_a"] if u_a is None else u_a)
    output = np.empty((len(action), 8, 32), dtype=np.float32)
    visual.eval(); predictor.eval()
    for start in range(0, len(action), batch_size):
        stop = min(start + batch_size, len(action))
        features = visual_batch(split, support, slice(start, stop)) if visual_override is None else np.array(visual_override[start:stop], copy=True)
        c_v = visual(torch.from_numpy(features).to(device))
        a = torch.from_numpy(np.array(action[start:stop], copy=True)).to(device)
        result = predictor(c_v, a)
        output[start:stop] = result[0].float().cpu().numpy()
    return output


def load_c4_fallbacks(config: Mapping[str, Any], device: torch.device):
    va, va_meta = load_fallback_checkpoint(ROOT / config["runtime"]["offline_va_checkpoint"], device)
    a, a_meta = load_fallback_checkpoint(ROOT / config["runtime"]["emergency_a_checkpoint"], device)
    if va.source != "VA" or a.source != "A":
        raise RuntimeError("C4 fallback source identity changed")
    return va.eval().requires_grad_(False), a.eval().requires_grad_(False), va_meta, a_meta


def load_full(config: Mapping[str, Any], device: torch.device):
    model, metadata = load_full_checkpoint(ROOT / config["runtime"]["full_checkpoint"], device)
    if model.source != "AH":
        raise RuntimeError("C5_FULL_PATH_REGRESSION: full checkpoint is not AH")
    return model.eval().requires_grad_(False), metadata


def validate_causal_selection(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    root = ROOT / config["runtime"]["artifact_root"]
    path, digest_path = root / "causal_visual_selection.json", root / "causal_visual_selection.sha256"
    if not path.is_file() or not digest_path.is_file():
        raise RuntimeError("C5 causal mean fallback is not frozen")
    digest = sha256_file(path)
    if digest != digest_path.read_text().split()[0]:
        raise RuntimeError("C5 causal selection hash mismatch")
    value = json.loads(path.read_text())
    if value.get("test_loaded") is not False or value.get("selected_via") != "VALIDATION ONLY":
        raise RuntimeError("invalid C5 mean selection protocol")
    if sha256_file(ROOT / value["checkpoint"]) != value["checkpoint_sha256"]:
        raise RuntimeError("selected C5 mean checkpoint changed")
    return value, digest


def load_selected_causal(config: Mapping[str, Any], device: torch.device):
    selection, digest = validate_causal_selection(config)
    offline_va, _, _, _ = load_c4_fallbacks(config, device)
    visual, predictor, metadata = load_causal_checkpoint(ROOT / selection["checkpoint"], device, frozen_f_va=offline_va)
    if metadata.get("test_loaded") is not False:
        raise RuntimeError("selected C5 mean predictor saw test")
    return visual.eval().requires_grad_(False), predictor.eval().requires_grad_(False), selection, digest


def validate_uncertainty_selection(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    root = ROOT / config["runtime"]["artifact_root"]
    path, digest_path = root / "uncertainty_selection.json", root / "uncertainty_selection.sha256"
    if not path.is_file() or not digest_path.is_file():
        raise RuntimeError("C5 uncertainty is not frozen")
    digest = sha256_file(path)
    if digest != digest_path.read_text().split()[0]:
        raise RuntimeError("C5 uncertainty selection hash mismatch")
    value = json.loads(path.read_text())
    if value.get("test_loaded") is not False or value.get("selected_via") != "VALIDATION ONLY":
        raise RuntimeError("invalid C5 uncertainty selection protocol")
    if sha256_file(ROOT / value["checkpoint"]) != value["checkpoint_sha256"]:
        raise RuntimeError("selected C5 uncertainty checkpoint changed")
    return value, digest


def load_selected_uncertainty(config: Mapping[str, Any], device: torch.device):
    selection, digest = validate_uncertainty_selection(config)
    model, metadata = load_c5_uncertainty_checkpoint(ROOT / selection["checkpoint"], device)
    if metadata.get("test_loaded") is not False:
        raise RuntimeError("selected C5 uncertainty saw test")
    return model.eval().requires_grad_(False), selection, digest
