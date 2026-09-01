"""Shared validation-only utilities for the S4.2-DR dynamics audit."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gr00t.contact_dynamics.evaluation import different_episode_permutation
from gr00t.contact_dynamics.models import (
    ContactDynamicsEncoder,
    ContactDynamicsModel,
    CurrentOnlyPredictor,
    DeltaMLPEncoder,
    LatentTransitionDecoder,
)


ROOT = Path(__file__).resolve().parents[2]
S42_PAIR_ROOT = ROOT / ".local/cache/simulation/s4_2/pairs"
CACHE_ROOT = ROOT / ".local/cache/simulation/s4_2r/contact_dynamics"
EXPERIMENT_ROOT = ROOT / ".local/experiments/simulation/s4_2r/contact_dynamics"
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_2dr"
DR_CACHE_ROOT = ROOT / ".local/cache/simulation/s4_2dr"
DR_EXPERIMENT_ROOT = ROOT / ".local/experiments/simulation/s4_2dr"
DR_LOG_ROOT = ROOT / ".local/logs/simulation/s4_2dr"
DR_TMP_ROOT = ROOT / ".local/tmp/simulation/s4_2dr"
CONTRACT_PATH = ROOT / "configs/simulation/s4_2_representation_contract.json"
SELECTION_PATH = ROOT / ".local/artifacts/simulation/s4_2r/contact_dynamics_selection.json"
HISTORICAL_PATH = ROOT / "configs/simulation/s4_2r_contact_dynamics_decision.json"
CONTACT_STATE_PATH = ROOT / "configs/simulation/s4_2r_contact_state_acceptance.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def seed_everything(seed: int = 4242) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(name: str) -> torch.nn.Module:
    if name == "C1":
        return CurrentOnlyPredictor()
    if name == "C2":
        return ContactDynamicsModel(DeltaMLPEncoder(), LatentTransitionDecoder())
    if name == "C3":
        return ContactDynamicsModel(ContactDynamicsEncoder(), LatentTransitionDecoder())
    raise ValueError(name)


def load_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    name: str,
    current: np.ndarray,
    future: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray | None]:
    predictions: list[np.ndarray] = []
    codes: list[np.ndarray] = []
    for start in range(0, len(current), batch_size):
        stop = min(start + batch_size, len(current))
        current_t = torch.from_numpy(current[start:stop]).to(device)
        future_t = torch.from_numpy(future[start:stop]).to(device)
        if name == "C1":
            predictions.append(model(current_t).cpu().numpy())
        else:
            output = model(current_t, future_t)
            predictions.append(output["future"].cpu().numpy())
            codes.append(output["code"].cpu().numpy())
    return np.concatenate(predictions), None if not codes else np.concatenate(codes)


@torch.inference_mode()
def decode(
    model: ContactDynamicsModel,
    code: np.ndarray,
    current: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for start in range(0, len(current), batch_size):
        stop = min(start + batch_size, len(current))
        code_t = torch.from_numpy(code[start:stop]).to(device)
        current_t = torch.from_numpy(current[start:stop]).to(device)
        predictions.append(model.decoder(code_t, current_t).cpu().numpy())
    return np.concatenate(predictions)


def model_paths() -> dict[str, Path]:
    selection = load_json(SELECTION_PATH)
    by_model = {row["model"]: ROOT / row["checkpoint"] for row in selection["trials"]}
    return {
        "C1": by_model["C1"],
        "C2": by_model["C2"],
        "C3": ROOT / selection["selected_checkpoint"],
    }


def per_sample_mse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.mean(np.square(prediction.astype(np.float64) - target.astype(np.float64)), axis=1)


def bootstrap_mean_ci(values: np.ndarray, samples: int = 5000, seed: int = 4242) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 100):
        stop = min(samples, start + 100)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).astype(float).tolist()


def metric_rows(
    predictions: dict[str, np.ndarray], target: np.ndarray, masks: dict[str, np.ndarray]
) -> dict[str, dict[str, dict[str, float | int]]]:
    errors = {name: per_sample_mse(value, target) for name, value in predictions.items()}
    return {
        regime: {
            name: {"mse": float(error[mask].mean()), "samples": int(mask.sum())}
            for name, error in errors.items()
        }
        for regime, mask in masks.items()
        if int(mask.sum()) > 0
    }


def control_predictions(
    model: ContactDynamicsModel,
    code: np.ndarray,
    current: np.ndarray,
    future: np.ndarray,
    episode_id: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(4242)
    shuffled = rng.permutation(len(code))
    different = different_episode_permutation(episode_id, seed=4242)
    _, reversed_code = predict(model, "C3", future, current, device)
    _, mismatch_code = predict(model, "C3", current, future[different], device)
    assert reversed_code is not None and mismatch_code is not None
    return {
        "zero": decode(model, np.zeros_like(code), current, device),
        "shuffled": decode(model, code[shuffled], current, device),
        "reversed": decode(model, reversed_code, current, device),
        "different_episode": decode(model, code[different], current, device),
        "mismatch": decode(model, mismatch_code, current, device),
    }


def ensure_local_roots() -> None:
    for path in (ARTIFACT_ROOT, DR_CACHE_ROOT, DR_EXPERIMENT_ROOT, DR_LOG_ROOT, DR_TMP_ROOT):
        path.mkdir(parents=True, exist_ok=True)
