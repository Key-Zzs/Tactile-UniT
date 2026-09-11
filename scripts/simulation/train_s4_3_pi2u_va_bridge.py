#!/usr/bin/env python3
"""Train and structurally validate the one preregistered VA-only bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
import shutil

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_3_pi2u_va import VAOnlyBridge, VALossWeights, va_bridge_loss  # noqa: E402


CONFIG = ROOT / "configs/simulation/s4_3_pi2u_va_bridge.json"
CACHE = ROOT / ".local/cache/simulation/s4_3_pi2u/va_bridge"
EXPERIMENT = ROOT / ".local/experiments/simulation/s4_3_pi2u/va_bridge"
ARTIFACT = ROOT / ".local/artifacts/simulation/s4_3_pi2u"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load(split: str) -> dict[str, np.ndarray]:
    with np.load(CACHE / f"{split}.npz", allow_pickle=False) as source:
        allowed = {"pair_id", "episode_id", "task", "source_trajectory_id", "anchor_step", "future_step", "z_v", "z_a"}
        if set(source.files) != allowed:
            raise RuntimeError(f"VA cache schema mismatch: {source.files}")
        return {name: np.array(source[name], copy=True) for name in source.files}


def numeric(values: np.ndarray) -> np.ndarray:
    return np.unique(values, return_inverse=True)[1].astype(np.int64)


@torch.inference_mode()
def encode(model: VAOnlyBridge, arrays: dict[str, np.ndarray], device: torch.device) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    shared = {name: np.empty((len(arrays["pair_id"]), 8, 32), np.float32) for name in model.modalities}
    recovered = {name: np.empty_like(value) for name, value in shared.items()}
    keys = {"vision": "z_v", "action": "z_a"}
    model.eval()
    for start in range(0, len(arrays["pair_id"]), 1024):
        stop = min(start + 1024, len(arrays["pair_id"]))
        for name in model.modalities:
            native = torch.from_numpy(arrays[keys[name]][start:stop]).to(device)
            value = model.encode(name, native)
            shared[name][start:stop] = value.float().cpu().numpy()
            recovered[name][start:stop] = model.recover(name, value).float().cpu().numpy()
    return shared, recovered


def normalize(value: np.ndarray) -> np.ndarray:
    flat = value.astype(np.float64).reshape(len(value), -1)
    return flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)


def different_episode_permutation(episode: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(episode))
    result = np.empty(len(episode), np.int64)
    pools = {int(key): order[episode[order] != key] for key in np.unique(episode)}
    offsets = {key: 0 for key in pools}
    for index, value in enumerate(episode):
        key = int(value)
        result[index] = pools[key][offsets[key] % len(pools[key])]
        offsets[key] += 1
    return result


def retrieval(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    query, candidate = normalize(left).astype(np.float32), normalize(right).astype(np.float32)
    ranks = np.empty(len(query), np.int64)
    for start in range(0, len(query), 512):
        stop = min(start + 512, len(query))
        similarity = query[start:stop] @ candidate.T
        positive = similarity[np.arange(stop - start), np.arange(start, stop)]
        ranks[start:stop] = 1 + np.sum(similarity > positive[:, None], axis=1)
    count = len(ranks)
    return {
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "chance": {"recall_at_1": 1 / count, "recall_at_5": 5 / count, "recall_at_10": 10 / count},
    }


def effective_rank(value: np.ndarray) -> float:
    flat = value.astype(np.float64).reshape(len(value), -1)
    singular = np.linalg.svd(flat - flat.mean(0, keepdims=True), compute_uv=False)
    probability = np.square(singular) / max(float(np.square(singular).sum()), 1e-12)
    return float(np.exp(-np.sum(probability * np.log(np.maximum(probability, 1e-12)))))


def geometry(value: np.ndarray) -> dict[str, float]:
    flat = value.astype(np.float64).reshape(len(value), -1)
    variance = flat.var(0)
    token = normalize(value.reshape(-1, 32)).reshape(len(value), 8, 32)
    cosine = np.einsum("bqd,bkd->bqk", token, token)
    off = ~np.eye(8, dtype=bool)
    return {
        "effective_rank": effective_rank(value),
        "mean_dimension_variance": float(variance.mean()),
        "near_zero_variance_fraction": float(np.mean(variance < 1e-8)),
        "mean_off_diagonal_token_cosine": float(cosine[:, off].mean()),
    }


def recovery(native: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    source, output = native.astype(np.float64).reshape(len(native), -1), predicted.astype(np.float64).reshape(len(native), -1)
    cosine = np.sum(source * output, axis=1) / np.maximum(np.linalg.norm(source, axis=1) * np.linalg.norm(output, axis=1), 1e-12)
    residual = np.square(output - source)
    denominator = np.square(source - source.mean(0, keepdims=True)).sum()
    return {"mse": float(residual.mean()), "cosine": float(cosine.mean()), "r2": 1 - float(residual.sum()) / max(float(denominator), 1e-12)}


def validate(model: VAOnlyBridge, arrays: dict[str, np.ndarray], config: dict[str, object], device: torch.device) -> dict[str, object]:
    shared, recovered = encode(model, arrays, device)
    episode = numeric(arrays["episode_id"])
    permutation = different_episode_permutation(episode, int(config["seed"]) + 91)
    vision, action = normalize(shared["vision"]), normalize(shared["action"])
    paired = np.sum(vision * action, axis=1)
    shuffled = np.sum(vision * action[permutation], axis=1)
    differences = paired - shuffled
    rng = np.random.default_rng(int(config["seed"]) + 92)
    means = np.empty(int(config["validation"]["bootstrap_samples"]), np.float64)
    for start in range(0, len(means), 250):
        stop = min(start + 250, len(means))
        indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        means[start:stop] = differences[indices].mean(axis=1)
    forward, reverse = retrieval(shared["vision"], shared["action"]), retrieval(shared["action"], shared["vision"])
    result = {
        "paired_cosine": float(paired.mean()),
        "different_episode_shuffled_cosine": float(shuffled.mean()),
        "paired_minus_shuffled_margin": float(differences.mean()),
        "margin_bootstrap_ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "retrieval": {"vision_to_action": forward, "action_to_vision": reverse},
        "geometry": {name: geometry(value) for name, value in shared.items()},
        "native_effective_rank": {"vision": effective_rank(arrays["z_v"]), "action": effective_rank(arrays["z_a"])},
        "recovery": {"vision": recovery(arrays["z_v"], recovered["vision"]), "action": recovery(arrays["z_a"], recovered["action"])},
    }
    gates_cfg = config["validation"]
    gates = {
        "paired_margin_positive_ci": result["margin_bootstrap_ci95"][0] > gates_cfg["paired_margin_ci_lower_min"],
        "vision_to_action_retrieval": forward["recall_at_10"] / forward["chance"]["recall_at_10"] >= gates_cfg["retrieval_r10_chance_multiplier_min"],
        "action_to_vision_retrieval": reverse["recall_at_10"] / reverse["chance"]["recall_at_10"] >= gates_cfg["retrieval_r10_chance_multiplier_min"],
        "no_collapse": all(value["near_zero_variance_fraction"] <= gates_cfg["near_zero_variance_fraction_max"] for value in result["geometry"].values()),
        "native_recovery_retained": all(value["cosine"] >= gates_cfg["recovery_cosine_min"] for value in result["recovery"].values()),
        "finite": all(np.isfinite(value).all() for value in shared.values()),
        "contact_absent": set(model.modalities) == {"vision", "action"} and not any("contact" in name.lower() for name, _ in model.named_parameters()),
    }
    result["gates"] = {name: "PASS" if value else "FAIL" for name, value in gates.items()}
    result["status"] = "PASS" if all(gates.values()) else "FAIL"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    if config["status"] != "FROZEN_BEFORE_VA_BRIDGE_TRAINING":
        raise RuntimeError("VA bridge protocol is not frozen")
    train, validation = load("train"), load("validation")
    seed = int(config["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(args.device)
    model = VAOnlyBridge(width=int(config["architecture"]["hidden_width"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    rng = np.random.default_rng(seed)
    episode = numeric(train["episode_id"])
    history = []
    model.train()
    for step in range(1, int(config["training"]["steps"]) + 1):
        rows = rng.choice(len(train["pair_id"]), int(config["training"]["batch_size"]), replace=False)
        loss, breakdown = va_bridge_loss(
            model,
            torch.from_numpy(train["z_v"][rows]).to(device),
            torch.from_numpy(train["z_a"][rows]).to(device),
            torch.from_numpy(episode[rows]).to(device),
            temperature=config["training"]["temperature"],
            weights=VALossWeights(**config["training"]["loss_weights"]),
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite VA bridge loss")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == config["training"]["steps"]:
            row = {"step": step, **{name: float(value) for name, value in breakdown.items()}}
            history.append(row); print(json.dumps(row), flush=True)
    metrics = validate(model, validation, config, device)
    checkpoint = EXPERIMENT / "final.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "tactile3d-unit.s4-3-pi2u-va-bridge-checkpoint.v1",
        "architecture": "VAOnlyBridge(slot,width=64)",
        "modalities": list(model.modalities),
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "config_sha256": sha256(CONFIG),
        "training_steps": int(config["training"]["steps"]),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }, checkpoint)
    frozen = EXPERIMENT / "frozen.pt"
    shutil.copy2(checkpoint, frozen)
    training = {"schema": "tactile3d-unit.s4-3-pi2u-va-bridge-training.v1", "status": metrics["status"], "device": str(device), "history": history, "checkpoint": "$PI2U_ROOT/experiments/va_bridge/frozen.pt", "checkpoint_sha256": sha256(frozen)}
    metrics.update({"schema": "tactile3d-unit.s4-3-pi2u-va-bridge-metrics.v1", "split": "DEV representation only", "checkpoint_sha256": sha256(frozen)})
    manifest = {"schema": "tactile3d-unit.s4-3-pi2u-va-bridge-checkpoint-manifest.v1", "status": metrics["status"], "checkpoint": "$PI2U_ROOT/experiments/va_bridge/frozen.pt", "sha256": sha256(frozen), "parameters": sum(parameter.numel() for parameter in model.parameters()), "modalities": list(model.modalities), "contact_parameters": 0, "selection": "single preregistered final checkpoint"}
    atomic_json(ARTIFACT / "va_bridge_training.json", training)
    atomic_json(ARTIFACT / "va_bridge_metrics.json", metrics)
    atomic_json(ARTIFACT / "va_bridge_checkpoint_manifest.json", manifest)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if metrics["status"] != "PASS":
        raise SystemExit("S4_3_PI2U_BVA_BRIDGE_FAIL")


if __name__ == "__main__":
    main()
