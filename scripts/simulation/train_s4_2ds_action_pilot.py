#!/usr/bin/env python3
"""Train and gate the single preregistered S4.2-DS Action pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.contact_dynamics.evaluation import (  # noqa: E402
    different_episode_permutation,
    query_diversity,
)
from gr00t.simulation.s4_2_dataset import sha256_file  # noqa: E402
from gr00t.simulation.s4_2ds import ActionPilot  # noqa: E402


PAIR_PATH = ROOT / ".local/cache/simulation/s4_2/pairs/train.npz"
CONTACT_PATH = ROOT / ".local/cache/simulation/s4_2r/contact_dynamics/train.npz"
SPLIT_PATH = ROOT / ".local/artifacts/simulation/s4_2ds/ds_split_manifest.json"
PROTOCOL_PATH = ROOT / ".local/artifacts/simulation/s4_2ds/protocol_freeze.json"
CONFIG_PATH = ROOT / "configs/simulation/s4_2ds_representation_selection.json"
EXPERIMENT_PATH = ROOT / ".local/experiments/simulation/s4_2ds/action_pilot.pt"
ARTIFACT_PATH = ROOT / ".local/artifacts/simulation/s4_2ds/action_pilot.json"
CACHE_PATH = ROOT / ".local/cache/simulation/s4_2ds/action_pilot.npz"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def quaternion_to_rotvec(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    value = value / np.maximum(norm, 1e-12)
    value = np.where(value[:, :1] < 0.0, -value, value)
    xyz = value[:, 1:]
    xyz_norm = np.linalg.norm(xyz, axis=1, keepdims=True)
    angle = 2.0 * np.arctan2(xyz_norm, np.clip(value[:, :1], 1e-12, None))
    scale = np.where(xyz_norm > 1e-10, angle / np.maximum(xyz_norm, 1e-12), 2.0)
    return xyz * scale


def policy_state(current_state: np.ndarray) -> np.ndarray:
    if current_state.ndim != 2 or current_state.shape[1] != 23:
        raise ValueError("raw current state must be [N,23]")
    result = np.concatenate(
        (current_state[:, :3], quaternion_to_rotvec(current_state[:, 3:7]), current_state[:, 7:]),
        axis=1,
    )
    if result.shape[1] != 22:
        raise AssertionError("policy-facing state contract is not 22D")
    return result.astype(np.float32)


def fit_stats(state: np.ndarray, action: np.ndarray, indices: np.ndarray) -> dict[str, np.ndarray]:
    selected_state = state[indices]
    selected_action = action[indices]
    relative = selected_action - selected_state[:, None]
    difference = np.empty_like(selected_action)
    difference[:, 0] = selected_action[:, 0] - selected_state
    difference[:, 1:] = np.diff(selected_action, axis=1)
    result = {}
    for name, value in (("absolute", selected_action), ("relative", relative), ("difference", difference)):
        result[f"{name}_mean"] = value.mean(axis=(0, 1)).astype(np.float32)
        result[f"{name}_std"] = np.maximum(value.std(axis=(0, 1)), 1e-6).astype(np.float32)
    result["state_mean"] = selected_state.mean(axis=0).astype(np.float32)
    result["state_std"] = np.maximum(selected_state.std(axis=0), 1e-6).astype(np.float32)
    return result


def features(state: np.ndarray, action: np.ndarray, stats: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    relative = action - state[:, None]
    difference = np.empty_like(action)
    difference[:, 0] = action[:, 0] - state
    difference[:, 1:] = np.diff(action, axis=1)
    value = np.concatenate(
        (
            (action - stats["absolute_mean"]) / stats["absolute_std"],
            (relative - stats["relative_mean"]) / stats["relative_std"],
            (difference - stats["difference_mean"]) / stats["difference_std"],
        ),
        axis=2,
    ).astype(np.float32)
    state_normalized = ((state - stats["state_mean"]) / stats["state_std"]).astype(np.float32)
    action_normalized = ((action - stats["absolute_mean"]) / stats["absolute_std"]).astype(np.float32)
    return value, state_normalized, action_normalized


@torch.inference_mode()
def infer(
    model: ActionPilot,
    feature: np.ndarray,
    state: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    code = np.empty((len(feature), 8, 32), dtype=np.float32)
    prediction = np.empty((len(feature), 27, 22), dtype=np.float32)
    model.eval()
    for start in range(0, len(feature), batch_size):
        stop = min(start + batch_size, len(feature))
        output = model(
            torch.from_numpy(np.array(feature[start:stop], copy=True)).to(device),
            torch.from_numpy(np.array(state[start:stop], copy=True)).to(device),
        )
        code[start:stop] = output["code"].float().cpu().numpy()
        prediction[start:stop] = output["action"].float().cpu().numpy()
    return code, prediction


def bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 100):
        stop = min(samples, start + 100)
        index = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[index].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).astype(float).tolist()


def effective_rank(code: np.ndarray) -> float:
    flat = code.reshape(len(code), -1).astype(np.float64)
    singular = np.linalg.svd(flat - flat.mean(axis=0), compute_uv=False)
    probability = np.square(singular) / max(float(np.square(singular).sum()), 1e-12)
    return float(np.exp(-np.sum(probability * np.log(np.maximum(probability, 1e-15)))))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["config_sha256"] != sha256_file(CONFIG_PATH):
        raise RuntimeError("S4.2-DS protocol changed after freeze")
    training = config["action_pilot"]["training"]
    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device(args.device)
    pairs = load_npz(PAIR_PATH)
    contact = load_npz(CONTACT_PATH)
    if not np.array_equal(pairs["pair_id"], contact["pair_id"]):
        raise RuntimeError("pair identity mismatch")
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    dev_groups = {tuple(value) for value in split["ds_dev_group_ids"]}
    dev_mask = np.asarray(
        [(str(task), str(source)) in dev_groups for task, source in zip(pairs["task"], pairs["source_trajectory_id"])]
    )
    train_indices = np.flatnonzero(~dev_mask)
    dev_indices = np.flatnonzero(dev_mask)
    state = policy_state(pairs["current_state"])
    action = np.asarray(pairs["action_chunk"], dtype=np.float32)
    if action.shape[1:] != (27, 22):
        raise RuntimeError("Action chunk violates [27,22] contract")
    stats = fit_stats(state, action, train_indices)
    feature, state_normalized, action_normalized = features(state, action, stats)
    model = ActionPilot(width=int(config["action_pilot"]["architecture"]["width"])).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters > int(config["action_pilot"]["architecture"]["parameter_max"]):
        raise RuntimeError("Action pilot exceeds parameter budget")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    rng = np.random.default_rng(seed)
    steps = int(training["steps"])
    batch_size = int(training["batch_size"])
    history = []
    model.train()
    for step in range(1, steps + 1):
        indices = rng.choice(train_indices, size=batch_size, replace=False)
        feature_t = torch.from_numpy(feature[indices]).to(device)
        state_t = torch.from_numpy(state_normalized[indices]).to(device)
        target_t = torch.from_numpy(action_normalized[indices]).to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(feature_t, state_t)
        loss = F.mse_loss(output["action"], target_t)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite Action pilot loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            history.append({"step": step, "train_mse": float(loss.detach())})
            print(json.dumps(history[-1]), flush=True)
    EXPERIMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema": "tactile3d-unit.s4-2ds-action-pilot.v1",
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "stats": {name: value.tolist() for name, value in stats.items()},
        "parameters": parameters,
        "steps": steps,
        "seed": seed,
        "formal_test_loaded": False,
    }
    torch.save(checkpoint, EXPERIMENT_PATH)
    model.eval()
    code, correct_prediction = infer(model, feature, state_normalized, device, 512)
    rng = np.random.default_rng(seed)
    dev_episode = contact["episode_id"][dev_indices]
    shuffled = rng.permutation(dev_indices)
    different_local = different_episode_permutation(dev_episode, seed=seed)
    different = dev_indices[different_local]
    control_actions = {
        "reversed": action[dev_indices, ::-1].copy(),
        "shuffled": action[shuffled],
        "different_episode": action[different],
    }
    target = action_normalized[dev_indices]
    errors = {
        "correct": np.square(correct_prediction[dev_indices] - target).mean(axis=(1, 2))
    }
    control_codes = {}
    for name, control_action in control_actions.items():
        control_feature, _, _ = features(state[dev_indices], control_action, stats)
        control_code, control_prediction = infer(
            model, control_feature, state_normalized[dev_indices], device, 512
        )
        control_codes[name] = control_code
        errors[name] = np.square(control_prediction - target).mean(axis=(1, 2))
    dynamic = contact["dynamic"][dev_indices].astype(bool)
    metrics = {"all": {}, "dynamic": {}}
    for subset, mask in (("all", np.ones(len(dev_indices), dtype=bool)), ("dynamic", dynamic)):
        correct = float(errors["correct"][mask].mean())
        metrics[subset]["correct_mse"] = correct
        for offset, name in enumerate(("reversed", "shuffled", "different_episode")):
            value = float(errors[name][mask].mean())
            metrics[subset][f"{name}_mse"] = value
            metrics[subset][f"{name}_over_correct"] = value / max(correct, 1e-12)
            metrics[subset][f"{name}_minus_correct_ci95"] = bootstrap_ci(
                errors[name][mask] - errors["correct"][mask], args.bootstrap_samples, seed + offset
            )
    stability_device = torch.device("cpu")
    cold = ActionPilot(width=int(config["action_pilot"]["architecture"]["width"])).to(stability_device)
    cold.load_state_dict(torch.load(EXPERIMENT_PATH, map_location="cpu", weights_only=False)["state_dict"], strict=True)
    cold.eval().requires_grad_(False)
    repeat_code, _ = infer(
        cold, feature[dev_indices[:64]], state_normalized[dev_indices[:64]], stability_device, 64
    )
    repeated_code, _ = infer(
        cold, feature[dev_indices[:64]], state_normalized[dev_indices[:64]], stability_device, 64
    )
    batch_one_code, _ = infer(
        cold, feature[dev_indices[:64]], state_normalized[dev_indices[:64]], stability_device, 1
    )
    max_repeat = float(np.max(np.abs(repeat_code - repeated_code)))
    max_batch = float(np.max(np.abs(repeat_code - batch_one_code)))
    geometry = {
        "effective_rank": effective_rank(code[dev_indices]),
        "query_diversity": query_diversity(code[dev_indices]),
        "finite": bool(np.isfinite(code).all()),
    }
    gates = {
        "reversed_ratio": metrics["dynamic"]["reversed_over_correct"] >= 1.05,
        "shuffled_ratio": metrics["dynamic"]["shuffled_over_correct"] >= 1.10,
        "different_episode_ratio": metrics["dynamic"]["different_episode_over_correct"] >= 2.0,
        "paired_improvement_cis": all(
            metrics["dynamic"][f"{name}_minus_correct_ci95"][0] > 0.0
            for name in ("reversed", "shuffled", "different_episode")
        ),
        "finite": geometry["finite"],
        "no_collapse": geometry["query_diversity"]["collapsed_sample_fraction"] == 0.0,
        "deterministic": max_repeat <= 1e-5 and max_batch <= 1e-5,
    }
    result = {
        "schema": "tactile3d-unit.s4-2ds-action-pilot-evaluation.v1",
        "stage": "DS4_ACTION",
        "role": "bridge pilot infrastructure; not formal S4.2-4",
        "input": {"action_chunk": [27, 22], "current_state": 22, "features": [27, 66]},
        "latent_shape": [8, 32],
        "parameters": parameters,
        "checkpoint": str(EXPERIMENT_PATH.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(EXPERIMENT_PATH),
        "training": {"steps": steps, "seed": seed, "history": history, "selection": training["selection"]},
        "metrics": metrics,
        "geometry": geometry,
        "stability": {
            "device": "cpu",
            "tolerance": 1e-5,
            "cold_reload_repeated_extraction_max_abs": max_repeat,
            "batch_size_change_max_abs": max_batch,
        },
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "overall": "PASS" if all(gates.values()) else "FAIL",
        "failure_decision": None if all(gates.values()) else "S4_2DS_ACTION_PILOT_FAIL",
        "formal_test_loaded": False,
    }
    np.savez_compressed(
        CACHE_PATH,
        pair_id=pairs["pair_id"],
        z_a=code,
        ds_train_indices=train_indices,
        ds_dev_indices=dev_indices,
    )
    atomic_json(ARTIFACT_PATH, result)
    print(json.dumps({"overall": result["overall"], "metrics": metrics["dynamic"], "parameters": parameters}, indent=2))
    if result["overall"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
