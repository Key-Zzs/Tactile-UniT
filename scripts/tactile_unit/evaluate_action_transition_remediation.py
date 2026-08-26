#!/usr/bin/env python3
"""Untouched-test evaluation for the selected A-R action-transition model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.trex_action_bootstrap import (  # noqa: E402
    QUERY_NUM,
    TREX_EMBODIMENT_ID,
    ReleasedTokenizerSource,
    effective_rank,
    query_diversity,
)
from gr00t.tactile_unit.trex_action_data import (  # noqa: E402
    RAW_ACTION_DIM,
    SEGMENTS,
    TReXActionCache,
    action_activity,
    different_episode_indices,
)
from gr00t.tactile_unit.trex_action_transition import (  # noqa: E402
    NativeTransitionActionModel,
    SharedTransitionActionModel,
    load_shared_transition_checkpoint,
    load_transition_checkpoint,
)
from scripts.tactile_unit.evaluate_trex_action_bootstrap import (  # noqa: E402
    code_usage,
    fit_frozen_probes,
    load_frozen_rq,
)


DEFAULT_CONFIG = ROOT / "configs/tactile_unit/s3_3_r_action_transition_remediation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--max-test-windows", type=int, default=None)
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def require_isolated_gpu(config: Mapping[str, Any]) -> tuple[torch.device, int]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("A-R evaluation requires exactly one visible CUDA device")
    physical = int(os.environ.get("TACTILE_PHYSICAL_GPU", "-1"))
    if physical not in set(map(int, config["gpu"]["allowed_physical"])):
        raise RuntimeError("A-R evaluation received a forbidden physical GPU")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("physical/logical GPU isolation mismatch")
    return torch.device("cuda:0"), physical


def selected_indices(length: int, count: int | None) -> np.ndarray:
    count = length if count is None else min(length, int(count))
    return np.floor((np.arange(count) + 0.5) * length / count).astype(np.int64)


def make_tensors(cache: TReXActionCache, indices: np.ndarray, device: torch.device):
    batch = cache.batch(indices)
    state = torch.from_numpy(batch["state"]).to(device)
    action = torch.from_numpy(batch["action"]).to(device)
    embodiment = torch.full((len(indices),), TREX_EMBODIMENT_ID, dtype=torch.long, device=device)
    return batch, state, action, embodiment


def reconstruction_per_sample(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(
        prediction[..., :RAW_ACTION_DIM].float(), target[..., :RAW_ACTION_DIM], reduction="none"
    ).mean(dim=(1, 2))


def load_selected(training: Mapping[str, Any], source: ReleasedTokenizerSource):
    path = resolve(training["checkpoint"]["relative_path"])
    if training["selected_candidate"] == "R1-N":
        return load_transition_checkpoint(path)
    return load_shared_transition_checkpoint(path, source)


def _empty_error() -> dict[str, float]:
    return defaultdict(float)


def _update_error(
    accumulator: dict[str, float],
    prediction: np.ndarray,
    target: np.ndarray,
    raw_prediction: np.ndarray,
    raw_target: np.ndarray,
) -> None:
    difference = prediction - target
    raw_difference = raw_prediction - raw_target
    accumulator["square"] += float(np.square(difference, dtype=np.float64).sum())
    accumulator["absolute"] += float(np.abs(difference).sum(dtype=np.float64))
    accumulator["raw_square"] += float(np.square(raw_difference, dtype=np.float64).sum())
    accumulator["raw_absolute"] += float(np.abs(raw_difference).sum(dtype=np.float64))
    accumulator["count"] += difference.size


def _final_error(value: Mapping[str, float]) -> dict[str, float | int]:
    count = max(float(value["count"]), 1.0)
    return {
        "normalized_mse": float(value["square"] / count),
        "normalized_mae": float(value["absolute"] / count),
        "raw_unit_mse": float(value["raw_square"] / count),
        "raw_unit_mae": float(value["raw_absolute"] / count),
        "element_count": int(value["count"]),
    }


@torch.no_grad()
def reconstruction_audit(
    model: torch.nn.Module,
    cache: TReXActionCache,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    dynamic_threshold: float,
) -> dict[str, Any]:
    accumulators = {"all": _empty_error(), "dynamic": _empty_error()}
    for name in SEGMENTS:
        accumulators[name] = _empty_error()
        accumulators[f"dynamic_{name}"] = _empty_error()
    for start in range(0, len(indices), batch_size):
        current = indices[start : start + batch_size]
        batch, state, action, embodiment = make_tensors(cache, current, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(state, action, embodiment)["prediction"]
        prediction = prediction[..., :RAW_ACTION_DIM].float().cpu().numpy()
        target = batch["action"][..., :RAW_ACTION_DIM]
        raw_prediction = cache.inverse_action(prediction)
        raw_target = batch["action_raw"]
        dynamic = action_activity(target)["magnitude"] > dynamic_threshold
        _update_error(accumulators["all"], prediction, target, raw_prediction, raw_target)
        if dynamic.any():
            _update_error(accumulators["dynamic"], prediction[dynamic], target[dynamic], raw_prediction[dynamic], raw_target[dynamic])
        for name, segment in SEGMENTS.items():
            _update_error(accumulators[name], prediction[..., segment], target[..., segment], raw_prediction[..., segment], raw_target[..., segment])
            if dynamic.any():
                _update_error(
                    accumulators[f"dynamic_{name}"], prediction[dynamic, ..., segment], target[dynamic, ..., segment],
                    raw_prediction[dynamic, ..., segment], raw_target[dynamic, ..., segment],
                )
    return {name: _final_error(value) for name, value in accumulators.items()}


@torch.no_grad()
def mean_train_token(model: torch.nn.Module, cache: TReXActionCache, count: int, batch_size: int, device: torch.device) -> torch.Tensor:
    indices = selected_indices(len(cache), count)
    total = None
    seen = 0
    for start in range(0, len(indices), batch_size):
        current = indices[start : start + batch_size]
        _, state, action, embodiment = make_tensors(cache, current, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z_action, _, _ = model.encode(state, action, embodiment)
        value = z_action.float().sum(dim=0)
        total = value if total is None else total + value
        seen += len(current)
    return total / seen


def paired_ratio_ci(correct: np.ndarray, control: np.ndarray, *, samples: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    ratios = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 100):
        take = min(100, samples - start)
        choices = rng.integers(0, len(correct), size=(take, len(correct)), endpoint=False)
        ratios[start : start + take] = control[choices].mean(axis=1) / np.maximum(correct[choices].mean(axis=1), 1e-12)
    return {
        "ratio": float(control.mean() / max(correct.mean(), 1e-12)),
        "ci95_lower": float(np.quantile(ratios, 0.025)),
        "ci95_upper": float(np.quantile(ratios, 0.975)),
        "bootstrap_samples": samples,
    }


@torch.no_grad()
def temporal_control_audit(
    model: torch.nn.Module,
    cache: TReXActionCache,
    indices: np.ndarray,
    *,
    mean_token: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dynamic_threshold: float,
    shuffle_seed: int,
    bootstrap_samples: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    rng = np.random.default_rng(shuffle_seed)
    all_errors: dict[str, list[np.ndarray]] = defaultdict(list)
    dynamic_errors: dict[str, list[np.ndarray]] = defaultdict(list)
    for start in range(0, len(indices), batch_size):
        current = indices[start : start + batch_size]
        batch, state, action, embodiment = make_tensors(cache, current, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(state, action, embodiment)
            zero_prediction = model.decode(torch.zeros_like(output["z_action"]), output["state_features"], embodiment)
            mean_prediction = model.decode(
                mean_token.to(device).unsqueeze(0).expand(len(current), -1, -1),
                output["state_features"],
                embodiment,
            )
        controls = {"correct": output["prediction"], "zero": zero_prediction, "mean": mean_prediction}
        controls["state_only"] = controls["zero"]
        other = cache.batch(different_episode_indices(cache, current))
        inputs = {
            "reversed": action.flip(1),
            "shuffled": action[:, torch.from_numpy(rng.permutation(16)).to(device)],
            "different_episode": torch.from_numpy(other["action"]).to(device),
        }
        for name, candidate in inputs.items():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                controls[name] = model(state, candidate, embodiment)["prediction"]
        dynamic = action_activity(batch["action"][..., :RAW_ACTION_DIM])["magnitude"] > dynamic_threshold
        for name, prediction in controls.items():
            error = reconstruction_per_sample(prediction, action).cpu().numpy()
            all_errors[name].append(error)
            dynamic_errors[name].append(error[dynamic])
    arrays = {name: np.concatenate(values) for name, values in all_errors.items()}
    dynamic_arrays = {name: np.concatenate(values) for name, values in dynamic_errors.items()}
    result: dict[str, Any] = {
        "windows": len(indices),
        "dynamic_windows": len(dynamic_arrays["correct"]),
        "all": {name: float(value.mean()) for name, value in arrays.items()},
        "dynamic": {name: float(value.mean()) for name, value in dynamic_arrays.items()},
        "paired_bootstrap": {"all": {}, "dynamic": {}},
    }
    for name in ("reversed", "shuffled", "different_episode", "zero", "mean", "state_only"):
        result["paired_bootstrap"]["all"][name] = paired_ratio_ci(
            arrays["correct"], arrays[name], samples=bootstrap_samples, seed=shuffle_seed + len(name)
        )
        result["paired_bootstrap"]["dynamic"][name] = paired_ratio_ci(
            dynamic_arrays["correct"], dynamic_arrays[name], samples=bootstrap_samples, seed=shuffle_seed + 100 + len(name)
        )
    return result, {**{f"all_{key}": value for key, value in arrays.items()}, **{f"dynamic_{key}": value for key, value in dynamic_arrays.items()}}


@torch.no_grad()
def extract_latents(model: torch.nn.Module, cache: TReXActionCache, indices: np.ndarray, batch_size: int, device: torch.device):
    latents = []
    labels: dict[str, list[np.ndarray]] = defaultdict(list)
    state_features = []
    states = []
    actions = []
    embodiments = []
    for start in range(0, len(indices), batch_size):
        current = indices[start : start + batch_size]
        batch, state, action, embodiment = make_tensors(cache, current, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z_action, features, _ = model.encode(state, action, embodiment)
        latents.append(z_action.float().cpu().numpy())
        state_features.append(features.detach().cpu())
        states.append(state.cpu())
        actions.append(action.cpu())
        embodiments.append(embodiment.cpu())
        activity = action_activity(batch["action"][..., :RAW_ACTION_DIM])
        for name, value in activity.items():
            labels[name].append(value)
        labels["primitive_id"].append(batch["primitive_id"])
    return (
        np.concatenate(latents),
        {name: np.concatenate(values) for name, values in labels.items()},
        torch.cat(state_features), torch.cat(states), torch.cat(actions), torch.cat(embodiments),
    )


def noncollapse_metrics(z_action: np.ndarray) -> dict[str, Any]:
    flat = z_action.reshape(len(z_action), -1)
    variance = flat.var(axis=0)
    norms = np.linalg.norm(z_action, axis=-1)
    pair_count = min(4096, len(z_action) // 2)
    distance = np.linalg.norm(flat[:pair_count] - flat[-pair_count:], axis=1)
    return {
        "windows": len(z_action),
        "per_dimension_variance": {
            "minimum": float(variance.min()), "median": float(np.median(variance)),
            "mean": float(variance.mean()), "near_zero_fraction": float((variance < 1e-8).mean()),
        },
        "effective_rank": effective_rank(torch.from_numpy(flat)),
        "token_norm": {"mean": float(norms.mean()), "std": float(norms.std()), "minimum": float(norms.min()), "maximum": float(norms.max())},
        "query_diversity": query_diversity(torch.from_numpy(z_action)),
        "collapsed_query_fraction": float((z_action.var(axis=(0, 2)) < 1e-8).mean()),
        "pairwise_sample_distance": {"mean": float(distance.mean()), "median": float(np.median(distance)), "minimum": float(distance.min())},
    }


@torch.no_grad()
def quantize(rq: torch.nn.Module, values: np.ndarray, batch_size: int, device: torch.device):
    outputs, codes = [], []
    rq.to(device).eval()
    for start in range(0, len(values), batch_size):
        tensor = torch.from_numpy(values[start : start + batch_size]).to(device)
        value, index, _ = rq(tensor)
        if index.ndim == 2:
            index = index.unsqueeze(-1)
        outputs.append(value.float().cpu().numpy())
        codes.append(index.cpu().numpy())
    return np.concatenate(outputs), np.concatenate(codes)


@torch.no_grad()
def decode_errors(model: torch.nn.Module, z_action: np.ndarray, state_features: torch.Tensor, actions: torch.Tensor, embodiments: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    errors = []
    for start in range(0, len(z_action), batch_size):
        stop = min(start + batch_size, len(z_action))
        z = torch.from_numpy(z_action[start:stop]).to(device)
        features = state_features[start:stop].to(device)
        target = actions[start:stop].to(device)
        embodiment = embodiments[start:stop].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model.decode(z, features, embodiment)
        errors.append(reconstruction_per_sample(prediction, target).cpu().numpy())
    return np.concatenate(errors)


@torch.no_grad()
def quantized_temporal_audit(
    model: torch.nn.Module,
    rq: torch.nn.Module,
    cache: TReXActionCache,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    dynamic_threshold: float,
    shuffle_seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(shuffle_seed)
    all_errors: dict[str, list[np.ndarray]] = defaultdict(list)
    dynamic_errors: dict[str, list[np.ndarray]] = defaultdict(list)
    for start in range(0, len(indices), batch_size):
        current = indices[start : start + batch_size]
        batch, state, action, embodiment = make_tensors(cache, current, device)
        other = cache.batch(different_episode_indices(cache, current))
        inputs = {
            "correct": action,
            "reversed": action.flip(1),
            "shuffled": action[:, torch.from_numpy(rng.permutation(16)).to(device)],
            "different_episode": torch.from_numpy(other["action"]).to(device),
        }
        dynamic = (
            action_activity(batch["action"][..., :RAW_ACTION_DIM])["magnitude"]
            > dynamic_threshold
        )
        for name, candidate in inputs.items():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z_action, state_features, _ = model.encode(state, candidate, embodiment)
                quantized, _, _ = rq(z_action)
                prediction = model.decode(quantized, state_features, embodiment)
            error = reconstruction_per_sample(prediction, action).cpu().numpy()
            all_errors[name].append(error)
            dynamic_errors[name].append(error[dynamic])
    all_means = {name: float(np.concatenate(value).mean()) for name, value in all_errors.items()}
    dynamic_means = {
        name: float(np.concatenate(value).mean()) for name, value in dynamic_errors.items()
    }
    return {
        "windows": len(indices),
        "all": all_means,
        "dynamic": dynamic_means,
        "all_ratios": {
            name: value / max(all_means["correct"], 1e-12)
            for name, value in all_means.items()
        },
        "dynamic_ratios": {
            name: value / max(dynamic_means["correct"], 1e-12)
            for name, value in dynamic_means.items()
        },
    }


@torch.no_grad()
def feature_ablation(model: torch.nn.Module, cache: TReXActionCache, count: int, batch_size: int, device: torch.device) -> dict[str, Any]:
    indices = selected_indices(len(cache), count)
    variants = {"full": (True, True, True), "absolute_only": (True, False, False), "absolute_relative": (True, True, False), "absolute_velocity": (True, False, True)}
    errors: dict[str, list[np.ndarray]] = defaultdict(list)
    for start in range(0, len(indices), batch_size):
        current = indices[start : start + batch_size]
        _, state, action, embodiment = make_tensors(cache, current, device)
        features = model.features(state[:, :RAW_ACTION_DIM], action[:, :, :RAW_ACTION_DIM])
        for name, keep in variants.items():
            candidate = features.clone()
            if not keep[0]: candidate[..., :58] = 0
            if not keep[1]: candidate[..., 58:116] = 0
            if not keep[2]: candidate[..., 116:] = 0
            if isinstance(model, NativeTransitionActionModel):
                z_action = model.encoder(candidate)
                state_features = state[:, :RAW_ACTION_DIM]
            else:
                adapted = model.transition_adapter(candidate)
                z_action, state_features, _ = model.base.encode(state, adapted, embodiment)
            prediction = model.decode(z_action, state_features, embodiment)
            errors[name].append(reconstruction_per_sample(prediction, action).cpu().numpy())
    means = {name: float(np.concatenate(values).mean()) for name, values in errors.items()}
    return {"selection_split": "validation", "windows": len(indices), "normalized_mse": means, "ratio_to_full": {name: value / max(means["full"], 1e-12) for name, value in means.items()}}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    device, physical = require_isolated_gpu(config)
    artifact_root = resolve(config["paths"]["artifact_root"])
    training = json.loads((artifact_root / "training_summary.json").read_text())
    source = ReleasedTokenizerSource.open(args.tokenizer_root)
    model, metadata = load_selected(training, source)
    model = model.to(device).eval()
    cache_root = resolve(config["data"]["cache_root"])
    normalization = resolve(config["data"]["normalization"])
    train_cache = TReXActionCache(cache_root, "train", normalization)
    val_cache = TReXActionCache(cache_root, "val", normalization)
    test_cache = TReXActionCache(cache_root, "test", normalization)
    baseline_training = json.loads(resolve(config["paths"]["bootstrap_training"]).read_text())
    dynamic_threshold = float(baseline_training["distribution"]["dynamic_threshold_normalized_rms_delta"])
    evaluation = config["evaluation"]
    batch_size = int(evaluation["batch_size"])
    test_indices = selected_indices(len(test_cache), args.max_test_windows)
    reconstruction = reconstruction_audit(
        model, test_cache, test_indices, batch_size=batch_size, device=device, dynamic_threshold=dynamic_threshold
    )
    mean_token = mean_train_token(model, train_cache, int(evaluation["probe_train_windows"]), batch_size, device)
    temporal_indices = selected_indices(len(test_cache), int(evaluation["temporal_windows"]))
    temporal, temporal_arrays = temporal_control_audit(
        model,
        test_cache,
        temporal_indices,
        mean_token=mean_token,
        batch_size=batch_size,
        device=device,
        dynamic_threshold=dynamic_threshold,
        shuffle_seed=int(config["diagnosis"]["shuffle_seed"]),
        bootstrap_samples=int(evaluation["bootstrap_samples"]),
    )
    probe_train = selected_indices(len(train_cache), int(evaluation["probe_train_windows"]))
    probe_test = selected_indices(len(test_cache), int(evaluation["probe_test_windows"]))
    train_z, train_labels, train_features, _, train_actions, train_embodiments = extract_latents(model, train_cache, probe_train, batch_size, device)
    test_z, test_labels, test_features, test_states, test_actions, test_embodiments = extract_latents(model, test_cache, probe_test, batch_size, device)
    probes, _, _ = fit_frozen_probes(train_z, train_labels, test_z, test_labels)
    noncollapse = noncollapse_metrics(test_z[: int(evaluation["latent_windows"])])
    rq = load_frozen_rq(source)
    quantized_train_z, train_codes = quantize(rq, train_z, batch_size, device)
    quantized_test_z, test_codes = quantize(rq, test_z, batch_size, device)
    _, quantized_probes, retention = fit_frozen_probes(
        train_z, train_labels, test_z, test_labels,
        quantized_train_z=quantized_train_z, quantized_test_z=quantized_test_z,
    )
    continuous_error = decode_errors(model, test_z, test_features, test_actions, test_embodiments, batch_size, device)
    quantized_error = decode_errors(model, quantized_test_z, test_features, test_actions, test_embodiments, batch_size, device)
    difference = quantized_test_z - test_z
    relative_distortion = np.linalg.norm(difference.reshape(len(test_z), -1), axis=1) / np.maximum(np.linalg.norm(test_z.reshape(len(test_z), -1), axis=1), 1e-12)
    cosine = np.sum(quantized_test_z * test_z, axis=-1) / np.maximum(np.linalg.norm(quantized_test_z, axis=-1) * np.linalg.norm(test_z, axis=-1), 1e-12)
    quantized_temporal = quantized_temporal_audit(
        model,
        rq,
        test_cache,
        selected_indices(len(test_cache), min(4096, int(evaluation["temporal_windows"]))),
        batch_size=batch_size,
        device=device,
        dynamic_threshold=dynamic_threshold,
        shuffle_seed=int(config["diagnosis"]["shuffle_seed"]),
    )
    rq_diagnostic = {
        "read_only": True,
        "relative_distortion": {"mean": float(relative_distortion.mean()), "median": float(np.median(relative_distortion))},
        "pre_post_cosine": {"mean": float(cosine.mean()), "median": float(np.median(cosine))},
        "code_usage": code_usage(test_codes, int(source.config["vq_cfg"]["n_e"])),
        "continuous_reconstruction_mse": float(continuous_error.mean()),
        "quantized_reconstruction_mse": float(quantized_error.mean()),
        "reconstruction_error_ratio": float(quantized_error.mean() / max(continuous_error.mean(), 1e-12)),
        "quantized_probes": quantized_probes,
        "semantic_retention_ratio": retention,
        "temporal_semantic_retention": {
            "quantized": quantized_temporal,
            "continuous_dynamic_reversed_ratio": temporal["paired_bootstrap"]["dynamic"]["reversed"]["ratio"],
            "continuous_dynamic_shuffled_ratio": temporal["paired_bootstrap"]["dynamic"]["shuffled"]["ratio"],
            "reversed_ratio_retention": quantized_temporal["dynamic_ratios"]["reversed"] / max(temporal["paired_bootstrap"]["dynamic"]["reversed"]["ratio"], 1e-12),
            "shuffled_ratio_retention": quantized_temporal["dynamic_ratios"]["shuffled"] / max(temporal["paired_bootstrap"]["dynamic"]["shuffled"]["ratio"], 1e-12),
        },
    }
    ablation = feature_ablation(model, val_cache, 4096, batch_size, device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, state, action, embodiment = make_tensors(test_cache, np.arange(64, dtype=np.int64), device)
        first = model(state, action, embodiment)["z_action"].float().cpu()
        second = model(state, action, embodiment)["z_action"].float().cpu()
    deterministic = {
        "repeat_exact": bool(torch.equal(first, second)),
        "finite": bool(torch.isfinite(first).all()),
        "shape": list(first.shape),
        "cold_reload": training["cold_reload"],
    }
    acceptance = evaluation["acceptance"]
    dynamic_bootstrap = temporal["paired_bootstrap"]["dynamic"]
    all_bootstrap = temporal["paired_bootstrap"]["all"]
    gates = {
        "raw_negative_nontrivial": bool(
            json.loads((artifact_root / "a_r0_raw_negative_strength.json").read_text())["raw_negative_strength"]["test"]["controls"]["reversed"]["dynamic"]["normalized_mse"] > 0.02
            and json.loads((artifact_root / "a_r0_raw_negative_strength.json").read_text())["raw_negative_strength"]["test"]["controls"]["shuffled"]["dynamic"]["normalized_mse"] > 0.01
        ),
        "reconstruction": bool(
            reconstruction["all"]["normalized_mse"]
            <= float(acceptance["normalized_mse_max"])
        ),
        "dynamic_reversed": bool(dynamic_bootstrap["reversed"]["ratio"] >= float(acceptance["dynamic_reversed_ratio_min"]) and dynamic_bootstrap["reversed"]["ci95_lower"] > float(acceptance["temporal_ci_lower_min"])),
        "dynamic_shuffled": bool(dynamic_bootstrap["shuffled"]["ratio"] >= float(acceptance["dynamic_shuffled_ratio_min"]) and dynamic_bootstrap["shuffled"]["ci95_lower"] > float(acceptance["temporal_ci_lower_min"])),
        "different_episode": bool(all_bootstrap["different_episode"]["ci95_lower"] > 1.0),
        "zero_token": bool(all_bootstrap["zero"]["ratio"] >= float(acceptance["zero_ratio_min"])),
        "mean_token": bool(all_bootstrap["mean"]["ratio"] >= float(acceptance["mean_ratio_min"])),
        "noncollapse": bool(noncollapse["effective_rank"] >= float(acceptance["effective_rank_min"]) and noncollapse["collapsed_query_fraction"] <= float(acceptance["collapsed_query_fraction_max"])),
        "interface_determinism": bool(deterministic["repeat_exact"] and deterministic["finite"] and deterministic["shape"][1:] == [QUERY_NUM, 32] and training["cold_reload"]["exact"]),
        "original_unit_preserved": bool(training["original_unit_preservation"]["old_rows_bit_identical"]),
    }
    ready = all(gates.values())
    rq_warning = rq_diagnostic["reconstruction_error_ratio"] > 1.5 or rq_diagnostic["relative_distortion"]["mean"] > 0.5
    if ready and rq_warning:
        decision = "ACTION_TRANSITION_READY_WITH_RQ_WARNING"
    elif ready and training["selected_candidate"] == "R1-P":
        decision = "ACTION_TRANSITION_READY_SHARED_PATH"
    elif ready:
        decision = "ACTION_TRANSITION_READY_NATIVE_PATH"
    elif gates["interface_determinism"] and gates["original_unit_preserved"]:
        decision = "ACTION_TRANSITION_PARTIAL" if any(gates[name] for name in ("dynamic_reversed", "dynamic_shuffled", "zero_token", "mean_token")) else "ACTION_TRANSITION_INSUFFICIENT"
    else:
        decision = "STRUCTURAL_FAIL"
    checkpoint_path = resolve(training["checkpoint"]["relative_path"])
    contract = {
        "encoder_type": training["selected_encoder_type"],
        "input": "current normalized state [B,128] plus planned target action chunk a_t:t+15 [B,16,128]; transition features are absolute + raw-state-relative + first difference",
        "output": "z_a [B,8,32] continuous pre-RQ",
        "normalization": str((artifact_root / "transition_feature_stats.json").relative_to(ROOT)),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "sha256": sha256_file(checkpoint_path),
        "decoder_interface": "decode(z_a, current_state_features, embodiment_id=31) -> planned target action [B,16,128]",
        "online_availability": "The action chunk is a planned/target transition representation, not a current observation.",
        "frozen_rq_status": "warning" if rq_warning else "compatible diagnostic",
    }
    result = {
        "schema": "tactile3d-unit.s3-3-r-held-out-evaluation.v1",
        "decision": decision,
        "ready": ready,
        "gpu": {"preferred_physical": int(config["gpu"]["preferred_physical"]), "actual_physical": physical, "fallback": physical != int(config["gpu"]["preferred_physical"]), "logical": "cuda:0", "isolation": "PASS"},
        "data": {"split": "untouched frozen T-Rex test episodes", "windows": len(test_indices), "leakage": test_cache.manifest["leakage"], "dynamic_threshold_source": "frozen train split only"},
        "selected_architecture": {"candidate": training["selected_candidate"], "encoder_type": training["selected_encoder_type"], "parameters": training["selected_parameter_summary"], "checkpoint_metadata": metadata},
        "reconstruction": reconstruction,
        "temporal_controls": temporal,
        "feature_ablation": ablation,
        "frozen_probes": probes,
        "noncollapse": noncollapse,
        "determinism": deterministic,
        "original_unit_preservation": training["original_unit_preservation"],
        "frozen_rq_diagnostic": rq_diagnostic,
        "gates": gates,
        "track_c_contract": contract,
    }
    atomic_json(artifact_root / "held_out_evaluation.json", result)
    atomic_json(artifact_root / "final_decision.json", {"decision": decision, "ready": ready, "gates": gates, "track_c_contract": contract})
    np.savez_compressed(
        artifact_root / "visualization_data.npz",
        test_z=test_z,
        quantized_test_z=quantized_test_z,
        magnitude=test_labels["magnitude"],
        active_side=test_labels["active_side"],
        primitive_id=test_labels["primitive_id"],
        **temporal_arrays,
    )
    print(json.dumps({"decision": decision, "ready": ready, "gates": gates, "gpu": result["gpu"]}, indent=2))


if __name__ == "__main__":
    main()
