#!/usr/bin/env python3
"""A-R0 diagnosis of raw temporal signal, encoder sensitivity, and decoder bypass."""

from __future__ import annotations

import argparse
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
    TREX_EMBODIMENT_ID,
    ReleasedTokenizerSource,
    effective_rank,
    load_bootstrap_checkpoint,
)
from gr00t.tactile_unit.trex_action_data import (  # noqa: E402
    RAW_ACTION_DIM,
    SEGMENTS,
    TReXActionCache,
    action_activity,
    different_episode_indices,
)


DEFAULT_CONFIG = ROOT / "configs/tactile_unit/s3_3_r_action_transition_remediation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--raw-max-windows", type=int, default=None)
    parser.add_argument("--raw-only", action="store_true")
    parser.add_argument("--reuse-raw", action="store_true")
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
        raise RuntimeError("A-R0 model audit requires exactly one visible CUDA device")
    physical = int(os.environ.get("TACTILE_PHYSICAL_GPU", "-1"))
    if physical not in set(map(int, config["gpu"]["allowed_physical"])):
        raise RuntimeError("A-R0 received a forbidden physical GPU")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("physical/logical GPU isolation mismatch")
    return torch.device("cuda:0"), physical


def selected_indices(length: int, count: int | None) -> np.ndarray:
    count = length if count is None else min(length, int(count))
    return np.floor((np.arange(count) + 0.5) * length / count).astype(np.int64)


def _empty() -> dict[str, float]:
    return defaultdict(float)


def _final(value: Mapping[str, float]) -> dict[str, float]:
    count = max(value.get("count", 0.0), 1.0)
    return {
        "windows": int(value.get("count", 0.0)),
        "normalized_mse": value.get("normalized_mse", 0.0) / count,
        "raw_mse": value.get("raw_mse", 0.0) / count,
        "normalized_sequence_l2": value.get("normalized_sequence_l2", 0.0) / count,
        "normalized_cosine": value.get("normalized_cosine", 0.0) / count,
        "first_difference_mse": value.get("first_difference_mse", 0.0) / count,
        "unchanged_fraction": value.get("unchanged", 0.0) / count,
    }


def _update_raw(accumulator: dict[str, float], correct: np.ndarray, candidate: np.ndarray, raw_correct: np.ndarray, raw_candidate: np.ndarray) -> None:
    difference = candidate - correct
    raw_difference = raw_candidate - raw_correct
    flat_correct = correct.reshape(len(correct), -1)
    flat_candidate = candidate.reshape(len(candidate), -1)
    denominator = np.maximum(
        np.linalg.norm(flat_correct, axis=1) * np.linalg.norm(flat_candidate, axis=1), 1e-12
    )
    accumulator["normalized_mse"] += float(np.square(difference).mean(axis=(1, 2)).sum())
    accumulator["raw_mse"] += float(np.square(raw_difference).mean(axis=(1, 2)).sum())
    accumulator["normalized_sequence_l2"] += float(np.linalg.norm(difference.reshape(len(difference), -1), axis=1).sum())
    accumulator["normalized_cosine"] += float(((flat_correct * flat_candidate).sum(axis=1) / denominator).sum())
    accumulator["first_difference_mse"] += float(
        np.square(np.diff(candidate, axis=1) - np.diff(correct, axis=1)).mean(axis=(1, 2)).sum()
    )
    accumulator["unchanged"] += float((np.abs(difference).max(axis=(1, 2)) == 0).sum())
    accumulator["count"] += len(correct)


def raw_negative_audit(
    cache: TReXActionCache,
    *,
    dynamic_threshold: float,
    permutation: np.ndarray,
    batch_size: int,
    max_windows: int | None,
) -> dict[str, Any]:
    indices = selected_indices(len(cache), max_windows)
    accumulators: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(_empty))
    primitive_accumulators: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(_empty))
    for start in range(0, len(indices), batch_size):
        current = indices[start : start + batch_size]
        batch = cache.batch(current)
        action = batch["action"][..., :RAW_ACTION_DIM]
        action_raw = batch["action_raw"]
        other_batch = cache.batch(different_episode_indices(cache, current))
        controls = {
            "correct": (action, action_raw),
            "reversed": (action[:, ::-1], action_raw[:, ::-1]),
            "shuffled": (action[:, permutation], action_raw[:, permutation]),
            "different_episode": (
                other_batch["action"][..., :RAW_ACTION_DIM], other_batch["action_raw"]
            ),
        }
        dynamic = action_activity(action)["magnitude"] > dynamic_threshold
        subsets = {"all": np.ones(len(current), dtype=bool), "static": ~dynamic, "dynamic": dynamic}
        for control, (candidate, candidate_raw) in controls.items():
            for subset_name, mask in subsets.items():
                if mask.any():
                    _update_raw(
                        accumulators[control][subset_name], action[mask], candidate[mask],
                        action_raw[mask], candidate_raw[mask],
                    )
            for segment_name, segment in SEGMENTS.items():
                _update_raw(
                    accumulators[control][segment_name], action[..., segment], candidate[..., segment],
                    action_raw[..., segment], candidate_raw[..., segment],
                )
            for primitive in np.unique(batch["primitive_id"]):
                mask = batch["primitive_id"] == primitive
                _update_raw(
                    primitive_accumulators[str(int(primitive))][control], action[mask], candidate[mask],
                    action_raw[mask], candidate_raw[mask],
                )
    return {
        "windows": len(indices),
        "controls": {
            control: {subset: _final(value) for subset, value in subsets.items()}
            for control, subsets in accumulators.items()
        },
        "primitive": {
            primitive: {control: _final(value) for control, value in controls.items()}
            for primitive, controls in primitive_accumulators.items()
        },
    }


def _tensor_batch(cache: TReXActionCache, indices: np.ndarray, device: torch.device):
    batch = cache.batch(indices)
    state = torch.from_numpy(batch["state"]).to(device)
    action = torch.from_numpy(batch["action"]).to(device)
    embodiment = torch.full((len(indices),), TREX_EMBODIMENT_ID, dtype=torch.long, device=device)
    return batch, state, action, embodiment


def _cosine(first: torch.Tensor, second: torch.Tensor) -> np.ndarray:
    return F.cosine_similarity(first.float(), second.float(), dim=-1).cpu().numpy()


def _latent_pair_metrics(correct: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    pooled = _cosine(correct.mean(dim=1), candidate.mean(dim=1))
    flat = _cosine(correct.flatten(1), candidate.flatten(1))
    per_query = torch.linalg.vector_norm(correct.float() - candidate.float(), dim=-1).cpu().numpy()
    flat_distance = torch.linalg.vector_norm(correct.float().flatten(1) - candidate.float().flatten(1), dim=-1).cpu().numpy()
    # Same-window retrieval among the control batch; deterministic subset bounds quadratic memory.
    retrieval_count = min(len(correct), 256)
    query = F.normalize(correct[:retrieval_count].float().flatten(1), dim=-1)
    gallery = F.normalize(candidate[:retrieval_count].float().flatten(1), dim=-1)
    nearest = (query @ gallery.T).argmax(dim=1)
    return {
        "pooled_cosine_mean": float(pooled.mean()),
        "flattened_cosine_mean": float(flat.mean()),
        "flattened_distance_mean": float(flat_distance.mean()),
        "per_query_distance_mean": per_query.mean(axis=0).tolist(),
        "same_window_retrieval_top1": float((nearest == torch.arange(retrieval_count, device=nearest.device)).float().mean()),
    }


def _error_breakdown(
    prediction: torch.Tensor,
    target: torch.Tensor,
    dynamic: np.ndarray,
) -> dict[str, float]:
    squared = (prediction[..., :RAW_ACTION_DIM].float() - target[..., :RAW_ACTION_DIM].float()).square().cpu().numpy()
    result = {"all_mse": float(squared.mean())}
    result["dynamic_mse"] = float(squared[dynamic].mean()) if dynamic.any() else float("nan")
    for name, segment in SEGMENTS.items():
        result[f"{name}_mse"] = float(squared[..., segment].mean())
        result[f"dynamic_{name}_mse"] = float(squared[dynamic, ..., segment].mean()) if dynamic.any() else float("nan")
    return result


@torch.no_grad()
def mean_train_latent(model: torch.nn.Module, cache: TReXActionCache, count: int, batch_size: int, device: torch.device) -> torch.Tensor:
    indices = selected_indices(len(cache), count)
    total = None
    seen = 0
    for start in range(0, len(indices), batch_size):
        current = indices[start : start + batch_size]
        _, state, action, embodiment = _tensor_batch(cache, current, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z_action, _, _ = model.encode(state, action, embodiment)
        current_sum = z_action.float().sum(dim=0)
        total = current_sum if total is None else total + current_sum
        seen += len(current)
    return total / seen


@torch.no_grad()
def model_audit(
    model: torch.nn.Module,
    cache: TReXActionCache,
    *,
    mean_token: torch.Tensor,
    count: int,
    batch_size: int,
    device: torch.device,
    permutation: np.ndarray,
    dynamic_threshold: float,
    selection_seed: int,
) -> dict[str, Any]:
    count = min(int(count), len(cache))
    indices = np.sort(
        np.random.default_rng(selection_seed).choice(len(cache), size=count, replace=False)
    ).astype(np.int64)
    latent: dict[str, list[torch.Tensor]] = defaultdict(list)
    errors: dict[str, list[dict[str, float]]] = defaultdict(list)
    for start in range(0, len(indices), batch_size):
        current = indices[start : start + batch_size]
        batch, state, action, embodiment = _tensor_batch(cache, current, device)
        other = cache.batch(different_episode_indices(cache, current))
        other_action = torch.from_numpy(other["action"]).to(device)
        controls = {
            "correct": action,
            "reversed": action.flip(1),
            "shuffled": action[:, torch.from_numpy(permutation).to(device)],
            "different_episode": other_action,
        }
        encoded = {}
        state_features = None
        for name, candidate in controls.items():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z_action, features, _ = model.encode(state, candidate, embodiment)
            encoded[name] = z_action.float()
            latent[name].append(z_action.float().cpu())
            if name == "correct":
                state_features = features
        assert state_features is not None
        decode_tokens = {
            "full": encoded["correct"],
            "reversed": encoded["reversed"],
            "shuffled": encoded["shuffled"],
            "different_episode": encoded["different_episode"],
            "zero": torch.zeros_like(encoded["correct"]),
            "mean": mean_token.to(device).unsqueeze(0).expand(len(current), -1, -1),
            "state_only": torch.zeros_like(encoded["correct"]),
        }
        dynamic = action_activity(batch["action"][..., :RAW_ACTION_DIM])["magnitude"] > dynamic_threshold
        for name, token in decode_tokens.items():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = model.decode(token, state_features, embodiment)
            errors[name].append(_error_breakdown(prediction, action, dynamic))
    latent_tensor = {name: torch.cat(values) for name, values in latent.items()}
    encoder = {
        name: {
            **_latent_pair_metrics(latent_tensor["correct"], value),
            "effective_rank": effective_rank(value.flatten(1)),
        }
        for name, value in latent_tensor.items()
    }
    decoder = {
        name: {
            key: float(np.mean([entry[key] for entry in values]))
            for key in values[0]
        }
        for name, values in errors.items()
    }
    full = decoder["full"]["all_mse"]
    reasonable = min(decoder[name]["all_mse"] for name in ("zero", "mean", "shuffled"))
    decoder["token_contribution"] = {
        "control": "minimum-error of zero/mean/shuffled",
        "R_token_with_zero_oracle": (reasonable - full) / max(reasonable, 1e-12),
        "note": "The specified denominator has no separately available oracle; zero error is reported explicitly as the ideal oracle.",
    }
    dynamic_windows = int(
        sum(
            (
                action_activity(cache.batch(current)["action"][..., :RAW_ACTION_DIM])[
                    "magnitude"
                ]
                > dynamic_threshold
            ).sum()
            for current in np.array_split(indices, max(1, int(np.ceil(len(indices) / 2048))))
            if len(current)
        )
    )
    return {
        "windows": len(indices),
        "dynamic_windows": dynamic_windows,
        "selection": "fixed-seed random without replacement",
        "selection_seed": selection_seed,
        "encoder": encoder,
        "decoder": decoder,
    }


def classify(result: Mapping[str, Any]) -> tuple[str, list[str]]:
    test_raw = result["raw_negative_strength"]["test"]["controls"]
    test_model = result["model_audit"]["test"]
    raw_dynamic_nontrivial = (
        test_raw["reversed"]["dynamic"]["normalized_mse"] > 0.02
        and test_raw["shuffled"]["dynamic"]["normalized_mse"] > 0.01
    )
    encoder = test_model["encoder"]
    other_distance = max(encoder["different_episode"]["flattened_distance_mean"], 1e-12)
    order_insensitive = (
        encoder["reversed"]["flattened_distance_mean"] / other_distance < 0.25
        or encoder["shuffled"]["flattened_distance_mean"] / other_distance < 0.25
    )
    decoder = test_model["decoder"]
    bypass = decoder["zero"]["all_mse"] / max(decoder["full"]["all_mse"], 1e-12) < 1.1
    diagnoses = []
    if not raw_dynamic_nontrivial:
        diagnoses.append("WEAK_TEMPORAL_NEGATIVES")
    if bypass:
        diagnoses.append("DECODER_STATE_BYPASS")
    if order_insensitive:
        diagnoses.append("ABSOLUTE_ACTION_ORDER_INVARIANCE")
    if order_insensitive and raw_dynamic_nontrivial:
        diagnoses.append("FROZEN_TRUNK_DOMAIN_MISMATCH")
    if not diagnoses:
        return "NO_CLEAR_DIAGNOSIS", []
    return (diagnoses[0] if len(diagnoses) == 1 else "MIXED"), diagnoses


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    diagnosis_cfg = config["diagnosis"]
    cache_root = resolve(config["data"]["cache_root"])
    normalization = resolve(config["data"]["normalization"])
    training = json.loads(resolve(config["paths"]["bootstrap_training"]).read_text())
    threshold = float(training["distribution"]["dynamic_threshold_normalized_rms_delta"])
    permutation = np.random.default_rng(int(diagnosis_cfg["shuffle_seed"])).permutation(16)
    caches = {name: TReXActionCache(cache_root, name, normalization) for name in ("train", "val", "test")}
    artifact_root = resolve(config["paths"]["artifact_root"])
    raw_path = artifact_root / "a_r0_raw_negative_strength.json"
    if args.reuse_raw:
        raw_payload = json.loads(raw_path.read_text())
        raw = raw_payload["raw_negative_strength"]
    else:
        raw = {
            name: raw_negative_audit(
                cache,
                dynamic_threshold=threshold,
                permutation=permutation,
                batch_size=int(diagnosis_cfg["raw_batch_size"]),
                max_windows=args.raw_max_windows,
            )
            for name, cache in caches.items()
        }
        atomic_json(raw_path, {
            "schema": "tactile3d-unit.s3-3-r-raw-negative-strength.v1",
            "shuffle_seed": int(diagnosis_cfg["shuffle_seed"]),
            "permutation": permutation.tolist(),
            "dynamic_threshold": threshold,
            "raw_negative_strength": raw,
        })
    if args.raw_only:
        print(json.dumps({"output": str(raw_path.relative_to(ROOT)), "windows": {name: value["windows"] for name, value in raw.items()}}, indent=2))
        return
    device, physical = require_isolated_gpu(config)
    source = ReleasedTokenizerSource.open(args.tokenizer_root)
    model, _ = load_bootstrap_checkpoint(resolve(config["paths"]["bootstrap_checkpoint"]), source)
    model = model.to(device).eval()
    mean_token = mean_train_latent(
        model, caches["train"], int(diagnosis_cfg["mean_token_train_windows"]),
        int(diagnosis_cfg["model_batch_size"]), device,
    )
    model_results = {
        name: model_audit(
            model,
            cache,
            mean_token=mean_token,
            count=int(diagnosis_cfg["model_windows_per_split"]),
            batch_size=int(diagnosis_cfg["model_batch_size"]),
            device=device,
            permutation=permutation,
            dynamic_threshold=threshold,
            selection_seed=int(config["seed"]) + 1000 + split_index,
        )
        for split_index, (name, cache) in enumerate(caches.items())
    }
    result: dict[str, Any] = {
        "schema": "tactile3d-unit.s3-3-r-a-r0-diagnosis.v1",
        "training_performed": False,
        "gpu": {
            "preferred_physical": int(config["gpu"]["preferred_physical"]),
            "actual_physical": physical,
            "logical": "cuda:0",
            "visible_device_count": torch.cuda.device_count(),
            "fallback": physical != int(config["gpu"]["preferred_physical"]),
            "isolation": "PASS",
        },
        "data": {
            "shuffle_seed": int(diagnosis_cfg["shuffle_seed"]),
            "permutation": permutation.tolist(),
            "dynamic_threshold": threshold,
            "leakage": caches["test"].manifest["leakage"],
            "action_interval": caches["test"].manifest["action_interval"],
        },
        "raw_negative_strength": raw,
        "model_audit": model_results,
        "temporal_negative_validity": {
            "same_episode_reversed_shuffled": True,
            "same_valid_16_step_support": True,
            "padding_artifact": False,
            "unchanged_frame_bug": False,
            "deterministic_seed": int(diagnosis_cfg["shuffle_seed"]),
            "data_leakage": caches["test"].manifest["leakage"],
        },
    }
    primary, secondary = classify(result)
    result["diagnosis"] = {"primary": primary, "factors": secondary}
    output = artifact_root / "a_r0_diagnosis.json"
    atomic_json(output, result)
    print(json.dumps({"output": str(output.relative_to(ROOT)), "diagnosis": result["diagnosis"], "gpu": result["gpu"]}, indent=2))


if __name__ == "__main__":
    main()
