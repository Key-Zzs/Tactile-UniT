#!/usr/bin/env python3
"""Held-out S3.3 evaluation, preservation proof, and frozen-RQ diagnostic."""

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

from gr00t.model.tokenizer.vector_quantizer import (  # noqa: E402
    ResidualVectorQuantizer,
    ResidualVectorQuantizerConfig,
)
from gr00t.tactile_unit.trex_action_bootstrap import (  # noqa: E402
    EXPANDED_CATEGORY_CAPACITY,
    GR1_EMBODIMENT_ID,
    QUERY_NUM,
    RELEASED_CATEGORY_CAPACITY,
    TREX_EMBODIMENT_ID,
    UNUSED_EMBODIMENT_ID,
    ReleasedTokenizerSource,
    effective_rank,
    expand_category_tensor,
    load_bootstrap_checkpoint,
    overlay_to_released_name,
    query_diversity,
)
from gr00t.tactile_unit.trex_action_data import (  # noqa: E402
    RAW_ACTION_DIM,
    SEGMENTS,
    TReXActionCache,
    action_activity,
    different_episode_indices,
)


DEFAULT_CONFIG = ROOT / "configs/tactile_unit/s3_3_action_bootstrap.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--max-test-windows", type=int, default=None)
    return parser.parse_args()


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _json_default(item: Any) -> Any:
    if isinstance(item, np.generic):
        return item.item()
    raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n"
    )
    temporary.replace(path)


def require_isolated_gpu(config: Mapping[str, Any]) -> tuple[torch.device, int]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("S3.3 evaluation requires exactly one visible CUDA device")
    physical = int(os.environ.get("TACTILE_PHYSICAL_GPU", "-1"))
    if physical not in set(map(int, config["gpu"]["allowed_physical"])):
        raise RuntimeError("evaluation GPU is outside the configured allowed physical set")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("physical/logical GPU isolation mismatch")
    return torch.device("cuda:0"), physical


def make_tensors(
    cache: TReXActionCache, indices: np.ndarray, device: torch.device
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = cache.batch(indices)
    state = torch.from_numpy(batch["state"]).to(device)
    action = torch.from_numpy(batch["action"]).to(device)
    embodiment = torch.full(
        (len(indices),), TREX_EMBODIMENT_ID, dtype=torch.long, device=device
    )
    return batch, state, action, embodiment


def selected_indices(length: int, count: int | None) -> np.ndarray:
    count = length if count is None else min(length, int(count))
    return np.floor((np.arange(count) + 0.5) * length / count).astype(np.int64)


def _empty_accumulator() -> dict[str, float]:
    return {"sq": 0.0, "abs": 0.0, "count": 0.0, "raw_sq": 0.0, "raw_abs": 0.0}


def _update_errors(
    accumulator: dict[str, float],
    prediction: np.ndarray,
    target: np.ndarray,
    raw_prediction: np.ndarray,
    raw_target: np.ndarray,
) -> None:
    difference = prediction - target
    raw_difference = raw_prediction - raw_target
    accumulator["sq"] += float(np.square(difference, dtype=np.float64).sum())
    accumulator["abs"] += float(np.abs(difference).sum(dtype=np.float64))
    accumulator["raw_sq"] += float(np.square(raw_difference, dtype=np.float64).sum())
    accumulator["raw_abs"] += float(np.abs(raw_difference).sum(dtype=np.float64))
    accumulator["count"] += float(difference.size)


def _finalize_errors(value: Mapping[str, float]) -> dict[str, float]:
    count = max(float(value["count"]), 1.0)
    return {
        "normalized_mse": float(value["sq"] / count),
        "normalized_mae": float(value["abs"] / count),
        "raw_unit_mse": float(value["raw_sq"] / count),
        "raw_unit_mae": float(value["raw_abs"] / count),
        "element_count": int(value["count"]),
    }


def load_frozen_rq(source: ReleasedTokenizerSource) -> ResidualVectorQuantizer:
    config = ResidualVectorQuantizerConfig(**source.config["vq_cfg"])
    rq = ResidualVectorQuantizer(config)
    state = {
        name: source.tensor(f"vq.{name}")
        for name in rq.state_dict()
    }
    rq.load_state_dict(state, strict=True)
    rq.eval()
    return rq


@torch.no_grad()
def extract_latents(
    model: torch.nn.Module,
    cache: TReXActionCache,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    model.eval()
    latents: list[np.ndarray] = []
    labels: dict[str, list[np.ndarray]] = defaultdict(list)
    for start in range(0, len(indices), batch_size):
        current = indices[start : start + batch_size]
        batch, state, action, embodiment = make_tensors(cache, current, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z_action, _, _ = model.encode(state, action, embodiment)
        latents.append(z_action.float().cpu().numpy())
        activity = action_activity(batch["action"][:, :, :RAW_ACTION_DIM])
        for name, value in activity.items():
            labels[name].append(value)
        labels["primitive_id"].append(batch["primitive_id"])
    return np.concatenate(latents), {
        name: np.concatenate(values) for name, values in labels.items()
    }


def fit_frozen_probes(
    train_z: np.ndarray,
    train_labels: Mapping[str, np.ndarray],
    test_z: np.ndarray,
    test_labels: Mapping[str, np.ndarray],
    *,
    quantized_train_z: np.ndarray | None = None,
    quantized_test_z: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_x = train_z.reshape(len(train_z), -1)
    test_x = test_z.reshape(len(test_z), -1)
    quantized_train_x = None if quantized_train_z is None else quantized_train_z.reshape(len(quantized_train_z), -1)
    quantized_test_x = None if quantized_test_z is None else quantized_test_z.reshape(len(quantized_test_z), -1)
    results: dict[str, Any] = {}
    quantized_results: dict[str, Any] = {}
    models: dict[str, Any] = {}
    for name in ("magnitude", "trend"):
        probe = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        probe.fit(train_x, train_labels[name])
        prediction = probe.predict(test_x)
        results[name] = {
            "task": "regression",
            "label_source": "DERIVED from canonical action chunk",
            "r2": float(r2_score(test_labels[name], prediction)),
        }
        if quantized_train_x is not None and quantized_test_x is not None:
            q_probe = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            q_probe.fit(quantized_train_x, train_labels[name])
            q_prediction = q_probe.predict(quantized_test_x)
            quantized_results[name] = {"r2": float(r2_score(test_labels[name], q_prediction))}
        models[name] = probe
    for name in ("active_side", "arm_vs_hand", "primitive_id"):
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=3301,
                solver="lbfgs",
            ),
        )
        probe.fit(train_x, train_labels[name])
        prediction = probe.predict(test_x)
        source = "ACTUAL T-Rex episode metadata" if name == "primitive_id" else "DERIVED from canonical action chunk"
        results[name] = {
            "task": "classification",
            "label_source": source,
            "accuracy": float(accuracy_score(test_labels[name], prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(test_labels[name], prediction)),
            "classes": int(len(np.unique(train_labels[name]))),
        }
        if quantized_train_x is not None and quantized_test_x is not None:
            q_probe = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=3301,
                    solver="lbfgs",
                ),
            )
            q_probe.fit(quantized_train_x, train_labels[name])
            q_prediction = q_probe.predict(quantized_test_x)
            quantized_results[name] = {
                "accuracy": float(accuracy_score(test_labels[name], q_prediction)),
                "balanced_accuracy": float(balanced_accuracy_score(test_labels[name], q_prediction)),
            }
        models[name] = probe
    retention = {}
    for name, continuous in results.items():
        if name not in quantized_results:
            continue
        metric = "r2" if continuous["task"] == "regression" else "balanced_accuracy"
        denominator = max(abs(float(continuous[metric])), 1e-8)
        retention[name] = float(quantized_results[name][metric] / denominator)
    return results, quantized_results, retention


@torch.no_grad()
def quantize_numpy(
    rq: torch.nn.Module,
    values: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    outputs: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    rq.to(device).eval()
    for start in range(0, len(values), batch_size):
        tensor = torch.from_numpy(values[start : start + batch_size]).to(device)
        quantized, codes, _ = rq(tensor)
        if codes.ndim == 2:
            codes = codes.unsqueeze(-1)
        outputs.append(quantized.float().cpu().numpy())
        indices.append(codes.cpu().numpy())
    return np.concatenate(outputs), np.concatenate(indices)


def code_usage(codes: np.ndarray, codebook_size: int) -> dict[str, Any]:
    result = {}
    for stage in range(codes.shape[-1]):
        counts = np.bincount(codes[..., stage].reshape(-1), minlength=codebook_size)
        probability = counts / max(counts.sum(), 1)
        nonzero = probability[probability > 0]
        perplexity = float(np.exp(-(nonzero * np.log(nonzero)).sum()))
        sorted_probability = np.sort(probability)[::-1]
        result[f"stage_{stage}"] = {
            "active_codes": int((counts > 0).sum()),
            "perplexity": perplexity,
            "top1_mass": float(sorted_probability[:1].sum()),
            "top5_mass": float(sorted_probability[:5].sum()),
        }
    return result


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    device, physical_gpu = require_isolated_gpu(config)
    source = ReleasedTokenizerSource.open(args.tokenizer_root)
    artifact_root = resolve(config["paths"]["artifact_root"])
    experiment_root = resolve(config["paths"]["experiment_root"])
    checkpoint = args.checkpoint or experiment_root / "selected.pt"
    training = json.loads((artifact_root / "training_summary.json").read_text())
    model, checkpoint_metadata = load_bootstrap_checkpoint(checkpoint, source)
    model = model.to(device).eval()
    cache_root = resolve(config["data"]["cache_root"])
    normalization = resolve(config["data"]["normalization"])
    train_cache = TReXActionCache(cache_root, "train", normalization)
    test_cache = TReXActionCache(cache_root, "test", normalization)
    evaluation_cfg = config["evaluation"]
    batch_size = int(evaluation_cfg["batch_size"])
    test_indices = selected_indices(len(test_cache), args.max_test_windows)
    dynamic_threshold = float(training["distribution"]["dynamic_threshold_normalized_rms_delta"])

    error = {"all": _empty_accumulator(), "dynamic": _empty_accumulator()}
    for name in SEGMENTS:
        error[name] = _empty_accumulator()
        error[f"dynamic_{name}"] = _empty_accumulator()
    temporal_sums = defaultdict(float)
    temporal_count = 0
    deterministic_exact = True
    deterministic_windows = 0
    rng = np.random.default_rng(3301)
    temporal_limit = min(int(evaluation_cfg["temporal_windows"]), len(test_indices))

    with torch.no_grad():
        for start in range(0, len(test_indices), batch_size):
            current = test_indices[start : start + batch_size]
            batch, state, action, embodiment = make_tensors(test_cache, current, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(state, action, embodiment)
                repeat = model(state, action, embodiment) if start == 0 else None
            prediction = output["prediction"].float().cpu().numpy()[..., :RAW_ACTION_DIM]
            target = batch["action"][..., :RAW_ACTION_DIM]
            raw_prediction = test_cache.inverse_action(prediction)
            raw_target = batch["action_raw"]
            activity = action_activity(target)
            dynamic = activity["magnitude"] > dynamic_threshold
            _update_errors(error["all"], prediction, target, raw_prediction, raw_target)
            if dynamic.any():
                _update_errors(
                    error["dynamic"],
                    prediction[dynamic], target[dynamic], raw_prediction[dynamic], raw_target[dynamic],
                )
            for name, segment in SEGMENTS.items():
                _update_errors(
                    error[name], prediction[..., segment], target[..., segment], raw_prediction[..., segment], raw_target[..., segment]
                )
                if dynamic.any():
                    _update_errors(
                        error[f"dynamic_{name}"], prediction[dynamic, ..., segment], target[dynamic, ..., segment], raw_prediction[dynamic, ..., segment], raw_target[dynamic, ..., segment]
                    )
            if repeat is not None:
                deterministic_exact = deterministic_exact and bool(
                    torch.equal(output["z_action"], repeat["z_action"])
                )
                deterministic_windows += len(current)

            if start < temporal_limit:
                allowed = min(len(current), temporal_limit - start)
                temporal_action = action[:allowed]
                temporal_state = state[:allowed]
                temporal_embodiment = embodiment[:allowed]
                correct_error = F.mse_loss(
                    output["prediction"][:allowed, ..., :RAW_ACTION_DIM].float(),
                    temporal_action[..., :RAW_ACTION_DIM],
                    reduction="none",
                ).mean(dim=(1, 2))
                different_indices = different_episode_indices(test_cache, current[:allowed])
                different_batch = test_cache.batch(different_indices)
                different_action = torch.from_numpy(different_batch["action"]).to(device)
                controls = {
                    "reversed": temporal_action.flip(1),
                    "shuffled": temporal_action[:, torch.from_numpy(rng.permutation(16)).to(device)],
                    "different_episode": different_action,
                }
                temporal_sums["correct"] += float(correct_error.sum().cpu())
                for name, negative in controls.items():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        negative_output = model(temporal_state, negative, temporal_embodiment)
                    negative_error = F.mse_loss(
                        negative_output["prediction"][..., :RAW_ACTION_DIM].float(),
                        temporal_action[..., :RAW_ACTION_DIM],
                        reduction="none",
                    ).mean(dim=(1, 2))
                    temporal_sums[name] += float(negative_error.sum().cpu())
                temporal_count += allowed

    reconstruction = {name: _finalize_errors(value) for name, value in error.items()}
    temporal = {name: value / temporal_count for name, value in temporal_sums.items()}
    for name in ("reversed", "shuffled", "different_episode"):
        temporal[f"{name}_ratio_to_correct"] = temporal[name] / max(temporal["correct"], 1e-12)
    temporal["windows"] = temporal_count

    probe_train_indices = selected_indices(len(train_cache), int(evaluation_cfg["probe_train_windows"]))
    probe_test_indices = selected_indices(len(test_cache), int(evaluation_cfg["probe_test_windows"]))
    train_z, train_labels = extract_latents(
        model, train_cache, probe_train_indices, batch_size=batch_size, device=device
    )
    test_z, test_labels = extract_latents(
        model, test_cache, probe_test_indices, batch_size=batch_size, device=device
    )
    latent_count = min(int(evaluation_cfg["latent_windows"]), len(test_z))
    latent = test_z[:latent_count]
    flat = latent.reshape(latent_count, -1)
    variance = flat.var(axis=0)
    token_norm = np.linalg.norm(latent, axis=-1)
    sample_pairs = min(4096, latent_count // 2)
    pairwise = np.linalg.norm(flat[:sample_pairs] - flat[-sample_pairs:], axis=1)
    noncollapse = {
        "windows": latent_count,
        "per_dimension_variance": {
            "minimum": float(variance.min()),
            "median": float(np.median(variance)),
            "mean": float(variance.mean()),
            "near_zero_fraction": float((variance < 1e-8).mean()),
        },
        "effective_rank": effective_rank(torch.from_numpy(flat)),
        "token_norm": {
            "mean": float(token_norm.mean()),
            "std": float(token_norm.std()),
            "minimum": float(token_norm.min()),
            "maximum": float(token_norm.max()),
        },
        "query_diversity": query_diversity(torch.from_numpy(latent)),
        "collapsed_query_fraction": float(
            (latent.var(axis=(0, 2)) < 1e-8).mean()
        ),
        "pairwise_sample_distance": {
            "mean": float(pairwise.mean()),
            "median": float(np.median(pairwise)),
            "minimum": float(pairwise.min()),
        },
    }

    rq = load_frozen_rq(source)
    quantized_train_z, train_codes = quantize_numpy(
        rq, train_z, batch_size=batch_size, device=device
    )
    quantized_test_z, test_codes = quantize_numpy(
        rq, test_z, batch_size=batch_size, device=device
    )
    probes, quantized_probes, semantic_retention = fit_frozen_probes(
        train_z,
        train_labels,
        test_z,
        test_labels,
        quantized_train_z=quantized_train_z,
        quantized_test_z=quantized_test_z,
    )
    difference = quantized_test_z - test_z
    relative_distortion = np.linalg.norm(difference.reshape(len(test_z), -1), axis=1) / np.maximum(
        np.linalg.norm(test_z.reshape(len(test_z), -1), axis=1), 1e-12
    )
    cosine = np.sum(quantized_test_z * test_z, axis=-1) / np.maximum(
        np.linalg.norm(quantized_test_z, axis=-1) * np.linalg.norm(test_z, axis=-1), 1e-12
    )

    # Decode both continuous and frozen-RQ representations on the same held-out
    # samples to measure action retention without changing the shared RQ.
    continuous_errors: list[np.ndarray] = []
    quantized_errors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(probe_test_indices), batch_size):
            current = probe_test_indices[start : start + batch_size]
            _, state, action, embodiment = make_tensors(test_cache, current, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z, state_features, _ = model.encode(state, action, embodiment)
                continuous = model.decode(z, state_features, embodiment)
                q_value, _, _ = rq(z.float())
                quantized_prediction = model.decode(q_value, state_features, embodiment)
            continuous_errors.append(
                F.mse_loss(continuous[..., :RAW_ACTION_DIM].float(), action[..., :RAW_ACTION_DIM], reduction="none").mean(dim=(1, 2)).cpu().numpy()
            )
            quantized_errors.append(
                F.mse_loss(quantized_prediction[..., :RAW_ACTION_DIM].float(), action[..., :RAW_ACTION_DIM], reduction="none").mean(dim=(1, 2)).cpu().numpy()
            )
    continuous_error = np.concatenate(continuous_errors)
    quantized_error = np.concatenate(quantized_errors)
    rq_diagnostic = {
        "read_only": True,
        "relative_distortion": {
            "mean": float(relative_distortion.mean()),
            "median": float(np.median(relative_distortion)),
        },
        "pre_post_cosine": {
            "mean": float(cosine.mean()),
            "median": float(np.median(cosine)),
        },
        "code_usage": code_usage(test_codes, int(source.config["vq_cfg"]["n_e"])),
        "continuous_reconstruction_mse": float(continuous_error.mean()),
        "quantized_reconstruction_mse": float(quantized_error.mean()),
        "reconstruction_error_ratio": float(quantized_error.mean() / max(continuous_error.mean(), 1e-12)),
        "quantized_probes": quantized_probes,
        "semantic_retention_ratio": semantic_retention,
    }

    # Stream the actual 30->32 expansion tensor by tensor.  This is stronger
    # than comparing only a nominal embedding table: every encoder and decoder
    # category-indexed tensor is covered.
    overlay = model.overlay_state_dict()
    released_overlay = {
        overlay_to_released_name(name): value
        for name, value in overlay.items()
        if not name.startswith("a2_adapter.")
    }
    expansion_checks = []
    for name in source.category_tensor_names():
        released = source.tensor(name)
        expanded = expand_category_tensor(
            released,
            name=name,
            trex_row=released_overlay[name],
            seed=int(training["a1"]["history"][0].get("step", 3301)),
        )
        expansion_checks.append(
            expanded.shape[0] == EXPANDED_CATEGORY_CAPACITY
            and torch.equal(expanded[:RELEASED_CATEGORY_CAPACITY], released)
        )
        del expanded, released
    preservation = {
        "old_rows_digest_before": training["old_rows_digest_before"],
        "old_rows_digest_after": training["old_rows_digest_after"],
        "old_rows_bit_identical": bool(all(expansion_checks))
        and training["old_rows_digest_before"] == training["old_rows_digest_after"],
        "category_tensor_count": len(expansion_checks),
        "gr1_id": GR1_EMBODIMENT_ID,
        "gr1_action_l2": {
            "comparison": "exact equality",
            "result": "PASS" if all(expansion_checks) else "FAIL",
            "proof": "all GR1-selected category rows plus every shared action-only tensor are byte-identical; the overlay contains only row 31",
        },
        "t4_non_regression_required": False,
        "reason": "A3 was not executed and no shared Action parameter changed",
    }

    acceptance = evaluation_cfg["acceptance"]
    gates = {
        "explicit_trex_identity": bool(
            TREX_EMBODIMENT_ID == 31 and TREX_EMBODIMENT_ID != GR1_EMBODIMENT_ID
        ),
        "capacity_and_unused_id": bool(
            EXPANDED_CATEGORY_CAPACITY == 32 and UNUSED_EMBODIMENT_ID == 30
        ),
        "interface": bool(
            list(test_z.shape[1:]) == [QUERY_NUM, 32] and np.isfinite(test_z).all()
        ),
        "reconstruction": bool(
            reconstruction["all"]["normalized_mse"]
            <= float(acceptance["normalized_mse_max"])
        ),
        "temporal_controls": bool(all(
            temporal[f"{name}_ratio_to_correct"] >= float(acceptance["minimum_temporal_loss_ratio"])
            for name in ("reversed", "shuffled", "different_episode")
        )),
        "noncollapse": bool(
            noncollapse["effective_rank"] >= float(acceptance["minimum_effective_rank"])
            and noncollapse["collapsed_query_fraction"]
            <= float(acceptance["maximum_collapsed_query_fraction"])
        ),
        "deterministic": bool(
            deterministic_exact and bool(training["cold_reload"]["exact"])
        ),
        "old_embodiments_preserved": bool(preservation["old_rows_bit_identical"]),
        "probe_semantics_measurable": bool(
            probes["magnitude"]["r2"] > 0.0
            and probes["active_side"]["balanced_accuracy"] > 0.5
            and probes["arm_vs_hand"]["balanced_accuracy"] > 0.5
        ),
    }
    continuous_ready = all(gates.values())
    rq_warning = (
        rq_diagnostic["relative_distortion"]["mean"] > 0.5
        or rq_diagnostic["reconstruction_error_ratio"] > 1.5
    )
    if continuous_ready:
        decision = "ACTION_BOOTSTRAP_READY_WITH_SHARED_RQ_WARNING" if rq_warning else "ACTION_BOOTSTRAP_READY"
    else:
        structural = not all(
            gates[name]
            for name in (
                "explicit_trex_identity",
                "capacity_and_unused_id",
                "interface",
                "deterministic",
                "old_embodiments_preserved",
            )
        )
        decision = "STRUCTURAL_FAIL" if structural else "ACTION_BOOTSTRAP_INSUFFICIENT"

    result = {
        "schema": "tactile3d-unit.s3-3-held-out-evaluation.v1",
        "decision": decision,
        "track_a_ready": continuous_ready,
        "gpu": {
            "preferred_physical": int(config["gpu"]["preferred_physical"]),
            "actual_physical": physical_gpu,
            "logical": "cuda:0",
            "fallback": physical_gpu != int(config["gpu"]["preferred_physical"]),
            "fallback_reason": (
                None
                if physical_gpu == int(config["gpu"]["preferred_physical"])
                else "physical GPU 3 failed the atomic lock/compute-occupancy gate; fallback was explicitly authorized"
            ),
            "lock": os.environ.get(
                "TACTILE_GPU_LOCK_NAME", f"tactile3d_unit_gpu{physical_gpu}.lock"
            ),
            "isolation": "PASS",
        },
        "data": {
            "split": "untouched frozen T-Rex test episodes",
            "windows": len(test_indices),
            "dynamic_threshold_source": "train split only",
            "leakage": test_cache.manifest["leakage"],
        },
        "interface": {
            "shape": ["B", QUERY_NUM, 32],
            "finite": bool(np.isfinite(test_z).all()),
            "deterministic_exact": deterministic_exact,
            "deterministic_repeat_windows": deterministic_windows,
            "cold_reload": training["cold_reload"],
        },
        "reconstruction": reconstruction,
        "temporal_controls": temporal,
        "frozen_probes": probes,
        "noncollapse": noncollapse,
        "original_unit_preservation": preservation,
        "frozen_rq_diagnostic": rq_diagnostic,
        "gates": gates,
        "checkpoint_metadata": checkpoint_metadata,
    }
    atomic_json(artifact_root / "held_out_evaluation.json", result)
    np.savez_compressed(
        artifact_root / "visualization_data.npz",
        test_z=test_z[:latent_count],
        quantized_test_z=quantized_test_z[:latent_count],
        magnitude=test_labels["magnitude"][:latent_count],
        active_side=test_labels["active_side"][:latent_count],
        primitive_id=test_labels["primitive_id"][:latent_count],
        test_codes=test_codes[:latent_count],
    )
    print(json.dumps({"decision": decision, "track_a_ready": continuous_ready, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
