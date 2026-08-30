#!/usr/bin/env python3
"""Run Q0 semantic-error diagnosis on the frozen ordinary two-stage Contact RQ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.lib.format import open_memmap

from contact_semantic_tokenizer_common import (
    DEFAULT_ARTIFACTS,
    DEFAULT_CACHE,
    DEFAULT_SPEC,
    apply_ridge,
    fit_ridge,
    load_runtime,
    probe_bundle,
    set_seed,
    verify_gpu,
    write_json,
)
from gr00t.tactile_teacher.evaluation import classification_metrics
from gr00t.tactile_unit.s3_2_r import build_contact_rq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACTS / "q0_diagnosis.json")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE / "q0")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--jacobian-samples", type=int, default=256)
    return parser.parse_args()


def stats(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
    }


def fit_flat_pca(
    values: np.ndarray, device: torch.device, batch_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dimension = int(np.prod(values.shape[1:]))
    total = torch.zeros(dimension, dtype=torch.float64, device=device)
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        batch = torch.from_numpy(np.array(values[start:stop], copy=True)).to(device).double()
        total += batch.reshape(len(batch), -1).sum(dim=0)
    mean = total / len(values)
    covariance = torch.zeros((dimension, dimension), dtype=torch.float64, device=device)
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        batch = torch.from_numpy(np.array(values[start:stop], copy=True)).to(device).double()
        centered = batch.reshape(len(batch), -1) - mean
        covariance += centered.T @ centered
    covariance /= max(len(values) - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    return (
        mean.float().cpu().numpy(),
        eigenvalues[order].float().cpu().numpy(),
        eigenvectors[:, order].float().cpu().numpy(),
    )


def project_to_cache(
    values: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    path: Path,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    result = open_memmap(path, mode="w+", dtype=np.float32, shape=(len(values), components.shape[1]))
    mean_t = torch.from_numpy(mean).to(device)
    components_t = torch.from_numpy(components).to(device)
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        batch = torch.from_numpy(np.array(values[start:stop], copy=True)).to(device)
        result[start:stop] = ((batch.reshape(len(batch), -1) - mean_t) @ components_t).cpu().numpy()
    result.flush()
    del result
    return np.load(path, mmap_mode="r")


def probe_direction_analysis(
    train: np.ndarray,
    native_test: np.ndarray,
    quantized_test: np.ndarray,
    train_arrays: dict[str, np.ndarray],
    test_arrays: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    definitions = {"contact_transition": ("contact_transition", 4), "force_trend": ("force_trend_class", 3)}
    masks = {
        "all": np.ones(len(native_test), dtype=bool),
        "dynamic": np.asarray(test_arrays["dynamic"], dtype=bool),
        "static": ~np.asarray(test_arrays["dynamic"], dtype=bool),
        "boundary": np.isin(np.asarray(test_arrays["contact_transition"]), [1, 3]),
        "non_boundary": np.isin(np.asarray(test_arrays["contact_transition"]), [0, 2]),
    }
    result = {}
    for name, (label_name, classes) in definitions.items():
        labels_train = np.asarray(train_arrays[label_name])
        labels_test = np.asarray(test_arrays[label_name])
        probe = fit_ridge(train, labels_train, classes, device, batch_size, 10.0, classes=classes)
        native_logits = apply_ridge(probe, native_test, device, batch_size)
        quantized_logits = apply_ridge(probe, quantized_test, device, batch_size)
        native_prediction = native_logits.argmax(axis=1)
        quantized_prediction = quantized_logits.argmax(axis=1)
        row: dict[str, Any] = {
            "native": classification_metrics(labels_test, native_prediction),
            "quantized": classification_metrics(labels_test, quantized_prediction),
            "subsets": {},
        }
        native_other = native_logits.copy()
        quantized_other = quantized_logits.copy()
        native_true = native_other[np.arange(len(labels_test)), labels_test]
        quantized_true = quantized_other[np.arange(len(labels_test)), labels_test]
        native_other[np.arange(len(labels_test)), labels_test] = -np.inf
        quantized_other[np.arange(len(labels_test)), labels_test] = -np.inf
        native_margin = native_true - native_other.max(axis=1)
        quantized_margin = quantized_true - quantized_other.max(axis=1)
        standardized_error = (
            quantized_test.reshape(len(quantized_test), -1)
            - native_test.reshape(len(native_test), -1)
        ) / probe["std"]
        direction_projection = standardized_error @ probe["weights"][:-1]
        semantic_direction_error = np.linalg.norm(direction_projection, axis=1)
        for subset_name, mask in masks.items():
            row["subsets"][subset_name] = {
                "windows": int(mask.sum()),
                "native_macro_f1": classification_metrics(labels_test[mask], native_prediction[mask])[
                    "macro_f1"
                ],
                "quantized_macro_f1": classification_metrics(
                    labels_test[mask], quantized_prediction[mask]
                )["macro_f1"],
                "margin_loss": stats((native_margin - quantized_margin)[mask]),
                "semantic_direction_error": stats(semantic_direction_error[mask]),
            }
        row["probe_weight_norm"] = float(np.linalg.norm(probe["weights"][:-1]))
        result[name] = row
    return result


def stage_error_analysis(
    checkpoint: Path,
    native_test: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, float | int]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    rq = build_contact_rq(stages=int(payload["stages"]), codes=int(payload["codes_per_stage"]))
    rq.load_state_dict(payload["state_dict"], strict=True)
    rq.eval().requires_grad_(False).to(device)
    squared = np.zeros(len(rq.layers), dtype=np.float64)
    count = 0
    with torch.inference_mode():
        for start in range(0, len(native_test), batch_size):
            stop = min(start + batch_size, len(native_test))
            z_c = torch.from_numpy(np.array(native_test[start:stop], copy=True)).to(device)
            residual = z_c
            cumulative = torch.zeros_like(z_c)
            for stage, layer in enumerate(rq.layers):
                code, _, _ = layer(residual)
                cumulative += code
                residual = residual - code
                squared[stage] += float((cumulative - z_c).square().sum().item())
            count += z_c.numel()
    return [{"stage": stage + 1, "cumulative_mse": float(value / count)} for stage, value in enumerate(squared)]


def decoder_sensitivity(
    decoder: torch.nn.Module,
    native: np.ndarray,
    quantized: np.ndarray,
    current: np.ndarray,
    *,
    device: torch.device,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    count = min(int(samples), len(native))
    if count < 1:
        return {"status": "NOT_AVAILABLE", "reason": "no samples requested"}
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(native), size=count, replace=False))
    z_c = torch.from_numpy(np.array(native[selected], copy=True)).to(device).requires_grad_(True)
    h_t = torch.from_numpy(np.array(current[selected], copy=True)).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    prediction = decoder(z_c, h_t)
    direction = torch.randn(prediction.shape, generator=generator, device=device)
    direction = direction / torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(1e-8)
    gradient = torch.autograd.grad((prediction * direction).sum(), z_c)[0]
    error = torch.from_numpy(np.array(quantized[selected] - native[selected], copy=True)).to(device)
    gradient_flat = gradient.flatten(1)
    error_flat = error.flatten(1)
    projection = torch.abs((gradient_flat * error_flat).sum(dim=1))
    cosine = projection / (
        torch.linalg.vector_norm(gradient_flat, dim=1)
        * torch.linalg.vector_norm(error_flat, dim=1)
    ).clamp_min(1e-8)
    return {
        "status": "AVAILABLE_HUTCHINSON_DIRECTIONAL_ESTIMATE",
        "samples": count,
        "gradient_norm": stats(torch.linalg.vector_norm(gradient_flat, dim=1).detach().cpu().numpy()),
        "absolute_error_projection": stats(projection.detach().cpu().numpy()),
        "absolute_projection_cosine": stats(cosine.detach().cpu().numpy()),
        "interpretation": "one deterministic output-space Hutchinson direction per sample; not a full Jacobian",
    }


def main() -> int:
    args = parse_args()
    device, physical_gpu = verify_gpu()
    spec = json.loads(args.spec.read_text())
    set_seed(int(spec["seed"]))
    runtime = load_runtime(spec_path=args.spec, source_root=args.source_root, device=device)
    train = runtime["codes"]["train"]
    test = runtime["codes"]["test"]
    quantized = np.load(runtime["paths"]["r0_cache"] / "private_test_quantized.npy", mmap_mode="r")
    train_quantized = np.load(runtime["paths"]["r0_cache"] / "private_train_quantized.npy", mmap_mode="r")
    if quantized.shape != test.shape or train_quantized.shape != train.shape:
        raise RuntimeError("Q_BASE_2 cache shape mismatch")
    error = np.asarray(test) - np.asarray(quantized)
    mean, eigenvalues, components = fit_flat_pca(train, device, args.batch_size)
    bands = spec["q0"]["variance_bands"]
    variance_total = float(eigenvalues.sum())
    band_results = {}
    for name in ("high", "mid", "low"):
        start, stop = map(int, bands[name])
        basis = components[:, start:stop]
        train_scores = project_to_cache(
            train, mean, basis, args.cache_dir / f"train_{name}_pc.npy", device, args.batch_size
        )
        test_scores = project_to_cache(
            test, mean, basis, args.cache_dir / f"test_{name}_pc.npy", device, args.batch_size
        )
        error_scores = error.reshape(len(error), -1) @ basis
        dynamic = np.asarray(runtime["arrays"]["test"]["dynamic"], dtype=bool)
        band_results[name] = {
            "rank": [start, stop],
            "explained_variance_fraction": float(eigenvalues[start:stop].sum() / variance_total),
            "probe_information": probe_bundle(
                train_scores,
                test_scores,
                runtime["arrays"]["train"],
                runtime["arrays"]["test"],
                device=device,
                batch_size=args.batch_size,
            ),
            "quantization_error_energy": float(np.square(error_scores).mean()),
            "dynamic_error_energy": float(np.square(error_scores[dynamic]).mean()),
            "static_error_energy": float(np.square(error_scores[~dynamic]).mean()),
        }
    per_query = [
        {
            "query": query,
            "mse": float(np.square(error[:, query]).mean()),
            "dynamic_mse": float(
                np.square(error[np.asarray(runtime["arrays"]["test"]["dynamic"], dtype=bool), query]).mean()
            ),
        }
        for query in range(8)
    ]
    probe_direction = probe_direction_analysis(
        train,
        test,
        quantized,
        runtime["arrays"]["train"],
        runtime["arrays"]["test"],
        device,
        args.batch_size,
    )
    transition = np.asarray(runtime["arrays"]["test"]["contact_transition"])
    boundary = np.isin(transition, [1, 3])
    sample_error = np.square(error).mean(axis=(1, 2))
    boundary_result = {
        "class_order": ["free_to_free", "free_to_contact", "contact_to_contact", "contact_to_free"],
        "per_class": {
            str(index): {"windows": int(np.sum(transition == index)), **stats(sample_error[transition == index])}
            for index in range(4)
        },
        "boundary_mse": float(sample_error[boundary].mean()),
        "non_boundary_mse": float(sample_error[~boundary].mean()),
        "boundary_to_non_boundary_ratio": float(
            sample_error[boundary].mean() / max(sample_error[~boundary].mean(), 1e-12)
        ),
    }
    stage_errors = stage_error_analysis(
        runtime["baselines"]["Q_BASE_2"]["checkpoint"], test, device, args.batch_size
    )
    decoder_result = decoder_sensitivity(
        runtime["s2"].decoder,
        test,
        quantized,
        runtime["arrays"]["test"]["current"],
        device=device,
        samples=args.jacobian_samples,
        seed=int(spec["seed"]),
    )
    low_contact = band_results["low"]["probe_information"]["contact_transition"]["macro_f1"]
    high_contact = band_results["high"]["probe_information"]["contact_transition"]["macro_f1"]
    boundary_ratio = boundary_result["boundary_to_non_boundary_ratio"]
    direction_loss = probe_direction["contact_transition"]["subsets"]["boundary"][
        "semantic_direction_error"
    ]["mean"]
    thresholds = spec["q0"]["decision_thresholds"]
    low_variance_loss = low_contact >= high_contact * float(
        thresholds["low_vs_high_contact_f1_ratio"]
    )
    boundary_underweighted = boundary_ratio >= float(
        thresholds["boundary_to_non_boundary_mse_ratio"]
    )
    objective_mismatch = direction_loss > float(
        thresholds["boundary_probe_direction_error"]
    )
    flags = [low_variance_loss, boundary_underweighted, objective_mismatch]
    if sum(flags) >= 2:
        diagnosis = "MIXED"
    elif low_variance_loss:
        diagnosis = "LOW_VARIANCE_SEMANTIC_LOSS"
    elif boundary_underweighted:
        diagnosis = "DYNAMIC_BOUNDARY_UNDERWEIGHTED"
    elif objective_mismatch:
        diagnosis = "EUCLIDEAN_OBJECTIVE_MISMATCH"
    else:
        diagnosis = "NO_CLEAR_DIAGNOSIS"
    output = {
        "schema": "tactile3d-unit.s3-2-q-q0-diagnosis.v1",
        "status": "COMPLETE",
        "training_performed": False,
        "test_used_for_model_selection": False,
        "physical_gpu": physical_gpu,
        "logical_device": str(device),
        "frozen_identity": runtime["identity"],
        "baseline_integrity": {
            "Q_BASE_2": runtime["baselines"]["Q_BASE_2"]["checkpoint_sha256"],
            "Q_BASE_3": runtime["baselines"]["Q_BASE_3"]["checkpoint_sha256"],
        },
        "global_quantization_error": stats(sample_error),
        "probe_direction": probe_direction,
        "variance_bands": {
            "rule": bands["rule"],
            "train_only_fit": True,
            "eigenvalues": eigenvalues,
            "bands": band_results,
        },
        "dynamic_boundary": boundary_result,
        "per_query_error": per_query,
        "per_stage_error": stage_errors,
        "decoder_sensitivity": decoder_result,
        "diagnostic_flags": {
            "low_variance_semantic_loss": low_variance_loss,
            "dynamic_boundary_underweighted": boundary_underweighted,
            "euclidean_objective_mismatch": objective_mismatch,
        },
        "decision_thresholds": thresholds,
        "diagnosis": diagnosis,
    }
    write_json(args.output, output)
    print(json.dumps({"status": "COMPLETE", "diagnosis": diagnosis, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
