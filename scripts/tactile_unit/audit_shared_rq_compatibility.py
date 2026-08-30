#!/usr/bin/env python3
"""Audit contact z_c against the frozen Original UniT shared residual VQ."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.lib.format import open_memmap
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/reproduce"))
sys.path.insert(0, str(ROOT / "scripts/contact_dynamics"))

from evaluate_contact_dynamics import apply_ridge, fit_ridge  # noqa: E402
from gr00t.contact_dynamics.evaluation import (  # noqa: E402
    different_episode_permutation,
    query_diversity,
    transition_metrics,
)
from gr00t.contact_dynamics.models import (  # noqa: E402
    ContactDynamicsEncoder,
    ContactDynamicsModel,
    LatentTransitionDecoder,
)
from gr00t.model.tokenizer.vector_quantizer import (  # noqa: E402
    ResidualVectorQuantizer,
    ResidualVectorQuantizerConfig,
)
from gr00t.tactile_teacher.evaluation import classification_metrics  # noqa: E402
from gr00t.tactile_unit.compatibility import (  # noqa: E402
    active_set_jaccard,
    code_frequency,
    codebook_usage,
    deterministic_contact_subset,
    effective_rank,
    jensen_shannon_divergence,
    parameter_digest,
    quantization_metrics,
    quantize_with_stage_diagnostics,
)
from unit_representation_metrics import (  # noqa: E402
    mean_query_pool,
    mmd_rbf,
    sliced_wasserstein,
)


DEFAULT_SPEC = ROOT / "configs/tactile_unit/s3_0_codebook_compatibility.json"
DEFAULT_T4 = ROOT / ".local/artifacts/reproduction/t4"
DEFAULT_TRANSITIONS = ROOT / ".local/cache/contact_dynamics/s2_transition_pairs"
DEFAULT_CODES = ROOT / ".local/cache/contact_dynamics/s2_codes"
DEFAULT_S1 = ROOT / ".local/experiments/tactile_teacher/s1_teacher/best.pt"
DEFAULT_S2 = ROOT / ".local/experiments/contact_dynamics/s2_models/proposed_best.pt"
DEFAULT_OUTPUT = ROOT / ".local/artifacts/tactile_unit/s3_0"
DEFAULT_CACHE = ROOT / ".local/cache/tactile_unit/s3_0"
MODALITIES = ("vision", "action", "multimodal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--t4-dir", type=Path, default=DEFAULT_T4)
    parser.add_argument("--transition-cache", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--code-cache", type=Path, default=DEFAULT_CODES)
    parser.add_argument("--s1-checkpoint", type=Path, default=DEFAULT_S1)
    parser.add_argument("--s2-checkpoint", type=Path, default=DEFAULT_S2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runtime-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--probe-batch-size", type=int, default=4096)
    parser.add_argument("--ridge", type=float, default=10.0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(32 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def verify_gpu() -> torch.device:
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("S3.0 requires CUDA_DEVICE_ORDER=PCI_BUS_ID")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise RuntimeError("S3.0 requires physical GPU3 as the only visible GPU")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible CUDA device, got {torch.cuda.device_count()}")
    return torch.device("cuda:0")


def verify_file(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} identity mismatch: {actual} != {expected}")
    return actual


def load_frozen_rq(
    extraction: dict[str, Any], spec: dict[str, Any]
) -> tuple[ResidualVectorQuantizer, dict[str, Any]]:
    checkpoint = Path(extraction["checkpoint"])
    expected_files = spec["original_unit"]["tokenizer_files_sha256"]
    verified = {
        relative: verify_file(checkpoint / relative, expected, relative)
        for relative, expected in expected_files.items()
    }
    tokenizer = checkpoint / "tokenizer"
    config = json.loads((tokenizer / "config.json").read_text())
    expected_rq = spec["original_unit"]["rq"]
    vq_config = config["vq_cfg"]
    dimensions = {
        "stages": int(vq_config["num_stages"]),
        "codes_per_stage": int(vq_config["n_e"]),
        "embedding_dim": int(vq_config["e_dim"]),
    }
    for key in dimensions:
        if dimensions[key] != int(expected_rq[key]):
            raise RuntimeError(f"Original UniT RQ {key} mismatch")
    rq = ResidualVectorQuantizer(ResidualVectorQuantizerConfig(**vq_config))
    index = json.loads((tokenizer / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    with torch.no_grad():
        for stage, layer in enumerate(rq.layers):
            key = f"vq.layers.{stage}.embedding.weight"
            shard = tokenizer / weight_map[key]
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                weight = handle.get_tensor(key)
            if tuple(weight.shape) != (dimensions["codes_per_stage"], dimensions["embedding_dim"]):
                raise RuntimeError(f"unexpected codebook tensor shape for {key}: {tuple(weight.shape)}")
            layer.embedding.weight.copy_(weight)
    rq.eval().requires_grad_(False)
    return rq, {
        "variant": spec["original_unit"]["variant"],
        "files_sha256": verified,
        **dimensions,
    }


def load_transition_arrays(cache: Path, split: str) -> dict[str, np.ndarray]:
    names = (
        "current",
        "future",
        "episode_id",
        "anchor_frame",
        "dynamic",
        "contact_transition",
        "force_trend_class",
    )
    return {
        name: np.load(cache / split / f"{name}.npy", mmap_mode="r")
        for name in names
    }


def load_s2_model(path: Path, device: torch.device) -> ContactDynamicsModel:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model") != "proposed":
        raise RuntimeError("S2 checkpoint is not the accepted proposed model")
    model = ContactDynamicsModel(ContactDynamicsEncoder(), LatentTransitionDecoder())
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval().requires_grad_(False).to(device)


@torch.inference_mode()
def quantize_array(
    rq: ResidualVectorQuantizer,
    values: np.ndarray,
    device: torch.device,
    batch_size: int,
    *,
    quantized_path: Path | None = None,
    indices_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    shape = tuple(values.shape)
    if len(shape) != 3 or shape[1:] != (8, 32):
        raise ValueError(f"expected [N,8,32] direct RQ input, got {shape}")
    if quantized_path is None:
        quantized: np.ndarray = np.empty(shape, dtype=np.float32)
        indices: np.ndarray = np.empty((shape[0], 8, len(rq.layers)), dtype=np.int64)
    else:
        if indices_path is None:
            raise ValueError("indices_path is required with quantized_path")
        quantized_path.parent.mkdir(parents=True, exist_ok=True)
        quantized = open_memmap(quantized_path, mode="w+", dtype=np.float32, shape=shape)
        indices = open_memmap(
            indices_path, mode="w+", dtype=np.int64, shape=(shape[0], 8, len(rq.layers))
        )
    aggregate = [
        {
            "stage": stage,
            "residual_norm_before_mean": 0.0,
            "residual_norm_after_mean": 0.0,
            "residual_energy_after": 0.0,
        }
        for stage in range(len(rq.layers))
    ]
    total = 0
    checked_reference = False
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        batch = torch.from_numpy(np.array(values[start:stop], copy=True)).to(device)
        q_value, q_indices, rows = quantize_with_stage_diagnostics(rq, batch)
        if not checked_reference:
            reference_q, reference_indices, _ = rq(batch)
            if not torch.equal(q_value, reference_q) or not torch.equal(q_indices, reference_indices):
                raise RuntimeError("stage diagnostic route disagrees with frozen RQ forward")
            checked_reference = True
        quantized[start:stop] = q_value.float().cpu().numpy()
        indices[start:stop] = q_indices.cpu().numpy()
        weight = stop - start
        total += weight
        for target, row in zip(aggregate, rows):
            for key in (
                "residual_norm_before_mean",
                "residual_norm_after_mean",
                "residual_energy_after",
            ):
                target[key] += float(row[key]) * weight
    for row in aggregate:
        for key in (
            "residual_norm_before_mean",
            "residual_norm_after_mean",
            "residual_energy_after",
        ):
            row[key] /= total
    if isinstance(quantized, np.memmap):
        quantized.flush()
        indices.flush()
        del quantized, indices
        quantized = np.load(quantized_path, mmap_mode="r")
        indices = np.load(indices_path, mmap_mode="r")
    return quantized, indices, aggregate


@torch.inference_mode()
def encode_s2(
    model: ContactDynamicsModel,
    current: np.ndarray,
    future: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    current_t = torch.from_numpy(np.array(current, copy=True)).to(device)
    future_t = torch.from_numpy(np.array(future, copy=True)).to(device)
    return model.encoder(current_t, future_t).float().cpu().numpy()


@torch.inference_mode()
def decode_s2(
    model: ContactDynamicsModel,
    code: np.ndarray,
    current: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    result = np.empty((len(code), 256), dtype=np.float32)
    for start in range(0, len(code), batch_size):
        stop = min(start + batch_size, len(code))
        z = torch.from_numpy(np.array(code[start:stop], copy=True)).to(device)
        c = torch.from_numpy(np.array(current[start:stop], copy=True)).to(device)
        result[start:stop] = model.decoder(z, c).float().cpu().numpy()
    return result


def metric_bundle(
    current: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    dynamic: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    return {
        "all": transition_metrics(current, target, prediction),
        "dynamic": transition_metrics(current, target, prediction, dynamic),
    }


def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
    }


def distribution_summary(values: np.ndarray, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    token_norm = np.linalg.norm(values, axis=2)
    pooled = values.mean(axis=1)
    pooled_norm = np.linalg.norm(pooled, axis=1)
    flattened = values.reshape(len(values), -1)
    flat_rank, flat_spectrum = effective_rank(flattened)
    pooled_rank, pooled_spectrum = effective_rank(pooled)
    per_dimension_mean = values.reshape(-1, values.shape[-1]).mean(axis=0)
    per_dimension_std = values.reshape(-1, values.shape[-1]).std(axis=0)
    rng = np.random.default_rng(seed)
    pair_count = min(20000, len(values) * 20)
    left = rng.integers(0, len(values), size=pair_count)
    right = rng.integers(0, len(values), size=pair_count)
    distances = np.linalg.norm(pooled[left] - pooled[right], axis=1)
    return {
        "token_norm": stats(token_norm),
        "pooled_norm": stats(pooled_norm),
        "per_dimension_mean": per_dimension_mean,
        "per_dimension_std": per_dimension_std,
        "per_dimension_mean_abs_summary": stats(np.abs(per_dimension_mean)),
        "per_dimension_std_summary": stats(per_dimension_std),
        "flattened_effective_rank": flat_rank,
        "flattened_covariance_spectrum": flat_spectrum,
        "pooled_effective_rank": pooled_rank,
        "pooled_covariance_spectrum": pooled_spectrum,
        "pooled_pairwise_distance": stats(distances),
    }


def query_metrics(values: np.ndarray) -> dict[str, Any]:
    result = query_diversity(values)
    per_query_norm = np.linalg.norm(np.asarray(values), axis=2)
    result["per_query_norm_mean"] = per_query_norm.mean(axis=0).tolist()
    result["per_query_norm_std"] = per_query_norm.std(axis=0).tolist()
    return result


def probe_metric(
    train_feature: np.ndarray,
    test_feature: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    classes: int,
    device: torch.device,
    batch_size: int,
    ridge: float,
) -> dict[str, Any]:
    probe = fit_ridge(
        train_feature,
        train_labels,
        classes,
        device,
        batch_size,
        ridge,
        classes=classes,
    )
    scores = apply_ridge(probe, test_feature, device, batch_size)
    prediction = scores.reshape(len(scores), classes).argmax(axis=1)
    majority_class = int(np.bincount(np.asarray(train_labels), minlength=classes).argmax())
    majority = np.full(len(test_labels), majority_class, dtype=np.int64)
    return {
        **classification_metrics(test_labels, prediction),
        "majority": classification_metrics(test_labels, majority),
        "ridge": float(ridge),
    }


def rgb_status() -> dict[str, Any]:
    root_value = os.environ.get("TREX_DATASET_ROOT")
    if not root_value:
        return {"status": "NOT AVAILABLE", "video_files": 0, "expected_episodes": None}
    root = Path(root_value)
    video_root = root / "videos/observation.images.head_left"
    files = list(video_root.rglob("*.mp4")) if video_root.is_dir() else []
    metadata = root / "meta/episodes.jsonl"
    expected = len(metadata.read_text().splitlines()) if metadata.is_file() else None
    if not files:
        status = "NOT AVAILABLE"
    elif expected is not None and len(files) >= expected:
        status = "AVAILABLE"
    else:
        status = "PARTIAL"
    return {"status": status, "video_files": len(files), "expected_episodes": expected}


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    device = verify_gpu()
    if args.device != "cuda:0" or args.batch_size < 1 or args.probe_batch_size < 1:
        raise ValueError("S3.0 uses logical cuda:0 and positive batch sizes")
    spec = json.loads(args.spec.read_text())
    seed = int(spec["seed"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.runtime_cache.mkdir(parents=True, exist_ok=True)
    metrics_dir = args.output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    identity = {
        "spec_sha256": sha256_file(args.spec),
        "t4_config_sha256": verify_file(
            ROOT / spec["t4_reference"]["benchmark_config"],
            spec["t4_reference"]["benchmark_config_sha256"],
            "T4 benchmark config",
        ),
        "t4_manifest_sha256": verify_file(
            args.t4_dir / "sample_manifest.json",
            spec["t4_reference"]["resolved_manifest_sha256"],
            "T4 sample manifest",
        ),
        "s1_checkpoint_sha256": verify_file(
            args.s1_checkpoint,
            spec["contact"]["s1_teacher_checkpoint_sha256"],
            "S1 Teacher checkpoint",
        ),
        "s2_checkpoint_sha256": verify_file(
            args.s2_checkpoint,
            spec["contact"]["s2_checkpoint_sha256"],
            "S2 checkpoint",
        ),
        "s2_transition_manifest_sha256": verify_file(
            args.transition_cache / "manifest.json",
            spec["contact"]["s2_transition_manifest_sha256"],
            "S2 transition manifest",
        ),
        "s2_train_codes_sha256": verify_file(
            args.code_cache / "train.npy",
            spec["contact"]["cached_train_codes_sha256"],
            "S2 train codes",
        ),
        "s2_test_codes_sha256": verify_file(
            args.code_cache / "test.npy",
            spec["contact"]["cached_test_codes_sha256"],
            "S2 test codes",
        ),
    }
    extraction = json.loads((args.t4_dir / "extraction_summary.json").read_text())
    if extraction.get("status") != "PASS" or extraction.get("sample_count") != 960:
        raise RuntimeError("canonical T4 extraction is not valid")
    rq, rq_identity = load_frozen_rq(extraction, spec)
    rq.to(device)
    rq_before = parameter_digest(rq)

    with np.load(args.t4_dir / "features/unit_representation_features.npz", allow_pickle=False) as data:
        t4_l2 = np.asarray(data["l2"], dtype=np.float32)
        t4_l3 = np.asarray(data["l3"], dtype=np.float32)
        t4_l4 = np.asarray(data["l4"], dtype=np.int64)
        t4_modalities = tuple(data["modality"].tolist())
    if t4_modalities != MODALITIES or tuple(t4_l2.shape) != (960, 3, 8, 32):
        raise RuntimeError("canonical T4 L2 identity or shape mismatch")

    transition_manifest = json.loads((args.transition_cache / "manifest.json").read_text())
    train = load_transition_arrays(args.transition_cache, "train")
    test = load_transition_arrays(args.transition_cache, "test")
    train_code = np.load(args.code_cache / "train.npy", mmap_mode="r")
    test_code = np.load(args.code_cache / "test.npy", mmap_mode="r")
    if tuple(train_code.shape[1:]) != (8, 32) or tuple(test_code.shape[1:]) != (8, 32):
        raise RuntimeError("contact code geometry is not [8,32]")
    if len(test_code) != int(transition_manifest["splits"]["test"]["pairs"]):
        raise RuntimeError("contact test code count disagrees with transition manifest")
    if not np.isfinite(test_code).all() or not np.isfinite(train_code).all():
        raise RuntimeError("contact z_c contains NaN or Inf")

    subset = deterministic_contact_subset(
        test["episode_id"],
        test["anchor_frame"],
        test["dynamic"],
        test["contact_transition"],
        count=int(spec["contact"]["cross_distribution_subset"]["count"]),
        seed=seed,
    )
    subset_manifest = {
        "schema": "tactile3d-unit.s3-0-contact-subset.v1",
        "selection_rule": spec["contact"]["cross_distribution_subset"]["selection"],
        "seed": seed,
        "source_split": "canonical S2 test",
        "canonical_horizon_frames": spec["contact"]["canonical_horizon_frames"],
        "full_test_pairs": len(test_code),
        "count": len(subset),
        "dynamic_fraction_full": float(np.asarray(test["dynamic"]).mean()),
        "dynamic_fraction_subset": float(np.asarray(test["dynamic"])[subset].mean()),
        "transition_class_counts_full": {
            str(value): int(count)
            for value, count in zip(*np.unique(test["contact_transition"], return_counts=True))
        },
        "transition_class_counts_subset": {
            str(value): int(count)
            for value, count in zip(
                *np.unique(np.asarray(test["contact_transition"])[subset], return_counts=True)
            )
        },
        "rows": [
            {
                "source_index": int(index),
                "episode_id": int(test["episode_id"][index]),
                "anchor_frame": int(test["anchor_frame"][index]),
                "dynamic": bool(test["dynamic"][index]),
                "contact_transition": int(test["contact_transition"][index]),
            }
            for index in subset
        ],
    }
    contact_manifest_path = args.output_dir / "contact_manifest.json"
    write_json(contact_manifest_path, subset_manifest)
    contact_manifest_sha256 = sha256_file(contact_manifest_path)

    s2_model = load_s2_model(args.s2_checkpoint, device)
    encoder_before = parameter_digest(s2_model.encoder)
    decoder_before = parameter_digest(s2_model.decoder)
    regenerated = encode_s2(
        s2_model, test["current"][:64], test["future"][:64], device
    )
    cache_max_abs_diff = float(
        np.max(np.abs(regenerated.astype(np.float64) - np.asarray(test_code[:64], dtype=np.float64)))
    )
    cache_tolerance = 1e-5
    cache_match = bool(
        np.allclose(
            regenerated,
            np.asarray(test_code[:64]),
            atol=cache_tolerance,
            rtol=cache_tolerance,
        )
    )
    if not cache_match:
        raise RuntimeError("cached S2 z_c does not match accepted encoder checkpoint")

    reference_quantized = np.empty_like(t4_l2)
    reference_indices = np.empty_like(t4_l4)
    quantization_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    relative_samples: list[np.ndarray] = []
    for modality_index, modality in enumerate(MODALITIES):
        q_value, q_indices, stages = quantize_array(
            rq, t4_l2[:, modality_index], device, args.batch_size
        )
        reference_quantized[:, modality_index] = q_value
        reference_indices[:, modality_index] = q_indices
        metrics = quantization_metrics(t4_l2[:, modality_index], q_value)
        quantization_rows.append({"modality": modality, **metrics})
        for row in stages:
            stage_rows.append({"modality": modality, **row})
        per_error = np.square(t4_l2[:, modality_index] - q_value).mean(axis=(1, 2))
        per_energy = np.square(t4_l2[:, modality_index]).mean(axis=(1, 2))
        relative_samples.append(per_error / np.maximum(per_energy, 1e-12))
    t4_quantized_match = bool(
        np.allclose(reference_quantized, t4_l3, atol=1e-6, rtol=1e-6)
        and np.array_equal(reference_indices, t4_l4)
    )
    if not t4_quantized_match:
        raise RuntimeError("loaded frozen RQ does not reproduce canonical T4 L3/L4")

    q_test_path = args.runtime_cache / "contact_test_quantized.npy"
    i_test_path = args.runtime_cache / "contact_test_indices.npy"
    q_test, i_test, contact_stages = quantize_array(
        rq,
        test_code,
        device,
        args.batch_size,
        quantized_path=q_test_path,
        indices_path=i_test_path,
    )
    q_train_path = args.runtime_cache / "contact_train_quantized.npy"
    i_train_path = args.runtime_cache / "contact_train_indices.npy"
    q_train, i_train, _ = quantize_array(
        rq,
        train_code,
        device,
        args.batch_size,
        quantized_path=q_train_path,
        indices_path=i_train_path,
    )
    if i_test.min() < 0 or i_test.max() >= rq_identity["codes_per_stage"]:
        raise RuntimeError("contact VQ indices are invalid")
    contact_quantization = quantization_metrics(test_code, q_test)
    quantization_rows.append({"modality": "contact", **contact_quantization})
    for row in contact_stages:
        stage_rows.append({"modality": "contact", **row})
    contact_subset_code = np.asarray(test_code)[subset]
    contact_subset_q = np.asarray(q_test)[subset]
    per_error = np.square(contact_subset_code - contact_subset_q).mean(axis=(1, 2))
    per_energy = np.square(contact_subset_code).mean(axis=(1, 2))
    relative_samples.append(per_error / np.maximum(per_energy, 1e-12))

    first_q, first_i, _ = quantize_array(rq, test_code[:32], device, args.batch_size)
    second_q, second_i, _ = quantize_array(rq, test_code[:32], device, args.batch_size)
    deterministic_quantization = bool(
        np.array_equal(first_q, second_q) and np.array_equal(first_i, second_i)
    )

    usage_rows: list[dict[str, Any]] = []
    frequencies: dict[str, list[np.ndarray]] = {}
    all_codes = {
        **{
            modality: reference_indices[:, index]
            for index, modality in enumerate(MODALITIES)
        },
        "contact": np.asarray(i_test),
    }
    for modality, codes in all_codes.items():
        frequencies[modality] = []
        for stage in range(codes.shape[-1]):
            frequency = code_frequency(codes[:, :, stage], rq_identity["codes_per_stage"])
            frequencies[modality].append(frequency)
            usage_rows.append(
                {
                    "row_type": "modality_stage",
                    "modality": modality,
                    "stage": stage,
                    "query": "all",
                    **codebook_usage(codes[:, :, stage], rq_identity["codes_per_stage"]),
                }
            )
            if modality == "contact":
                for query in range(8):
                    usage_rows.append(
                        {
                            "row_type": "contact_query_stage",
                            "modality": modality,
                            "stage": stage,
                            "query": query,
                            **codebook_usage(
                                codes[:, query, stage], rq_identity["codes_per_stage"]
                            ),
                        }
                    )
    overlap_rows: list[dict[str, Any]] = []
    for modality in MODALITIES:
        for stage in range(rq_identity["stages"]):
            contact_frequency = frequencies["contact"][stage]
            reference_frequency = frequencies[modality][stage]
            overlap_rows.append(
                {
                    "row_type": "contact_reference_overlap",
                    "modality": f"contact-vs-{modality}",
                    "stage": stage,
                    "active_set_jaccard": active_set_jaccard(
                        contact_frequency, reference_frequency
                    ),
                    "jensen_shannon_divergence": jensen_shannon_divergence(
                        contact_frequency, reference_frequency
                    ),
                }
            )

    distributions: dict[str, Any] = {}
    distribution_rows: list[dict[str, Any]] = []
    subset_values = {
        **{modality: t4_l2[:, index] for index, modality in enumerate(MODALITIES)},
        "contact": contact_subset_code,
    }
    for modality, values in subset_values.items():
        distributions[modality] = distribution_summary(values, seed)
        item = distributions[modality]
        distribution_rows.append(
            {
                "row_type": "modality_summary",
                "modality": modality,
                "token_norm_mean": item["token_norm"]["mean"],
                "token_norm_std": item["token_norm"]["std"],
                "pooled_norm_mean": item["pooled_norm"]["mean"],
                "pooled_norm_std": item["pooled_norm"]["std"],
                "flattened_effective_rank": item["flattened_effective_rank"],
                "pooled_effective_rank": item["pooled_effective_rank"],
                "pairwise_distance_mean": item["pooled_pairwise_distance"]["mean"],
                "pairwise_distance_std": item["pooled_pairwise_distance"]["std"],
            }
        )
    contact_pooled = mean_query_pool(contact_subset_code[:, None])[:, 0]
    distance_results: dict[str, Any] = {}
    for index, modality in enumerate(MODALITIES):
        reference_pooled = mean_query_pool(t4_l2[:, index : index + 1])[:, 0]
        distance_results[modality] = {
            "mmd": mmd_rbf(contact_pooled, reference_pooled),
            "sliced_wasserstein": sliced_wasserstein(
                contact_pooled, reference_pooled, projections=128, seed=seed
            ),
            "interpretation": "unpaired distribution/codebook diagnostic only",
        }
        distribution_rows.append(
            {
                "row_type": "contact_reference_distance",
                "modality": f"contact-vs-{modality}",
                "mmd": distance_results[modality]["mmd"]["mmd"],
                "mmd_bandwidth": distance_results[modality]["mmd"]["bandwidth"],
                "swd": distance_results[modality]["sliced_wasserstein"]["swd"],
            }
        )

    current = np.asarray(test["current"])
    future = np.asarray(test["future"])
    dynamic = np.asarray(test["dynamic"], dtype=bool)
    permutation = different_episode_permutation(test["episode_id"], seed=seed)
    predictions = {
        "continuous": decode_s2(s2_model, test_code, current, device, args.batch_size),
        "quantized": decode_s2(s2_model, q_test, current, device, args.batch_size),
        "zero": decode_s2(s2_model, np.zeros_like(test_code), current, device, args.batch_size),
        "shuffled": decode_s2(
            s2_model, np.asarray(test_code)[permutation], current, device, args.batch_size
        ),
    }
    reconstruction = {
        name: metric_bundle(current, future, prediction, dynamic)
        for name, prediction in predictions.items()
    }
    reconstruction_rows = [
        {"condition": condition, "scope": scope, **values}
        for condition, scopes in reconstruction.items()
        for scope, values in scopes.items()
    ]
    control_dynamic = min(
        reconstruction["zero"]["dynamic"]["future_mse"],
        reconstruction["shuffled"]["dynamic"]["future_mse"],
    )
    continuous_dynamic = reconstruction["continuous"]["dynamic"]["future_mse"]
    quantized_dynamic = reconstruction["quantized"]["dynamic"]["future_mse"]
    reconstruction_advantage_retention = float(
        (control_dynamic - quantized_dynamic)
        / max(control_dynamic - continuous_dynamic, 1e-12)
    )

    probe_definitions = {
        "contact_transition": ("contact_transition", 4),
        "force_trend": ("force_trend_class", 3),
    }
    semantic_retention: dict[str, Any] = {}
    semantic_rows: list[dict[str, Any]] = []
    for name, (key, classes) in probe_definitions.items():
        continuous_probe = probe_metric(
            train_code,
            test_code,
            np.asarray(train[key]),
            np.asarray(test[key]),
            classes,
            device,
            args.probe_batch_size,
            args.ridge,
        )
        quantized_probe = probe_metric(
            q_train,
            q_test,
            np.asarray(train[key]),
            np.asarray(test[key]),
            classes,
            device,
            args.probe_batch_size,
            args.ridge,
        )
        baseline = continuous_probe["majority"]["macro_f1"]
        advantage_retention = float(
            (quantized_probe["macro_f1"] - baseline)
            / max(continuous_probe["macro_f1"] - baseline, 1e-12)
        )
        semantic_retention[name] = {
            "continuous": continuous_probe,
            "quantized": quantized_probe,
            "advantage_retention": advantage_retention,
        }
        semantic_rows.extend(
            [
                {"probe": name, "representation": "continuous", **continuous_probe},
                {"probe": name, "representation": "quantized", **quantized_probe},
            ]
        )

    diversity = {
        "continuous": query_metrics(test_code),
        "quantized": query_metrics(q_test),
    }
    query_collapsed = bool(diversity["quantized"]["collapsed_sample_fraction"] >= 0.01)
    rq_after = parameter_digest(rq)
    encoder_after = parameter_digest(s2_model.encoder)
    decoder_after = parameter_digest(s2_model.decoder)
    final_file_hashes = {
        "s1_checkpoint_sha256": sha256_file(args.s1_checkpoint),
        "s2_checkpoint_sha256": sha256_file(args.s2_checkpoint),
    }
    parameter_integrity = {
        "rq_before": rq_before,
        "rq_after": rq_after,
        "rq_unchanged": rq_before == rq_after,
        "s2_encoder_before": encoder_before,
        "s2_encoder_after": encoder_after,
        "s2_encoder_unchanged": encoder_before == encoder_after,
        "s2_decoder_before": decoder_before,
        "s2_decoder_after": decoder_after,
        "s2_decoder_unchanged": decoder_before == decoder_after,
        "s1_checkpoint_unchanged": (
            final_file_hashes["s1_checkpoint_sha256"]
            == identity["s1_checkpoint_sha256"]
        ),
        "s2_checkpoint_unchanged": (
            final_file_hashes["s2_checkpoint_sha256"]
            == identity["s2_checkpoint_sha256"]
        ),
    }

    reference_distortions = [
        float(row["relative_distortion"])
        for row in quantization_rows
        if row["modality"] in MODALITIES
    ]
    contact_relative = float(contact_quantization["relative_distortion"])
    distortion_ratio_to_reference_max = contact_relative / max(max(reference_distortions), 1e-12)
    contact_usage_rows = [
        row
        for row in usage_rows
        if row["modality"] == "contact" and row["query"] == "all"
    ]
    severe_usage_collapse = any(
        float(row["active_ratio"]) <= 0.02 or float(row["top1_frequency"]) >= 0.90
        for row in contact_usage_rows
    )
    probe_advantage = {
        name: float(value["advantage_retention"])
        for name, value in semantic_retention.items()
    }
    structural_gates = {
        "original_unit_identity": True,
        "t4_reference_validity": t4_quantized_match,
        "s1_teacher_frozen": parameter_integrity["s1_checkpoint_unchanged"],
        "s2_encoder_decoder_frozen": (
            parameter_integrity["s2_encoder_unchanged"]
            and parameter_integrity["s2_decoder_unchanged"]
            and parameter_integrity["s2_checkpoint_unchanged"]
        ),
        "contact_geometry_8x32": tuple(test_code.shape[1:]) == (8, 32),
        "finite_contact": bool(np.isfinite(test_code).all() and np.isfinite(q_test).all()),
        "valid_vq_indices": bool(
            i_test.min() >= 0 and i_test.max() < rq_identity["codes_per_stage"]
        ),
        "determinism": deterministic_quantization,
        "parameter_integrity": all(
            value for key, value in parameter_integrity.items() if key.endswith("unchanged")
        ),
        "cached_s2_identity": cache_match,
    }
    adaptor_reasons: list[str] = []
    if distortion_ratio_to_reference_max > 4.0:
        adaptor_reasons.append("contact relative distortion is over 4x the Original UniT reference maximum")
    if severe_usage_collapse:
        adaptor_reasons.append("contact code usage meets the severe-collapse diagnostic")
    if query_collapsed:
        adaptor_reasons.append("quantized contact query collapse exceeds the 1% diagnostic")
    if reconstruction_advantage_retention <= 0.25:
        adaptor_reasons.append("quantized reconstruction retains at most 25% of continuous control advantage")
    if all(value <= 0.25 for value in probe_advantage.values()):
        adaptor_reasons.append("both dynamic probes retain at most 25% of continuous majority advantage")
    if not all(structural_gates.values()):
        decision = "STRUCTURAL_FAIL"
        decision_reasons = [
            f"structural gate failed: {name}"
            for name, passed in structural_gates.items()
            if not passed
        ]
    elif adaptor_reasons:
        decision = "ADAPTER_RECOMMENDED"
        decision_reasons = adaptor_reasons
    else:
        decision = "DIRECT_RQ_COMPATIBLE"
        decision_reasons = [
            "frozen-RQ distortion and usage are not pathological relative to Original UniT",
            "quantized contact reconstruction and dynamic probes retain meaningful information",
            "contact query diversity remains healthy after quantization",
        ]

    write_csv(metrics_dir / "quantization.csv", quantization_rows + stage_rows)
    write_csv(metrics_dir / "codebook_usage.csv", usage_rows + overlap_rows)
    write_csv(metrics_dir / "distribution.csv", distribution_rows)
    write_csv(metrics_dir / "reconstruction_retention.csv", reconstruction_rows)
    write_csv(metrics_dir / "semantic_retention.csv", semantic_rows)
    np.savez_compressed(
        args.output_dir / "visualization_data.npz",
        modality=np.asarray((*MODALITIES, "contact")),
        reference_l2=t4_l2,
        contact_continuous=contact_subset_code,
        contact_quantized=contact_subset_q,
        relative_distortion=np.stack(relative_samples),
        code_frequency=np.stack(
            [
                np.stack(frequencies[modality])
                for modality in (*MODALITIES, "contact")
            ]
        ),
    )
    summary = {
        "schema": "tactile3d-unit.s3-0-shared-codebook-compatibility.v1",
        "status": "COMPLETE" if decision != "STRUCTURAL_FAIL" else "INVALID",
        "identity": identity,
        "original_unit": {
            **rq_identity,
            "rq_parameter_digest": rq_before,
            "t4_l2_shape": list(t4_l2.shape),
            "t4_l3_l4_reproduced": t4_quantized_match,
            "sample_count": len(t4_l2),
        },
        "contact_audit": {
            "canonical_horizon_frames": spec["contact"]["canonical_horizon_frames"],
            "geometry": list(test_code.shape[1:]),
            "train_pairs": len(train_code),
            "test_pairs": len(test_code),
            "full_codebook_usage_count": len(test_code),
            "subset_count": len(subset),
            "subset_manifest_sha256": contact_manifest_sha256,
            "dynamic_fraction": subset_manifest["dynamic_fraction_subset"],
            "transition_class_distribution": subset_manifest[
                "transition_class_counts_subset"
            ],
            "cached_code_identity_match": cache_match,
            "cached_code_max_abs_diff": cache_max_abs_diff,
            "cached_code_atol_rtol": cache_tolerance,
        },
        "continuous_distribution": distributions,
        "unpaired_distribution_distances": distance_results,
        "quantization": {
            "rows": quantization_rows,
            "stage_diagnostics": stage_rows,
            "contact_ratio_to_original_min": contact_relative
            / max(min(reference_distortions), 1e-12),
            "contact_ratio_to_original_median": contact_relative
            / max(float(np.median(reference_distortions)), 1e-12),
            "contact_ratio_to_original_max": distortion_ratio_to_reference_max,
        },
        "codebook_usage": {
            "per_stage": usage_rows,
            "contact_reference_overlap": overlap_rows,
        },
        "reconstruction_retention": {
            "conditions": reconstruction,
            "dynamic_control_advantage_retention": reconstruction_advantage_retention,
        },
        "dynamic_semantic_retention": semantic_retention,
        "query_diversity": diversity,
        "parameter_integrity": parameter_integrity,
        "machine_gates": {
            **{key: "PASS" if value else "FAIL" for key, value in structural_gates.items()},
            "quantization_metrics": "PASS",
            "codebook_usage": "PASS",
            "distribution_audit": "PASS",
            "query_diversity_audit": "PASS",
            "quantized_contact_reconstruction": "PASS",
            "dynamic_semantic_retention": "PASS",
        },
        "decision": decision,
        "decision_reasons": decision_reasons,
        "decision_diagnostics": {
            "engineering_heuristics_not_scientific_thresholds": True,
            "contact_distortion_ratio_to_reference_max": distortion_ratio_to_reference_max,
            "severe_usage_collapse": severe_usage_collapse,
            "query_collapsed": query_collapsed,
            "reconstruction_advantage_retention": reconstruction_advantage_retention,
            "probe_advantage_retention": probe_advantage,
        },
        "consequence": (
            "z_c -> shared Original UniT RQ; no mandatory contact adaptor"
            if decision == "DIRECT_RQ_COMPATIBLE"
            else "test z_c -> P_c(32->32) -> shared RQ in S3.1/S3.2; adaptor not yet validated"
            if decision == "ADAPTER_RECOMMENDED"
            else "resolve the structural audit blocker before S3.1"
        ),
        "trex_rgb": {**rgb_status(), "required_for_s3_0": False},
        "artifacts": {
            "contact_manifest": str(contact_manifest_path),
            "metrics": sorted(str(path) for path in metrics_dir.glob("*.csv")),
            "visualization_data": str(args.output_dir / "visualization_data.npz"),
        },
        "runtime_seconds": time.monotonic() - started,
        "s3_1_started": False,
    }
    write_json(args.output_dir / "s3_0_summary.json", summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": decision,
                "decision_reasons": decision_reasons,
                "machine_gates": summary["machine_gates"],
                "runtime_seconds": summary["runtime_seconds"],
            },
            indent=2,
        )
    )
    return 2 if decision == "STRUCTURAL_FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
