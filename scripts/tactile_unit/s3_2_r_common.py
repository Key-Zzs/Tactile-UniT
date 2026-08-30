"""Shared runtime and evaluation helpers for the S3.2-R decision tree."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.lib.format import open_memmap


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tactile_unit"))

from audit_shared_rq_compatibility import (  # noqa: E402
    load_frozen_rq,
    load_s2_model,
    probe_metric,
    sha256_file,
    verify_file,
)
from contact_adapter_common import (  # noqa: E402
    DEFAULT_CODES,
    DEFAULT_S1,
    DEFAULT_S2,
    DEFAULT_T4,
    DEFAULT_TRANSITIONS,
    decode_codes,
    ensure_validation_codes,
    load_arrays,
    load_s1_teacher,
    reconstruction_bundle,
)
from gr00t.tactile_unit.compatibility import (  # noqa: E402
    parameter_digest,
    quantization_metrics,
)
from gr00t.tactile_unit.s3_2_r import (  # noqa: E402
    FrozenDigestGuard,
    assert_disjoint_splits,
    collapse_diagnostics,
    linear_cka,
    representation_structure,
)


DEFAULT_SPEC = ROOT / "configs/tactile_unit/s3_2_r_diagnostics.json"
DEFAULT_ROOT = ROOT / ".local"
DEFAULT_CACHE = DEFAULT_ROOT / "cache/tactile_unit/s3_2_r"
DEFAULT_EXPERIMENTS = DEFAULT_ROOT / "experiments/tactile_unit/s3_2_r"
DEFAULT_ARTIFACTS = DEFAULT_ROOT / "artifacts/tactile_unit/s3_2_r"
DEFAULT_LOGS = DEFAULT_ROOT / "logs/tactile_unit/s3_2_r"
S3_0_CACHE = DEFAULT_ROOT / "cache/tactile_unit/s3_0"
S3_2_CACHE = DEFAULT_ROOT / "cache/tactile_unit/s3_2"
S3_2_EXPERIMENTS = DEFAULT_ROOT / "experiments/tactile_unit/s3_2"
S3_2_ARTIFACTS = DEFAULT_ROOT / "artifacts/tactile_unit/s3_2"


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_gpu() -> torch.device:
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("S3.2-R requires CUDA_DEVICE_ORDER=PCI_BUS_ID")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise RuntimeError("S3.2-R requires physical GPU3 as the only visible GPU")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible CUDA device, got {torch.cuda.device_count()}")
    return torch.device("cuda:0")


def load_runtime(
    *,
    spec_path: Path = DEFAULT_SPEC,
    transition_cache: Path = DEFAULT_TRANSITIONS,
    code_cache: Path = DEFAULT_CODES,
    s1_checkpoint: Path = DEFAULT_S1,
    s2_checkpoint: Path = DEFAULT_S2,
    t4_dir: Path = DEFAULT_T4,
    device: torch.device,
) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text())
    identity_spec = spec["frozen_identity"]
    identity = {
        "spec_sha256": sha256_file(spec_path),
        "s1_checkpoint_sha256": verify_file(
            s1_checkpoint, identity_spec["s1_teacher_checkpoint_sha256"], "S1 checkpoint"
        ),
        "s2_checkpoint_sha256": verify_file(
            s2_checkpoint, identity_spec["s2_checkpoint_sha256"], "S2 checkpoint"
        ),
        "s2_transition_manifest_sha256": verify_file(
            transition_cache / "manifest.json",
            identity_spec["s2_transition_manifest_sha256"],
            "S2 transition manifest",
        ),
        "s2_train_codes_sha256": verify_file(
            code_cache / "train.npy",
            identity_spec["s2_train_codes_sha256"],
            "S2 train codes",
        ),
        "s2_test_codes_sha256": verify_file(
            code_cache / "test.npy",
            identity_spec["s2_test_codes_sha256"],
            "S2 test codes",
        ),
    }
    manifest = json.loads((transition_cache / "manifest.json").read_text())
    expected = spec["canonical_contact"]["pairs"]
    split_names = {"train": "train", "validation": "val", "test": "test"}
    arrays = {}
    for public_name, cache_name in split_names.items():
        if int(manifest["splits"][cache_name]["pairs"]) != int(expected[public_name]):
            raise RuntimeError(f"canonical {public_name} pair count mismatch")
        arrays[public_name] = load_arrays(transition_cache, public_name)
    assert_disjoint_splits(
        np.unique(arrays["train"]["episode_id"]),
        np.unique(arrays["validation"]["episode_id"]),
        np.unique(arrays["test"]["episode_id"]),
    )
    s3_0_spec = json.loads((ROOT / identity_spec["original_unit_spec"]).read_text())
    extraction = json.loads((t4_dir / "extraction_summary.json").read_text())
    original_rq, original_rq_identity = load_frozen_rq(extraction, s3_0_spec)
    original_rq.eval().requires_grad_(False).to(device)
    s1 = load_s1_teacher(s1_checkpoint, device)
    s2 = load_s2_model(s2_checkpoint, device)
    codes = {
        "train": np.load(code_cache / "train.npy", mmap_mode="r"),
        "test": np.load(code_cache / "test.npy", mmap_mode="r"),
    }
    codes["validation"] = ensure_validation_codes(
        s2,
        arrays["validation"],
        DEFAULT_CACHE / "canonical_validation_z_c.npy",
        device,
        2048,
    )
    for name, values in codes.items():
        if tuple(values.shape) != (int(expected[name]), 8, 32):
            raise RuntimeError(f"canonical {name} z_c geometry mismatch")
    return {
        "spec": spec,
        "identity": identity,
        "manifest": manifest,
        "arrays": arrays,
        "codes": codes,
        "s1": s1,
        "s2": s2,
        "original_rq": original_rq,
        "original_rq_identity": original_rq_identity,
    }


def frozen_guard(runtime: dict[str, Any]) -> FrozenDigestGuard:
    return FrozenDigestGuard.capture(
        s1_teacher=runtime["s1"],
        s2_encoder=runtime["s2"].encoder,
        s2_decoder=runtime["s2"].decoder,
        original_unit_rq=runtime["original_rq"],
    )


def verify_frozen(guard: FrozenDigestGuard, runtime: dict[str, Any]) -> dict[str, Any]:
    return guard.verify(
        s1_teacher=runtime["s1"],
        s2_encoder=runtime["s2"].encoder,
        s2_decoder=runtime["s2"].decoder,
        original_unit_rq=runtime["original_rq"],
    )


@torch.inference_mode()
def quantize_to_cache(
    rq: torch.nn.Module,
    values: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    quantized_path: Path,
    indices_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    shape = tuple(values.shape)
    quantized_path.parent.mkdir(parents=True, exist_ok=True)
    quantized = open_memmap(quantized_path, mode="w+", dtype=np.float32, shape=shape)
    indices = open_memmap(
        indices_path, mode="w+", dtype=np.int64, shape=(shape[0], shape[1], len(rq.layers))
    )
    residual_sums = [
        {"count": 0, "norm_before": 0.0, "norm_after": 0.0, "energy_after": 0.0}
        for _ in rq.layers
    ]
    distance_values: list[list[np.ndarray]] = [[] for _ in rq.layers]
    rq.eval()
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        batch = torch.from_numpy(np.array(values[start:stop], copy=True)).to(device)
        residual = batch
        total = torch.zeros_like(batch)
        batch_indices = []
        for stage, layer in enumerate(rq.layers):
            before = torch.linalg.vector_norm(residual, dim=-1)
            code, code_indices, _ = layer(residual)
            raw_weight = layer.embedding.weight
            weight = torch.nn.functional.normalize(raw_weight, dim=1) if getattr(
                layer.config, "l2_norm", False
            ) else raw_weight
            nearest = torch.cdist(residual.reshape(-1, residual.shape[-1]), weight).min(dim=1).values
            if sum(item.size for item in distance_values[stage]) < 200000:
                distance_values[stage].append(nearest[: min(len(nearest), 200000)].cpu().numpy())
            total += code
            residual = residual - code
            after = torch.linalg.vector_norm(residual, dim=-1)
            count = int(before.numel())
            row = residual_sums[stage]
            row["count"] += count
            row["norm_before"] += float(before.sum().item())
            row["norm_after"] += float(after.sum().item())
            row["energy_after"] += float(residual.square().sum().item())
            batch_indices.append(code_indices)
        quantized[start:stop] = total.float().cpu().numpy()
        indices[start:stop] = torch.stack(batch_indices, dim=-1).cpu().numpy()
    quantized.flush()
    indices.flush()
    del quantized, indices
    diagnostics = {"stages": []}
    for stage, row in enumerate(residual_sums):
        distances = np.concatenate(distance_values[stage]) if distance_values[stage] else np.empty(0)
        diagnostics["stages"].append(
            {
                "stage": stage,
                "residual_norm_before_mean": row["norm_before"] / row["count"],
                "residual_norm_after_mean": row["norm_after"] / row["count"],
                "residual_energy_after": row["energy_after"] / (row["count"] * shape[-1]),
                "nearest_code_distance": {
                    "mean": float(distances.mean()),
                    "p05": float(np.quantile(distances, 0.05)),
                    "median": float(np.median(distances)),
                    "p95": float(np.quantile(distances, 0.95)),
                },
            }
        )
    return (
        np.load(quantized_path, mmap_mode="r"),
        np.load(indices_path, mmap_mode="r"),
        diagnostics,
    )


def semantic_probe_bundle(
    train_feature: np.ndarray,
    test_feature: np.ndarray,
    train_arrays: dict[str, np.ndarray],
    test_arrays: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    definitions = {
        "contact_transition": ("contact_transition", 4),
        "force_trend": ("force_trend_class", 3),
    }
    return {
        name: probe_metric(
            train_feature,
            test_feature,
            np.asarray(train_arrays[key]),
            np.asarray(test_arrays[key]),
            classes,
            device,
            batch_size,
            10.0,
        )
        for name, (key, classes) in definitions.items()
    }


def fit_ridge_recovery(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 4096,
    alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
) -> dict[str, Any]:
    """Fit multi-output ridge from compressed to native z_c using train sufficient statistics."""

    x_dim = int(np.prod(train_x.shape[1:]))
    y_dim = int(np.prod(train_y.shape[1:]))
    x_sum = torch.zeros(x_dim, dtype=torch.float64, device=device)
    y_sum = torch.zeros(y_dim, dtype=torch.float64, device=device)
    count = 0
    for start in range(0, len(train_x), batch_size):
        stop = min(start + batch_size, len(train_x))
        x = torch.from_numpy(np.array(train_x[start:stop], copy=True)).to(device).double().reshape(-1, x_dim)
        y = torch.from_numpy(np.array(train_y[start:stop], copy=True)).to(device).double().reshape(-1, y_dim)
        x_sum += x.sum(dim=0)
        y_sum += y.sum(dim=0)
        count += len(x)
    x_mean = x_sum / count
    y_mean = y_sum / count
    xtx = torch.zeros((x_dim, x_dim), dtype=torch.float64, device=device)
    xty = torch.zeros((x_dim, y_dim), dtype=torch.float64, device=device)
    for start in range(0, len(train_x), batch_size):
        stop = min(start + batch_size, len(train_x))
        x = torch.from_numpy(np.array(train_x[start:stop], copy=True)).to(device).double().reshape(-1, x_dim) - x_mean
        y = torch.from_numpy(np.array(train_y[start:stop], copy=True)).to(device).double().reshape(-1, y_dim) - y_mean
        xtx += x.T @ x
        xty += x.T @ y

    def evaluate(weight: torch.Tensor, x_values: np.ndarray, y_values: np.ndarray) -> dict[str, float]:
        squared_error = 0.0
        squared_target = 0.0
        total_values = 0
        for start in range(0, len(x_values), batch_size):
            stop = min(start + batch_size, len(x_values))
            x = torch.from_numpy(np.array(x_values[start:stop], copy=True)).to(device).double().reshape(-1, x_dim)
            y = torch.from_numpy(np.array(y_values[start:stop], copy=True)).to(device).double().reshape(-1, y_dim)
            prediction = (x - x_mean) @ weight + y_mean
            squared_error += float((prediction - y).square().sum().item())
            squared_target += float((y - y_mean).square().sum().item())
            total_values += y.numel()
        mse = squared_error / total_values
        target_energy = squared_target / total_values
        return {
            "mse": mse,
            "normalized_mse": mse / max(target_energy, 1e-12),
            "r2": 1.0 - squared_error / max(squared_target, 1e-12),
        }

    candidates = []
    eye = torch.eye(x_dim, dtype=torch.float64, device=device)
    for alpha in alphas:
        weight = torch.linalg.solve(xtx + float(alpha) * eye, xty)
        candidates.append({"alpha": alpha, **evaluate(weight, val_x, val_y)})
    best = min(candidates, key=lambda row: row["normalized_mse"])
    weight = torch.linalg.solve(xtx + float(best["alpha"]) * eye, xty)
    return {
        "selection_partition": "validation",
        "candidates": candidates,
        "selected_alpha": best["alpha"],
        "test": evaluate(weight, test_x, test_y),
    }


def candidate_metric_bundle(
    *,
    quantized: np.ndarray,
    indices: np.ndarray,
    native: np.ndarray,
    decoder: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    codebook_size: int,
    stage_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    prediction = decode_codes(decoder, quantized, arrays["current"], device, batch_size)
    return {
        "reconstruction": reconstruction_bundle(
            np.asarray(arrays["current"]),
            np.asarray(arrays["future"]),
            prediction,
            np.asarray(arrays["dynamic"], dtype=bool),
        ),
        "quantization": quantization_metrics(native, quantized),
        "stage_diagnostics": stage_diagnostics,
        "collapse": collapse_diagnostics(indices, quantized, codebook_size=codebook_size),
        "linear_cka_native_compressed": linear_cka(
            np.asarray(native).reshape(len(native), -1),
            np.asarray(quantized).reshape(len(quantized), -1),
        ),
        "structure": representation_structure(quantized),
    }


def checkpoint_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def component_digests(runtime: dict[str, Any]) -> dict[str, str]:
    return {
        "s1_teacher": parameter_digest(runtime["s1"]),
        "s2_encoder": parameter_digest(runtime["s2"].encoder),
        "s2_decoder": parameter_digest(runtime["s2"].decoder),
        "original_unit_rq": parameter_digest(runtime["original_rq"]),
    }
