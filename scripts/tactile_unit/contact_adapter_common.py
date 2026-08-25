"""Shared runtime helpers for the S3.2 contact-adaptor experiment."""

from __future__ import annotations

import json
import os
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
    sha256_file,
    verify_file,
)
from gr00t.contact_dynamics.evaluation import transition_metrics  # noqa: E402
from gr00t.tactile_teacher.models import PredictiveContactTeacher  # noqa: E402
from gr00t.tactile_unit.compatibility import parameter_digest, quantization_metrics  # noqa: E402


DEFAULT_SPEC = ROOT / "configs/tactile_unit/s3_2_contact_adapter.json"
DEFAULT_TRANSITIONS = ROOT / ".local/cache/contact_dynamics/s2_transition_pairs"
DEFAULT_CODES = ROOT / ".local/cache/contact_dynamics/s2_codes"
DEFAULT_S1 = ROOT / ".local/experiments/tactile_teacher/s1_teacher/best.pt"
DEFAULT_S2 = ROOT / ".local/experiments/contact_dynamics/s2_models/proposed_best.pt"
DEFAULT_T4 = ROOT / ".local/artifacts/reproduction/t4"
DEFAULT_CACHE = ROOT / ".local/cache/tactile_unit/s3_2"
SPLIT_NAMES = {"train": "train", "validation": "val", "test": "test"}


def verify_gpu() -> torch.device:
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("S3.2 requires CUDA_DEVICE_ORDER=PCI_BUS_ID")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise RuntimeError("S3.2 requires physical GPU3 as the only visible GPU")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible CUDA device, got {torch.cuda.device_count()}")
    return torch.device("cuda:0")


def load_arrays(cache: Path, split: str) -> dict[str, np.ndarray]:
    names = (
        "current",
        "future",
        "episode_id",
        "anchor_frame",
        "dynamic",
        "contact_transition",
        "force_trend_class",
        "finger_change",
    )
    directory = cache / SPLIT_NAMES[split]
    return {name: np.load(directory / f"{name}.npy", mmap_mode="r") for name in names}


def load_s1_teacher(path: Path, device: torch.device) -> PredictiveContactTeacher:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "tactile3d-unit.s1-contact-teacher-checkpoint.v1":
        raise RuntimeError("S1 checkpoint schema mismatch")
    model = PredictiveContactTeacher(
        latent_dim=int(checkpoint["latent_dim"]), channels=int(checkpoint["channels"])
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval().requires_grad_(False).to(device)


def load_runtime(
    spec_path: Path,
    transition_cache: Path,
    code_cache: Path,
    s1_checkpoint: Path,
    s2_checkpoint: Path,
    t4_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text())
    frozen = spec["frozen_components"]
    identity = {
        "spec_sha256": sha256_file(spec_path),
        "s1_checkpoint_sha256": verify_file(
            s1_checkpoint, frozen["s1_teacher_checkpoint_sha256"], "S1 checkpoint"
        ),
        "s2_checkpoint_sha256": verify_file(
            s2_checkpoint, frozen["s2_checkpoint_sha256"], "S2 checkpoint"
        ),
        "s2_transition_manifest_sha256": verify_file(
            transition_cache / "manifest.json",
            frozen["s2_transition_manifest_sha256"],
            "S2 transition manifest",
        ),
        "s2_train_codes_sha256": verify_file(
            code_cache / "train.npy", frozen["s2_train_codes_sha256"], "S2 train codes"
        ),
        "s2_test_codes_sha256": verify_file(
            code_cache / "test.npy", frozen["s2_test_codes_sha256"], "S2 test codes"
        ),
    }
    s3_0_spec_path = ROOT / frozen["original_unit_spec"]
    s3_0_spec = json.loads(s3_0_spec_path.read_text())
    extraction = json.loads((t4_dir / "extraction_summary.json").read_text())
    rq, rq_identity = load_frozen_rq(extraction, s3_0_spec)
    rq.eval().requires_grad_(False).to(device)
    s1 = load_s1_teacher(s1_checkpoint, device)
    s2 = load_s2_model(s2_checkpoint, device)
    manifest = json.loads((transition_cache / "manifest.json").read_text())
    expected_pairs = spec["data"]["pairs"]
    for public, cache_name in SPLIT_NAMES.items():
        if int(manifest["splits"][cache_name]["pairs"]) != int(expected_pairs[public]):
            raise RuntimeError(f"{public} pair count mismatch")
    if int(manifest["canonical_contract"]["horizon_frames"]) != int(
        spec["data"]["canonical_horizon_frames"]
    ):
        raise RuntimeError("canonical S2 horizon mismatch")
    return {
        "spec": spec,
        "identity": identity,
        "rq": rq,
        "rq_identity": rq_identity,
        "s1": s1,
        "s2": s2,
        "manifest": manifest,
    }


@torch.inference_mode()
def ensure_validation_codes(
    s2: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    path: Path,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    shape = (len(arrays["current"]), 8, 32)
    if path.is_file():
        cached = np.load(path, mmap_mode="r")
        if tuple(cached.shape) != shape or not np.isfinite(cached).all():
            raise RuntimeError("invalid cached S3.2 validation contact code")
        regenerated = s2.encoder(
            torch.from_numpy(np.array(arrays["current"][:64], copy=True)).to(device),
            torch.from_numpy(np.array(arrays["future"][:64], copy=True)).to(device),
        ).cpu().numpy()
        if not np.allclose(regenerated, np.asarray(cached[:64]), atol=1e-5, rtol=1e-5):
            raise RuntimeError("validation code cache does not match frozen S2 encoder")
        return cached
    path.parent.mkdir(parents=True, exist_ok=True)
    result = open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    for start in range(0, len(result), batch_size):
        stop = min(start + batch_size, len(result))
        current = torch.from_numpy(np.array(arrays["current"][start:stop], copy=True)).to(device)
        future = torch.from_numpy(np.array(arrays["future"][start:stop], copy=True)).to(device)
        result[start:stop] = s2.encoder(current, future).float().cpu().numpy()
    result.flush()
    del result
    return np.load(path, mmap_mode="r")


@torch.inference_mode()
def transform_codes(
    adaptor: torch.nn.Module,
    rq: torch.nn.Module,
    values: np.ndarray,
    device: torch.device,
    batch_size: int,
    *,
    adapted_path: Path | None = None,
    quantized_path: Path | None = None,
    indices_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = tuple(values.shape)
    if len(shape) != 3 or shape[1:] != (8, 32):
        raise ValueError(f"expected contact codes [N,8,32], got {shape}")
    paths = (adapted_path, quantized_path, indices_path)
    if any(path is not None for path in paths) and not all(path is not None for path in paths):
        raise ValueError("all transform cache paths must be supplied together")
    if adapted_path is None:
        adapted = np.empty(shape, dtype=np.float32)
        quantized = np.empty(shape, dtype=np.float32)
        indices = np.empty((shape[0], 8, len(rq.layers)), dtype=np.int64)
    else:
        adapted_path.parent.mkdir(parents=True, exist_ok=True)
        adapted = open_memmap(adapted_path, mode="w+", dtype=np.float32, shape=shape)
        quantized = open_memmap(quantized_path, mode="w+", dtype=np.float32, shape=shape)
        indices = open_memmap(
            indices_path, mode="w+", dtype=np.int64, shape=(shape[0], 8, len(rq.layers))
        )
    adaptor.eval()
    rq.eval()
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        batch = torch.from_numpy(np.array(values[start:stop], copy=True)).to(device)
        mapped = adaptor(batch)
        q_value, q_indices, _ = rq(mapped)
        adapted[start:stop] = mapped.float().cpu().numpy()
        quantized[start:stop] = q_value.float().cpu().numpy()
        indices[start:stop] = q_indices.cpu().numpy()
    if isinstance(adapted, np.memmap):
        adapted.flush()
        quantized.flush()
        indices.flush()
        del adapted, quantized, indices
        adapted = np.load(adapted_path, mmap_mode="r")
        quantized = np.load(quantized_path, mmap_mode="r")
        indices = np.load(indices_path, mmap_mode="r")
    return adapted, quantized, indices


@torch.inference_mode()
def decode_codes(
    decoder: torch.nn.Module,
    codes: np.ndarray,
    current: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    result = np.empty((len(codes), 256), dtype=np.float32)
    decoder.eval()
    for start in range(0, len(codes), batch_size):
        stop = min(start + batch_size, len(codes))
        code = torch.from_numpy(np.array(codes[start:stop], copy=True)).to(device)
        state = torch.from_numpy(np.array(current[start:stop], copy=True)).to(device)
        result[start:stop] = decoder(code, state).float().cpu().numpy()
    return result


def reconstruction_bundle(
    current: np.ndarray,
    future: np.ndarray,
    prediction: np.ndarray,
    dynamic: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    return {
        "all": transition_metrics(current, future, prediction),
        "dynamic": transition_metrics(current, future, prediction, dynamic),
    }


def evaluate_transformed(
    decoder: torch.nn.Module,
    adapted: np.ndarray,
    quantized: np.ndarray,
    arrays: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    current = np.asarray(arrays["current"])
    future = np.asarray(arrays["future"])
    dynamic = np.asarray(arrays["dynamic"], dtype=bool)
    continuous_prediction = decode_codes(decoder, adapted, current, device, batch_size)
    quantized_prediction = decode_codes(decoder, quantized, current, device, batch_size)
    return {
        "quantization": quantization_metrics(adapted, quantized),
        "adapted_continuous_reconstruction": reconstruction_bundle(
            current, future, continuous_prediction, dynamic
        ),
        "adapted_quantized_reconstruction": reconstruction_bundle(
            current, future, quantized_prediction, dynamic
        ),
    }


def component_digests(runtime: dict[str, Any]) -> dict[str, str]:
    return {
        "s1_teacher": parameter_digest(runtime["s1"]),
        "s2_encoder": parameter_digest(runtime["s2"].encoder),
        "s2_decoder": parameter_digest(runtime["s2"].decoder),
        "original_unit_rq": parameter_digest(runtime["rq"]),
    }
