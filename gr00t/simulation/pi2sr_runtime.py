"""Prospective canonical Contact-State runtime helpers for PI2S-R."""

from __future__ import annotations

import hashlib
import socket
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from gr00t.simulation.pi1d_runtime import recv_exact
from gr00t.simulation.s4_3_act import TorchTactileNormalization
from gr00t.simulation.s4_3_pi1 import CONTACT_STATE_DIM, HISTORY_STEPS, TACTILE_DIM
from gr00t.simulation.sim_contact_models import load_teacher_checkpoint

FRAME_HEADER = struct.Struct("!I")
CANONICAL_CHECKPOINT_SHA256 = "3d4519195e7a0d9f5af63399840a48121d5f614ea587c80b9461fc696a372a19"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_canonical_torch() -> dict[str, Any]:
    """Apply the frozen CPU/float32 canonical settings before model execution."""

    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as error:
        if "cannot set number of interop threads" not in str(error).lower():
            raise
    torch.backends.mkldnn.enabled = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    return {
        "device": "cpu",
        "dtype": "float32",
        "batch_shape": [1, HISTORY_STEPS, TACTILE_DIM],
        "eval": True,
        "inference_mode": True,
        "autocast": False,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "mkldnn": bool(torch.backends.mkldnn.enabled),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch_version": torch.__version__,
    }


def validate_float32_array(value: np.ndarray, shape: tuple[int, ...], role: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float32:
        raise ValueError(f"{role} must have dtype float32")
    if array.shape != shape:
        raise ValueError(f"{role} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{role} must be finite")
    return np.ascontiguousarray(array)


def send_float32_frame(connection: socket.socket, value: np.ndarray) -> None:
    """Send a contiguous float32 array with the historical H service framing."""

    array = np.asarray(value)
    if array.dtype != np.float32 or not array.flags.c_contiguous:
        raise ValueError("framed array must be contiguous float32")
    payload = array.tobytes(order="C")
    connection.sendall(FRAME_HEADER.pack(len(payload)) + payload)


def receive_float32_frame(connection: socket.socket, shape: tuple[int, ...]) -> np.ndarray:
    expected_bytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize
    size = FRAME_HEADER.unpack(recv_exact(connection, FRAME_HEADER.size))[0]
    if size != expected_bytes:
        raise ValueError(f"framed payload has {size} bytes, expected {expected_bytes}")
    result = np.frombuffer(recv_exact(connection, size), dtype=np.float32).copy().reshape(shape)
    if not np.isfinite(result).all():
        raise ValueError("framed payload is non-finite")
    return result


class FramedFloat32UnixClient:
    """Shape-strict AF_UNIX client shared by transport and compute qualification."""

    def __init__(
        self,
        socket_path: Path,
        request_shape: tuple[int, ...],
        response_shape: tuple[int, ...],
        timeout_seconds: float = 30.0,
    ) -> None:
        self.request_shape = request_shape
        self.response_shape = response_shape
        self.connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.connection.settimeout(timeout_seconds)
        self.connection.connect(str(socket_path))
        self.requests = 0

    def request(self, value: np.ndarray) -> np.ndarray:
        request = validate_float32_array(value, self.request_shape, "request")
        send_float32_frame(self.connection, request)
        response = receive_float32_frame(self.connection, self.response_shape)
        self.requests += 1
        return response

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "FramedFloat32UnixClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


@dataclass
class DirectContactEncoder:
    """Exact accepted E_T under the frozen PI2S-R canonical CPU settings."""

    checkpoint_path: Path
    expected_sha256: str = CANONICAL_CHECKPOINT_SHA256

    def __post_init__(self) -> None:
        import torch

        self.checkpoint_path = Path(self.checkpoint_path)
        observed = sha256_file(self.checkpoint_path)
        if observed != self.expected_sha256:
            raise ValueError(f"canonical E_T checkpoint SHA mismatch: {observed}")
        self.settings = configure_canonical_torch()
        self.teacher, checkpoint = load_teacher_checkpoint(self.checkpoint_path, "cpu")
        self.normalization = TorchTactileNormalization(checkpoint["normalization"])
        self.teacher.eval().requires_grad_(False)
        self.normalization.eval().requires_grad_(False)
        if checkpoint.get("schema") != "tactile3d-unit.s4-2-sim-contact-teacher.v1":
            raise ValueError("canonical E_T checkpoint schema mismatch")
        if checkpoint["normalization"].get("candidate") != "N1":
            raise ValueError("canonical E_T normalization is not N1")
        self._torch = torch

    def encode(self, history: np.ndarray) -> np.ndarray:
        value = validate_float32_array(
            np.asarray(history), (HISTORY_STEPS, TACTILE_DIM), "tactile history"
        )
        tensor = self._torch.from_numpy(value.copy()).reshape(1, HISTORY_STEPS, TACTILE_DIM)
        with self._torch.inference_mode():
            latent = self.teacher(self.normalization(tensor))["latent"]
        result = latent.to(dtype=self._torch.float32, device="cpu").numpy()[0].copy()
        return validate_float32_array(result, (CONTACT_STATE_DIM,), "contact state")


def numeric_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """Report exact and numeric discrepancy without embedding an acceptance threshold."""

    lhs = np.asarray(left, dtype=np.float32)
    rhs = np.asarray(right, dtype=np.float32)
    if lhs.shape != rhs.shape:
        raise ValueError(f"comparison shape mismatch: {lhs.shape} != {rhs.shape}")
    delta = lhs.astype(np.float64) - rhs.astype(np.float64)
    absolute = np.abs(delta)
    denominator = np.maximum(np.abs(lhs.astype(np.float64)), np.abs(rhs.astype(np.float64)))
    relative = np.divide(absolute, denominator, out=np.zeros_like(absolute), where=denominator > 0)
    lhs_flat = lhs.astype(np.float64).reshape(-1)
    rhs_flat = rhs.astype(np.float64).reshape(-1)
    norm_product = float(np.linalg.norm(lhs_flat) * np.linalg.norm(rhs_flat))
    cosine = (
        1.0
        if norm_product == 0.0 and np.array_equal(lhs, rhs)
        else (float(np.dot(lhs_flat, rhs_flat) / norm_product) if norm_product else 0.0)
    )
    return {
        "shape": list(lhs.shape),
        "dtype": str(lhs.dtype),
        "array_equal": bool(np.array_equal(lhs, rhs)),
        "byte_equal": bool(lhs.tobytes() == rhs.tobytes()),
        "max_absolute_error": float(absolute.max(initial=0.0)),
        "mean_absolute_error": float(absolute.mean()) if absolute.size else 0.0,
        "max_relative_error": float(relative.max(initial=0.0)),
        "cosine_similarity": cosine,
        "left_nan_count": int(np.isnan(lhs).sum()),
        "right_nan_count": int(np.isnan(rhs).sum()),
        "left_inf_count": int(np.isinf(lhs).sum()),
        "right_inf_count": int(np.isinf(rhs).sum()),
    }


def repeatability_metrics(values: np.ndarray) -> dict[str, Any]:
    samples = np.asarray(values, dtype=np.float32)
    if samples.ndim < 2 or len(samples) < 2:
        raise ValueError("repeatability requires at least two repeated outputs")
    reference = samples[0]
    comparisons = [numeric_metrics(reference, current) for current in samples[1:]]
    return {
        "repeats": len(samples),
        "all_byte_equal": all(row["byte_equal"] for row in comparisons),
        "max_absolute_error": max(row["max_absolute_error"] for row in comparisons),
        "mean_absolute_error": float(np.mean([row["mean_absolute_error"] for row in comparisons])),
    }


def latency_percentiles(samples_ns: Iterable[int]) -> dict[str, float]:
    values = np.asarray(list(samples_ns), dtype=np.float64) / 1_000_000.0
    if not len(values):
        raise ValueError("latency sample is empty")
    return {
        "count": int(len(values)),
        "p50_ms": float(np.quantile(values, 0.50)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "p99_ms": float(np.quantile(values, 0.99)),
        "max_ms": float(values.max()),
    }


def measure_requests(function, values: Iterable[np.ndarray]) -> tuple[list[np.ndarray], list[int]]:
    outputs: list[np.ndarray] = []
    latencies: list[int] = []
    for value in values:
        started = time.perf_counter_ns()
        outputs.append(function(value))
        latencies.append(time.perf_counter_ns() - started)
    return outputs, latencies
