#!/usr/bin/env python3
"""Isolated DexJoCo embodiment slot for the frozen Original UniT tokenizer.

The official implementation stores embodiment-specific tensors in dense banks.
This module uses torch parametrizations to expose the byte-identical official
bank as a frozen prefix and one separately optimizable DexJoCo row as the final
slot.  No official source file or released checkpoint is modified.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize


OFFICIAL_CATEGORY_COUNT = 30
DEXJOCO_CATEGORY_ID = 30
EXTENDED_CATEGORY_COUNT = 31
EXPECTED_ADAPTER_PARAMETERS = 23_485_568
TOKENIZER_STATE_DIM = 128
TOKENIZER_ACTION_DIM = 128
TOKENIZER_ACTION_HORIZON = 16
CONTROL_HZ = 50.0
CANONICAL_CONTROL_STEPS = 27
CANONICAL_HORIZON_SECONDS = CANONICAL_CONTROL_STEPS / CONTROL_HZ
POLICY_DATASET_HZ = 30.0
POLICY_DATASET_FRAMES = 16
POLICY_DATASET_HORIZON_SECONDS = POLICY_DATASET_FRAMES / POLICY_DATASET_HZ

# The 27 simulator commands represent the half-open interval [t, t+27).
# These are the nearest 50 Hz commands to the official 30 Hz action slots
# t + j/30, j=0..15.  The future image remains the observation at t+27.
ACTION_RESAMPLE_INDICES = np.rint(
    np.arange(TOKENIZER_ACTION_HORIZON, dtype=np.float64) * CONTROL_HZ / POLICY_DATASET_HZ
).astype(np.int64)


class AppendDexJoCoSlot(nn.Module):
    """Concatenate a frozen official category bank with one trainable row."""

    def __init__(self, original: torch.Tensor, parameter_name: str) -> None:
        super().__init__()
        shape = (1, *original.shape[1:])
        if parameter_name == "W":
            value = torch.randn(shape, dtype=original.dtype, device=original.device) * 0.02
        elif parameter_name in {"b", "bias"}:
            value = torch.zeros(shape, dtype=original.dtype, device=original.device)
        elif parameter_name == "weight":
            value = torch.ones(shape, dtype=original.dtype, device=original.device)
        else:
            raise ValueError(f"unsupported category parameter: {parameter_name}")
        self.adapter = nn.Parameter(value)

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        return torch.cat((original, self.adapter), dim=0)


def install_dexjoco_adapter(model: nn.Module, seed: int = 42) -> dict[str, Any]:
    """Freeze the tokenizer and append one independently trainable category."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    installed: list[dict[str, Any]] = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for module_name, module in list(model.named_modules()):
            if getattr(module, "num_categories", None) != OFFICIAL_CATEGORY_COUNT:
                continue
            local_parameters = list(module.named_parameters(recurse=False))
            for parameter_name, parameter in local_parameters:
                if parameter.ndim < 1 or parameter.shape[0] != OFFICIAL_CATEGORY_COUNT:
                    raise RuntimeError(
                        f"malformed official category bank {module_name}.{parameter_name}: "
                        f"{tuple(parameter.shape)}"
                    )
                parameter.requires_grad_(False)
                parametrization = AppendDexJoCoSlot(parameter, parameter_name)
                parametrize.register_parametrization(
                    module, parameter_name, parametrization, unsafe=True
                )
                installed.append(
                    {
                        "name": f"{module_name}.parametrizations.{parameter_name}.0.adapter",
                        "official_bank_shape": list(parameter.shape),
                        "adapter_shape": [1, *parameter.shape[1:]],
                        "parameters": int(parameter[0].numel()),
                        "initialization": (
                            "normal_std_0.02"
                            if parameter_name == "W"
                            else "ones"
                            if parameter_name == "weight"
                            else "zeros"
                        ),
                    }
                )
            module.num_categories = EXTENDED_CATEGORY_COUNT

    if hasattr(model, "action_branch"):
        model.action_branch.unified_embodiment_id = DEXJOCO_CATEGORY_ID
        model.action_branch.config.max_num_embodiments = EXTENDED_CATEGORY_COUNT
    if hasattr(model, "action_decoder"):
        model.action_decoder.unified_embodiment_id = DEXJOCO_CATEGORY_ID
        model.action_decoder.config.max_num_embodiments = EXTENDED_CATEGORY_COUNT
    if hasattr(model, "config"):
        model.config.action_encoder_cfg["max_num_embodiments"] = EXTENDED_CATEGORY_COUNT
        model.config.action_decoder_cfg["max_num_embodiments"] = EXTENDED_CATEGORY_COUNT
        model.config.unified_embodiment_id = DEXJOCO_CATEGORY_ID

    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for _, parameter in trainable)
    if trainable_count != EXPECTED_ADAPTER_PARAMETERS:
        raise RuntimeError(
            f"DexJoCo adapter count {trainable_count} != {EXPECTED_ADAPTER_PARAMETERS}"
        )
    if any(".adapter" not in name for name, _ in trainable):
        raise RuntimeError("a non-adapter tokenizer parameter remains trainable")
    if len(installed) != len(trainable):
        raise RuntimeError("adapter installation and trainable parameter lists disagree")
    return {
        "category_id": DEXJOCO_CATEGORY_ID,
        "official_category_count": OFFICIAL_CATEGORY_COUNT,
        "extended_category_count": EXTENDED_CATEGORY_COUNT,
        "trainable_parameters": trainable_count,
        "tensors": installed,
    }


def adapter_named_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    result = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    if not result or any(".adapter" not in name for name, _ in result):
        raise RuntimeError("optimizer parameter selection is not DexJoCo-adapter-only")
    if sum(value.numel() for _, value in result) != EXPECTED_ADAPTER_PARAMETERS:
        raise RuntimeError("optimizer parameter count is not the frozen adapter count")
    return result


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in adapter_named_parameters(model)
    }


def load_adapter_state_dict(model: nn.Module, values: dict[str, torch.Tensor]) -> None:
    parameters = dict(adapter_named_parameters(model))
    if set(parameters) != set(values):
        missing = sorted(set(parameters) - set(values))
        unexpected = sorted(set(values) - set(parameters))
        raise RuntimeError(f"adapter state mismatch: missing={missing}, unexpected={unexpected}")
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = values[name]
            if value.shape != parameter.shape:
                raise RuntimeError(f"adapter tensor shape mismatch for {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def frozen_parameter_digest(model: nn.Module, chunk_bytes: int = 32 * 1024 * 1024) -> str:
    """Hash every non-trainable parameter without materializing one giant copy."""

    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            continue
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tuple(parameter.shape)).encode("ascii") + b"\0")
        flat = parameter.detach().contiguous().view(torch.uint8).reshape(-1)
        elements = max(1, chunk_bytes)
        for start in range(0, flat.numel(), elements):
            digest.update(flat[start : start + elements].cpu().numpy().tobytes())
    return digest.hexdigest()


def quaternion_wxyz_to_rotation_6d(quaternion: np.ndarray) -> np.ndarray:
    """Match pytorch3d quaternion_to_matrix -> matrix_to_rotation_6d."""

    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape[-1] != 4:
        raise ValueError("quaternion input must end in four wxyz values")
    value = value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = np.moveaxis(value, -1, 0)
    matrix = np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(*value.shape[:-1], 3, 3)
    return matrix[..., :2, :].reshape(*value.shape[:-1], 6).astype(np.float32)


def dexjoco_state_view(state23: np.ndarray) -> np.ndarray:
    value = np.asarray(state23)
    if value.shape[-1] != 23:
        raise ValueError("DexJoCo state must be xyz3 + quaternion-wxyz4 + Allegro16")
    result = np.concatenate(
        (value[..., :3], quaternion_wxyz_to_rotation_6d(value[..., 3:7]), value[..., 7:]),
        axis=-1,
    ).astype(np.float32)
    if result.shape[-1] != 25 or not np.isfinite(result).all():
        raise RuntimeError("invalid DexJoCo tokenizer state view")
    return result


def dexjoco_action_view(action27x22: np.ndarray) -> np.ndarray:
    value = np.asarray(action27x22, dtype=np.float32)
    if value.shape[-2:] != (CANONICAL_CONTROL_STEPS, 22):
        raise ValueError("DexJoCo action window must be [27,22]")
    result = value[..., ACTION_RESAMPLE_INDICES, :]
    if result.shape[-2:] != (TOKENIZER_ACTION_HORIZON, 22) or not np.isfinite(result).all():
        raise RuntimeError("invalid DexJoCo tokenizer action view")
    return result


def normalize_and_pad(
    value: np.ndarray, mean: np.ndarray, std: np.ndarray, target_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(value, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape != (value.shape[-1],) or std.shape != mean.shape:
        raise ValueError("normalization statistic shape mismatch")
    normalized = np.zeros_like(value)
    nonzero = std != 0
    normalized[..., nonzero] = (value[..., nonzero] - mean[nonzero]) / std[nonzero]
    normalized[..., ~nonzero] = value[..., ~nonzero]
    output = np.zeros((*value.shape[:-1], target_dim), dtype=np.float32)
    mask = np.zeros_like(output)
    output[..., : value.shape[-1]] = normalized
    mask[..., : value.shape[-1]] = 1.0
    if not np.isfinite(output).all():
        raise RuntimeError("normalization produced non-finite values")
    return output, mask


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(32 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trainable_audit(model: nn.Module) -> dict[str, Any]:
    trainable = []
    frozen = []
    for name, parameter in model.named_parameters():
        row = {"name": name, "shape": list(parameter.shape), "parameters": parameter.numel()}
        (trainable if parameter.requires_grad else frozen).append(row)
    return {
        "trainable": trainable,
        "frozen": frozen,
        "trainable_tensor_count": len(trainable),
        "frozen_tensor_count": len(frozen),
        "trainable_parameter_count": sum(row["parameters"] for row in trainable),
        "frozen_parameter_count": sum(row["parameters"] for row in frozen),
        "shared_official_parameters_trainable": 0
        if all(".adapter" in row["name"] for row in trainable)
        else -1,
    }


def finite_nonzero_adapter_gradients(parameters: Iterable[nn.Parameter]) -> dict[str, Any]:
    gradients = [parameter.grad for parameter in parameters]
    return {
        "all_present": all(gradient is not None for gradient in gradients),
        "all_finite": all(
            gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients
        ),
        "nonzero_tensor_count": sum(
            gradient is not None and bool(torch.count_nonzero(gradient)) for gradient in gradients
        ),
        "tensor_count": len(gradients),
    }
