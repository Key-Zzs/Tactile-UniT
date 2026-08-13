#!/usr/bin/env python3
"""Validate the official UniT checkpoint load and one real GR1 inference.

This is a diagnostics-only wrapper around the repository's official
``Gr00tUniTPolicy`` and GR1 dataset path.  It captures the underlying
Transformers loading info because the custom UniT loader intentionally catches
some exceptions; a caught exception must not be reported as a successful load.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _install_loading_info_capture() -> list[dict[str, Any]]:
    """Capture loading info for the two repository-defined model classes."""
    from transformers import PreTrainedModel

    original = PreTrainedModel.from_pretrained
    original_function = original.__func__
    records: list[dict[str, Any]] = []
    captured_classes = {"GR00T_N1_5_UniT", "GR00T_Tokenizer"}

    def capture(cls, *args, **kwargs):
        if cls.__name__ not in captured_classes:
            return original_function(cls, *args, **kwargs)

        kwargs = dict(kwargs)
        kwargs["output_loading_info"] = True
        loaded, info = original_function(cls, *args, **kwargs)
        records.append(
            {
                "class": cls.__name__,
                "missing_keys": list(info.get("missing_keys", [])),
                "unexpected_keys": list(info.get("unexpected_keys", [])),
                "mismatched_keys": [list(item) for item in info.get("mismatched_keys", [])],
                "error_msgs": list(info.get("error_msgs", [])),
            }
        )
        return loaded

    PreTrainedModel.from_pretrained = classmethod(capture)
    return records


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _flatten_action(action: dict[str, Any]) -> np.ndarray:
    arrays = []
    for key in sorted(action):
        if not key.startswith("action."):
            continue
        value = np.asarray(action[key])
        arrays.append(value.reshape(value.shape[0], -1) if value.ndim > 1 else value.reshape(1, -1))
    if not arrays:
        raise RuntimeError(f"Policy returned no action.* keys: {sorted(action)}")
    return np.concatenate(arrays, axis=-1)


def _checkpoint_index_summary(checkpoint: Path) -> dict[str, Any]:
    index_path = checkpoint / "model.safetensors.index.json"
    with index_path.open() as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map", {})
    return {
        "index": str(index_path),
        "weight_count": len(weight_map),
        "shards": sorted(set(weight_map.values())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.environ.get("UNIT_FULLDATA_CKPT"), required=False)
    parser.add_argument("--dataset-root", default=os.environ.get("GR1_DATASET_DIR"), required=False)
    parser.add_argument(
        "--task",
        default="gr1_unified.PnPCupToDrawerClose",
        help="One real GR1 dataset directory name.",
    )
    parser.add_argument("--output-dir", default=".local/artifacts/reproduction/phase3")
    parser.add_argument("--data-config", default="fourier_gr1_arms_waist_gausNorm_crop_cam_ego_joints_only")
    parser.add_argument("--denoising-steps", type=int, default=4)
    args = parser.parse_args()

    if not args.checkpoint or not args.dataset_root:
        parser.error("--checkpoint/UNIT_FULLDATA_CKPT and --dataset-root/GR1_DATASET_DIR are required")

    checkpoint = Path(args.checkpoint).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    task_path = dataset_root / args.task
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Phase 3 checker")

    from transformers import AutoConfig
    from gr00t.data.dataset import LeRobotSingleDatasetWithGoalImage
    from gr00t.experiment.data_config_unit import load_data_config
    from gr00t.model.policy_unit import Gr00tUniTPolicy

    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    root_config = json.loads((checkpoint / "config.json").read_text())
    tokenizer_config = json.loads((checkpoint / "tokenizer" / "config.json").read_text())
    model_cfg = root_config["unit_cfg"]
    backbone_cfg = root_config["backbone_cfg"]
    config_summary = {
        "architecture": root_config.get("architectures"),
        "model_type": root_config.get("model_type"),
        "action_horizon": root_config.get("action_horizon"),
        "action_dim": root_config.get("action_dim"),
        "state_dim": root_config.get("state_dim"),
        "backbone": backbone_cfg,
        "unit_cfg": model_cfg,
        "nested_tokenizer_path": str(checkpoint / "tokenizer"),
        "nested_tokenizer_exists": (checkpoint / "tokenizer").is_dir(),
        "nested_tokenizer_architecture": tokenizer_config.get("architectures"),
        "checkpoint_index": _checkpoint_index_summary(checkpoint),
    }

    records = _install_loading_info_capture()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.time()

    data_config = load_data_config(
        args.data_config,
        eagle_path=backbone_cfg["eagle_path"],
        use_bridge=model_cfg["use_bridge"],
        ignore_lang_prefix=root_config.get("ignore_lang_prefix", False),
        num_bridge_tokens=model_cfg["num_bridge_tokens"],
    )
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    tokenizer = modality_transform.transforms[-1].eagle_processor.tokenizer
    policy = Gr00tUniTPolicy(
        model_path=str(checkpoint),
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag="gr1",
        denoising_steps=args.denoising_steps,
        device="cuda",
        tokenizer_len=len(tokenizer),
        compute_bridge_loss=False,
    )
    load_seconds = time.time() - load_started

    loading_failures = []
    for record in records:
        for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            if record[field]:
                loading_failures.append({"class": record["class"], "field": field, "values": record[field]})
    if not records:
        loading_failures.append({"field": "loading_info", "values": ["No underlying Transformers load record captured"]})

    model = policy.model
    parameters = list(model.parameters())
    model_summary = {
        "model_class": type(model).__name__,
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
        "dtype": str(next(model.parameters()).dtype),
        "device": str(next(model.parameters()).device),
        "training": bool(model.training),
        "cuda_allocated_before_bytes": 0,
        "cuda_allocated_after_bytes": torch.cuda.memory_allocated(),
        "cuda_reserved_bytes": torch.cuda.memory_reserved(),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
    }

    dataset = LeRobotSingleDatasetWithGoalImage(
        dataset_path=task_path,
        modality_configs=policy.get_modality_config(),
        video_backend="decord",
        video_backend_kwargs=None,
        transforms=None,
        embodiment_tag="gr1",
        split="[-2:]",
    )
    trajectory_id = int(dataset.trajectory_ids[0])
    raw_sample = dataset.get_step_data(trajectory_id, 0)
    action = policy.get_action(raw_sample)
    flat_action = _flatten_action(action)
    finite = bool(np.isfinite(flat_action).all())
    expected_horizon = int(root_config["action_horizon"])
    expected_action_dim = sum(
        int(np.asarray(raw_sample[key]).shape[-1]) for key in data_config.action_keys
    )
    sample_summary = {
        "task": args.task,
        "trajectory_id": trajectory_id,
        "raw_input_keys": sorted(raw_sample),
        "input_shapes": {key: list(np.asarray(value).shape) for key, value in raw_sample.items()},
        "action_keys": sorted(action),
        "action_shapes": {key: list(np.asarray(value).shape) for key, value in action.items()},
        "flattened_action_shape": list(flat_action.shape),
        "expected_action_shape": [expected_horizon, expected_action_dim],
        "model_latent_action_dim": int(root_config["action_dim"]),
        "finite": finite,
        "min": float(flat_action.min()),
        "max": float(flat_action.max()),
        "norm": float(np.linalg.norm(flat_action)),
        "dtype": str(flat_action.dtype),
    }

    summary = {
        "checkpoint": str(checkpoint),
        "offline_env": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
        "config": config_summary,
        "loading": {
            "seconds": load_seconds,
            "records": records,
            "failures": loading_failures,
            "passed": not loading_failures,
        },
        "model": model_summary,
        "real_gr1_sample": sample_summary,
        "status": "PASS" if not loading_failures and finite and tuple(flat_action.shape) == (expected_horizon, expected_action_dim) else "FAIL",
    }

    (output_dir / "phase3_summary.json").write_text(json.dumps(_json_safe(summary), indent=2) + "\n")
    (output_dir / "sample_inference.json").write_text(json.dumps(_json_safe(sample_summary), indent=2) + "\n")
    print(json.dumps(_json_safe(summary), indent=2))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
