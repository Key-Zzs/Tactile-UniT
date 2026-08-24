#!/usr/bin/env python3
"""Incremental, GPU-backed verification for the UniT training environment."""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
from importlib import metadata


def version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not-installed"


def check_imports() -> None:
    import torch
    import transformers
    import accelerate
    import gr00t
    import decord
    import qwen_vl_utils
    import lpips
    from gr00t.data.dataset import LeRobotMixtureDataset
    from gr00t.model.gr00t_n1_tokenizer_unit import GR00T_Tokenizer
    from gr00t.model.gr00t_n1_unit import GR00T_N1_5_UniT

    print(f"Python: {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"torch: {torch.__version__}")
    print(f"transformers: {transformers.__version__}")
    print(f"accelerate: {accelerate.__version__}")
    print(f"gr00t: {version('gr00t')}")
    print(f"decord: {decord.__version__}")
    print(f"qwen-vl-utils: {version('qwen-vl-utils')}")
    print(f"lpips: {version('lpips')}")
    print("UniT imports: GR00T_Tokenizer, GR00T_N1_5_UniT, LeRobotMixtureDataset")
    print("TRAINING_ENV PASS")


def check_cuda() -> None:
    import torch

    print(f"torch version: {torch.__version__}")
    print(f"torch CUDA version: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    count = torch.cuda.device_count()
    print(f"GPU count: {count}")
    if not torch.cuda.is_available() or count != 4:
        raise RuntimeError(f"Expected 4 CUDA GPUs, found {count}")
    for index in range(count):
        props = torch.cuda.get_device_properties(index)
        try:
            bf16 = torch.cuda.is_bf16_supported(index)
        except TypeError:
            with torch.cuda.device(index):
                bf16 = torch.cuda.is_bf16_supported()
        print(
            f"GPU {index}: {props.name}; cc={props.major}.{props.minor}; "
            f"VRAM={props.total_memory / 2**30:.2f} GiB; bf16={bf16}"
        )
    # Keep the kernel on a specified physical device so the audit still sees all four GPUs.
    device_index = int(__import__('os').environ.get("S0_TRAIN_GPU", "3"))
    if device_index >= count:
        raise RuntimeError(f"S0_TRAIN_GPU={device_index} is outside visible GPU range")
    device = torch.device(f"cuda:{device_index}")
    left = torch.randn((256, 256), device=device, dtype=torch.bfloat16)
    right = torch.randn((256, 256), device=device, dtype=torch.bfloat16)
    value = (left @ right).float().mean()
    torch.cuda.synchronize(device)
    if not torch.isfinite(value):
        raise RuntimeError("BF16 matmul returned a non-finite value")
    print(f"BF16 matmul: device={device}; mean={value.item():.6f}")
    print("CUDA PASS")


def check_flash_attention() -> None:
    import torch
    flash_attn = importlib.import_module("flash_attn")
    from flash_attn import flash_attn_func

    device_index = int(__import__('os').environ.get("S0_TRAIN_GPU", "3"))
    device = torch.device(f"cuda:{device_index}")
    q = torch.randn((1, 16, 4, 64), device=device, dtype=torch.float16)
    output = flash_attn_func(q, q, q)
    torch.cuda.synchronize(device)
    if output.shape != q.shape or not torch.isfinite(output).all():
        raise RuntimeError(f"Unexpected FlashAttention result: shape={tuple(output.shape)}")
    print(f"flash-attn: {getattr(flash_attn, '__version__', version('flash-attn'))}")
    print(f"FlashAttention forward: shape={tuple(output.shape)}; finite=True; device={device}")
    print("FLASH_ATTENTION PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("imports", "cuda", "flash-attn", "all"), default="all")
    args = parser.parse_args()
    if args.stage in ("imports", "all"):
        check_imports()
    if args.stage in ("cuda", "all"):
        check_cuda()
    if args.stage in ("flash-attn", "all"):
        check_flash_attention()


if __name__ == "__main__":
    main()
