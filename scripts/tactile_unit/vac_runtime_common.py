"""Runtime-only helpers shared by Track C1/C2 commands."""

from __future__ import annotations

import fcntl
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any, IO

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> tuple[torch.device, IO[str] | None, dict[str, Any]]:
    if requested == "cpu":
        return torch.device("cpu"), None, {
            "preferred_physical": 3,
            "actual_physical": None,
            "logical": "cpu",
            "fallback": True,
        }
    if requested != "cuda:0":
        raise RuntimeError("Track C GPU processes must use logical cuda:0")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("CUDA_DEVICE_ORDER=PCI_BUS_ID is required")
    physical = os.environ.get("CUDA_VISIBLE_DEVICES")
    if physical not in {"1", "2", "3"}:
        raise RuntimeError("Track C permits GPU1 only by explicit authorization; GPU0 is forbidden")
    completed = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    lock_path = Path(completed.stdout.strip()) / f"tactile3d_unit_gpu{physical}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(f"physical GPU{physical} advisory lock is busy") from error
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        handle.close()
        raise RuntimeError("Track C requires exactly one visible CUDA device")
    return torch.device("cuda:0"), handle, {
        "preferred_physical": 3,
        "actual_physical": int(physical),
        "logical": "cuda:0",
        "fallback": physical != "3",
    }


def ensure_no_native_gradients(*models: torch.nn.Module) -> None:
    for model in models:
        model.eval().requires_grad_(False)
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("a frozen native encoder still has trainable parameters")
