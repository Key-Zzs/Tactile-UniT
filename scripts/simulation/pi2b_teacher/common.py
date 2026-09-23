"""Small shared helpers for the PI2B matched-teacher scripts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = ROOT / "configs/simulation/pi2b_teacher"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def tensor_digest(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def git_output(*arguments: str) -> str:
    import subprocess

    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def workspace() -> dict[str, Any]:
    return load_json(ROOT / ".local/config/pi2b_workspace.json")


def stage_root() -> Path:
    return Path(workspace()["write_root"])


def artifact_root() -> Path:
    return stage_root() / "artifacts"


def status_root() -> Path:
    return stage_root() / "status"


def history_cache(name: str) -> Path:
    return ROOT / ".local/refs/base/cache/simulation/s4_2_formal" / name


def model_parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
