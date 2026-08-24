"""Load the exact accepted S1 Teacher as an immutable inference module."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from gr00t.tactile_teacher.models import PredictiveContactTeacher


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_frozen_teacher(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[PredictiveContactTeacher, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "tactile3d-unit.s1-contact-teacher-checkpoint.v1":
        raise ValueError("checkpoint is not the accepted S1 Teacher schema")
    model = PredictiveContactTeacher(
        latent_dim=int(checkpoint["latent_dim"]),
        channels=int(checkpoint["channels"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    model.to(device)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("S1 Teacher was not frozen")
    identity = {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "parameter_sha256": parameter_digest(model),
        "schema": checkpoint["schema"],
        "latent_dim": int(checkpoint["latent_dim"]),
        "channels": int(checkpoint["channels"]),
        "epoch": int(checkpoint["epoch"]),
        "validation": checkpoint.get("val_metrics", {}),
    }
    return model, identity
