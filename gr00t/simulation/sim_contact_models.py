"""Neural representations for S4.2 simulated Contact state and dynamics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from gr00t.tactile_teacher.models import (
    CurrentMLP,
    FlattenedHistoryMLP,
    PredictiveContactTeacher,
    TemporalCNN,
)


HISTORY_STEPS = 26
TACTILE_DIM = 30
TEACHER_FUTURE_STEPS = 13
CONTACT_STATE_DIM = 256


def build_sim_contact_teacher(name: str, *, reconstruction_weight: float = 0.25) -> nn.Module:
    if name == "B0":
        return CurrentMLP(
            input_dim=TACTILE_DIM,
            latent_dim=CONTACT_STATE_DIM,
            future_steps=TEACHER_FUTURE_STEPS,
        )
    if name == "B1":
        return FlattenedHistoryMLP(
            input_dim=TACTILE_DIM,
            history_steps=HISTORY_STEPS,
            latent_dim=CONTACT_STATE_DIM,
            future_steps=TEACHER_FUTURE_STEPS,
        )
    if name == "B2":
        return TemporalCNN(
            input_dim=TACTILE_DIM,
            latent_dim=CONTACT_STATE_DIM,
            future_steps=TEACHER_FUTURE_STEPS,
            channels=192,
        )
    if name == "B3":
        model = PredictiveContactTeacher(
            input_dim=TACTILE_DIM,
            history_steps=HISTORY_STEPS,
            future_steps=TEACHER_FUTURE_STEPS,
            latent_dim=CONTACT_STATE_DIM,
            channels=256,
        )
        model.reconstruction_weight = float(reconstruction_weight)  # type: ignore[attr-defined]
        return model
    raise ValueError(f"unknown simulated Contact teacher {name!r}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def save_teacher_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    candidate: str,
    normalization: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "tactile3d-unit.s4-2-sim-contact-teacher.v1",
            "candidate": candidate,
            "normalization": normalization,
            "metadata": metadata,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_teacher_checkpoint(
    path: Path, map_location: str | torch.device = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("schema") != "tactile3d-unit.s4-2-sim-contact-teacher.v1":
        raise ValueError("unsupported simulated Contact teacher checkpoint")
    candidate = checkpoint["candidate"]
    family = "B3" if candidate.startswith("B3-") else candidate
    reconstruction_weight = float(checkpoint["metadata"].get("reconstruction_weight", 0.25))
    model = build_sim_contact_teacher(family, reconstruction_weight=reconstruction_weight)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model, checkpoint
