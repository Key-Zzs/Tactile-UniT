"""Small calibrated shared-error uncertainty estimator for Track C4."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from .c4_availability_conditioning import AvailabilityMode, sha256_file
from .continuous_vac_shared_space import state_dict_digest


PREDICTED_MODES = (
    AvailabilityMode.FULL_AH,
    AvailabilityMode.FALLBACK_VA,
    AvailabilityMode.FALLBACK_A,
)
MODE_TO_ID = {mode: index for index, mode in enumerate(PREDICTED_MODES)}


class ContactUncertaintyEstimator(nn.Module):
    """Predict scalar log variance from prediction/source summaries and mode."""

    def __init__(self, hidden: int = 64, log_variance_min: float = -12.0, log_variance_max: float = 4.0):
        super().__init__()
        if hidden > 96:
            raise ValueError("C4 uncertainty model exceeds bounded capacity")
        self.hidden = int(hidden)
        self.log_variance_min = float(log_variance_min)
        self.log_variance_max = float(log_variance_max)
        self.mode_embedding = nn.Embedding(3, 8)
        # mean/std/max/rms summaries for prediction and source, each over width 32.
        self.network = nn.Sequential(
            nn.Linear(32 * 8 + 8, hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1),
        )
        if self.parameter_count() > 50_000:
            raise ValueError("C4 uncertainty estimator must be <=50k parameters")

    @staticmethod
    def _summary(value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or value.shape[-1] != 32:
            raise ValueError("uncertainty inputs must be [B,T,32]")
        return torch.cat(
            (
                value.mean(1), value.std(1, unbiased=False),
                value.abs().amax(1), torch.sqrt(value.square().mean(1) + 1e-8),
            ),
            dim=1,
        )

    def forward(
        self,
        mode: AvailabilityMode | str,
        prediction: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        mode = AvailabilityMode(mode)
        if mode not in MODE_TO_ID:
            raise ValueError("ABSTAIN has no finite calibrated uncertainty")
        if len(prediction) != len(source):
            raise ValueError("unaligned uncertainty inputs")
        mode_id = torch.full(
            (len(prediction),), MODE_TO_ID[mode], dtype=torch.long,
            device=prediction.device,
        )
        features = torch.cat(
            (self._summary(prediction), self._summary(source), self.mode_embedding(mode_id)),
            dim=1,
        )
        return self.network(features).squeeze(1).clamp(
            self.log_variance_min, self.log_variance_max
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def heteroscedastic_nll(
    log_variance: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if target.requires_grad:
        target = target.detach()
    error = torch.square(prediction.detach() - target).flatten(1).mean(1)
    return (error / (2.0 * torch.exp(log_variance)) + 0.5 * log_variance).mean()


def save_uncertainty_checkpoint(
    path: Path, model: ContactUncertaintyEstimator, metadata: Mapping[str, Any]
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema": "tactile3d-unit.vac-c4-uncertainty.v1",
            "hidden": model.hidden,
            "log_variance_min": model.log_variance_min,
            "log_variance_max": model.log_variance_max,
            "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "state_dict_sha256": state_dict_digest(model),
            "metadata": dict(metadata),
        },
        temporary,
    )
    temporary.replace(path)
    return sha256_file(path)


def load_uncertainty_checkpoint(
    path: Path, device: str | torch.device = "cpu"
) -> tuple[ContactUncertaintyEstimator, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "tactile3d-unit.vac-c4-uncertainty.v1":
        raise ValueError("unsupported C4 uncertainty checkpoint")
    model = ContactUncertaintyEstimator(
        hidden=int(payload["hidden"]),
        log_variance_min=float(payload["log_variance_min"]),
        log_variance_max=float(payload["log_variance_max"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    if state_dict_digest(model) != payload.get("state_dict_sha256"):
        raise ValueError("C4 uncertainty state digest mismatch")
    return model.to(device), dict(payload.get("metadata", {}))
