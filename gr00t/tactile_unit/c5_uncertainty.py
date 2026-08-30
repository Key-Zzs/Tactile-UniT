"""Mode-aware C5 uncertainty for frozen causal/runtime mean predictors."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from .c4_availability_conditioning import sha256_file
from .continuous_vac_shared_space import state_dict_digest


class C5RuntimeMode(str, Enum):
    FULL_AH = "FULL_AH"
    FALLBACK_CAUSAL_VA = "FALLBACK_CAUSAL_VA"
    FALLBACK_A = "FALLBACK_A"
    ABSTAIN_NO_ACTION = "ABSTAIN_NO_ACTION"


CALIBRATED_MODES = (C5RuntimeMode.FULL_AH, C5RuntimeMode.FALLBACK_CAUSAL_VA, C5RuntimeMode.FALLBACK_A)
MODE_TO_ID = {mode: index for index, mode in enumerate(CALIBRATED_MODES)}


class C5ContactUncertaintyEstimator(nn.Module):
    def __init__(self, hidden: int = 64, log_variance_min: float = -12.0, log_variance_max: float = 4.0):
        super().__init__()
        if hidden > 96:
            raise ValueError("C5 uncertainty exceeds bounded capacity")
        self.hidden, self.log_variance_min, self.log_variance_max = int(hidden), float(log_variance_min), float(log_variance_max)
        self.mode_embedding = nn.Embedding(3, 8)
        self.network = nn.Sequential(
            nn.Linear(32 * 8 + 8 + 1, hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1),
        )
        if self.parameter_count() > 50_000:
            raise ValueError("C5 uncertainty must be <=50k parameters")

    @staticmethod
    def _summary(value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or value.shape[-1] != 32:
            raise ValueError("uncertainty inputs must be [B,T,32]")
        return torch.cat((value.mean(1), value.std(1, unbiased=False), value.abs().amax(1), torch.sqrt(value.square().mean(1) + 1e-8)), dim=1)

    def forward(self, mode: C5RuntimeMode | str, prediction: torch.Tensor, source: torch.Tensor, plan_ood_score: torch.Tensor | None = None) -> torch.Tensor:
        mode = C5RuntimeMode(mode)
        if mode not in MODE_TO_ID:
            raise ValueError("ABSTAIN has no calibrated uncertainty")
        if len(prediction) != len(source):
            raise ValueError("unaligned uncertainty inputs")
        if plan_ood_score is None:
            plan_ood_score = prediction.new_zeros(len(prediction))
        if plan_ood_score.shape != (len(prediction),):
            raise ValueError("plan_ood_score must be [B]")
        mode_ids = torch.full((len(prediction),), MODE_TO_ID[mode], dtype=torch.long, device=prediction.device)
        features = torch.cat((self._summary(prediction), self._summary(source), self.mode_embedding(mode_ids), plan_ood_score[:, None]), dim=1)
        return self.network(features).squeeze(1).clamp(self.log_variance_min, self.log_variance_max)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class CalibratedC5Uncertainty(nn.Module):
    """Runtime view that converts frozen log-variance to calibrated variance."""

    def __init__(self, estimator: C5ContactUncertaintyEstimator, calibration_scale: float):
        super().__init__()
        if calibration_scale <= 0:
            raise ValueError("calibration_scale must be positive")
        self.estimator = estimator.eval().requires_grad_(False)
        self.register_buffer("calibration_scale", torch.tensor(float(calibration_scale)))

    def forward(self, mode, prediction, source, plan_ood_score=None):
        log_variance = self.estimator(mode, prediction, source, plan_ood_score)
        return torch.exp(log_variance) * self.calibration_scale.to(log_variance)


def heteroscedastic_nll(log_variance: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = torch.square(prediction.detach() - target.detach()).flatten(1).mean(1)
    return (error / (2.0 * torch.exp(log_variance)) + 0.5 * log_variance).mean()


def save_c5_uncertainty_checkpoint(path: Path, model: C5ContactUncertaintyEstimator, metadata: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "schema": "tactile3d-unit.vac-c5-uncertainty.v1", "hidden": model.hidden,
        "log_variance_min": model.log_variance_min, "log_variance_max": model.log_variance_max,
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "state_dict_sha256": state_dict_digest(model), "metadata": dict(metadata),
    }, temporary)
    temporary.replace(path)
    return sha256_file(path)


def load_c5_uncertainty_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[C5ContactUncertaintyEstimator, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "tactile3d-unit.vac-c5-uncertainty.v1":
        raise ValueError("unsupported C5 uncertainty checkpoint")
    model = C5ContactUncertaintyEstimator(int(payload["hidden"]), float(payload["log_variance_min"]), float(payload["log_variance_max"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    if state_dict_digest(model) != payload["state_dict_sha256"]:
        raise ValueError("C5 uncertainty state digest mismatch")
    return model.to(device), dict(payload.get("metadata", {}))
