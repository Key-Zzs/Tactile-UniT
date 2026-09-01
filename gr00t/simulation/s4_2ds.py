"""Fixed pilot models for S4.2-DS representation selection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalResidualBlock(nn.Module):
    """Small residual TCN block with explicit temporal dilation."""

    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Conv1d(width, width, kernel_size=3, padding=dilation, dilation=dilation)
        self.projection = nn.Linear(width, width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.conv(F.gelu(self.norm(value)).transpose(1, 2)).transpose(1, 2)
        return value + self.projection(F.gelu(residual))


class ActionPilot(nn.Module):
    """Encode a 27x22 planned action transition into eight 32D slots."""

    def __init__(self, width: int = 128) -> None:
        super().__init__()
        self.input = nn.Linear(66, width)
        self.position = nn.Parameter(torch.zeros(27, width))
        self.temporal = nn.ModuleList(
            [TemporalResidualBlock(width, dilation) for dilation in (1, 2, 4)]
        )
        self.to_token = nn.Linear(width, 32)
        self.token_norm = nn.LayerNorm(32)
        self.decoder = nn.Sequential(
            nn.Linear(8 * 32 + 22, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 27 * 22),
        )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or tuple(features.shape[1:]) != (27, 66):
            raise ValueError("Action pilot features must be [B,27,66]")
        value = self.input(features) + self.position
        for block in self.temporal:
            value = block(value)
        pooled = F.adaptive_avg_pool1d(value.transpose(1, 2), 8).transpose(1, 2)
        return self.token_norm(self.to_token(pooled))

    def decode(self, code: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
        if code.ndim != 3 or tuple(code.shape[1:]) != (8, 32):
            raise ValueError("Action pilot code must be [B,8,32]")
        if current_state.shape != (len(code), 22):
            raise ValueError("Action pilot current state must be [B,22]")
        return self.decoder(torch.cat((code.flatten(1), current_state), dim=1)).view(-1, 27, 22)

    def forward(self, features: torch.Tensor, current_state: torch.Tensor) -> dict[str, torch.Tensor]:
        code = self.encode(features)
        return {"code": code, "action": self.decode(code, current_state)}


class ResidualSlotProjector(nn.Module):
    """Independent token-wise mapping into a common VAC slot space."""

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.mapping = nn.Sequential(nn.Linear(32, width), nn.GELU(), nn.Linear(width, 32))
        self.norm = nn.LayerNorm(32)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or tuple(value.shape[1:]) != (8, 32):
            raise ValueError("native representation must be [B,8,32]")
        return self.norm(value + self.mapping(value))


class PilotVACBridge(nn.Module):
    """Equal-budget independent VAC projectors and native recovery heads."""

    modalities = ("vision", "action", "contact")

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.projectors = nn.ModuleDict(
            {name: ResidualSlotProjector(width) for name in self.modalities}
        )
        self.recovery = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(32, width), nn.GELU(), nn.Linear(width, 32)
                )
                for name in self.modalities
            }
        )

    def encode(self, modality: str, value: torch.Tensor) -> torch.Tensor:
        if modality not in self.modalities:
            raise ValueError(f"unknown modality {modality}")
        return self.projectors[modality](value)

    def recover(self, modality: str, shared: torch.Tensor) -> torch.Tensor:
        if modality not in self.modalities:
            raise ValueError(f"unknown modality {modality}")
        if shared.ndim != 3 or tuple(shared.shape[1:]) != (8, 32):
            raise ValueError("shared representation must be [B,8,32]")
        return shared + self.recovery[modality](shared)
