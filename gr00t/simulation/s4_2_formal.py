"""Formal S4.2-4--8 models with explicit offline VAC interfaces."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gr00t.simulation.s4_2ds import TemporalResidualBlock


class FormalActionEncoder(nn.Module):
    """One of the three frozen action candidates, always producing [B,8,32]."""

    def __init__(self, candidate: str, width: int = 128) -> None:
        super().__init__()
        if candidate not in {"A0", "A1", "A2"}:
            raise ValueError(f"unknown Action candidate {candidate}")
        self.candidate = candidate
        self.width = width
        if candidate == "A0":
            self.temporal = nn.Sequential(
                nn.Flatten(),
                nn.Linear(27 * 66, 512),
                nn.GELU(),
                nn.LayerNorm(512),
                nn.Linear(512, 8 * 32),
            )
        else:
            self.input = nn.Linear(66, width)
            self.position = nn.Parameter(torch.zeros(27, width))
            if candidate == "A1":
                self.temporal = nn.ModuleList(
                    [TemporalResidualBlock(width, dilation) for dilation in (1, 2, 4)]
                )
            else:
                layer = nn.TransformerEncoderLayer(
                    width,
                    4,
                    4 * width,
                    dropout=0.0,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
                self.temporal = nn.TransformerEncoder(layer, num_layers=2)
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
            raise ValueError("Action features must be [B,27,66]")
        if self.candidate == "A0":
            return self.token_norm(self.temporal(features).view(-1, 8, 32))
        value = self.input(features) + self.position
        if self.candidate == "A1":
            for block in self.temporal:
                value = block(value)
        else:
            value = self.temporal(value)
        value = F.adaptive_avg_pool1d(value.transpose(1, 2), 8).transpose(1, 2)
        return self.token_norm(self.to_token(value))

    def decode(self, code: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
        if code.ndim != 3 or tuple(code.shape[1:]) != (8, 32):
            raise ValueError("Action code must be [B,8,32]")
        if current_state.shape != (len(code), 22):
            raise ValueError("current state must be [B,22]")
        return self.decoder(torch.cat((code.flatten(1), current_state), dim=1)).view(-1, 27, 22)

    def forward(
        self, features: torch.Tensor, current_state: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        code = self.encode(features)
        return {"code": code, "action": self.decode(code, current_state)}


class VACAdapter(nn.Module):
    """Independent modality adapter; no counterpart modality is an input."""

    def __init__(self, kind: str, width: int = 64) -> None:
        super().__init__()
        self.kind = kind
        if kind == "affine":
            self.mapping = nn.Linear(32, 32)
        elif kind == "mlp":
            self.mapping = nn.Sequential(nn.Linear(32, width), nn.GELU(), nn.Linear(width, 32))
        elif kind == "slot":
            self.attention = nn.MultiheadAttention(32, 4, batch_first=True)
            self.mapping = nn.Sequential(nn.Linear(32, width), nn.GELU(), nn.Linear(width, 32))
        else:
            raise ValueError(f"unknown VAC adapter {kind}")
        self.norm = nn.LayerNorm(32)

    def forward(self, native: torch.Tensor) -> torch.Tensor:
        if native.ndim != 3 or tuple(native.shape[1:]) != (8, 32):
            raise ValueError("native VAC representation must be [B,8,32]")
        value = native
        if self.kind == "slot":
            attended, _ = self.attention(value, value, value, need_weights=False)
            value = value + attended
        return self.norm(value + self.mapping(value))


class FormalVACBridge(nn.Module):
    """Continuous independent VAC encoders plus native recovery heads."""

    modalities = ("vision", "action", "contact")

    def __init__(self, adapter: str, width: int = 64) -> None:
        super().__init__()
        self.adapter = adapter
        self.projectors = nn.ModuleDict(
            {name: VACAdapter(adapter, width) for name in self.modalities}
        )
        self.recovery = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(32, width), nn.GELU(), nn.Linear(width, 32))
                for name in self.modalities
            }
        )

    def encode(self, modality: str, native: torch.Tensor) -> torch.Tensor:
        if modality not in self.modalities:
            raise ValueError(f"unknown modality {modality}")
        return self.projectors[modality](native)

    def recover(self, modality: str, shared: torch.Tensor) -> torch.Tensor:
        if modality not in self.modalities:
            raise ValueError(f"unknown modality {modality}")
        if shared.ndim != 3 or tuple(shared.shape[1:]) != (8, 32):
            raise ValueError("shared VAC representation must be [B,8,32]")
        return shared + self.recovery[modality](shared)


class SharedPrivateDecomposer(nn.Module):
    """Decompose native modality tokens into shared-aligned and private residual codes."""

    modalities = ("vision", "action", "contact")

    def __init__(self, private_dim: int = 16) -> None:
        super().__init__()
        self.private_dim = private_dim
        self.private = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(64, 48), nn.GELU(), nn.Linear(48, private_dim))
                for name in self.modalities
            }
        )
        self.native_decoder = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(32 + private_dim, 64), nn.GELU(), nn.Linear(64, 32))
                for name in self.modalities
            }
        )
        self.cross_decoder = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
                for name in self.modalities
            }
        )

    def encode_private(
        self, modality: str, native: torch.Tensor, shared: torch.Tensor
    ) -> torch.Tensor:
        return self.private[modality](torch.cat((native, shared), dim=-1))

    def reconstruct(
        self, modality: str, shared: torch.Tensor, private: torch.Tensor
    ) -> torch.Tensor:
        return self.native_decoder[modality](torch.cat((shared, private), dim=-1))

    def cross_predict(self, modality: str, shared: torch.Tensor) -> torch.Tensor:
        return self.cross_decoder[modality](shared)


class ConditionalContactPredictor(nn.Module):
    """Predict shared Contact transition tokens from an explicit conditioning set."""

    def __init__(self, modalities: tuple[str, ...], hidden: int = 384) -> None:
        super().__init__()
        allowed = {"vision", "action", "history"}
        if not modalities or not set(modalities) <= allowed:
            raise ValueError("invalid conditional modalities")
        self.modalities = modalities
        input_dim = 256 * len(modalities)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 256),
        )

    def forward(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        missing = set(self.modalities) - set(values)
        if missing:
            raise ValueError(f"missing conditioning modalities: {sorted(missing)}")
        flat = [values[name].reshape(len(values[name]), 256) for name in self.modalities]
        return self.network(torch.cat(flat, dim=1)).view(-1, 8, 32)


class ScalarLogVarianceHead(nn.Module):
    """Heteroscedastic scalar uncertainty head for a frozen mean predictor."""

    def __init__(self, modalities: tuple[str, ...], hidden: int = 128) -> None:
        super().__init__()
        self.modalities = modalities
        self.network = nn.Sequential(
            nn.Linear(256 * len(modalities), hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        flat = [values[name].reshape(len(values[name]), 256) for name in self.modalities]
        return self.network(torch.cat(flat, dim=1)).squeeze(1).clamp(-10.0, 8.0)
