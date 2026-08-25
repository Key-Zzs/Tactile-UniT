"""Lightweight per-token adaptors from S2 contact codes to the UniT shared RQ."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


AdapterArchitecture = Literal["identity", "affine", "mlp"]


class ContactCodebookAdaptor(nn.Module):
    """Apply one shared 32-to-32 mapping to every contact query token.

    The module deliberately leaves the query axis outside the learned mapping.
    PyTorch's final-dimension linear operations therefore share the exact same
    parameters across all eight query positions.
    """

    def __init__(
        self,
        architecture: AdapterArchitecture,
        *,
        token_dim: int = 32,
        hidden_dim: int = 128,
        queries: int = 8,
    ) -> None:
        super().__init__()
        if token_dim < 1 or hidden_dim < 1 or queries < 1:
            raise ValueError("token_dim, hidden_dim, and queries must be positive")
        self.architecture = architecture
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.queries = queries
        if architecture == "identity":
            self.net: nn.Module = nn.Identity()
        elif architecture == "affine":
            self.net = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim))
            self.reset_identity()
        elif architecture == "mlp":
            self.net = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, token_dim),
            )
        else:
            raise ValueError(f"unknown contact adaptor architecture: {architecture}")

    def reset_identity(self) -> None:
        """Initialize the affine candidate as an identity calibration."""

        if self.architecture != "affine":
            return
        layer = self.net[1]
        if not isinstance(layer, nn.Linear):
            raise AssertionError("affine adaptor layout changed")
        with torch.no_grad():
            nn.init.eye_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        expected = (self.queries, self.token_dim)
        if value.ndim != 3 or tuple(value.shape[1:]) != expected:
            raise ValueError(f"expected contact code [B,{self.queries},{self.token_dim}]")
        return self.net(value)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_contact_adaptor(architecture: AdapterArchitecture) -> ContactCodebookAdaptor:
    """Build a canonical S3.2 adaptor candidate."""

    return ContactCodebookAdaptor(architecture)
