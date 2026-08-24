"""Continuous S2 contact-transition encoders and latent decoder."""

from __future__ import annotations

import torch
import torch.nn as nn


def _validate_pair(current: torch.Tensor, future: torch.Tensor, latent_dim: int) -> None:
    if current.ndim != 2 or future.shape != current.shape or current.shape[1] != latent_dim:
        raise ValueError(
            f"expected current/future [B,{latent_dim}], got {tuple(current.shape)} and "
            f"{tuple(future.shape)}"
        )


class ContactDynamicsEncoder(nn.Module):
    """Encode explicit current/future/delta views into eight continuous tokens."""

    def __init__(
        self,
        latent_dim: int = 256,
        queries: int = 8,
        token_dim: int = 32,
        hidden_dim: int = 384,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.queries = queries
        self.token_dim = token_dim
        self.current_projection = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.GELU(), nn.LayerNorm(256)
        )
        self.future_projection = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.GELU(), nn.LayerNorm(256)
        )
        self.delta_projection = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.GELU(), nn.LayerNorm(256)
        )
        self.transition = nn.Sequential(
            nn.Linear(3 * 256, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, queries * token_dim),
        )
        self.query_bias = nn.Parameter(torch.randn(queries, token_dim) * 0.02)
        self.output_norm = nn.LayerNorm(token_dim)

    def forward(self, current: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        _validate_pair(current, future, self.latent_dim)
        value = torch.cat(
            [
                self.current_projection(current),
                self.future_projection(future),
                self.delta_projection(future - current),
            ],
            dim=-1,
        )
        tokens = self.transition(value).view(-1, self.queries, self.token_dim)
        return self.output_norm(tokens + self.query_bias)


class DeltaMLPEncoder(nn.Module):
    """C2 matched-capacity compressed delta MLP baseline."""

    def __init__(
        self,
        latent_dim: int = 256,
        queries: int = 8,
        token_dim: int = 32,
        hidden_dim: int = 384,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.queries = queries
        self.token_dim = token_dim
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, queries * token_dim),
        )
        self.output_norm = nn.LayerNorm(token_dim)

    def forward(self, current: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        _validate_pair(current, future, self.latent_dim)
        value = torch.cat([current, future - current], dim=-1)
        return self.output_norm(
            self.net(value).view(-1, self.queries, self.token_dim)
        )


class LatentTransitionDecoder(nn.Module):
    """Decode transition tokens conditioned on the current contact state."""

    def __init__(
        self,
        latent_dim: int = 256,
        queries: int = 8,
        token_dim: int = 32,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.queries = queries
        self.token_dim = token_dim
        self.net = nn.Sequential(
            nn.Linear(queries * token_dim + latent_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, tokens: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1:] != (self.queries, self.token_dim):
            raise ValueError(
                f"expected tokens [B,{self.queries},{self.token_dim}], got {tuple(tokens.shape)}"
            )
        if current.shape != (len(tokens), self.latent_dim):
            raise ValueError(f"expected current [B,{self.latent_dim}]")
        delta = self.net(torch.cat([tokens.flatten(1), current], dim=-1))
        return current + delta


class ContactDynamicsModel(nn.Module):
    """Pair an S2 transition encoder with the shared decoder interface."""

    def __init__(self, encoder: nn.Module, decoder: LatentTransitionDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, current: torch.Tensor, future: torch.Tensor) -> dict[str, torch.Tensor]:
        code = self.encoder(current, future)
        prediction = self.decoder(code, current)
        return {"code": code, "future": prediction}


class CurrentOnlyPredictor(nn.Module):
    """C1: predict the future Teacher latent from the current latent only."""

    def __init__(self, latent_dim: int = 256, hidden_dim: int = 512) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        if current.ndim != 2 or current.shape[1] != self.latent_dim:
            raise ValueError(f"expected current [B,{self.latent_dim}]")
        return current + self.net(current)
