"""Continuous S1 baselines and a T-Rex-style temporal VQ baseline.

The VQ encoder/EMA design is adapted from the sibling T-Rex repository's
MIT-licensed tactile VQ-VAE (Copyright 2026, Regents of the University of
California). It is kept independent from the proposed continuous teacher.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class FutureRegressor(nn.Module):
    """Common interface for future-predictive continuous baselines."""

    latent_dim: int
    future_steps: int
    input_dim: int

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, history: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encode(history)
        future = self.future_head(latent).view(-1, self.future_steps, self.input_dim)
        return {"latent": latent, "future": future}


class CurrentMLP(FutureRegressor):
    """B0: encode only the anchor frame."""

    def __init__(self, input_dim: int = 60, latent_dim: int = 256, future_steps: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.future_steps = future_steps
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 384),
            nn.GELU(),
            nn.LayerNorm(384),
            nn.Linear(384, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.future_head = nn.Linear(latent_dim, future_steps * input_dim)

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        return self.encoder(history[:, -1])


class FlattenedHistoryMLP(FutureRegressor):
    """B1: encode the complete fixed-length history as one vector."""

    def __init__(
        self,
        input_dim: int = 60,
        history_steps: int = 16,
        latent_dim: int = 256,
        future_steps: int = 8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.history_steps = history_steps
        self.latent_dim = latent_dim
        self.future_steps = future_steps
        self.encoder = nn.Sequential(
            nn.Linear(history_steps * input_dim, 768),
            nn.GELU(),
            nn.LayerNorm(768),
            nn.Linear(768, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.future_head = nn.Linear(latent_dim, future_steps * input_dim)

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        return self.encoder(history.flatten(1))


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation),
            nn.GroupNorm(8, channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.gelu(value + self.net(value))


class TemporalCNN(FutureRegressor):
    """B2: order-sensitive dilated temporal convolution baseline."""

    def __init__(
        self,
        input_dim: int = 60,
        latent_dim: int = 256,
        future_steps: int = 8,
        channels: int = 192,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.future_steps = future_steps
        self.stem = nn.Conv1d(input_dim, channels, 3, padding=1)
        self.temporal = nn.Sequential(
            ResidualTemporalBlock(channels, 1),
            ResidualTemporalBlock(channels, 2),
            ResidualTemporalBlock(channels, 4),
        )
        self.project = nn.Sequential(
            nn.Linear(2 * channels, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.future_head = nn.Linear(latent_dim, future_steps * input_dim)

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        features = self.temporal(self.stem(history.transpose(1, 2)))
        pooled = torch.cat([features.mean(dim=-1), features[:, :, -1]], dim=-1)
        return self.project(pooled)


class PredictiveContactTeacher(nn.Module):
    """S1.3 continuous teacher: residual TCN plus learned query pooling."""

    def __init__(
        self,
        input_dim: int = 60,
        history_steps: int = 16,
        future_steps: int = 8,
        latent_dim: int = 256,
        channels: int = 256,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.latent_dim = latent_dim
        # First differences are deterministic features of the same 60-D wrench history.
        self.stem = nn.Conv1d(2 * input_dim, channels, 3, padding=1)
        self.position = nn.Parameter(torch.randn(1, channels, history_steps) * 0.02)
        self.temporal = nn.Sequential(
            ResidualTemporalBlock(channels, 1),
            ResidualTemporalBlock(channels, 2),
            ResidualTemporalBlock(channels, 4),
            ResidualTemporalBlock(channels, 8),
        )
        self.query = nn.Parameter(torch.randn(channels) * channels**-0.5)
        self.project = nn.Sequential(
            nn.Linear(2 * channels, 384),
            nn.GELU(),
            nn.LayerNorm(384),
            nn.Linear(384, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.history_head = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.Linear(512, history_steps * input_dim),
        )
        self.future_head = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.Linear(512, future_steps * input_dim),
        )

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        if history.shape[1:] != (self.history_steps, self.input_dim):
            raise ValueError(
                f"expected history [B,{self.history_steps},{self.input_dim}], "
                f"got {tuple(history.shape)}"
            )
        delta = torch.diff(history, dim=1, prepend=history[:, :1])
        value = torch.cat([history, delta], dim=-1).transpose(1, 2)
        features = self.temporal(self.stem(value) + self.position)
        scores = torch.einsum("bct,c->bt", features, self.query) / features.shape[1] ** 0.5
        weights = scores.softmax(dim=-1)
        queried = torch.einsum("bct,bt->bc", features, weights)
        pooled = torch.cat([queried, features[:, :, -1]], dim=-1)
        return self.project(pooled)

    def forward(self, history: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encode(history)
        reconstruction = self.history_head(latent).view(
            -1, self.history_steps, self.input_dim
        )
        future = self.future_head(latent).view(-1, self.future_steps, self.input_dim)
        return {"latent": latent, "reconstruction": reconstruction, "future": future}


class VQEMA(nn.Module):
    """EMA codebook with periodic dead-code revival, following T-Rex."""

    def __init__(
        self,
        codebook_size: int,
        embed_dim: int,
        commitment_weight: float = 0.25,
        decay: float = 0.99,
        revive_every: int = 200,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.commitment_weight = commitment_weight
        self.decay = decay
        self.revive_every = revive_every
        embedding = torch.randn(codebook_size, embed_dim) * 0.02
        self.register_buffer("embedding", embedding)
        self.register_buffer("embedding_sum", embedding.clone())
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def _update(self, inputs: torch.Tensor, indices: torch.Tensor) -> None:
        assignments = F.one_hot(indices, self.codebook_size).to(inputs.dtype)
        counts = assignments.sum(0)
        sums = assignments.transpose(0, 1) @ inputs
        self.cluster_size.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
        self.embedding_sum.mul_(self.decay).add_(sums, alpha=1.0 - self.decay)
        total = self.cluster_size.sum()
        smooth = (self.cluster_size + 1e-5) / (total + self.codebook_size * 1e-5) * total
        self.embedding.copy_(self.embedding_sum / smooth.clamp_min(1e-5).unsqueeze(1))
        self.updates.add_(1)
        if int(self.updates.item()) % self.revive_every == 0:
            dead = self.cluster_size < 1.0
            count = int(dead.sum().item())
            if count:
                selected = torch.randint(len(inputs), (count,), device=inputs.device)
                replacement = inputs[selected]
                self.embedding[dead] = replacement
                self.embedding_sum[dead] = replacement
                self.cluster_size[dead] = 2.0

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distances = (
            inputs.square().sum(1, keepdim=True)
            + self.embedding.square().sum(1)
            - 2.0 * inputs @ self.embedding.transpose(0, 1)
        )
        indices = distances.argmin(1)
        quantized = self.embedding[indices]
        if self.training:
            self._update(inputs.detach(), indices)
        commitment = self.commitment_weight * F.mse_loss(inputs, quantized.detach())
        straight_through = inputs + (quantized - inputs).detach()
        return straight_through, indices, commitment


@dataclass(frozen=True)
class VQConfig:
    input_dim: int = 60
    history_steps: int = 16
    latent_dim: int = 256
    future_steps: int = 8
    fingers: int = 10
    finger_dim: int = 6
    codebook_size: int = 64


class TactileVQBaseline(nn.Module):
    """B3: shared per-finger temporal VQ encoder with one code per finger."""

    def __init__(self, config: VQConfig = VQConfig()):
        super().__init__()
        self.config = config
        if config.fingers * config.finger_dim != config.input_dim:
            raise ValueError("fingers * finger_dim must equal input_dim")
        self.stem = nn.Sequential(
            nn.Conv1d(config.finger_dim, 128, 5, padding=2),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv1d(128, 128, 5, stride=2, padding=2),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv1d(128, config.latent_dim, 5, stride=2, padding=2),
            nn.GroupNorm(8, config.latent_dim),
            nn.GELU(),
        )
        self.finger_embedding = nn.Embedding(config.fingers, 128)
        self.quantizer = VQEMA(config.codebook_size, config.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, 512),
            nn.GELU(),
            nn.Linear(512, config.history_steps * config.finger_dim),
        )
        self.pool = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
        )
        self.future_head = nn.Linear(
            config.latent_dim, config.future_steps * config.input_dim
        )

    def _encode_fingers(
        self, history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, steps, _ = history.shape
        cfg = self.config
        fingers = history.view(batch, steps, cfg.fingers, cfg.finger_dim)
        value = fingers.permute(0, 2, 3, 1).reshape(
            batch * cfg.fingers, cfg.finger_dim, steps
        )
        stem = self.stem[0](value)
        ids = torch.arange(cfg.fingers, device=history.device).repeat(batch)
        stem = stem + self.finger_embedding(ids).unsqueeze(-1)
        encoded = self.stem[1:](stem).mean(-1)
        quantized, indices, commitment = self.quantizer(encoded)
        return quantized.view(batch, cfg.fingers, -1), indices.view(batch, -1), commitment

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        quantized, _, _ = self._encode_fingers(history)
        return self.pool(quantized.mean(1))

    def forward(self, history: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = history.shape[0]
        cfg = self.config
        quantized, indices, commitment = self._encode_fingers(history)
        decoded = self.decoder(quantized).view(
            batch, cfg.fingers, cfg.history_steps, cfg.finger_dim
        )
        reconstruction = decoded.permute(0, 2, 1, 3).reshape(
            batch, cfg.history_steps, cfg.input_dim
        )
        latent = self.pool(quantized.mean(1))
        future = self.future_head(latent).view(batch, cfg.future_steps, cfg.input_dim)
        return {
            "latent": latent,
            "future": future,
            "reconstruction": reconstruction,
            "indices": indices,
            "commitment": commitment,
        }


def build_baseline(name: str, latent_dim: int = 256) -> nn.Module:
    constructors = {
        "B0": lambda: CurrentMLP(latent_dim=latent_dim),
        "B1": lambda: FlattenedHistoryMLP(latent_dim=latent_dim),
        "B2": lambda: TemporalCNN(latent_dim=latent_dim),
        "B3": lambda: TactileVQBaseline(VQConfig(latent_dim=latent_dim)),
    }
    try:
        return constructors[name]()
    except KeyError as error:
        raise ValueError(f"unknown baseline {name!r}; expected one of {sorted(constructors)}") from error
