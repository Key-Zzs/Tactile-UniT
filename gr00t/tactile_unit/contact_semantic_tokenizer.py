"""Contact-native semantic tokenizers and S3.2-Q protocol utilities.

The classes in this module deliberately do not import or optimize the Original
UniT codebook.  They operate on the accepted continuous Contact transition
latent ``z_c`` with shape ``[B, 8, 32]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import torch
import torch.nn as nn

from gr00t.tactile_unit.s3_2_r import build_contact_rq


WhiteningKind = Literal["pca", "zca"]
InterfaceType = Literal["SINGLE_SEMANTIC", "SEMANTIC_PLUS_PRIVATE", "CONTINUOUS"]


def nominal_index_bitrate(*, queries: int, stages: int, codes: int) -> float:
    """Return the nominal fixed-width index rate in bits per transition."""

    if queries < 1 or stages < 1 or codes < 2:
        raise ValueError("queries/stages must be positive and codes must be at least two")
    return float(queries * stages * np.log2(codes))


def reconstruction_retention(e_rep: float, e_control: float, e_cont: float) -> float:
    """Return raw reconstruction advantage retention without display clipping."""

    denominator = float(e_control) - float(e_cont)
    if denominator <= 0:
        raise ValueError("reconstruction retention requires E_control > E_cont")
    return (float(e_control) - float(e_rep)) / denominator


def semantic_retention(f1_rep: float, f1_cont: float, f1_majority: float) -> float:
    """Return raw semantic advantage retention without display clipping."""

    denominator = float(f1_cont) - float(f1_majority)
    if denominator <= 0:
        raise ValueError("semantic retention requires F1_cont > F1_majority")
    return (float(f1_rep) - float(f1_majority)) / denominator


def assert_episode_disjoint(*splits: Iterable[int]) -> None:
    """Reject any episode overlap between train, validation, and test."""

    episode_sets = [set(map(int, split)) for split in splits]
    for left in range(len(episode_sets)):
        for right in range(left + 1, len(episode_sets)):
            overlap = episode_sets[left] & episode_sets[right]
            if overlap:
                raise ValueError(f"split leakage: {len(overlap)} overlapping episodes")


def deterministic_different_episode_permutation(
    episode_ids: np.ndarray, seed: int = 42
) -> np.ndarray:
    """Construct a deterministic permutation with no same-episode matches."""

    values = np.asarray(episode_ids)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("episode IDs must be a non-trivial vector")
    order = np.argsort(values, kind="stable")
    _, counts = np.unique(values[order], return_counts=True)
    if len(counts) < 2 or int(counts.max()) * 2 > len(values):
        raise ValueError("cannot construct an all-different episode permutation")
    # Seed is part of the frozen protocol even though the block rotation is deterministic.
    _ = int(seed)
    candidate = np.roll(order, int(counts.max()))
    permutation = np.empty(len(values), dtype=np.int64)
    permutation[order] = candidate
    if not np.all(values[permutation] != values):
        raise AssertionError("different-episode permutation construction failed")
    return permutation


def same_episode_horizon_links(
    episode_ids: np.ndarray, anchor_frames: np.ndarray, offset_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic source/target row links for an exact future offset."""

    episodes = np.asarray(episode_ids)
    anchors = np.asarray(anchor_frames)
    if episodes.shape != anchors.shape or episodes.ndim != 1 or offset_frames < 1:
        raise ValueError("invalid episode/anchor arrays or offset")
    lookup: dict[tuple[int, int], int] = {}
    for index, (episode, anchor) in enumerate(zip(episodes, anchors)):
        key = (int(episode), int(anchor))
        if key in lookup:
            raise ValueError("duplicate episode/anchor rows are not deterministic")
        lookup[key] = index
    source: list[int] = []
    target: list[int] = []
    for index, (episode, anchor) in enumerate(zip(episodes, anchors)):
        match = lookup.get((int(episode), int(anchor) + int(offset_frames)))
        if match is not None:
            source.append(index)
            target.append(match)
    source_array = np.asarray(source, dtype=np.int64)
    target_array = np.asarray(target, dtype=np.int64)
    if len(source_array) and not np.all(episodes[source_array] == episodes[target_array]):
        raise AssertionError("multi-horizon link crossed episodes")
    return source_array, target_array


@dataclass(frozen=True)
class WhiteningStatistics:
    """Train-only affine whitening transform shared across Contact queries."""

    mean: np.ndarray
    transform: np.ndarray
    inverse: np.ndarray
    eigenvalues: np.ndarray
    kind: WhiteningKind
    regularization: float

    @classmethod
    def fit(
        cls,
        train_z: np.ndarray,
        *,
        kind: WhiteningKind,
        regularization: float,
    ) -> "WhiteningStatistics":
        values = np.asarray(train_z)
        if values.ndim != 3 or values.shape[1:] != (8, 32):
            raise ValueError("whitening fit requires train z_c [N,8,32]")
        if kind not in ("pca", "zca") or regularization <= 0:
            raise ValueError("invalid whitening kind or regularization")
        pooled = values.reshape(-1, 32).astype(np.float64)
        mean = pooled.mean(axis=0)
        centered = pooled - mean
        covariance = centered.T @ centered / max(len(centered) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]
        scale = 1.0 / np.sqrt(eigenvalues + float(regularization))
        inverse_scale = np.sqrt(eigenvalues + float(regularization))
        if kind == "pca":
            transform = eigenvectors * scale[None, :]
            inverse = inverse_scale[:, None] * eigenvectors.T
        else:
            transform = (eigenvectors * scale[None, :]) @ eigenvectors.T
            inverse = (eigenvectors * inverse_scale[None, :]) @ eigenvectors.T
        return cls(
            mean=mean.astype(np.float32),
            transform=transform.astype(np.float32),
            inverse=inverse.astype(np.float32),
            eigenvalues=eigenvalues.astype(np.float32),
            kind=kind,
            regularization=float(regularization),
        )

    def whiten_numpy(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        return (array - self.mean) @ self.transform

    def inverse_numpy(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        return array @ self.inverse + self.mean

    def inverse_consistency_error(self, values: np.ndarray) -> float:
        reconstructed = self.inverse_numpy(self.whiten_numpy(values))
        return float(np.max(np.abs(reconstructed - np.asarray(values))))


class FrozenWhitening(nn.Module):
    """Torch wrapper around fixed train-derived whitening statistics."""

    def __init__(self, statistics: WhiteningStatistics) -> None:
        super().__init__()
        self.kind = statistics.kind
        self.regularization = statistics.regularization
        self.register_buffer("mean", torch.from_numpy(statistics.mean.copy()))
        self.register_buffer("transform", torch.from_numpy(statistics.transform.copy()))
        self.register_buffer("inverse_transform", torch.from_numpy(statistics.inverse.copy()))

    def whiten(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) @ self.transform

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        return values @ self.inverse_transform + self.mean


class ResidualQueryMapper(nn.Module):
    """Shared per-query residual MLP initialized as an identity map."""

    def __init__(self, token_dim: int = 32, hidden_dim: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1:] != (8, 32):
            raise ValueError("Contact mapper requires [B,8,32]")
        return values + self.net(values)


class ContactHorizonDecoder(nn.Module):
    """Predict a future Contact-state latent from tokens and the current state."""

    def __init__(self, hidden_dim: int = 384) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8 * 32 + 256, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
        )

    def forward(self, tokens: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1:] != (8, 32):
            raise ValueError("Contact horizon decoder requires tokens [B,8,32]")
        if current.shape != (len(tokens), 256):
            raise ValueError("Contact horizon decoder requires current [B,256]")
        return current + self.net(torch.cat([tokens.flatten(1), current], dim=1))


class ContactSemanticTokenizer(nn.Module):
    """Single semantic stream, optionally with a Contact-private residual stream."""

    def __init__(
        self,
        *,
        semantic_stages: int,
        codes: int = 128,
        whitening: WhiteningStatistics | None = None,
        private_stages: int = 0,
        hidden_dim: int = 96,
    ) -> None:
        super().__init__()
        if semantic_stages < 1 or private_stages not in (0, 1):
            raise ValueError("semantic stages must be positive; private stages must be zero or one")
        self.semantic_stages = int(semantic_stages)
        self.private_stages = int(private_stages)
        self.codes = int(codes)
        self.whitening = FrozenWhitening(whitening) if whitening is not None else None
        self.semantic_encoder = ResidualQueryMapper(hidden_dim=hidden_dim)
        self.semantic_quantizer = build_contact_rq(stages=semantic_stages, codes=codes)
        self.semantic_decoder = ResidualQueryMapper(hidden_dim=hidden_dim)
        self.private_quantizer = (
            build_contact_rq(stages=private_stages, codes=codes) if private_stages else None
        )
        self.horizon_decoders = nn.ModuleDict(
            {str(horizon): ContactHorizonDecoder() for horizon in (16, 24, 32)}
        )

    @property
    def interface_type(self) -> InterfaceType:
        return "SEMANTIC_PLUS_PRIVATE" if self.private_quantizer is not None else "SINGLE_SEMANTIC"

    @property
    def semantic_bits(self) -> float:
        return nominal_index_bitrate(queries=8, stages=self.semantic_stages, codes=self.codes)

    @property
    def private_bits(self) -> float:
        if self.private_quantizer is None:
            return 0.0
        return nominal_index_bitrate(queries=8, stages=self.private_stages, codes=self.codes)

    def _preprocess(self, z_c: torch.Tensor) -> torch.Tensor:
        return self.whitening.whiten(z_c) if self.whitening is not None else z_c

    def _postprocess(self, values: torch.Tensor) -> torch.Tensor:
        return self.whitening.inverse(values) if self.whitening is not None else values

    def semantic_forward(self, z_c: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.semantic_encoder(self._preprocess(z_c))
        semantic, semantic_indices, semantic_vq_loss = self.semantic_quantizer(encoded)
        semantic_native = self._postprocess(self.semantic_decoder(semantic))
        return {
            "semantic_prequantized": encoded,
            "semantic": semantic,
            "semantic_indices": semantic_indices,
            "semantic_vq_loss": semantic_vq_loss,
            "semantic_native": semantic_native,
        }

    def forward(self, z_c: torch.Tensor) -> dict[str, torch.Tensor]:
        if z_c.ndim != 3 or z_c.shape[1:] != (8, 32):
            raise ValueError("Contact tokenizer requires z_c [B,8,32]")
        result = self.semantic_forward(z_c)
        if self.private_quantizer is None:
            result["full_native"] = result["semantic_native"]
            return result
        residual = z_c - result["semantic_native"].detach()
        private, private_indices, private_vq_loss = self.private_quantizer(residual)
        result.update(
            {
                "private_prequantized": residual,
                "private": private,
                "private_indices": private_indices,
                "private_vq_loss": private_vq_loss,
                "full_native": result["semantic_native"] + private,
            }
        )
        return result

    def predict_horizon(
        self, tokens_native: torch.Tensor, current: torch.Tensor, horizon: int
    ) -> torch.Tensor:
        """Decode one preregistered future horizon from a native-space representation."""

        key = str(int(horizon))
        if key not in self.horizon_decoders:
            raise ValueError(f"unsupported Contact prediction horizon: {horizon}")
        return self.horizon_decoders[key](tokens_native, current)


def classify_single_stream(
    *,
    r_recon: float,
    r_contact: float,
    r_force: float,
    rare_boundary_pass: bool,
    temporal_controls_pass: bool,
    collapse: bool,
) -> bool:
    """Apply the preregistered complete single-stream engineering gate."""

    return bool(
        r_recon >= 0.80
        and r_contact >= 0.90
        and r_force >= 0.90
        and rare_boundary_pass
        and temporal_controls_pass
        and not collapse
    )


def classify_shared_private(
    *,
    semantic_r_contact: float,
    semantic_r_force: float,
    full_r_recon: float,
    rare_boundary_pass: bool,
    temporal_controls_pass: bool,
    bypass: bool,
    collapse: bool,
) -> bool:
    """Apply the preregistered hierarchical semantic/private engineering gate."""

    return bool(
        semantic_r_contact >= 0.90
        and semantic_r_force >= 0.90
        and full_r_recon >= 0.80
        and rare_boundary_pass
        and temporal_controls_pass
        and not bypass
        and not collapse
    )


def private_stream_bypass(
    *,
    private_only_error: float,
    full_error: float,
    semantic_zero_error: float,
    near_full_ratio: float = 1.05,
    minimum_relative_impact: float = 0.05,
) -> bool:
    """Detect the preregistered private-only/semantic-zero bypass pattern."""

    if full_error <= 0 or near_full_ratio < 1 or minimum_relative_impact < 0:
        raise ValueError("invalid anti-bypass inputs")
    relative_impact = (float(semantic_zero_error) - float(full_error)) / float(full_error)
    return bool(
        float(private_only_error) <= float(full_error) * float(near_full_ratio)
        and relative_impact < float(minimum_relative_impact)
    )
