"""Shared models, metrics, and safety guards for the S3.2-R decision tree."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import torch
import torch.nn as nn

from gr00t.contact_dynamics.evaluation import different_episode_permutation, query_diversity
from gr00t.model.tokenizer.vector_quantizer import (
    ResidualVectorQuantizer,
    ResidualVectorQuantizerConfig,
)
from gr00t.tactile_unit.compatibility import codebook_usage, effective_rank, parameter_digest


Sufficiency = Literal["STRONG_PASS", "PARTIAL", "FAIL"]
BridgeArchitecture = Literal["identity", "affine", "residual_mlp"]


def reconstruction_retention(
    e_rep: float, e_zero: float, e_shuffle: float, e_cont: float
) -> float:
    """Return the raw, deliberately unclipped reconstruction advantage retention."""

    control = min(float(e_zero), float(e_shuffle))
    denominator = control - float(e_cont)
    if denominator <= 0:
        raise ValueError("reconstruction retention requires E_control > E_cont")
    return (control - float(e_rep)) / denominator


def semantic_retention(f1_q: float, f1_cont: float, f1_majority: float) -> float:
    """Return the raw, deliberately unclipped semantic advantage retention."""

    denominator = float(f1_cont) - float(f1_majority)
    if denominator <= 0:
        raise ValueError("semantic retention requires F1_cont > F1_majority")
    return (float(f1_q) - float(f1_majority)) / denominator


def classify_sufficiency(
    r_recon: float,
    r_contact: float,
    r_force: float,
    *,
    hard_code_collapse: bool,
    query_collapse: bool,
) -> Sufficiency:
    """Apply the pre-registered S3.2-R engineering gates without clipping."""

    if hard_code_collapse or query_collapse:
        return "FAIL"
    values = (float(r_recon), float(r_contact), float(r_force))
    if all(value >= threshold for value, threshold in zip(values, (0.80, 0.90, 0.90))):
        return "STRONG_PASS"
    if all(value >= threshold for value, threshold in zip(values, (0.60, 0.75, 0.75))):
        return "PARTIAL"
    return "FAIL"


def build_contact_rq(*, stages: int = 2, codes: int = 128, dim: int = 32) -> ResidualVectorQuantizer:
    """Build the repository-native RQ used for the private Contact ceiling."""

    if stages < 1 or codes < 2 or dim < 1:
        raise ValueError("invalid Contact RQ geometry")
    config = ResidualVectorQuantizerConfig(
        stages=[
            {
                "n_e": codes,
                "e_dim": dim,
                "beta": 0.25,
                "legacy": True,
                "code_restart": True,
                "restart_interval": 100,
                "max_restart_steps": 50000,
                "l2_norm": False,
            }
            for _ in range(stages)
        ]
    )
    return ResidualVectorQuantizer(config)


class ContactDecoderBridge(nn.Module):
    """A shared per-query q_c -> native-token bridge with no side inputs."""

    def __init__(
        self,
        architecture: BridgeArchitecture,
        *,
        token_dim: int = 32,
        queries: int = 8,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.token_dim = int(token_dim)
        self.queries = int(queries)
        if architecture == "identity":
            self.net: nn.Module = nn.Identity()
        elif architecture == "affine":
            layer = nn.Linear(token_dim, token_dim)
            nn.init.eye_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.net = layer
        elif architecture == "residual_mlp":
            self.net = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, token_dim),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        else:
            raise ValueError(f"unknown decoder bridge architecture: {architecture}")

    def forward(self, q_c: torch.Tensor) -> torch.Tensor:
        if q_c.ndim != 3 or tuple(q_c.shape[1:]) != (self.queries, self.token_dim):
            raise ValueError(f"expected q_c [B,{self.queries},{self.token_dim}]")
        value = self.net(q_c)
        return q_c + value if self.architecture == "residual_mlp" else value

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def assert_disjoint_splits(*splits: Iterable[int]) -> None:
    """Reject episode leakage across Contact partitions."""

    sets = [set(map(int, values)) for values in splits]
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            overlap = sets[left] & sets[right]
            if overlap:
                raise ValueError(f"Contact episode leakage: {len(overlap)} overlapping IDs")


def select_gr1_rehearsal_episodes(
    ordered_episode_ids: Iterable[int], *, held_out_count: int = 10
) -> tuple[list[int], list[int]]:
    """Return official-training and canonical-T4 episode IDs from metadata order."""

    values = [int(value) for value in ordered_episode_ids]
    if held_out_count < 1 or len(values) <= held_out_count:
        raise ValueError("insufficient GR1 episodes for rehearsal/T4 separation")
    train, held_out = values[:-held_out_count], values[-held_out_count:]
    if set(train) & set(held_out):
        raise ValueError("GR1 rehearsal overlaps canonical T4 episodes")
    return train, held_out


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    """Linear CKA for paired [N,D] representations, allowing unequal D."""

    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError("linear CKA requires paired [N,D] matrices")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = x.T @ y
    numerator = float(np.square(cross).sum())
    denominator = float(np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y))
    return numerator / max(denominator, 1e-12)


def representation_structure(values: np.ndarray) -> dict[str, object]:
    """Summarize held-out relational and query structure consistently."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("representation structure requires [N,Q,D]")
    flattened = array.reshape(len(array), -1)
    pooled = array.reshape(-1, array.shape[-1])
    flat_rank, _ = effective_rank(flattened)
    variance = pooled.var(axis=0)
    diversity = query_diversity(array)
    return {
        "flattened_effective_rank": float(flat_rank),
        "per_dimension_variance": variance.tolist(),
        "per_dimension_variance_mean": float(variance.mean()),
        "near_zero_variance_fraction": float(np.mean(variance < 1e-8)),
        **diversity,
    }


def collapse_diagnostics(
    indices: np.ndarray, values: np.ndarray, *, codebook_size: int
) -> dict[str, object]:
    """Apply explicit hard-code and query-collapse diagnostics."""

    index_array = np.asarray(indices)
    if index_array.ndim != 3:
        raise ValueError("indices must be [N,Q,S]")
    usage = [
        {"stage": stage, **codebook_usage(index_array[..., stage], codebook_size)}
        for stage in range(index_array.shape[-1])
    ]
    structure = representation_structure(values)
    hard = any(row["active_codes"] <= 1 or row["top1_frequency"] >= 0.90 for row in usage)
    query = bool(
        structure["collapsed_sample_fraction"] > 0
        or structure["near_zero_variance_fraction"] >= 0.90
    )
    return {
        "usage": usage,
        "structure": structure,
        "hard_code_collapse": bool(hard),
        "query_collapse": query,
    }


@dataclass(frozen=True)
class FrozenDigestGuard:
    """Capture and later verify module state identity."""

    digests: dict[str, str]

    @classmethod
    def capture(cls, **modules: nn.Module) -> "FrozenDigestGuard":
        return cls({name: parameter_digest(module) for name, module in modules.items()})

    def verify(self, **modules: nn.Module) -> dict[str, dict[str, object]]:
        if set(modules) != set(self.digests):
            raise ValueError("frozen component set changed")
        result = {}
        for name, module in modules.items():
            after = parameter_digest(module)
            result[name] = {
                "before": self.digests[name],
                "after": after,
                "unchanged": after == self.digests[name],
            }
        if not all(bool(row["unchanged"]) for row in result.values()):
            raise RuntimeError("frozen component identity changed")
        return result


def deterministic_different_episode_shuffle(episode_ids: np.ndarray, seed: int = 42) -> np.ndarray:
    """Public alias used by every S3.2-R stage and its tests."""

    return different_episode_permutation(episode_ids, seed=seed)
