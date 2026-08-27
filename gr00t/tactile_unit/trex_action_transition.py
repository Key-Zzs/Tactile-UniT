"""Transition-centered continuous action representations for T-Rex.

The public contract remains a planned 16-step joint-target chunk conditioned
on the current joint state.  Relative and velocity features are computed in
raw joint-position space and normalized with statistics fitted on the frozen
training split only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gr00t.tactile_unit.trex_action_bootstrap import (
    L2_DIM,
    QUERY_NUM,
    TREX_EMBODIMENT_ID,
    ReleasedTokenizerSource,
    TReXActionBootstrap,
)
from gr00t.tactile_unit.trex_action_data import (
    ACTION_HORIZON,
    RAW_ACTION_DIM,
    SEGMENTS,
    TReXActionCache,
)


FEATURE_MULTIPLIER = 3
TRANSITION_FEATURE_DIM = RAW_ACTION_DIM * FEATURE_MULTIPLIER


def _safe_std(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return np.where(value > 1e-8, value, 1.0)


@dataclass(frozen=True)
class TransitionFeatureStats:
    """Train-only statistics and the accepted state/action normalizer."""

    state_mean: list[float]
    state_std: list[float]
    action_mean: list[float]
    action_std: list[float]
    relative_mean: list[float]
    relative_std: list[float]
    velocity_mean: list[float]
    velocity_std: list[float]
    fit_split: str = "frozen train split only"
    velocity_t0: str = "zero sentinel after normalization"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["schema"] = "tactile3d-unit.s3-3-r-transition-feature-stats.v1"
        canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
        result["canonical_sha256"] = hashlib.sha256(canonical).hexdigest()
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TransitionFeatureStats":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value[key] for key in allowed})


def fit_transition_feature_stats(cache: TReXActionCache) -> TransitionFeatureStats:
    """Fit relative/delta moments on every window in the frozen train cache."""

    if cache.split != "train":
        raise ValueError("transition statistics may only be fitted on train")
    relative_sum = np.zeros(RAW_ACTION_DIM, dtype=np.float64)
    relative_square = np.zeros(RAW_ACTION_DIM, dtype=np.float64)
    velocity_sum = np.zeros(RAW_ACTION_DIM, dtype=np.float64)
    velocity_square = np.zeros(RAW_ACTION_DIM, dtype=np.float64)
    relative_count = 0
    velocity_count = 0
    for start in range(0, len(cache), 2048):
        stop = min(start + 2048, len(cache))
        batch = cache.batch(np.arange(start, stop, dtype=np.int64))
        state = batch["state_raw"]
        action = batch["action_raw"]
        relative = np.asarray(action - state[:, None, :], dtype=np.float64)
        velocity = np.asarray(np.diff(action, axis=1), dtype=np.float64)
        relative_sum += relative.sum(axis=(0, 1))
        relative_square += np.square(relative).sum(axis=(0, 1))
        velocity_sum += velocity.sum(axis=(0, 1))
        velocity_square += np.square(velocity).sum(axis=(0, 1))
        relative_count += relative.shape[0] * relative.shape[1]
        velocity_count += velocity.shape[0] * velocity.shape[1]
    relative_mean = relative_sum / relative_count
    velocity_mean = velocity_sum / velocity_count
    relative_std = np.sqrt(np.maximum(relative_square / relative_count - relative_mean**2, 0.0))
    velocity_std = np.sqrt(np.maximum(velocity_square / velocity_count - velocity_mean**2, 0.0))
    return TransitionFeatureStats(
        state_mean=np.asarray(cache.state_mean, dtype=np.float64).tolist(),
        state_std=_safe_std(cache.state_std).tolist(),
        action_mean=np.asarray(cache.action_mean, dtype=np.float64).tolist(),
        action_std=_safe_std(cache.action_std).tolist(),
        relative_mean=relative_mean.tolist(),
        relative_std=_safe_std(relative_std).tolist(),
        velocity_mean=velocity_mean.tolist(),
        velocity_std=_safe_std(velocity_std).tolist(),
    )


class TransitionFeatureTransform(nn.Module):
    """Invert accepted normalization, then build absolute/relative/delta features."""

    def __init__(self, stats: TransitionFeatureStats):
        super().__init__()
        for name in (
            "state_mean", "state_std", "action_mean", "action_std",
            "relative_mean", "relative_std", "velocity_mean", "velocity_std",
        ):
            self.register_buffer(name, torch.tensor(getattr(stats, name), dtype=torch.float32))

    def raw_values(self, state58: torch.Tensor, action58: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state_raw = state58 * self.state_std + self.state_mean
        action_raw = action58 * self.action_std + self.action_mean
        return state_raw, action_raw

    def forward(self, state58: torch.Tensor, action58: torch.Tensor) -> torch.Tensor:
        if state58.ndim != 2 or state58.shape[-1] != RAW_ACTION_DIM:
            raise ValueError("state58 must be [B,58]")
        if action58.ndim != 3 or action58.shape[1:] != (ACTION_HORIZON, RAW_ACTION_DIM):
            raise ValueError("action58 must be [B,16,58]")
        state_raw, action_raw = self.raw_values(state58, action58)
        relative = (action_raw - state_raw[:, None] - self.relative_mean) / self.relative_std
        velocity = torch.zeros_like(action58)
        velocity[:, 1:] = (
            torch.diff(action_raw, dim=1) - self.velocity_mean
        ) / self.velocity_std
        return torch.cat((action58, relative, velocity), dim=-1)

    def relative_target(self, state58: torch.Tensor, action58: torch.Tensor) -> torch.Tensor:
        state_raw, action_raw = self.raw_values(state58, action58)
        return (action_raw - state_raw[:, None] - self.relative_mean) / self.relative_std

    def velocity_target(self, action58: torch.Tensor) -> torch.Tensor:
        action_raw = action58 * self.action_std + self.action_mean
        return (torch.diff(action_raw, dim=1) - self.velocity_mean) / self.velocity_std


class UnifiedTransitionAdapter(nn.Module):
    """R1-P adapter, identity-initialized on absolute action channels."""

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(TRANSITION_FEATURE_DIM, 128)
        self.norm = nn.LayerNorm(128)
        self.temporal = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.position = nn.Parameter(torch.zeros(ACTION_HORIZON, 128))
        nn.init.zeros_(self.input.weight)
        nn.init.zeros_(self.input.bias)
        with torch.no_grad():
            self.input.weight[:RAW_ACTION_DIM, :RAW_ACTION_DIM] = torch.eye(RAW_ACTION_DIM)
        nn.init.zeros_(self.temporal.weight)
        nn.init.zeros_(self.temporal.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = self.input(features) + self.position
        residual = self.temporal(F.gelu(self.norm(value)).transpose(1, 2)).transpose(1, 2)
        return value + residual


class SharedTransitionActionModel(nn.Module):
    """R1-P: transition preprocessing followed by the selected shared Action path."""

    candidate = "R1-P"
    encoder_type = "shared"

    def __init__(self, base: TReXActionBootstrap, stats: TransitionFeatureStats):
        super().__init__()
        self.base = base
        self.feature_stats = stats
        self.features = TransitionFeatureTransform(stats)
        self.transition_adapter = UnifiedTransitionAdapter()

    def forward(self, state: torch.Tensor, action: torch.Tensor, embodiment_id: torch.Tensor) -> dict[str, torch.Tensor]:
        transition = self.features(state[:, :RAW_ACTION_DIM], action[:, :, :RAW_ACTION_DIM])
        adapted = self.transition_adapter(transition)
        z_action, state_features, l1 = self.base.encode(state, adapted, embodiment_id)
        return {
            "prediction": self.base.decode(z_action, state_features, embodiment_id),
            "z_action": z_action,
            "state_features": state_features,
            "l1": l1,
            "transition_features": transition,
        }

    def encode(self, state: torch.Tensor, action: torch.Tensor, embodiment_id: torch.Tensor):
        transition = self.features(state[:, :RAW_ACTION_DIM], action[:, :, :RAW_ACTION_DIM])
        adapted = self.transition_adapter(transition)
        return self.base.encode(state, adapted, embodiment_id)

    def decode(self, z_action: torch.Tensor, state_features: torch.Tensor, embodiment_id: torch.Tensor) -> torch.Tensor:
        return self.base.decode(z_action, state_features, embodiment_id)

    def parameter_summary(self) -> dict[str, int]:
        total = sum(value.numel() for value in self.parameters())
        trainable = sum(value.numel() for value in self.parameters() if value.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


class ResidualTemporalBlock(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.proj = nn.Linear(width, width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.norm(value)
        residual = self.conv(residual.transpose(1, 2)).transpose(1, 2)
        return value + self.proj(F.gelu(residual))


class GroupTemporalBranch(nn.Module):
    def __init__(self, input_dim: int, width: int):
        super().__init__()
        self.input = nn.Linear(input_dim, width)
        self.blocks = nn.Sequential(ResidualTemporalBlock(width, 1), ResidualTemporalBlock(width, 2))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.input(value))


def _feature_segment(features: torch.Tensor, segment: slice) -> torch.Tensor:
    return torch.cat(
        tuple(features[..., offset + segment.start : offset + segment.stop] for offset in (0, 58, 116)),
        dim=-1,
    )


class GroupedTransitionEncoder(nn.Module):
    """Four anatomical branches, temporal fusion, and eight learned queries."""

    def __init__(self, hidden_size: int = 96):
        super().__init__()
        group_width = hidden_size // 2
        self.branches = nn.ModuleDict({
            name: GroupTemporalBranch((segment.stop - segment.start) * FEATURE_MULTIPLIER, group_width)
            for name, segment in SEGMENTS.items()
        })
        self.side_embedding = nn.Parameter(torch.zeros(2, group_width))
        self.kind_embedding = nn.Parameter(torch.zeros(2, group_width))
        self.position = nn.Parameter(torch.zeros(ACTION_HORIZON, hidden_size))
        self.fusion = nn.Linear(group_width * 4, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.queries = nn.Parameter(torch.randn(QUERY_NUM, hidden_size) * 0.02)
        self.cross_attention = nn.MultiheadAttention(hidden_size, 4, dropout=0.0, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, L2_DIM))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        values = []
        for index, (name, segment) in enumerate(SEGMENTS.items()):
            value = self.branches[name](_feature_segment(features, segment))
            side = 0 if name.startswith("left") else 1
            kind = 0 if name.endswith("arm") else 1
            values.append(value + self.side_embedding[side] + self.kind_embedding[kind])
        fused = self.fusion(torch.cat(values, dim=-1)) + self.position
        memory = self.temporal(fused)
        query = self.queries.unsqueeze(0).expand(features.shape[0], -1, -1)
        attended, _ = self.cross_attention(query, memory, memory, need_weights=False)
        return self.output(attended + query)


class TransitionDecoder(nn.Module):
    """Compact token-gated decoder; state remains a legal, bounded condition."""

    def __init__(self, hidden_size: int = 96):
        super().__init__()
        self.token = nn.Sequential(
            nn.Flatten(), nn.Linear(QUERY_NUM * L2_DIM, ACTION_HORIZON * hidden_size), nn.GELU()
        )
        self.state = nn.Sequential(nn.Linear(RAW_ACTION_DIM, hidden_size), nn.GELU())
        self.state_gate_logit = nn.Parameter(torch.tensor(-2.0))
        self.position = nn.Parameter(torch.zeros(ACTION_HORIZON, hidden_size))
        self.blocks = nn.Sequential(ResidualTemporalBlock(hidden_size, 1), ResidualTemporalBlock(hidden_size, 2))
        self.output = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, RAW_ACTION_DIM))

    def forward(self, z_action: torch.Tensor, state58: torch.Tensor) -> torch.Tensor:
        value = self.token(z_action).view(z_action.shape[0], ACTION_HORIZON, -1)
        state_condition = self.state(state58).unsqueeze(1)
        value = value + torch.sigmoid(self.state_gate_logit) * state_condition + self.position
        return self.output(self.blocks(value))


class NativeTransitionActionModel(nn.Module):
    """R1-N compact T-Rex-native continuous transition autoencoder."""

    candidate = "R1-N"
    encoder_type = "native"

    def __init__(self, stats: TransitionFeatureStats, hidden_size: int = 96):
        super().__init__()
        self.feature_stats = stats
        self.features = TransitionFeatureTransform(stats)
        self.encoder = GroupedTransitionEncoder(hidden_size)
        self.decoder = TransitionDecoder(hidden_size)

    @staticmethod
    def _validate(state: torch.Tensor, action: torch.Tensor, embodiment_id: torch.Tensor) -> None:
        if state.ndim != 2 or state.shape[-1] != 128:
            raise ValueError("canonical state must be [B,128]")
        if action.ndim != 3 or action.shape[1:] != (ACTION_HORIZON, 128):
            raise ValueError("canonical action must be [B,16,128]")
        if embodiment_id.shape != (state.shape[0],) or torch.any(embodiment_id != TREX_EMBODIMENT_ID):
            raise ValueError("native transition path requires T-Rex embodiment ID 31")

    def encode(self, state: torch.Tensor, action: torch.Tensor, embodiment_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate(state, action, embodiment_id)
        features = self.features(state[:, :RAW_ACTION_DIM], action[:, :, :RAW_ACTION_DIM])
        z_action = self.encoder(features)
        if z_action.shape[1:] != (QUERY_NUM, L2_DIM) or not torch.isfinite(z_action).all():
            raise FloatingPointError("invalid native z_action")
        return z_action, state[:, :RAW_ACTION_DIM], features

    def decode(self, z_action: torch.Tensor, state_features: torch.Tensor, embodiment_id: torch.Tensor) -> torch.Tensor:
        if z_action.ndim != 3 or z_action.shape[1:] != (QUERY_NUM, L2_DIM):
            raise ValueError("z_action must be [B,8,32]")
        if embodiment_id.shape != (z_action.shape[0],) or torch.any(embodiment_id != TREX_EMBODIMENT_ID):
            raise ValueError("native decoder requires T-Rex embodiment ID 31")
        prediction58 = self.decoder(z_action, state_features[:, :RAW_ACTION_DIM])
        prediction = prediction58.new_zeros((len(prediction58), ACTION_HORIZON, 128))
        prediction[:, :, :RAW_ACTION_DIM] = prediction58
        return prediction

    def forward(self, state: torch.Tensor, action: torch.Tensor, embodiment_id: torch.Tensor) -> dict[str, torch.Tensor]:
        z_action, state_features, features = self.encode(state, action, embodiment_id)
        return {
            "prediction": self.decode(z_action, state_features, embodiment_id),
            "z_action": z_action,
            "state_features": state_features,
            "transition_features": features,
        }

    def parameter_summary(self) -> dict[str, int]:
        total = sum(value.numel() for value in self.parameters())
        trainable = sum(value.numel() for value in self.parameters() if value.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


def save_transition_checkpoint(path: Path, model: nn.Module, metadata: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(model, NativeTransitionActionModel):
        raise TypeError("deployment checkpoint is defined for the selected native model")
    payload = {
        "schema": "tactile3d-unit.s3-3-r-native-transition.v1",
        "candidate": model.candidate,
        "encoder_type": model.encoder_type,
        "feature_stats": model.feature_stats.to_dict(),
        "hidden_size": model.encoder.fusion.out_features,
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "metadata": dict(metadata),
    }
    torch.save(payload, path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_shared_transition_checkpoint(path: Path, model: SharedTransitionActionModel, metadata: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "tactile3d-unit.s3-3-r-shared-transition.v1",
        "candidate": model.candidate,
        "encoder_type": model.encoder_type,
        "source_identity": model.base.source_identity,
        "feature_stats": model.feature_stats.to_dict(),
        "base_initialization": model.base.initialization,
        "base_seed": model.base.seed,
        "base_overlay": model.base.overlay_state_dict(),
        "transition_adapter": {name: value.detach().cpu() for name, value in model.transition_adapter.state_dict().items()},
        "metadata": dict(metadata),
    }
    torch.save(payload, path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_shared_transition_checkpoint(
    path: Path,
    source: ReleasedTokenizerSource,
    map_location: str | torch.device = "cpu",
) -> tuple[SharedTransitionActionModel, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema") != "tactile3d-unit.s3-3-r-shared-transition.v1":
        raise ValueError("unsupported shared action-transition checkpoint")
    if payload.get("source_identity") != source.identity:
        raise ValueError("shared transition checkpoint does not match Original UniT")
    base = TReXActionBootstrap(
        source,
        initialization=str(payload["base_initialization"]),
        seed=int(payload["base_seed"]),
        enable_a2_adapter=True,
    )
    base.load_overlay_state_dict(payload["base_overlay"])
    base.configure_trainable(stage="A2")
    model = SharedTransitionActionModel(base, TransitionFeatureStats.from_dict(payload["feature_stats"]))
    model.transition_adapter.load_state_dict(payload["transition_adapter"], strict=True)
    return model, dict(payload.get("metadata", {}))


def load_transition_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> tuple[NativeTransitionActionModel, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema") != "tactile3d-unit.s3-3-r-native-transition.v1":
        raise ValueError("unsupported action-transition checkpoint")
    stats = TransitionFeatureStats.from_dict(payload["feature_stats"])
    model = NativeTransitionActionModel(stats, hidden_size=int(payload["hidden_size"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, dict(payload.get("metadata", {}))


def build_shared_candidate(
    source: ReleasedTokenizerSource,
    bootstrap_checkpoint: Path,
    stats: TransitionFeatureStats,
) -> SharedTransitionActionModel:
    from gr00t.tactile_unit.trex_action_bootstrap import load_bootstrap_checkpoint

    base, _ = load_bootstrap_checkpoint(bootstrap_checkpoint, source)
    base.configure_trainable(stage="A2")
    return SharedTransitionActionModel(base, stats)
