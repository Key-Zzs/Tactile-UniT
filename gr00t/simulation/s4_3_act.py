"""Canonical strictly causal ACT policy and frozen S4.2 auxiliary stack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gr00t.simulation.s4_2_formal import (
    ConditionalContactPredictor,
    FormalActionEncoder,
    FormalVACBridge,
)
from gr00t.simulation.sim_contact_models import load_teacher_checkpoint
from scripts.simulation.s4_2dr_common import load_model as load_contact_dynamics

ROOT = Path(__file__).resolve().parents[2]
ACTION_STEPS = 27
ACTION_DIM = 22
HISTORY_STEPS = 26
TACTILE_DIM = 30
HIDDEN_DIM = 256


@dataclass(frozen=True)
class PolicyNormalization:
    """Per-task POLICY_EXPERT_TRAIN-only normalization statistics."""

    proprio_mean: np.ndarray
    proprio_std: np.ndarray
    action_min: np.ndarray
    action_max: np.ndarray
    tactile_mean: np.ndarray
    tactile_std: np.ndarray

    def __post_init__(self) -> None:
        expected = {
            "proprio_mean": (ACTION_DIM,),
            "proprio_std": (ACTION_DIM,),
            "action_min": (ACTION_DIM,),
            "action_max": (ACTION_DIM,),
            "tactile_mean": (TACTILE_DIM,),
            "tactile_std": (TACTILE_DIM,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
            object.__setattr__(self, name, value)
        if np.any(self.proprio_std <= 0) or np.any(self.tactile_std <= 0):
            raise ValueError("normalization standard deviations must be positive")
        if np.any(self.action_max <= self.action_min):
            raise ValueError("Action bounds must have positive range")

    @classmethod
    def fit(
        cls, proprio: np.ndarray, action: np.ndarray, tactile: np.ndarray
    ) -> "PolicyNormalization":
        proprio = np.asarray(proprio, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        tactile = np.asarray(tactile, dtype=np.float32)
        if proprio.ndim != 2 or proprio.shape[1] != ACTION_DIM:
            raise ValueError("proprio fit data must be [N,22]")
        if action.ndim != 2 or action.shape[1] != ACTION_DIM:
            raise ValueError("Action fit data must be [N,22]")
        if tactile.ndim != 2 or tactile.shape[1] != TACTILE_DIM:
            raise ValueError("tactile fit data must be [N,30]")
        if not all(np.isfinite(value).all() for value in (proprio, action, tactile)):
            raise ValueError("policy fit data must be finite")
        action_min = action.min(axis=0)
        action_max = action.max(axis=0)
        center = 0.5 * (action_min + action_max)
        half_range = np.maximum(0.5 * (action_max - action_min), 5e-7)
        return cls(
            proprio_mean=proprio.mean(axis=0),
            proprio_std=np.maximum(proprio.std(axis=0), 1e-6),
            action_min=center - half_range,
            action_max=center + half_range,
            tactile_mean=tactile.mean(axis=0),
            tactile_std=np.maximum(tactile.std(axis=0), 1e-6),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            name: np.asarray(getattr(self, name), dtype=np.float32).tolist()
            for name in (
                "proprio_mean",
                "proprio_std",
                "action_min",
                "action_max",
                "tactile_mean",
                "tactile_std",
            )
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "PolicyNormalization":
        return cls(**{name: np.asarray(raw, dtype=np.float32) for name, raw in value.items()})


class CausalRawTactileEncoder(nn.Module):
    """Fixed small left-padded causal Conv1d encoder for P1."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(TACTILE_DIM, 64, kernel_size=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        self.projection = nn.Linear(128, HIDDEN_DIM)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or tuple(history.shape[1:]) != (HISTORY_STEPS, TACTILE_DIM):
            raise ValueError("raw tactile history must be [B,26,30]")
        value = history.transpose(1, 2)
        value = F.gelu(self.conv1(F.pad(value, (2, 0))))
        value = F.gelu(self.conv2(F.pad(value, (2, 0))))
        return self.projection(value[:, :, -1])

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class CausalACTPolicy(nn.Module):
    """Minimal repository-owned action-chunking CVAE Transformer."""

    def __init__(self, variant: str, normalization: PolicyNormalization) -> None:
        super().__init__()
        if variant not in {"P0", "P1", "P2", "P3"}:
            raise ValueError(f"unknown ACT variant {variant!r}")
        self.variant = variant
        self.vision_projection = nn.Linear(32, HIDDEN_DIM)
        self.proprio_projection = nn.Linear(ACTION_DIM, HIDDEN_DIM)
        self.raw_tactile = CausalRawTactileEncoder() if variant == "P1" else None
        self.contact_projection = nn.Linear(256, HIDDEN_DIM) if variant in {"P2", "P3"} else None
        observation_tokens = 10 if variant != "P0" else 9
        self.observation_position = nn.Parameter(torch.zeros(observation_tokens, HIDDEN_DIM))

        self.posterior_cls = nn.Parameter(torch.zeros(1, 1, HIDDEN_DIM))
        self.posterior_action_projection = nn.Linear(ACTION_DIM, HIDDEN_DIM)
        self.posterior_position = nn.Parameter(torch.zeros(ACTION_STEPS + 2, HIDDEN_DIM))
        posterior_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.posterior = nn.TransformerEncoder(posterior_layer, num_layers=4)
        self.posterior_mu = nn.Linear(HIDDEN_DIM, 32)
        self.posterior_log_variance = nn.Linear(HIDDEN_DIM, 32)

        self.latent_projection = nn.Linear(32, HIDDEN_DIM)
        self.action_queries = nn.Parameter(torch.zeros(ACTION_STEPS, HIDDEN_DIM))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=HIDDEN_DIM,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.action_head = nn.Linear(HIDDEN_DIM, ACTION_DIM)

        self.register_buffer("proprio_mean", torch.from_numpy(normalization.proprio_mean.copy()))
        self.register_buffer("proprio_std", torch.from_numpy(normalization.proprio_std.copy()))
        self.register_buffer("action_min", torch.from_numpy(normalization.action_min.copy()))
        self.register_buffer("action_max", torch.from_numpy(normalization.action_max.copy()))
        self.register_buffer("tactile_mean", torch.from_numpy(normalization.tactile_mean.copy()))
        self.register_buffer("tactile_std", torch.from_numpy(normalization.tactile_std.copy()))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.observation_position, std=0.02)
        nn.init.normal_(self.posterior_cls, std=0.02)
        nn.init.normal_(self.posterior_position, std=0.02)
        nn.init.normal_(self.action_queries, std=0.02)

    def normalize_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        return (proprio - self.proprio_mean) / self.proprio_std

    def normalize_tactile(self, history: torch.Tensor) -> torch.Tensor:
        return (history - self.tactile_mean) / self.tactile_std

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return 2.0 * (action - self.action_min) / (self.action_max - self.action_min) - 1.0

    def denormalize_action(self, normalized: torch.Tensor) -> torch.Tensor:
        return self.action_min + 0.5 * (normalized + 1.0) * (self.action_max - self.action_min)

    def observation_memory(
        self,
        vision: torch.Tensor,
        proprio: torch.Tensor,
        *,
        tactile_history: torch.Tensor | None = None,
        contact_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if vision.ndim != 3 or tuple(vision.shape[1:]) != (8, 32):
            raise ValueError("current-frame frozen Vision must be [B,8,32]")
        if proprio.shape != (len(vision), ACTION_DIM):
            raise ValueError("current proprio must be [B,22]")
        tokens = [
            self.vision_projection(vision),
            self.proprio_projection(self.normalize_proprio(proprio))[:, None],
        ]
        if self.variant == "P1":
            if tactile_history is None or self.raw_tactile is None:
                raise ValueError("P1 requires raw tactile history")
            tokens.append(self.raw_tactile(self.normalize_tactile(tactile_history))[:, None])
        elif self.variant in {"P2", "P3"}:
            if contact_state is None or self.contact_projection is None:
                raise ValueError(f"{self.variant} requires frozen Contact-State")
            if contact_state.shape != (len(vision), 256):
                raise ValueError("Contact-State must be [B,256]")
            tokens.append(self.contact_projection(contact_state)[:, None])
        memory = torch.cat(tokens, dim=1)
        if memory.shape[1:] != self.observation_position.shape:
            raise AssertionError("ACT observation token contract changed")
        return memory + self.observation_position

    def posterior_distribution(
        self, normalized_proprio: torch.Tensor, normalized_target_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if normalized_target_action.shape != (len(normalized_proprio), ACTION_STEPS, ACTION_DIM):
            raise ValueError("posterior target Action must be [B,27,22]")
        cls = self.posterior_cls.expand(len(normalized_proprio), -1, -1)
        state = self.proprio_projection(normalized_proprio)[:, None]
        action = self.posterior_action_projection(normalized_target_action)
        encoded = self.posterior(torch.cat((cls, state, action), dim=1) + self.posterior_position)
        return self.posterior_mu(encoded[:, 0]), self.posterior_log_variance(encoded[:, 0])

    def forward(
        self,
        vision: torch.Tensor,
        proprio: torch.Tensor,
        *,
        tactile_history: torch.Tensor | None = None,
        contact_state: torch.Tensor | None = None,
        target_action: torch.Tensor | None = None,
        sample_posterior: bool = True,
    ) -> dict[str, torch.Tensor]:
        memory = self.observation_memory(
            vision,
            proprio,
            tactile_history=tactile_history,
            contact_state=contact_state,
        )
        if target_action is None:
            mu = torch.zeros(len(vision), 32, device=vision.device, dtype=vision.dtype)
            log_variance = torch.zeros_like(mu)
            latent = mu
        else:
            normalized_target = self.normalize_action(target_action)
            mu, log_variance = self.posterior_distribution(
                self.normalize_proprio(proprio), normalized_target
            )
            latent = mu
            if sample_posterior:
                latent = mu + torch.exp(0.5 * log_variance) * torch.randn_like(mu)
        query = self.action_queries[None].expand(len(vision), -1, -1)
        query = query + self.latent_projection(latent)[:, None]
        decoded = self.decoder(query, memory)
        normalized_action = torch.tanh(self.action_head(decoded))
        physical_action = self.denormalize_action(normalized_action)
        return {
            "normalized_action": normalized_action,
            "physical_action": physical_action,
            "posterior_mu": mu,
            "posterior_log_variance": log_variance,
        }

    def act_loss(
        self, output: Mapping[str, torch.Tensor], target_action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        target = self.normalize_action(target_action).clamp(-1.0, 1.0)
        action_l1 = F.l1_loss(output["normalized_action"], target)
        mu = output["posterior_mu"]
        log_variance = output["posterior_log_variance"]
        kl = -0.5 * (1.0 + log_variance - mu.square() - log_variance.exp()).mean()
        return {"action_l1": action_l1, "kl": kl, "act_total": action_l1 + 10.0 * kl}

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class TorchTactileNormalization(nn.Module):
    """Exact torch form of the accepted frozen S4.2 N1 normalization."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        super().__init__()
        if payload["candidate"] != "N1":
            raise ValueError("S4.3 requires the accepted S4.2 N1 normalization")
        self.register_buffer("force_mean", torch.tensor(payload["force_mean"], dtype=torch.float32))
        self.register_buffer("force_std", torch.tensor(payload["force_std"], dtype=torch.float32))
        self.register_buffer("cop_mean", torch.tensor(payload["cop_mean"], dtype=torch.float32))
        self.register_buffer("cop_std", torch.tensor(payload["cop_std"], dtype=torch.float32))
        self.force_scale_newton = float(payload.get("force_scale_newton", 1.0))

    def forward(self, tactile: torch.Tensor) -> torch.Tensor:
        if tactile.shape[-1] != TACTILE_DIM:
            raise ValueError("S4.2 tactile input must end in 30 features")
        shape = tactile.shape
        matrix = tactile.reshape(*shape[:-1], 5, 6)
        occupancy = matrix[..., 0] > 0.5
        force = torch.log1p(torch.clamp_min(matrix[..., 1:3], 0.0) / self.force_scale_newton)
        normalized_force = (force - self.force_mean) / self.force_std
        normalized_cop = (matrix[..., 3:6] - self.cop_mean) / self.cop_std
        normalized_cop = torch.where(
            occupancy[..., None], normalized_cop, torch.zeros_like(normalized_cop)
        )
        output = torch.cat(
            (occupancy[..., None].to(tactile.dtype), normalized_force, normalized_cop), dim=-1
        )
        return output.reshape(shape)


class FrozenS42PolicyStack(nn.Module):
    """Read-only E_T/C3/A0/B3/A+H graph used by P2/P3."""

    CHECKPOINTS = {
        "contact_state": ROOT / ".local/experiments/simulation/s4_2r/contact_state/accepted.pt",
        "contact_C3": ROOT / ".local/experiments/simulation/s4_2r/contact_dynamics/selected.pt",
        "action_A0": ROOT / ".local/experiments/simulation/s4_2_formal/action/selected.pt",
        "bridge_B3": ROOT / ".local/experiments/simulation/s4_2_formal/bridge/selected.pt",
        "conditional_A_plus_H": ROOT
        / ".local/experiments/simulation/s4_2_formal/s4_2_7/A_plus_H.pt",
    }

    def __init__(self) -> None:
        super().__init__()
        teacher, teacher_payload = load_teacher_checkpoint(self.CHECKPOINTS["contact_state"], "cpu")
        self.tactile_normalization = TorchTactileNormalization(teacher_payload["normalization"])
        self.contact_state = teacher

        action_payload = torch.load(
            self.CHECKPOINTS["action_A0"], map_location="cpu", weights_only=False
        )
        if action_payload["candidate"] != "A0":
            raise RuntimeError("frozen Action checkpoint is not A0")
        self.action_A0 = FormalActionEncoder("A0")
        self.action_A0.load_state_dict(action_payload["state_dict"], strict=True)
        for name, value in action_payload["stats"].items():
            self.register_buffer(f"action_stat_{name}", torch.tensor(value, dtype=torch.float32))

        bridge_payload = torch.load(
            self.CHECKPOINTS["bridge_B3"], map_location="cpu", weights_only=False
        )
        if bridge_payload["adapter"] != "slot":
            raise RuntimeError("frozen bridge checkpoint is not selected B3 slot")
        self.bridge_B3 = FormalVACBridge("slot")
        self.bridge_B3.load_state_dict(bridge_payload["state_dict"], strict=True)

        predictor_payload = torch.load(
            self.CHECKPOINTS["conditional_A_plus_H"], map_location="cpu", weights_only=False
        )
        if tuple(predictor_payload["modalities"]) != ("action", "history"):
            raise RuntimeError("frozen policy auxiliary is not causal A+H")
        self.conditional_A_plus_H = ConditionalContactPredictor(("action", "history"))
        self.conditional_A_plus_H.load_state_dict(predictor_payload["state_dict"], strict=True)

        contact_dynamics, contact_payload = load_contact_dynamics(
            self.CHECKPOINTS["contact_C3"], torch.device("cpu")
        )
        if contact_payload["model"] != "C3":
            raise RuntimeError("frozen Contact dynamics checkpoint is not C3")
        self.contact_C3 = contact_dynamics
        self.eval().requires_grad_(False)

    def train(self, mode: bool = True) -> "FrozenS42PolicyStack":
        super().train(False)
        return self

    def encode_contact_state(self, tactile_history: torch.Tensor) -> torch.Tensor:
        if tactile_history.ndim != 3 or tuple(tactile_history.shape[1:]) != (26, 30):
            raise ValueError("Contact-State input must be [B,26,30]")
        return self.contact_state(self.tactile_normalization(tactile_history))["latent"]

    def action_features(
        self, current_state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if current_state.shape != (len(action), 22) or action.shape[1:] != (27, 22):
            raise ValueError("A0 requires current [B,22] and Action [B,27,22]")
        relative = action - current_state[:, None]
        difference = torch.cat(
            (action[:, :1] - current_state[:, None], action[:, 1:] - action[:, :-1]), dim=1
        )
        absolute = (action - self.action_stat_absolute_mean) / self.action_stat_absolute_std
        relative = (relative - self.action_stat_relative_mean) / self.action_stat_relative_std
        difference = (
            difference - self.action_stat_difference_mean
        ) / self.action_stat_difference_std
        features = torch.cat((absolute, relative, difference), dim=-1)
        state = (current_state - self.action_stat_state_mean) / self.action_stat_state_std
        return features, state

    def predict_shared_contact(
        self, current_state: torch.Tensor, planned_action: torch.Tensor, h_current: torch.Tensor
    ) -> torch.Tensor:
        features, normalized_state = self.action_features(current_state, planned_action)
        action_code = self.action_A0(features, normalized_state)["code"]
        shared_action = self.bridge_B3.encode("action", action_code)
        return self.conditional_A_plus_H(
            {"action": shared_action, "history": h_current.reshape(len(h_current), 8, 32)}
        )

    @torch.no_grad()
    def target_shared_contact(
        self, current_history: torch.Tensor, future_history: torch.Tensor
    ) -> torch.Tensor:
        h_current = self.encode_contact_state(current_history)
        h_future = self.encode_contact_state(future_history)
        contact_code = self.contact_C3(h_current, h_future)["code"]
        return self.bridge_B3.encode("contact", contact_code)

    def contact_auxiliary_loss(
        self,
        current_state: torch.Tensor,
        planned_action: torch.Tensor,
        current_history: torch.Tensor,
        target_shared_contact: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_current = self.encode_contact_state(current_history)
        prediction = self.predict_shared_contact(current_state, planned_action, h_current)
        target = target_shared_contact.detach()
        return F.mse_loss(prediction, target), prediction

    @property
    def frozen_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
