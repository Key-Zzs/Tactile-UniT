"""Repository-owned tactile conditioning adapters for official DexJoCo OpenPI.

This module is imported only from the ignored writable copy of the pinned
official OpenPI tree.  It does not implement a policy or a flow-matching
algorithm: it subclasses the official pi0.5 model and adds observation-prefix
tokens plus an optional training-only diagnostic head.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import flax.traverse_util
from flax import nnx, struct
import jax
import jax.numpy as jnp
import numpy as np
import optax

from gr00t.simulation.s4_3_pi1 import (
    CONTACT_STATE_DIM,
    CONTACT_TOKENS,
    CONTACT_TOKEN_WIDTH,
    PHYSICAL_TARGET_HORIZON,
    TactileUnitMode,
)

from openpi import transforms as _transforms
from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_config as _pi0_config
from openpi.policies import single_arm_policy
from openpi.shared import array_typing as at
from openpi.shared import download
from openpi.shared import nnx_utils
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import weight_loaders


_OfficialObservation = _model.Observation
_official_preprocess_observation = _model.preprocess_observation
_official_create_torch_dataset = _data_loader.create_torch_dataset
_HOOKS_INSTALLED = False


@struct.dataclass
class TactileObservation(_OfficialObservation):
    """Official observation plus optional training-only physical targets."""

    contact_state: Any | None = None
    contact_shared_target: Any | None = None
    physical_aux_valid: Any | None = None
    va_shared_target: Any | None = None
    va_aux_valid: Any | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[Any]) -> "TactileObservation":
        base = _OfficialObservation.from_dict(data)
        return cls(
            images=base.images,
            image_masks=base.image_masks,
            state=base.state,
            tokenized_prompt=base.tokenized_prompt,
            tokenized_prompt_mask=base.tokenized_prompt_mask,
            token_ar_mask=base.token_ar_mask,
            token_loss_mask=base.token_loss_mask,
            contact_state=data.get("contact_state"),
            contact_shared_target=data.get("contact_shared_target"),
            physical_aux_valid=data.get("physical_aux_valid"),
            va_shared_target=data.get("va_shared_target"),
            va_aux_valid=data.get("va_aux_valid"),
        )


def _preprocess_preserving_tactile(rng, observation, **kwargs):
    processed = _official_preprocess_observation(rng, observation, **kwargs)
    if not isinstance(observation, TactileObservation):
        return processed
    return TactileObservation(
        images=processed.images,
        image_masks=processed.image_masks,
        state=processed.state,
        tokenized_prompt=processed.tokenized_prompt,
        tokenized_prompt_mask=processed.tokenized_prompt_mask,
        token_ar_mask=processed.token_ar_mask,
        token_loss_mask=processed.token_loss_mask,
        contact_state=observation.contact_state,
        contact_shared_target=observation.contact_shared_target,
        physical_aux_valid=observation.physical_aux_valid,
        va_shared_target=observation.va_shared_target,
        va_aux_valid=observation.va_aux_valid,
    )


class _SidecarDataset:
    """Strict-superset view over the immutable official LeRobot dataset."""

    def __init__(self, dataset, sidecar_path: Path, mode: TactileUnitMode):
        self._dataset = dataset
        self._mode = mode
        with np.load(sidecar_path, allow_pickle=False) as source:
            self._index = source["index"].copy()
            if mode is TactileUnitMode.VA_PHYSICAL_AUX:
                if set(source.files) != {"index", "va_shared_target", "va_aux_valid"}:
                    raise ValueError(f"BVA sidecar must contain only index/V/A target/validity: {source.files}")
                self._va_shared_target = source["va_shared_target"].copy()
                self._va_aux_valid = source["va_aux_valid"].copy()
                values = (self._index, self._va_shared_target, self._va_aux_valid)
            else:
                self._contact_state = source["contact_state"].copy()
                self._contact_shared_target = source["contact_shared_target"].copy()
                self._physical_aux_valid = source["physical_aux_valid"].copy()
                values = (
                    self._index,
                    self._contact_state,
                    self._contact_shared_target,
                    self._physical_aux_valid,
                )
        if len(self._index) != len(dataset):
            raise ValueError(f"sidecar/dataset length mismatch: {len(self._index)} != {len(dataset)}")
        for value in values:
            value.flags.writeable = False

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index):
        position = index.__index__()
        item = self._dataset[index]
        official_index = int(np.asarray(item["index"]))
        if official_index != int(self._index[position]) or official_index != position:
            raise RuntimeError(
                f"sidecar index mismatch: request={position}, official={official_index}, sidecar={self._index[position]}"
            )
        if self._mode is TactileUnitMode.VA_PHYSICAL_AUX:
            return {
                **item,
                "va_shared_target": self._va_shared_target[position],
                "va_aux_valid": self._va_aux_valid[position],
            }
        return {
            **item,
            "contact_state": self._contact_state[position],
            "contact_shared_target": self._contact_shared_target[position],
            "physical_aux_valid": self._physical_aux_valid[position],
        }


@dataclasses.dataclass(frozen=True)
class TactileDataConfig(_config.DataConfig):
    sidecar_path: Path | None = None
    mode: TactileUnitMode = TactileUnitMode.CONTACT_STATE_TOKENS


def _create_tactile_dataset(data_config, action_horizon, model_config):
    dataset = _official_create_torch_dataset(data_config, action_horizon, model_config)
    if isinstance(data_config, TactileDataConfig):
        if data_config.sidecar_path is None:
            raise ValueError("tactile data config requires a sidecar")
        dataset = _SidecarDataset(dataset, Path(data_config.sidecar_path), data_config.mode)
    return dataset


def install_openpi_runtime_hooks() -> None:
    """Install narrow type/data hooks before the official loader is built."""

    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return
    _model.Observation = TactileObservation
    _model.preprocess_observation = _preprocess_preserving_tactile
    _data_loader.create_torch_dataset = _create_tactile_dataset
    _HOOKS_INSTALLED = True


@dataclasses.dataclass(frozen=True)
class TactileSingleArmInputs(_transforms.DataTransformFn):
    model_type: _model.ModelType
    mode: TactileUnitMode = TactileUnitMode.CONTACT_STATE_TOKENS

    def __call__(self, data: dict) -> dict:
        result = single_arm_policy.SingleArmInputs(model_type=self.model_type)(data)
        if self.mode is TactileUnitMode.VA_PHYSICAL_AUX:
            target = np.asarray(data["va_shared_target"], dtype=np.float32)
            if target.shape != (CONTACT_TOKENS, CONTACT_TOKEN_WIDTH):
                raise ValueError(f"VA target must be [8,32], got {target.shape}")
            result["va_shared_target"] = target
            result["va_aux_valid"] = np.asarray(data["va_aux_valid"], dtype=np.bool_)
            return result
        contact_state = np.asarray(data["contact_state"], dtype=np.float32)
        if contact_state.shape != (CONTACT_STATE_DIM,):
            raise ValueError(f"contact_state must be [{CONTACT_STATE_DIM}], got {contact_state.shape}")
        result["contact_state"] = contact_state
        if self.mode is TactileUnitMode.CONTACT_STATE_TOKENS_PHYSICAL_AUX:
            target = np.asarray(data["contact_shared_target"], dtype=np.float32)
            if target.shape != (CONTACT_TOKENS, CONTACT_TOKEN_WIDTH):
                raise ValueError(f"contact target must be [8,32], got {target.shape}")
            result["contact_shared_target"] = target
            result["physical_aux_valid"] = np.asarray(data["physical_aux_valid"], dtype=np.bool_)
        return result


@dataclasses.dataclass(frozen=True)
class TactileSingleArmDataConfig(_config.DataConfigFactory):
    root: Path = Path(".")
    sidecar_path: Path = Path("sidecar.npz")
    mode: TactileUnitMode = TactileUnitMode.CONTACT_STATE_TOKENS
    action_sequence_keys: tuple[str, ...] = ("action",)
    base_img_name: str | None = None

    def create(self, assets_dirs: Path, model_config: _model.BaseModelConfig) -> TactileDataConfig:
        if self.mode is TactileUnitMode.NONE:
            raise ValueError("NONE must use the unmodified official SingleArmDataConfig")
        base_img_name = self.base_img_name or "observation.images.front"
        mapping = {
            "base": base_img_name,
            "wrist": "observation.images.wrist",
            "state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        }
        if self.mode is TactileUnitMode.VA_PHYSICAL_AUX:
            mapping.update({"va_shared_target": "va_shared_target", "va_aux_valid": "va_aux_valid"})
        else:
            mapping["contact_state"] = "contact_state"
        if self.mode is TactileUnitMode.CONTACT_STATE_TOKENS_PHYSICAL_AUX:
            mapping.update(
                {
                    "contact_shared_target": "contact_shared_target",
                    "physical_aux_valid": "physical_aux_valid",
                }
            )
        repack = _transforms.Group(inputs=[_transforms.RepackTransform(mapping)])
        data_transforms = _transforms.Group(
            inputs=[TactileSingleArmInputs(model_type=model_config.model_type, mode=self.mode)],
            outputs=[single_arm_policy.SingleArmOutputs()],
        )
        model_transforms = _config.ModelTransformFactory()(model_config)
        base = self.create_base_config(assets_dirs, model_config)
        values = {field.name: getattr(base, field.name) for field in dataclasses.fields(_config.DataConfig)}
        values.update(
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            root=self.root,
            sidecar_path=self.sidecar_path,
            mode=self.mode,
        )
        return TactileDataConfig(**values)


class ContactTokenAdapter(nnx.Module):
    def __init__(self, prefix_width: int, rngs: nnx.Rngs):
        self.norm = nnx.LayerNorm(CONTACT_STATE_DIM, rngs=rngs)
        self.fc1 = nnx.Linear(CONTACT_STATE_DIM, CONTACT_STATE_DIM, rngs=rngs)
        self.fc2 = nnx.Linear(CONTACT_STATE_DIM, CONTACT_STATE_DIM, rngs=rngs)
        self.shared_projection = nnx.Linear(CONTACT_TOKEN_WIDTH, prefix_width, rngs=rngs)

    def __call__(self, contact_state):
        hidden = self.norm(contact_state)
        hidden = self.fc1(hidden)
        hidden = nnx.gelu(hidden)
        hidden = self.fc2(hidden)
        hidden = hidden.reshape((*hidden.shape[:-1], CONTACT_TOKENS, CONTACT_TOKEN_WIDTH))
        return self.shared_projection(hidden)


class PhysicalAuxiliaryHead(nnx.Module):
    def __init__(self, action_width: int, rngs: nnx.Rngs):
        self.norm = nnx.LayerNorm(action_width, rngs=rngs)
        self.fc1 = nnx.Linear(action_width, 512, rngs=rngs)
        self.fc2 = nnx.Linear(512, CONTACT_STATE_DIM, rngs=rngs)

    def __call__(self, action_hidden):
        pooled = jnp.mean(action_hidden[:, :PHYSICAL_TARGET_HORIZON], axis=1)
        hidden = self.norm(pooled)
        hidden = nnx.gelu(self.fc1(hidden))
        return self.fc2(hidden).reshape((-1, CONTACT_TOKENS, CONTACT_TOKEN_WIDTH))


@dataclasses.dataclass(frozen=True)
class TactilePi0Config(_pi0_config.Pi0Config):
    tactile_unit_mode: TactileUnitMode = TactileUnitMode.NONE
    lambda_phys: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        physical_modes = {
            TactileUnitMode.CONTACT_STATE_TOKENS_PHYSICAL_AUX,
            TactileUnitMode.VA_PHYSICAL_AUX,
        }
        if self.tactile_unit_mode not in physical_modes and self.lambda_phys != 0:
            raise ValueError("lambda_phys must be zero unless physical auxiliary mode is active")
        if not 0.0 <= self.lambda_phys <= 0.1:
            raise ValueError("lambda_phys must be in [0, 0.1]")

    def create(self, rng: at.KeyArrayLike):
        if self.tactile_unit_mode is TactileUnitMode.NONE:
            return _pi0.Pi0(self, rngs=nnx.Rngs(rng))
        return TactilePi0(self, rngs=nnx.Rngs(rng))

    def inputs_spec(self, *, batch_size: int = 1):
        official_obs, action_spec = super().inputs_spec(batch_size=batch_size)
        if self.tactile_unit_mode is TactileUnitMode.NONE:
            return official_obs, action_spec
        contact_target = None
        contact_valid = None
        va_target = None
        va_valid = None
        if self.tactile_unit_mode is TactileUnitMode.CONTACT_STATE_TOKENS_PHYSICAL_AUX:
            contact_target = jax.ShapeDtypeStruct([batch_size, CONTACT_TOKENS, CONTACT_TOKEN_WIDTH], jnp.float32)
            contact_valid = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        elif self.tactile_unit_mode is TactileUnitMode.VA_PHYSICAL_AUX:
            va_target = jax.ShapeDtypeStruct([batch_size, CONTACT_TOKENS, CONTACT_TOKEN_WIDTH], jnp.float32)
            va_valid = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        contact_state = None
        if self.tactile_unit_mode in {
            TactileUnitMode.CONTACT_STATE_TOKENS,
            TactileUnitMode.CONTACT_STATE_TOKENS_PHYSICAL_AUX,
        }:
            contact_state = jax.ShapeDtypeStruct([batch_size, CONTACT_STATE_DIM], jnp.float32)
        return TactileObservation(
            images=official_obs.images,
            image_masks=official_obs.image_masks,
            state=official_obs.state,
            tokenized_prompt=official_obs.tokenized_prompt,
            tokenized_prompt_mask=official_obs.tokenized_prompt_mask,
            token_ar_mask=official_obs.token_ar_mask,
            token_loss_mask=official_obs.token_loss_mask,
            contact_state=contact_state,
            contact_shared_target=contact_target,
            physical_aux_valid=contact_valid,
            va_shared_target=va_target,
            va_aux_valid=va_valid,
        ), action_spec


class TactilePi0(_pi0.Pi0):
    """Official pi0.5 with eight appended observation-prefix tokens."""

    def __init__(self, config: TactilePi0Config, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.tactile_unit_mode = config.tactile_unit_mode
        self.lambda_phys = config.lambda_phys
        action_width = _gemma.get_config(config.action_expert_variant).width
        if self.tactile_unit_mode in {
            TactileUnitMode.CONTACT_STATE_TOKENS,
            TactileUnitMode.CONTACT_STATE_TOKENS_PHYSICAL_AUX,
        }:
            prefix_width = _gemma.get_config(config.paligemma_variant).width
            self.contact_adapter = ContactTokenAdapter(prefix_width, rngs)
        if self.tactile_unit_mode in {
            TactileUnitMode.CONTACT_STATE_TOKENS_PHYSICAL_AUX,
            TactileUnitMode.VA_PHYSICAL_AUX,
        }:
            self.physical_auxiliary = PhysicalAuxiliaryHead(action_width, rngs)

    def embed_prefix(self, obs):
        tokens, input_mask, ar_mask = super().embed_prefix(obs)
        if self.tactile_unit_mode is TactileUnitMode.VA_PHYSICAL_AUX:
            return tokens, input_mask, ar_mask
        if obs.contact_state is None:
            raise ValueError("contact_state [B,256] is required in tactile modes")
        contact = jnp.asarray(obs.contact_state, dtype=jnp.float32)
        if contact.shape[-1] != CONTACT_STATE_DIM:
            raise ValueError(f"contact_state must end in {CONTACT_STATE_DIM}, got {contact.shape}")
        contact_tokens = self.contact_adapter(contact).astype(tokens.dtype)
        tokens = jnp.concatenate([tokens, contact_tokens], axis=1)
        input_mask = jnp.concatenate(
            [input_mask, jnp.ones(contact_tokens.shape[:2], dtype=jnp.bool_)], axis=1
        )
        ar_mask = jnp.concatenate([ar_mask, jnp.zeros((CONTACT_TOKENS,), dtype=jnp.bool_)], axis=0)
        return tokens, input_mask, ar_mask

    def compute_loss_with_info(self, rng, observation, actions, *, train=False):
        chunked_loss, action_hidden = super().compute_loss_with_hidden(rng, observation, actions, train=train)
        official_loss = jnp.mean(chunked_loss)
        info = {"official_loss": official_loss}
        if self.tactile_unit_mode in {
            TactileUnitMode.CONTACT_STATE_TOKENS_PHYSICAL_AUX,
            TactileUnitMode.VA_PHYSICAL_AUX,
        }:
            physical_loss = self._physical_loss(observation, action_hidden)
            total_loss = official_loss + self.lambda_phys * physical_loss
            info.update(physical_loss=physical_loss, lambda_phys=jnp.asarray(self.lambda_phys))
            return total_loss, info
        return official_loss, info

    def _physical_loss(self, observation, action_hidden):
        if self.tactile_unit_mode is TactileUnitMode.VA_PHYSICAL_AUX:
            target_value, valid_value = observation.va_shared_target, observation.va_aux_valid
            label = "BVA"
        else:
            target_value, valid_value = observation.contact_shared_target, observation.physical_aux_valid
            label = "PI1C"
        if target_value is None or valid_value is None:
            raise ValueError(f"physical auxiliary target and validity mask are required for {label} training")
        prediction = self.physical_auxiliary(action_hidden)
        target = jax.lax.stop_gradient(jnp.asarray(target_value, dtype=prediction.dtype))
        valid = jnp.asarray(valid_value, dtype=jnp.bool_)
        per_sample = jnp.mean(jnp.square(prediction - target), axis=(-2, -1))
        valid_float = valid.astype(per_sample.dtype)
        return jnp.sum(per_sample * valid_float) / jnp.maximum(jnp.sum(valid_float), 1.0)

    def compute_physical_auxiliary_loss(self, rng, observation, actions, *, train=False):
        """Expose the isolated training-only loss for a structural gradient gate."""
        _, action_hidden = super().compute_loss_with_hidden(rng, observation, actions, train=train)
        return self._physical_loss(observation, action_hidden)

    def sample_actions(self, rng, observation, **kwargs):
        if getattr(observation, "contact_shared_target", None) is not None:
            raise ValueError("training-only contact_shared_target is forbidden during inference")
        if getattr(observation, "physical_aux_valid", None) is not None:
            raise ValueError("training-only physical_aux_valid is forbidden during inference")
        if getattr(observation, "va_shared_target", None) is not None:
            raise ValueError("training-only va_shared_target is forbidden during inference")
        if getattr(observation, "va_aux_valid", None) is not None:
            raise ValueError("training-only va_aux_valid is forbidden during inference")
        return super().sample_actions(rng, observation, **kwargs)

    def training_gradient_metrics(self, grads):
        def norm(pattern: str):
            selected = grads.filter(nnx_utils.PathRegex(pattern))
            return optax.global_norm(selected) if jax.tree.leaves(selected) else jnp.asarray(0.0)

        metrics = {"pi05_trainable_grad_norm": norm(".*(lora|action_(in|out)_proj|time_mlp).*")}
        if self.tactile_unit_mode in {
            TactileUnitMode.CONTACT_STATE_TOKENS,
            TactileUnitMode.CONTACT_STATE_TOKENS_PHYSICAL_AUX,
        }:
            metrics["contact_adapter_grad_norm"] = norm(".*contact_adapter.*")
        if self.tactile_unit_mode in {
            TactileUnitMode.CONTACT_STATE_TOKENS_PHYSICAL_AUX,
            TactileUnitMode.VA_PHYSICAL_AUX,
        }:
            metrics["physical_auxiliary_grad_norm"] = norm(".*physical_auxiliary.*")
        return metrics


@dataclasses.dataclass(frozen=True)
class TactileCheckpointWeightLoader:
    """Load exact pi05_base and initialize only explicitly added leaves."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return weight_loaders._merge_params(  # noqa: SLF001 - pinned upstream compatibility hook
            loaded,
            params,
            missing_regex=".*(lora|contact_adapter|physical_auxiliary).*",
        )


def count_named_parameters(model) -> dict[str, int]:
    flat = flax.traverse_util.flatten_dict(nnx.state(model).to_pure_dict(), sep="/")
    counts = {"contact_adapter": 0, "physical_auxiliary": 0}
    for name, value in flat.items():
        for group in counts:
            if group in name:
                counts[group] += int(np.prod(value.shape))
    return counts
