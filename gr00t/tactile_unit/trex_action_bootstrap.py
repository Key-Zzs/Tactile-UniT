"""T-Rex-only bootstrap for the released UniT Action Branch.

The released checkpoint contains 30 category rows (IDs 0..29), while the
repository reserves ID 31 for ``new_embodiment``.  Loading the checkpoint into
a naively enlarged Action Branch would either fail on shape mismatches or
silently randomize every category-specific tensor.  This module instead:

* reads the released checkpoint without loading the vision branch;
* copies all shared Action-Branch weights exactly;
* reduces each 30-row category tensor to one isolated, trainable T-Rex row;
* exposes the real global ID 31 and the canonical action-only L2 interface;
* serializes only the new row (and an optional A2 residual adapter), never old
  rows or the frozen Original-UniT RQ.

The compact representation is a training implementation detail.  Deployment
can materialize a 32-row tensor with :func:`expand_category_tensor`; rows
0..29 are copied bit-for-bit, row 30 stays explicitly unused, and row 31 is
filled from the learned overlay.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import nn

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.model.tokenizer.action_branch_decoder import ActionDecoder, ActionDecoderConfig
from gr00t.model.tokenizer.action_branch_encoder import (
    ActionEncoder,
    ActionEncoderConfig,
    CategorySpecificCausalConv1D,
    CategorySpecificLayerNorm,
    CategorySpecificLinear,
)


RELEASED_CATEGORY_CAPACITY = 30
EXPANDED_CATEGORY_CAPACITY = 32
UNUSED_EMBODIMENT_ID = 30
TREX_EMBODIMENT_ID = 31
GR1_EMBODIMENT_ID = 24
TREX_EMBODIMENT_TAG = "trex"
ACTION_HORIZON = 16
CANONICAL_DIM = 128
QUERY_NUM = 8
L2_DIM = 32

CATEGORY_PREFIXES = (
    "action_branch.action_conv_encoder.",
    "action_branch.state_encoder.",
    "action_decoder.resnet_decoder.",
)


def validate_registration() -> None:
    """Fail closed if the process-wide embodiment registry drifts."""

    if EMBODIMENT_TAG_MAPPING.get(TREX_EMBODIMENT_TAG) != TREX_EMBODIMENT_ID:
        raise ValueError("T-Rex must be explicitly registered as embodiment ID 31")
    if EMBODIMENT_TAG_MAPPING.get("gr1") != GR1_EMBODIMENT_ID:
        raise ValueError("released GR1 embodiment ID changed")
    if TREX_EMBODIMENT_ID == GR1_EMBODIMENT_ID:
        raise ValueError("T-Rex must not alias GR1")
    if UNUSED_EMBODIMENT_ID in EMBODIMENT_TAG_MAPPING.values():
        raise ValueError("embodiment ID 30 must remain explicitly unused")


def _sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ReleasedTokenizerSource:
    """Read-only tensor access to a sharded released tokenizer checkpoint."""

    root: Path
    config: dict[str, Any]
    weight_map: dict[str, str]

    @classmethod
    def open(cls, root: Path | str) -> "ReleasedTokenizerSource":
        root = Path(root)
        config_path = root / "config.json"
        index_path = root / "model.safetensors.index.json"
        if not config_path.is_file() or not index_path.is_file():
            raise FileNotFoundError("released tokenizer config/index is missing")
        config = json.loads(config_path.read_text())
        weight_map = json.loads(index_path.read_text())["weight_map"]
        source = cls(root=root, config=config, weight_map=weight_map)
        source.validate_contract()
        return source

    def validate_contract(self) -> None:
        encoder = self.config["action_encoder_cfg"]
        decoder = self.config["action_decoder_cfg"]
        expected = {
            "action_horizon": ACTION_HORIZON,
            "action_dim": CANONICAL_DIM,
            "state_dim": CANONICAL_DIM,
            "query_num": QUERY_NUM,
        }
        for key, value in expected.items():
            if int(self.config[key]) != value:
                raise ValueError(f"released tokenizer {key} contract changed")
        if int(encoder["max_num_embodiments"]) != RELEASED_CATEGORY_CAPACITY:
            raise ValueError("released Action encoder must contain exactly 30 rows")
        if int(decoder["max_num_embodiments"]) != RELEASED_CATEGORY_CAPACITY:
            raise ValueError("released Action decoder must contain exactly 30 rows")
        if int(self.config["vq_cfg"]["e_dim"]) != L2_DIM:
            raise ValueError("released pre-RQ dimension must be 32")

    @property
    def identity(self) -> dict[str, str]:
        return {
            "config_sha256": _sha256_file(self.root / "config.json"),
            "index_sha256": _sha256_file(self.root / "model.safetensors.index.json"),
        }

    def tensor(self, key: str) -> torch.Tensor:
        from safetensors import safe_open

        shard = self.weight_map.get(key)
        if shard is None:
            raise KeyError(f"tensor absent from released tokenizer: {key}")
        with safe_open(self.root / shard, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)

    def category_tensor_names(self) -> list[str]:
        names = []
        for name in self.weight_map:
            if not name.startswith(CATEGORY_PREFIXES):
                continue
            shape = self.tensor_shape(name)
            if shape and shape[0] == RELEASED_CATEGORY_CAPACITY:
                names.append(name)
        return sorted(names)

    def tensor_shape(self, key: str) -> tuple[int, ...]:
        from safetensors import safe_open

        shard = self.weight_map[key]
        with safe_open(self.root / shard, framework="pt", device="cpu") as handle:
            return tuple(handle.get_slice(key).get_shape())

    def old_rows_digest(self) -> str:
        """Hash all released category tensors, including names/shapes/dtypes."""

        digest = hashlib.sha256()
        for name in self.category_tensor_names():
            value = self.tensor(name).contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(value.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()


def _deterministic_random_row(name: str, reference: torch.Tensor, seed: int) -> torch.Tensor:
    shape = (1, *reference.shape[1:])
    if name.endswith("norm.weight") or name.endswith("final_norm.weight"):
        return torch.ones(shape, dtype=reference.dtype)
    if name.endswith(".bias") or name.endswith(".b"):
        return torch.zeros(shape, dtype=reference.dtype)
    generator = torch.Generator(device="cpu")
    name_seed = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "little")
    generator.manual_seed((int(seed) + name_seed) % (2**63 - 1))
    return torch.randn(shape, generator=generator, dtype=torch.float32).mul_(0.02).to(reference.dtype)


def initialized_category_row(
    name: str,
    released: torch.Tensor,
    *,
    initialization: str,
    seed: int,
) -> torch.Tensor:
    if released.shape[0] != RELEASED_CATEGORY_CAPACITY:
        raise ValueError("category initialization requires a 30-row released tensor")
    if initialization == "old_rows_mean":
        return released.float().mean(dim=0, keepdim=True).to(released.dtype)
    if initialization == "deterministic_small_random":
        return _deterministic_random_row(name, released, seed)
    if initialization == "generic_new_embodiment":
        raise ValueError("released checkpoint has no legal generic/new-embodiment row")
    raise ValueError(f"unknown T-Rex initialization: {initialization}")


def expand_category_tensor(
    released: torch.Tensor,
    *,
    name: str,
    trex_row: torch.Tensor,
    unused_row_initialization: str = "old_rows_mean",
    seed: int = 0,
) -> torch.Tensor:
    """Materialize 30 -> 32 without mutating or reordering any released row."""

    if released.shape[0] != RELEASED_CATEGORY_CAPACITY:
        raise ValueError("released category tensor does not have 30 rows")
    if trex_row.shape != (1, *released.shape[1:]):
        raise ValueError("learned T-Rex row has an incompatible shape")
    row30 = initialized_category_row(
        name,
        released,
        initialization=unused_row_initialization,
        seed=seed,
    )
    expanded = torch.cat((released, row30, trex_row.to(released.dtype)), dim=0)
    if expanded.shape[0] != EXPANDED_CATEGORY_CAPACITY:
        raise AssertionError("category expansion did not produce 32 rows")
    if not torch.equal(expanded[:RELEASED_CATEGORY_CAPACITY], released):
        raise AssertionError("category expansion modified a released row")
    return expanded


def _load_compact_submodule(
    module: nn.Module,
    source: ReleasedTokenizerSource,
    prefix: str,
    *,
    initialization: str,
    seed: int,
) -> None:
    target = module.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    for local_name, target_value in target.items():
        source_name = f"{prefix}.{local_name}"
        released = source.tensor(source_name)
        if released.shape == target_value.shape:
            loaded[local_name] = released
        elif (
            target_value.ndim > 0
            and target_value.shape[0] == 1
            and released.ndim == target_value.ndim
            and released.shape[0] == RELEASED_CATEGORY_CAPACITY
            and released.shape[1:] == target_value.shape[1:]
        ):
            loaded[local_name] = initialized_category_row(
                source_name,
                released,
                initialization=initialization,
                seed=seed,
            )
        else:
            raise ValueError(
                f"cannot load {source_name}: released {tuple(released.shape)} vs "
                f"compact {tuple(target_value.shape)}"
            )
    missing, unexpected = module.load_state_dict(loaded, strict=True)
    if missing or unexpected:
        raise AssertionError(f"compact load mismatch: {missing=} {unexpected=}")


class FrozenActionOnlyProjection(nn.Module):
    """Released action-only fusion route and continuous L2 bridge."""

    def __init__(self, hidden_size: int, query_num: int, l2_dim: int):
        super().__init__()
        self.align_action = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.shared_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.vq_down_resampler = nn.Sequential(nn.Linear(hidden_size, l2_dim))
        self.bridge_projector = nn.Sequential(nn.Linear(l2_dim, hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(query_num, hidden_size))

    def encode_l2(self, action_query_features: torch.Tensor) -> torch.Tensor:
        unit = self.shared_projection(self.align_action(action_query_features))
        return self.vq_down_resampler(unit)

    def decode_l2(self, z_action: torch.Tensor) -> torch.Tensor:
        return self.bridge_projector(z_action) + self.pos_embed.unsqueeze(0)


class TReXResidualAdapter(nn.Module):
    """A2-only small residual calibration in the 32-D continuous interface."""

    def __init__(self, l2_dim: int = L2_DIM):
        super().__init__()
        self.norm = nn.LayerNorm(l2_dim)
        self.proj = nn.Linear(l2_dim, l2_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.proj(self.norm(value))


class TReXActionBootstrap(nn.Module):
    """Row-isolated T-Rex action autoencoder using Original-UniT shared paths."""

    def __init__(
        self,
        source: ReleasedTokenizerSource,
        *,
        initialization: str = "old_rows_mean",
        seed: int = 0,
        enable_a2_adapter: bool = False,
    ):
        super().__init__()
        validate_registration()
        self.initialization = initialization
        self.seed = int(seed)
        self.source_identity = source.identity
        encoder_cfg = dict(source.config["action_encoder_cfg"])
        decoder_cfg = dict(source.config["action_decoder_cfg"])
        encoder_cfg["max_num_embodiments"] = 1
        decoder_cfg["max_num_embodiments"] = 1
        self.action_encoder = ActionEncoder(ActionEncoderConfig(**encoder_cfg))
        self.action_decoder = ActionDecoder(ActionDecoderConfig(**decoder_cfg))
        self.action_path = FrozenActionOnlyProjection(
            hidden_size=int(source.config["hidden_size"]),
            query_num=int(source.config["query_num"]),
            l2_dim=int(source.config["vq_cfg"]["e_dim"]),
        )
        self.a2_adapter = TReXResidualAdapter() if enable_a2_adapter else nn.Identity()

        _load_compact_submodule(
            self.action_encoder,
            source,
            "action_branch",
            initialization=initialization,
            seed=seed,
        )
        _load_compact_submodule(
            self.action_decoder,
            source,
            "action_decoder",
            initialization=initialization,
            seed=seed,
        )
        projection_state = {}
        for local_name in self.action_path.state_dict():
            if local_name.startswith("align_action."):
                source_name = f"fusion.{local_name}"
            elif local_name.startswith("shared_projection."):
                source_name = f"fusion.{local_name}"
            else:
                source_name = local_name
            projection_state[local_name] = source.tensor(source_name)
        self.action_path.load_state_dict(projection_state, strict=True)
        self.configure_trainable(stage="A2" if enable_a2_adapter else "A1")

    @staticmethod
    def _is_category_parameter(module: nn.Module, local_name: str) -> bool:
        return isinstance(
            module,
            (CategorySpecificCausalConv1D, CategorySpecificLayerNorm, CategorySpecificLinear),
        ) and local_name in {"W", "b", "weight", "bias"}

    def configure_trainable(self, *, stage: str) -> None:
        if stage not in {"A0", "A1", "A2"}:
            raise ValueError("stage must be A0, A1, or A2")
        self.requires_grad_(False)
        if stage in {"A1", "A2"}:
            for root in (self.action_encoder, self.action_decoder):
                for module in root.modules():
                    for name, parameter in module.named_parameters(recurse=False):
                        if self._is_category_parameter(module, name):
                            parameter.requires_grad_(True)
        if stage == "A2":
            if isinstance(self.a2_adapter, nn.Identity):
                raise ValueError("A2 requires an enabled residual adapter")
            self.a2_adapter.requires_grad_(True)
        self.train_stage = stage

    def _validate_inputs(
        self, state: torch.Tensor, action: torch.Tensor, embodiment_id: torch.Tensor
    ) -> torch.Tensor:
        if state.ndim != 2 or state.shape[-1] != CANONICAL_DIM:
            raise ValueError("canonical T-Rex state must have shape [B,128]")
        if action.ndim != 3 or action.shape[1:] != (ACTION_HORIZON, CANONICAL_DIM):
            raise ValueError("canonical T-Rex action must have shape [B,16,128]")
        if embodiment_id.shape != (state.shape[0],):
            raise ValueError("embodiment_id must have shape [B]")
        if torch.any(embodiment_id < 0) or torch.any(
            embodiment_id >= EXPANDED_CATEGORY_CAPACITY
        ):
            raise IndexError("embodiment ID is outside expanded 32-slot capacity")
        if torch.any(embodiment_id != TREX_EMBODIMENT_ID):
            raise ValueError("compact Track A path accepts only explicit T-Rex ID 31")
        return torch.zeros_like(embodiment_id)

    def encode(
        self, state: torch.Tensor, action: torch.Tensor, embodiment_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_id = self._validate_inputs(state, action, embodiment_id)
        # The canonical data contract stores one [128] state vector.  The
        # released CategorySpecificMLP consumes it as a one-token sequence.
        l1, state_features = self.action_encoder(action, state.unsqueeze(1), local_id)
        z_action = self.a2_adapter(self.action_path.encode_l2(l1))
        if z_action.shape[1:] != (QUERY_NUM, L2_DIM):
            raise AssertionError("T-Rex Action L2 interface is not [B,8,32]")
        if not torch.isfinite(z_action).all():
            raise FloatingPointError("T-Rex Action L2 contains NaN/Inf")
        return z_action, state_features, l1

    def decode(
        self,
        z_action: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
    ) -> torch.Tensor:
        if z_action.ndim != 3 or z_action.shape[1:] != (QUERY_NUM, L2_DIM):
            raise ValueError("z_action must have shape [B,8,32]")
        if embodiment_id.shape != (z_action.shape[0],) or torch.any(
            embodiment_id != TREX_EMBODIMENT_ID
        ):
            raise ValueError("decoder requires explicit T-Rex ID 31")
        local_id = torch.zeros_like(embodiment_id)
        latent = self.action_path.decode_l2(z_action)
        output = self.action_decoder._decode_internal(latent, state_features, local_id)
        if output.shape[1:] != (ACTION_HORIZON, CANONICAL_DIM):
            raise AssertionError("decoded action does not satisfy [B,16,128]")
        return output

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, embodiment_id: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        z_action, state_features, l1 = self.encode(state, action, embodiment_id)
        prediction = self.decode(z_action, state_features, embodiment_id)
        return {"prediction": prediction, "z_action": z_action, "l1": l1}

    def trainable_summary(self) -> dict[str, int | float]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "trainable_fraction": trainable / max(total, 1),
        }

    def overlay_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only T-Rex-owned values; shared/old/RQ tensors are excluded."""

        result: dict[str, torch.Tensor] = {}
        for root_name in ("action_encoder", "action_decoder"):
            root = getattr(self, root_name)
            for module_name, module in root.named_modules():
                for parameter_name, parameter in module.named_parameters(recurse=False):
                    if self._is_category_parameter(module, parameter_name):
                        key = ".".join(
                            value
                            for value in (root_name, module_name, parameter_name)
                            if value
                        )
                        result[key] = parameter.detach().cpu().clone()
        if not isinstance(self.a2_adapter, nn.Identity):
            for name, value in self.a2_adapter.state_dict().items():
                result[f"a2_adapter.{name}"] = value.detach().cpu().clone()
        return result

    def load_overlay_state_dict(self, values: Mapping[str, torch.Tensor]) -> None:
        expected = self.overlay_state_dict()
        if set(values) != set(expected):
            missing = sorted(set(expected) - set(values))
            unexpected = sorted(set(values) - set(expected))
            raise ValueError(f"overlay key mismatch: {missing=} {unexpected=}")
        own = dict(self.named_parameters())
        for name, value in values.items():
            if name not in own:
                raise ValueError(f"overlay value is not an owned parameter: {name}")
            if own[name].shape != value.shape:
                raise ValueError(f"overlay shape mismatch for {name}")
            own[name].data.copy_(value.to(device=own[name].device, dtype=own[name].dtype))


def save_bootstrap_checkpoint(
    path: Path | str,
    model: TReXActionBootstrap,
    *,
    metadata: Mapping[str, Any],
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "tactile3d-unit.s3-3-trex-action-overlay.v1",
        "source_identity": model.source_identity,
        "initialization": model.initialization,
        "seed": model.seed,
        "stage": model.train_stage,
        "capacity": EXPANDED_CATEGORY_CAPACITY,
        "trex_embodiment_id": TREX_EMBODIMENT_ID,
        "unused_embodiment_id": UNUSED_EMBODIMENT_ID,
        "metadata": dict(metadata),
        "overlay": model.overlay_state_dict(),
    }
    torch.save(payload, path)
    return _sha256_file(path)


def load_bootstrap_checkpoint(
    path: Path | str,
    source: ReleasedTokenizerSource,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[TReXActionBootstrap, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("schema") != "tactile3d-unit.s3-3-trex-action-overlay.v1":
        raise ValueError("unsupported T-Rex Action checkpoint schema")
    if payload.get("source_identity") != source.identity:
        raise ValueError("T-Rex overlay does not match the released tokenizer source")
    if int(payload.get("capacity", -1)) != EXPANDED_CATEGORY_CAPACITY:
        raise ValueError("T-Rex overlay does not declare 32 category slots")
    if int(payload.get("trex_embodiment_id", -1)) != TREX_EMBODIMENT_ID:
        raise ValueError("T-Rex overlay embodiment ID changed")
    stage = str(payload.get("stage"))
    model = TReXActionBootstrap(
        source,
        initialization=str(payload["initialization"]),
        seed=int(payload["seed"]),
        enable_a2_adapter=stage == "A2",
    )
    model.load_overlay_state_dict(payload["overlay"])
    model.configure_trainable(stage=stage)
    return model, dict(payload.get("metadata", {}))


def overlay_to_released_name(compact_name: str) -> str:
    """Translate an overlay key to the released checkpoint tensor name."""

    if compact_name.startswith("action_encoder."):
        return "action_branch." + compact_name.removeprefix("action_encoder.")
    if compact_name.startswith("action_decoder."):
        return compact_name
    raise ValueError("A2 adapter parameters do not map to released category tensors")


def materialize_expanded_category_rows(
    source: ReleasedTokenizerSource,
    overlay: Mapping[str, torch.Tensor],
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Build the 32-row category portion of a deployable Action checkpoint."""

    expanded: dict[str, torch.Tensor] = {}
    mapped = {
        overlay_to_released_name(name): value
        for name, value in overlay.items()
        if not name.startswith("a2_adapter.")
    }
    expected = set(source.category_tensor_names())
    if set(mapped) != expected:
        raise ValueError("overlay does not cover every encoder/decoder category tensor")
    for name in sorted(expected):
        released = source.tensor(name)
        expanded[name] = expand_category_tensor(
            released,
            name=name,
            trex_row=mapped[name],
            seed=seed,
        )
    return expanded


def latent_noncollapse_losses(
    z_action: torch.Tensor,
    *,
    variance_floor: float = 0.05,
    diversity_margin: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Variance and query-separation penalties used by A1/A2."""

    flat = z_action.reshape(z_action.shape[0], -1)
    std = torch.sqrt(flat.var(dim=0, unbiased=False) + 1e-6)
    variance_loss = torch.relu(variance_floor - std).mean()
    normalized = torch.nn.functional.normalize(z_action, dim=-1)
    similarity = normalized @ normalized.transpose(1, 2)
    eye = torch.eye(QUERY_NUM, device=z_action.device, dtype=torch.bool).unsqueeze(0)
    off_diagonal = similarity.masked_select(~eye)
    diversity_loss = torch.relu(off_diagonal.abs() - (1.0 - diversity_margin)).mean()
    return variance_loss, diversity_loss


def effective_rank(values: torch.Tensor, eps: float = 1e-12) -> float:
    """Entropy effective rank of a two-dimensional feature matrix."""

    if values.ndim != 2:
        raise ValueError("effective_rank expects [samples,features]")
    centered = values.float() - values.float().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    probabilities = energy / energy.sum().clamp_min(eps)
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum()
    return float(torch.exp(entropy).item())


def query_diversity(z_action: torch.Tensor) -> dict[str, float]:
    normalized = torch.nn.functional.normalize(z_action.float(), dim=-1)
    distance = 1.0 - normalized @ normalized.transpose(1, 2)
    mask = ~torch.eye(QUERY_NUM, dtype=torch.bool, device=z_action.device)
    selected = distance[:, mask]
    return {
        "mean_cosine_distance": float(selected.mean().item()),
        "minimum_cosine_distance": float(selected.min().item()),
        "collapsed_pair_fraction": float((selected < 1e-3).float().mean().item()),
    }


def parameter_digest(values: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values, key=lambda item: item[0]):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


validate_registration()
