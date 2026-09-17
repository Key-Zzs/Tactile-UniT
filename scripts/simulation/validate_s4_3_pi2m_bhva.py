#!/usr/bin/env python3
"""Validate B_HVA mode parity, data isolation, masking, and inference guards."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OPENPI = ROOT / ".local/external/simulation/s4_3_pi1/openpi"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2m"
TARGET = ROOT / ".local/datasets/simulation/s4_3_pi2m/pinch_tongs_va_t27/sidecar.npz"
B2_SIDECAR = ROOT / ".local/datasets/simulation/s4_3_pi1/pinch_tongs_official_tactile/sidecar.npz"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def state_hash(model, contains: str) -> str:
    import flax.traverse_util
    from flax import nnx
    import jax

    digest = hashlib.sha256()
    matches = 0
    flat = flax.traverse_util.flatten_dict(nnx.state(model).to_pure_dict(), sep="/")
    for name, value in sorted(flat.items()):
        if contains not in name:
            continue
        array = np.asarray(jax.device_get(value))
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
        matches += 1
    if not matches:
        raise RuntimeError(f"no model state matched {contains}")
    return digest.hexdigest()


def make_observation(*, target: float | None = None, valid: bool = True):
    import jax.numpy as jnp
    from gr00t.simulation.pi05_tactile_unit import TactileObservation

    return TactileObservation(
        images={
            "base_0_rgb": jnp.linspace(-1, 1, 224 * 224 * 3).reshape(1, 224, 224, 3),
            "left_wrist_0_rgb": jnp.zeros((1, 224, 224, 3), dtype=jnp.float32),
            "right_wrist_0_rgb": jnp.ones((1, 224, 224, 3), dtype=jnp.float32),
        },
        image_masks={
            "base_0_rgb": jnp.ones((1,), dtype=jnp.bool_),
            "left_wrist_0_rgb": jnp.ones((1,), dtype=jnp.bool_),
            "right_wrist_0_rgb": jnp.zeros((1,), dtype=jnp.bool_),
        },
        state=jnp.linspace(-0.5, 0.5, 32).reshape(1, 32),
        tokenized_prompt=jnp.arange(8, dtype=jnp.int32).reshape(1, 8),
        tokenized_prompt_mask=jnp.ones((1, 8), dtype=jnp.bool_),
        contact_state=jnp.linspace(-1, 1, 256).reshape(1, 256),
        va_shared_target=(
            None if target is None else jnp.full((1, 8, 32), target, dtype=jnp.float32)
        ),
        va_aux_valid=(None if target is None else jnp.asarray([valid], dtype=jnp.bool_)),
    )


def sidecar_fixture_gates() -> dict[str, bool]:
    from gr00t.simulation.pi05_tactile_unit import _SidecarDataset
    from gr00t.simulation.s4_3_pi1 import TactileUnitMode

    class FixtureDataset:
        def __len__(self):
            return 3

        def __getitem__(self, index):
            return {"index": np.asarray(index, dtype=np.int64), "official": np.asarray(index + 10)}

    with tempfile.TemporaryDirectory(prefix="s43_pi2m_sidecar_") as directory:
        directory = Path(directory)
        contact = directory / "contact.npz"
        target = directory / "target.npz"
        np.savez(
            contact,
            index=np.arange(3, dtype=np.int64),
            contact_state=np.arange(3 * 256, dtype=np.float32).reshape(3, 256),
        )
        np.savez(
            target,
            index=np.arange(3, dtype=np.int64),
            va_shared_target=np.arange(3 * 8 * 32, dtype=np.float32).reshape(3, 8, 32),
            va_aux_valid=np.asarray([True, False, True]),
        )
        wrapped = _SidecarDataset(
            FixtureDataset(),
            contact,
            TactileUnitMode.CONTACT_STATE_TOKENS_VA_PHYSICAL_AUX,
            target_sidecar_path=target,
        )
        item = wrapped[1]
        forbidden = {"contact_shared_target", "physical_aux_valid", "tactile_sim", "tactile_history"}
        return {
            "dual_sidecar_index_identity": int(item["index"]) == 1,
            "official_fields_preserved": int(item["official"]) == 11,
            "contact_state_shape": item["contact_state"].shape == (256,),
            "VA_target_shape": item["va_shared_target"].shape == (8, 32),
            "VA_valid_boolean": np.asarray(item["va_aux_valid"]).dtype == np.bool_,
            "contact_target_fields_not_read_or_emitted": not (set(item) & forbidden),
        }


def real_loader_gates() -> dict[str, bool]:
    import dataclasses
    from openpi.training import data_loader
    from scripts.simulation.train_s4_3_pi1 import build_config as build_b2_config
    from scripts.simulation.train_s4_3_pi2m_bhva import build_config as build_bhva_config

    b2_config = dataclasses.replace(
        build_b2_config(
            "CONTACT_STATE_TOKENS_PHYSICAL_AUX", "pi2m_b2_loader_fixture", 0.01
        ),
        batch_size=4,
        num_workers=0,
    )
    bhva_config = dataclasses.replace(
        build_bhva_config("pi2m_bhva_loader_fixture", 0.01, 1),
        batch_size=4,
        num_workers=0,
    )
    b2_observation, b2_actions = next(
        iter(data_loader.create_data_loader(b2_config, shuffle=False, num_batches=1))
    )
    bhva_observation, bhva_actions = next(
        iter(data_loader.create_data_loader(bhva_config, shuffle=False, num_batches=1))
    )
    same_images = set(b2_observation.images) == set(bhva_observation.images) and all(
        np.array_equal(
            np.asarray(b2_observation.images[name]), np.asarray(bhva_observation.images[name])
        )
        for name in b2_observation.images
    )
    return {
        "real_loader_same_images_as_B2": same_images,
        "real_loader_same_state_as_B2": np.array_equal(
            np.asarray(b2_observation.state), np.asarray(bhva_observation.state)
        ),
        "real_loader_same_prompt_as_B2": np.array_equal(
            np.asarray(b2_observation.tokenized_prompt),
            np.asarray(bhva_observation.tokenized_prompt),
        ),
        "real_loader_same_actions_as_B2": np.array_equal(
            np.asarray(b2_actions), np.asarray(bhva_actions)
        ),
        "real_loader_same_contact_state_as_B2": np.array_equal(
            np.asarray(b2_observation.contact_state),
            np.asarray(bhva_observation.contact_state),
        ),
        "real_loader_same_valid_mask_as_B2": np.array_equal(
            np.asarray(b2_observation.physical_aux_valid),
            np.asarray(bhva_observation.va_aux_valid),
        ),
        "real_loader_BHVA_has_VA_target": bhva_observation.va_shared_target.shape == (4, 8, 32),
        "real_loader_BHVA_has_no_contact_target": bhva_observation.contact_shared_target is None
        and bhva_observation.physical_aux_valid is None,
    }


def main() -> None:
    os.chdir(OPENPI)
    sys.path[:0] = [str(ROOT), str(OPENPI / "src")]
    import flax.traverse_util
    from flax import nnx
    import jax
    import jax.numpy as jnp
    import optax
    from gr00t.simulation.pi05_tactile_unit import (
        TactilePi0Config,
        count_named_parameters,
        install_openpi_runtime_hooks,
    )
    from gr00t.simulation.s4_3_pi1 import TactileUnitMode
    from openpi.models import pi0 as official_pi0
    from openpi.models.pi0_config import Pi0Config
    from openpi.shared import nnx_utils

    install_openpi_runtime_hooks()
    common = dict(
        pi05=True,
        action_dim=32,
        action_horizon=30,
        max_token_len=8,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        dtype="bfloat16",
    )
    init_key = jax.random.key(42)
    loss_key = jax.random.key(432)
    actions = jnp.linspace(-0.75, 0.75, 30 * 32).reshape(1, 30, 32)
    noise = jnp.linspace(0.9, -0.9, 30 * 32).reshape(1, 30, 32)
    inference_observation = make_observation()
    official_observation = dataclasses.replace(inference_observation, contact_state=None)
    low_target = make_observation(target=0.0)
    high_target = make_observation(target=1.0)
    invalid_target = make_observation(target=1.0, valid=False)

    official = Pi0Config(**common).create(init_key)
    none = TactilePi0Config(**common, tactile_unit_mode=TactileUnitMode.NONE).create(init_key)
    b1 = TactilePi0Config(
        **common, tactile_unit_mode=TactileUnitMode.CONTACT_STATE_TOKENS
    ).create(init_key)
    bva = TactilePi0Config(
        **common,
        tactile_unit_mode=TactileUnitMode.VA_PHYSICAL_AUX,
        lambda_phys=0.01,
    ).create(init_key)
    b2 = TactilePi0Config(
        **common,
        tactile_unit_mode=TactileUnitMode.CONTACT_STATE_TOKENS_PHYSICAL_AUX,
        lambda_phys=0.01,
    ).create(init_key)
    bhva_config = TactilePi0Config(
        **common,
        tactile_unit_mode=TactileUnitMode.CONTACT_STATE_TOKENS_VA_PHYSICAL_AUX,
        lambda_phys=0.01,
    )
    bhva = bhva_config.create(init_key)

    b2_prefix = tuple(np.asarray(jax.device_get(value)) for value in b2.embed_prefix(inference_observation))
    bhva_prefix = tuple(
        np.asarray(jax.device_get(value)) for value in bhva.embed_prefix(inference_observation)
    )
    b2_action = np.asarray(
        jax.device_get(b2.sample_actions(loss_key, inference_observation, num_steps=1, noise=noise))
    )
    bhva_action = np.asarray(
        jax.device_get(bhva.sample_actions(loss_key, inference_observation, num_steps=1, noise=noise))
    )
    official_action = np.asarray(
        jax.device_get(
            official.sample_actions(loss_key, official_observation, num_steps=1, noise=noise)
        )
    )
    none_action = np.asarray(
        jax.device_get(none.sample_actions(loss_key, official_observation, num_steps=1, noise=noise))
    )
    b1_action = np.asarray(
        jax.device_get(b1.sample_actions(loss_key, inference_observation, num_steps=1, noise=noise))
    )
    bva_action = np.asarray(
        jax.device_get(bva.sample_actions(loss_key, official_observation, num_steps=1, noise=noise))
    )
    low_total, low_info = bhva.compute_loss_with_info(loss_key, low_target, actions, train=False)
    high_total, high_info = bhva.compute_loss_with_info(loss_key, high_target, actions, train=False)
    invalid_total, invalid_info = bhva.compute_loss_with_info(
        loss_key, invalid_target, actions, train=False
    )
    low_total, low_info, high_total, high_info, invalid_total, invalid_info = jax.device_get(
        (low_total, low_info, high_total, high_info, invalid_total, invalid_info)
    )

    def auxiliary_loss(candidate):
        return candidate.compute_physical_auxiliary_loss(
            loss_key, high_target, actions, train=False
        )

    gradient_filter = nnx.All(
        nnx.Param,
        nnx.Any(
            nnx_utils.PathRegex(".*contact_adapter.*"),
            nnx_utils.PathRegex(".*physical_auxiliary.*"),
            nnx_utils.PathRegex(".*PaliGemma/llm.*"),
        ),
    )
    auxiliary_value, gradients = nnx.value_and_grad(
        auxiliary_loss, argnums=nnx.DiffState(0, gradient_filter)
    )(bhva)

    def norm(pattern: str) -> float:
        selected = gradients.filter(nnx_utils.PathRegex(pattern))
        return float(jax.device_get(optax.global_norm(selected))) if jax.tree.leaves(selected) else 0.0

    inference_guard = False
    try:
        bhva.sample_actions(loss_key, high_target, num_steps=1, noise=noise)
    except ValueError as error:
        inference_guard = "training-only" in str(error)

    b2_counts = count_named_parameters(b2)
    bhva_counts = count_named_parameters(bhva)
    bva_counts = count_named_parameters(bva)
    spec = bhva_config.inputs_spec(batch_size=2)[0]
    with np.load(TARGET, allow_pickle=False) as target_source, np.load(
        B2_SIDECAR, allow_pickle=False
    ) as contact_source:
        data_gates = {
            "target_index_equal_contact_index": np.array_equal(
                target_source["index"], contact_source["index"]
            ),
            "target_valid_equal_B2": np.array_equal(
                target_source["va_aux_valid"], contact_source["physical_aux_valid"]
            ),
            "target_sidecar_has_no_contact_fields": set(target_source.files)
            == {"index", "va_shared_target", "va_aux_valid"},
        }
    flat_names = list(flax.traverse_util.flatten_dict(nnx.state(bhva).to_pure_dict(), sep="/"))
    gates = {
        **sidecar_fixture_gates(),
        **real_loader_gates(),
        **data_gates,
        "NONE_remains_exact_official_class": type(none) is official_pi0.Pi0,
        "NONE_inference_remains_bit_identical": np.array_equal(none_action, official_action),
        "B1_inference_remains_bit_identical_to_B2": np.array_equal(b1_action, b2_action),
        "BVA_inference_remains_bit_identical_to_official": np.array_equal(
            bva_action, official_action
        ),
        "BVA_remains_without_contact_adapter": bva_counts["contact_adapter"] == 0,
        "BVA_auxiliary_head_unchanged": bva_counts["physical_auxiliary"]
        == b2_counts["physical_auxiliary"],
        "mode_exact": bhva.tactile_unit_mode
        is TactileUnitMode.CONTACT_STATE_TOKENS_VA_PHYSICAL_AUX,
        "same_contact_adapter_initialization_as_B2": state_hash(bhva, "contact_adapter")
        == state_hash(b2, "contact_adapter"),
        "same_auxiliary_head_initialization_as_B2": state_hash(bhva, "physical_auxiliary")
        == state_hash(b2, "physical_auxiliary"),
        "same_prefix_tokens_positions_and_masks_as_B2": all(
            np.array_equal(left, right) for left, right in zip(bhva_prefix, b2_prefix, strict=True)
        ),
        "same_inference_action_as_B2": np.array_equal(bhva_action, b2_action),
        "inference_action_shape": bhva_action.shape == (1, 30, 32),
        "VA_target_changes_only_auxiliary_loss": np.array_equal(
            np.asarray(low_info["official_loss"]), np.asarray(high_info["official_loss"])
        )
        and not np.array_equal(
            np.asarray(low_info["physical_loss"]), np.asarray(high_info["physical_loss"])
        ),
        "VA_target_changes_total_training_loss": not np.array_equal(
            np.asarray(low_total), np.asarray(high_total)
        ),
        "invalid_mask_zeros_auxiliary_loss": float(invalid_info["physical_loss"]) == 0.0,
        "invalid_mask_keeps_BC_loss": np.array_equal(
            np.asarray(invalid_total), np.asarray(invalid_info["official_loss"])
        ),
        "auxiliary_loss_finite": bool(np.isfinite(float(jax.device_get(auxiliary_value)))),
        "auxiliary_head_gradient_nonzero": np.isfinite(norm(".*physical_auxiliary.*"))
        and norm(".*physical_auxiliary.*") > 0,
        "action_hidden_gradient_nonzero": np.isfinite(norm(".*PaliGemma/llm.*"))
        and norm(".*PaliGemma/llm.*") > 0,
        "hard_training_target_inference_guard": inference_guard,
        "inference_observation_has_current_contact": inference_observation.contact_state.shape
        == (1, 256),
        "inference_observation_has_no_target": inference_observation.va_shared_target is None
        and inference_observation.va_aux_valid is None
        and inference_observation.contact_shared_target is None
        and inference_observation.physical_aux_valid is None,
        "training_spec_uses_VA_not_contact_target": spec.contact_state.shape == (2, 256)
        and spec.va_shared_target.shape == (2, 8, 32)
        and spec.va_aux_valid.shape == (2,)
        and spec.contact_shared_target is None
        and spec.physical_aux_valid is None,
        "parameter_counts_match_B2": bhva_counts == b2_counts,
        "contact_adapter_parameter_family_present": any("contact_adapter" in name for name in flat_names),
        "physical_auxiliary_parameter_family_present": any(
            "physical_auxiliary" in name for name in flat_names
        ),
        "no_S4_2_teacher_parameters": not any(
            any(label in name for label in ("E_T", "A0", "C3", "B3", "A_H"))
            for name in flat_names
        ),
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2m-bhva-mode-parity.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "mode": TactileUnitMode.CONTACT_STATE_TOKENS_VA_PHYSICAL_AUX.value,
        "parameter_counts": {"B2": b2_counts, "B_HVA": bhva_counts},
        "gradient_norms": {
            "contact_adapter": norm(".*contact_adapter.*"),
            "physical_auxiliary": norm(".*physical_auxiliary.*"),
            "action_hidden": norm(".*PaliGemma/llm.*"),
        },
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
    }
    atomic_json(ARTIFACTS / "mode_parity.json", payload)
    print(json.dumps({"status": payload["status"], "failed": [k for k, v in gates.items() if not v]}))
    if payload["status"] != "PASS":
        raise SystemExit("PI2M_BHVA_MODE_PARITY_FAIL")


if __name__ == "__main__":
    main()
