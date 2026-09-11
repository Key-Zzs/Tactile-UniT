#!/usr/bin/env python3
"""Validate BVA NONE parity, contact absence, and training-only target semantics."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OPENPI = ROOT / ".local/external/simulation/s4_3_pi1/openpi"
SIDECAR = ROOT / ".local/datasets/simulation/s4_3_pi2u/pinch_tongs_va/sidecar.npz"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2u"


def write_json(name: str, payload: dict) -> None:
    target = ARTIFACTS / name
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)


def tree_hash(model, *, exclude: str | None = None) -> str:
    import flax.traverse_util
    from flax import nnx
    import jax

    digest = hashlib.sha256()
    flat = flax.traverse_util.flatten_dict(nnx.state(model).to_pure_dict(), sep="/")
    for name, value in sorted(flat.items()):
        if exclude and exclude in name:
            continue
        array = np.asarray(jax.device_get(value))
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def observation(target_value: float | None):
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
        va_shared_target=(
            None if target_value is None else jnp.full((1, 8, 32), target_value, dtype=jnp.float32)
        ),
        va_aux_valid=(None if target_value is None else jnp.ones((1,), dtype=jnp.bool_)),
    )


def main() -> None:
    os.chdir(OPENPI)
    sys.path[:0] = [str(ROOT), str(OPENPI / "src")]
    import flax.traverse_util
    from flax import nnx
    import jax
    import jax.numpy as jnp
    from gr00t.simulation.pi05_tactile_unit import (
        TactilePi0Config,
        count_named_parameters,
        install_openpi_runtime_hooks,
    )
    from gr00t.simulation.s4_3_pi1 import TactileUnitMode
    from openpi.models import pi0 as official_pi0
    from openpi.models.pi0_config import Pi0Config

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
    no_target = observation(None)

    official = Pi0Config(**common).create(init_key)
    none = TactilePi0Config(**common, tactile_unit_mode=TactileUnitMode.NONE).create(init_key)
    official_prefix = tuple(np.asarray(jax.device_get(value)) for value in official.embed_prefix(no_target))
    none_prefix = tuple(np.asarray(jax.device_get(value)) for value in none.embed_prefix(no_target))
    official_loss = np.asarray(jax.device_get(official.compute_loss(loss_key, no_target, actions)))
    none_loss = np.asarray(jax.device_get(none.compute_loss(loss_key, no_target, actions)))
    parity_gates = {
        "NONE_is_exact_official_class": type(none) is official_pi0.Pi0,
        "NONE_model_state_bit_identical": tree_hash(official) == tree_hash(none),
        "NONE_prefix_tokens_bit_identical": all(np.array_equal(a, b) for a, b in zip(official_prefix, none_prefix)),
        "NONE_loss_bit_identical": np.array_equal(official_loss, none_loss),
    }
    write_json(
        "bva_none_parity.json",
        {
            "schema": "tactile3d-unit.s4-3-pi2u-bva-none-parity.v1",
            "status": "PASS" if all(parity_gates.values()) else "FAIL",
            "gates": {name: "PASS" if value else "FAIL" for name, value in parity_gates.items()},
        },
    )

    bva_config = TactilePi0Config(
        **common, tactile_unit_mode=TactileUnitMode.VA_PHYSICAL_AUX, lambda_phys=0.01
    )
    b2_config = TactilePi0Config(
        **common,
        tactile_unit_mode=TactileUnitMode.CONTACT_STATE_TOKENS_PHYSICAL_AUX,
        lambda_phys=0.01,
    )
    bva = bva_config.create(init_key)
    b2 = b2_config.create(init_key)
    low_observation = observation(0.0)
    high_observation = observation(1.0)
    bva_prefix = tuple(np.asarray(jax.device_get(value)) for value in bva.embed_prefix(low_observation))
    low_total, low_info = bva.compute_loss_with_info(loss_key, low_observation, actions, train=False)
    high_total, high_info = bva.compute_loss_with_info(loss_key, high_observation, actions, train=False)
    low_total, high_total, low_info, high_info = jax.device_get((low_total, high_total, low_info, high_info))
    flat_names = list(flax.traverse_util.flatten_dict(nnx.state(bva).to_pure_dict(), sep="/"))
    with np.load(SIDECAR, allow_pickle=False) as source:
        sidecar_fields = set(source.files)
        target_shape = source["va_shared_target"].shape
        valid_dtype = source["va_aux_valid"].dtype
    bva_counts = count_named_parameters(bva)
    b2_counts = count_named_parameters(b2)
    gates = {
        "BVA_prefix_exactly_official": all(np.array_equal(a, b) for a, b in zip(official_prefix, bva_prefix)),
        "BVA_official_loss_invariant_to_target": np.array_equal(
            np.asarray(low_info["official_loss"]), np.asarray(high_info["official_loss"])
        ),
        "BVA_auxiliary_loss_changes_with_target": not np.array_equal(
            np.asarray(low_info["physical_loss"]), np.asarray(high_info["physical_loss"])
        ),
        "BVA_total_loss_changes_with_target": not np.array_equal(np.asarray(low_total), np.asarray(high_total)),
        "BVA_graph_has_no_contact_parameter": not any("contact" in name.lower() for name in flat_names),
        "BVA_has_physical_head": bva_counts["physical_auxiliary"] > 0,
        "BVA_head_parameter_count_equals_B2": bva_counts["physical_auxiliary"] == b2_counts["physical_auxiliary"],
        "BVA_has_no_contact_adapter": bva_counts["contact_adapter"] == 0,
        "sidecar_fields_exact": sidecar_fields == {"index", "va_shared_target", "va_aux_valid"},
        "sidecar_target_shape": target_shape[1:] == (8, 32),
        "sidecar_valid_boolean": valid_dtype == np.bool_,
        "inference_spec_has_no_target": bva_config.inputs_spec(batch_size=1)[0].contact_state is None,
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2u-bva-mode-contract.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "mode": "VA_PHYSICAL_AUX",
        "runtime_modalities": ["official pi0.5 observations only"],
        "training_only_target": "va_shared_target[8,32]",
        "sidecar_fields": sorted(sidecar_fields),
        "parameter_counts": {"BVA": bva_counts, "B2": b2_counts},
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
    }
    write_json("bva_mode_contract.json", payload)
    print(json.dumps({"status": payload["status"], "none": all(parity_gates.values())}, sort_keys=True))
    if payload["status"] != "PASS" or not all(parity_gates.values()):
        raise SystemExit("BVA structural validation failed")


if __name__ == "__main__":
    main()
