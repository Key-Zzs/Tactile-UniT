#!/usr/bin/env python3
"""Prove that pi05_base's official loss differentiates into the PI1B adapter."""

from __future__ import annotations

import dataclasses
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
OPENPI = ROOT / ".local/external/simulation/s4_3_pi1/openpi"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi1"


def write_json(name: str, payload: dict) -> None:
    target = ARTIFACTS / name
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)


def main() -> None:
    os.chdir(OPENPI)
    sys.path[:0] = [str(ROOT), str(OPENPI / "src")]

    import flax.traverse_util
    from flax import nnx
    import jax
    import jax.numpy as jnp
    import optax
    from gr00t.simulation.pi05_tactile_unit import install_openpi_runtime_hooks
    from openpi.shared import array_typing as at
    from openpi.shared import nnx_utils
    from openpi.training import data_loader
    from scripts.simulation.train_s4_3_pi1 import build_config

    install_openpi_runtime_hooks()
    config = dataclasses.replace(
        build_config("CONTACT_STATE_TOKENS", "loaded_base_gradient_gate", 0.0),
        batch_size=1,
        num_workers=0,
    )
    batch = next(iter(data_loader.create_data_loader(config, shuffle=False, num_batches=1)))

    # Reproduce the exact model RNG used by official train.init_train_state.
    _, init_rng = jax.random.split(jax.random.key(42))
    _, model_rng = jax.random.split(init_rng)
    model = config.model.create(model_rng)
    graphdef, state = nnx.split(model)
    reference = state.to_pure_dict()
    loaded = config.weight_loader.load(reference)
    at.check_pytree_equality(expected=reference, got=loaded, check_shapes=True, check_dtypes=True)

    flat_reference = flax.traverse_util.flatten_dict(reference, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded, sep="/")
    initialized_new_paths = sorted(
        name
        for name in flat_reference
        if name not in flat_loaded or any(token in name for token in ("contact_adapter", "lora"))
    )
    state.replace_by_pure_dict(loaded)
    model = nnx.merge(graphdef, state)

    observation, actions = batch
    loss_rng = jax.random.key(4302)

    def loss_fn(candidate):
        return candidate.compute_loss_with_info(loss_rng, observation, actions, train=True)[0]

    adapter_filter = nnx.All(nnx.Param, nnx_utils.PathRegex(".*contact_adapter.*"))
    loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, adapter_filter))(model)
    gradient_norm = float(jax.device_get(optax.global_norm(grads)))
    loss_value = float(jax.device_get(loss))

    zero_observation = dataclasses.replace(observation, contact_state=jnp.zeros_like(observation.contact_state))
    loaded_loss = float(
        jax.device_get(model.compute_loss_with_info(loss_rng, observation, actions, train=False)[0])
    )
    zero_contact_loss = float(
        jax.device_get(model.compute_loss_with_info(loss_rng, zero_observation, actions, train=False)[0])
    )
    gates = {
        "exact_weight_tree_shape_and_dtype": True,
        "pi05_base_not_B0_checkpoint": "pi05_base/params" in str(config.weight_loader.params_path),
        "official_loss_finite": math.isfinite(loss_value),
        "contact_adapter_gradient_finite": math.isfinite(gradient_norm),
        "contact_adapter_gradient_nonzero": gradient_norm > 0,
        "contact_state_conditions_official_loss": loaded_loss != zero_contact_loss,
        "action_shape_unchanged": actions.shape == (1, 30, 32),
        "no_E_T_optimizer_parameters": not any("E_T" in name for name in flat_reference),
        "only_declared_missing_parameter_families": all(
            any(token in name for token in ("contact_adapter", "lora")) for name in initialized_new_paths
        ),
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi1-loaded-base-adapter-gradient.v1",
        "mode": "CONTACT_STATE_TOKENS",
        "initialization": "$REPO_ROOT/.local/external/s4_3_pi0/models/DexJoCo-Pi05/pi05_base/params",
        "seed": 42,
        "fixed_loader_positions": [0],
        "official_loss": loss_value,
        "adapter_gradient_norm": gradient_norm,
        "loss_with_contact_state": loaded_loss,
        "loss_with_zero_contact_state": zero_contact_loss,
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "status": "PASS" if all(gates.values()) else "FAIL",
    }
    write_json("pi1b_loaded_base_gradient.json", payload)

    b1_path = ARTIFACTS / "pi1b_mode_validation.json"
    b1 = json.loads(b1_path.read_text())
    b1["loaded_base_adapter_gradient_norm"] = gradient_norm
    b1["gates"]["loaded_pi05_base_official_loss_adapter_gradient"] = payload["status"]
    b1["status"] = "PASS" if all(value == "PASS" for value in b1["gates"].values()) else "FAIL"
    write_json("pi1b_mode_validation.json", b1)

    freeze_path = ARTIFACTS / "training_protocol_freeze.json"
    freeze = json.loads(freeze_path.read_text())
    freeze["loaded_base_adapter_gradient_gate"] = payload["status"]
    freeze["status"] = "PASS" if payload["status"] == "PASS" and b1["status"] == "PASS" else "FAIL"
    write_json("training_protocol_freeze.json", freeze)
    print(json.dumps({"status": payload["status"], "official_loss": loss_value, "adapter_gradient_norm": gradient_norm}))
    if payload["status"] != "PASS":
        raise SystemExit("S4_3_PI1_MODE_IMPLEMENTATION_FAIL")


if __name__ == "__main__":
    main()
