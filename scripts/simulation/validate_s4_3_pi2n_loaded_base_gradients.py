#!/usr/bin/env python3
"""Validate PI2N auxiliary gradients from the exact loaded pi05_base."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
OPENPI = ROOT / ".local/external/simulation/s4_3_pi1/openpi"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2n"
BASE_PARAMS = ROOT / ".local/external/s4_3_pi0/models/DexJoCo-Pi05/pi05_base/params"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", choices=("B_VA27", "B_VAC_V"), required=True)
    args = parser.parse_args()
    artifact = ARTIFACTS / f"{args.model_id.lower()}_loaded_base_gradient_gate.json"
    if artifact.exists():
        raise SystemExit(f"refusing to overwrite gradient gate: {artifact}")
    os.chdir(OPENPI)
    sys.path[:0] = [str(ROOT), str(OPENPI / "src")]
    import dataclasses
    import flax.traverse_util
    from flax import nnx
    import jax
    import jax.numpy as jnp
    import optax
    from gr00t.simulation.pi05_tactile_unit import TactilePi0, count_named_parameters, install_openpi_runtime_hooks
    from gr00t.simulation.s4_3_pi1 import TactileUnitMode
    from openpi.shared import array_typing as at
    from openpi.shared import nnx_utils
    from openpi.training import data_loader
    from scripts.simulation.train_s4_3_pi2n import build_config

    install_openpi_runtime_hooks()
    config = dataclasses.replace(
        build_config(args.model_id, f"s43_pi2n_{args.model_id.lower()}_loaded_base_gradient_gate", 0.01, 1),
        batch_size=1,
        num_workers=0,
    )
    observation, actions = next(iter(data_loader.create_data_loader(config, shuffle=False, num_batches=1)))
    _, init_rng = jax.random.split(jax.random.key(config.seed))
    _, model_rng = jax.random.split(init_rng)
    model = config.model.create(model_rng)
    graphdef, state = nnx.split(model)
    reference = state.to_pure_dict()
    loaded = config.weight_loader.load(reference)
    at.check_pytree_equality(expected=reference, got=loaded, check_shapes=True, check_dtypes=True)
    state.replace_by_pure_dict(loaded)
    model = nnx.merge(graphdef, state)
    loss_rng = jax.random.key(4303)

    gradient_filter = nnx.All(
        nnx.Param,
        nnx.Any(
            nnx_utils.PathRegex(".*contact_adapter.*"),
            nnx_utils.PathRegex(".*physical_auxiliary.*"),
            nnx_utils.PathRegex(".*lora.*"),
        ),
    )

    def auxiliary_loss(candidate):
        return candidate.compute_physical_auxiliary_loss(loss_rng, observation, actions, train=True)

    def policy_loss(candidate):
        _, info = candidate.compute_loss_with_info(loss_rng, observation, actions, train=True)
        return info["official_loss"]

    auxiliary_value, auxiliary_grads = nnx.value_and_grad(
        auxiliary_loss, argnums=nnx.DiffState(0, gradient_filter)
    )(model)
    policy_value, policy_grads = nnx.value_and_grad(
        policy_loss, argnums=nnx.DiffState(0, gradient_filter)
    )(model)

    def selected(grads, pattern: str):
        return grads.filter(nnx_utils.PathRegex(pattern))

    def norm(grads, pattern: str) -> float:
        values = selected(grads, pattern)
        return float(jax.device_get(optax.global_norm(values))) if jax.tree.leaves(values) else 0.0

    policy_lora = selected(policy_grads, ".*lora.*")
    auxiliary_lora = selected(auxiliary_grads, ".*lora.*")
    policy_leaves = jax.tree.leaves(policy_lora)
    auxiliary_leaves = jax.tree.leaves(auxiliary_lora)
    dot = sum((jnp.vdot(left, right) for left, right in zip(policy_leaves, auxiliary_leaves, strict=True)), jnp.asarray(0.0))
    policy_norm = optax.global_norm(policy_lora)
    auxiliary_norm = optax.global_norm(auxiliary_lora)
    cosine = dot / jnp.maximum(policy_norm * auxiliary_norm, jnp.asarray(1e-12))

    norms = {
        "g_pi_lora": norm(policy_grads, ".*lora.*"),
        "g_aux_lora": norm(auxiliary_grads, ".*lora.*"),
        "g_aux_physical_head": norm(auxiliary_grads, ".*physical_auxiliary.*"),
        "g_aux_contact_adapter": norm(auxiliary_grads, ".*contact_adapter.*"),
        "cos_g_pi_g_aux_common_lora": float(jax.device_get(cosine)),
    }
    expected_mode = TactileUnitMode.VA_PHYSICAL_AUX if args.model_id == "B_VA27" else TactileUnitMode.CONTACT_STATE_TOKENS_VAC_V_PHYSICAL_AUX
    flat = flax.traverse_util.flatten_dict(nnx.state(model).to_pure_dict(), sep="/")
    counts = count_named_parameters(model)
    gates = {
        "exact_weight_tree_shape_and_dtype": True,
        "exact_pi05_base_source": config.weight_loader.params_path == str(BASE_PARAMS),
        "mode_exact": type(model) is TactilePi0 and model.tactile_unit_mode is expected_mode,
        "initial_step_before_optimizer_updates": True,
        "seed42": config.seed == 42,
        "fixed_TRAIN_loader_position_zero": actions.shape == (1, 30, 32),
        "B_VA27_NO_H": args.model_id != "B_VA27" or observation.contact_state is None,
        "B_VAC_V_H_present": args.model_id != "B_VAC_V" or observation.contact_state.shape == (1, 256),
        "auxiliary_loss_finite_positive": math.isfinite(float(jax.device_get(auxiliary_value))) and float(jax.device_get(auxiliary_value)) > 0,
        "policy_loss_finite_positive": math.isfinite(float(jax.device_get(policy_value))) and float(jax.device_get(policy_value)) > 0,
        "auxiliary_gradient_reaches_new_head": math.isfinite(norms["g_aux_physical_head"]) and norms["g_aux_physical_head"] > 0,
        "auxiliary_gradient_reaches_authorized_pi05_lora": math.isfinite(norms["g_aux_lora"]) and norms["g_aux_lora"] > 0,
        "B_VAC_V_auxiliary_gradient_reaches_contact_adapter": args.model_id != "B_VAC_V" or (math.isfinite(norms["g_aux_contact_adapter"]) and norms["g_aux_contact_adapter"] > 0),
        "B_VA27_has_no_contact_adapter": args.model_id != "B_VA27" or counts["contact_adapter"] == 0,
        "physical_auxiliary_parameter_count_658176": counts["physical_auxiliary"] == 658_176,
        "no_S4_2_parameters": not any(any(label in name for label in ("E_T", "A0", "C3", "B3", "A_H")) for name in flat),
        "cosine_finite": math.isfinite(norms["cos_g_pi_g_aux_common_lora"]),
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2n-loaded-base-gradient-gate.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "model_id": args.model_id,
        "mode": expected_mode.value,
        "seed": 42,
        "fixed_train_loader_positions": [0],
        "initialization": "$REPO_ROOT/.local/external/s4_3_pi0/models/DexJoCo-Pi05/pi05_base/params",
        "policy_loss": float(jax.device_get(policy_value)),
        "auxiliary_loss": float(jax.device_get(auxiliary_value)),
        "gradient_norms": norms,
        "parameter_counts": counts,
        "interpretation": "single common-initialization TRAIN diagnostic; a cosine is not a causal closed-loop result",
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
    }
    atomic_json(artifact, payload)
    print(json.dumps({"status": payload["status"], "model_id": args.model_id, "gradient_norms": norms}, sort_keys=True))
    if payload["status"] != "PASS":
        failed = [name for name, value in gates.items() if not value]
        raise SystemExit("PI2N_LOADED_BASE_GRADIENT_FAIL:" + ",".join(failed))


if __name__ == "__main__":
    main()
