#!/usr/bin/env python3
"""Validate B_HVA gradients from the exact loaded pi05_base before training."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
OPENPI = ROOT / ".local/external/simulation/s4_3_pi1/openpi"
ARTIFACT = ROOT / ".local/artifacts/simulation/s4_3_pi2m/loaded_base_gradient_gate.json"
BASE_PARAMS = ROOT / ".local/external/s4_3_pi0/models/DexJoCo-Pi05/pi05_base/params"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(32 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if ARTIFACT.exists():
        raise SystemExit("refusing to overwrite the PI2M loaded-base gradient gate")
    os.chdir(OPENPI)
    sys.path[:0] = [str(ROOT), str(OPENPI / "src")]
    import flax.traverse_util
    from flax import nnx
    import jax
    import optax
    from gr00t.simulation.pi05_tactile_unit import (
        TactilePi0,
        count_named_parameters,
        install_openpi_runtime_hooks,
    )
    from gr00t.simulation.s4_3_pi1 import TactileUnitMode
    from openpi.shared import array_typing as at
    from openpi.shared import nnx_utils
    from openpi.training import data_loader
    from scripts.simulation.train_s4_3_pi2m_bhva import build_config

    install_openpi_runtime_hooks()
    config = dataclasses.replace(
        build_config("s43_pi2m_bhva_loaded_base_gradient_gate", 0.01, 1),
        batch_size=1,
        num_workers=0,
    )
    observation, actions = next(
        iter(data_loader.create_data_loader(config, shuffle=False, num_batches=1))
    )
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

    def auxiliary_loss(candidate):
        return candidate.compute_physical_auxiliary_loss(
            loss_rng, observation, actions, train=True
        )

    gradient_filter = nnx.All(
        nnx.Param,
        nnx.Any(
            nnx_utils.PathRegex(".*contact_adapter.*"),
            nnx_utils.PathRegex(".*physical_auxiliary.*"),
            nnx_utils.PathRegex(".*lora.*"),
        ),
    )
    loss, gradients = nnx.value_and_grad(
        auxiliary_loss, argnums=nnx.DiffState(0, gradient_filter)
    )(model)

    def norm(pattern: str) -> float:
        selected = gradients.filter(nnx_utils.PathRegex(pattern))
        return float(jax.device_get(optax.global_norm(selected))) if jax.tree.leaves(selected) else 0.0

    gradient_norms = {
        "contact_adapter": norm(".*contact_adapter.*"),
        "physical_auxiliary": norm(".*physical_auxiliary.*"),
        "pi05_lora": norm(".*lora.*"),
    }
    flat = flax.traverse_util.flatten_dict(nnx.state(model).to_pure_dict(), sep="/")
    counts = count_named_parameters(model)
    gates = {
        "exact_weight_tree_shape_and_dtype": True,
        "exact_pi05_base_source": config.weight_loader.params_path == str(BASE_PARAMS),
        "mode_B_HVA": type(model) is TactilePi0
        and model.tactile_unit_mode
        is TactileUnitMode.CONTACT_STATE_TOKENS_VA_PHYSICAL_AUX,
        "initial_step_before_optimizer_updates": True,
        "seed42": config.seed == 42,
        "batch_is_TRAIN_only_position_zero": actions.shape == (1, 30, 32),
        "current_contact_state_present": observation.contact_state.shape == (1, 256),
        "VA_target_present": observation.va_shared_target.shape == (1, 8, 32),
        "contact_target_absent": observation.contact_shared_target is None
        and observation.physical_aux_valid is None,
        "auxiliary_loss_finite_positive": math.isfinite(float(jax.device_get(loss)))
        and float(jax.device_get(loss)) > 0,
        "auxiliary_gradient_reaches_contact_adapter": math.isfinite(
            gradient_norms["contact_adapter"]
        )
        and gradient_norms["contact_adapter"] > 0,
        "auxiliary_gradient_reaches_new_head": math.isfinite(
            gradient_norms["physical_auxiliary"]
        )
        and gradient_norms["physical_auxiliary"] > 0,
        "auxiliary_gradient_reaches_authorized_pi05_lora": math.isfinite(
            gradient_norms["pi05_lora"]
        )
        and gradient_norms["pi05_lora"] > 0,
        "contact_adapter_parameter_count_199680": counts["contact_adapter"] == 199_680,
        "auxiliary_head_parameter_count_658176": counts["physical_auxiliary"] == 658_176,
        "no_S4_2_parameters": not any(
            any(label in name for label in ("E_T", "A0", "C3", "B3", "A_H"))
            for name in flat
        ),
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2m-loaded-base-gradient-gate.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "mode": TactileUnitMode.CONTACT_STATE_TOKENS_VA_PHYSICAL_AUX.value,
        "seed": 42,
        "fixed_train_loader_positions": [0],
        "initialization": "$REPO_ROOT/.local/external/s4_3_pi0/models/DexJoCo-Pi05/pi05_base/params",
        "auxiliary_loss": float(jax.device_get(loss)),
        "gradient_norms": gradient_norms,
        "parameter_counts": counts,
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    temporary = ARTIFACT.with_suffix(ARTIFACT.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(ARTIFACT)
    print(json.dumps({"status": payload["status"], "gradient_norms": gradient_norms}, sort_keys=True))
    if payload["status"] != "PASS":
        failed = [name for name, value in gates.items() if not value]
        raise SystemExit("PI2M_LOADED_BASE_GRADIENT_FAIL:" + ",".join(failed))


if __name__ == "__main__":
    main()
