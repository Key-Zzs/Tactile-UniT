#!/usr/bin/env python3
"""Pre-training parity and structural gates for all frozen PI1 modes."""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OPENPI = ROOT / ".local/external/simulation/s4_3_pi1/openpi"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi1"


def setup_imports() -> None:
    os.chdir(OPENPI)
    sys.path[:0] = [str(ROOT), str(OPENPI / "src")]


def symbolic(path: Path) -> str:
    return "$REPO_ROOT/" + path.resolve().relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(name: str, value: Any) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    target = ARTIFACTS / name
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)


def array_hash(tree, *, contains: str) -> str:
    from flax import nnx
    import flax.traverse_util
    import jax

    flat = flax.traverse_util.flatten_dict(nnx.state(tree).to_pure_dict(), sep="/")
    digest = hashlib.sha256()
    matched = 0
    for name, value in sorted(flat.items()):
        if contains not in name:
            continue
        array = np.asarray(jax.device_get(value))
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
        matched += 1
    if matched == 0:
        raise RuntimeError(f"no parameter path contains {contains}")
    return digest.hexdigest()


def prefix_arrays(model, observation):
    import jax

    tokens, mask, ar_mask = model.embed_prefix(observation)
    return tuple(np.asarray(jax.device_get(value)) for value in (tokens, mask, ar_mask))


def norm_matching(grads, pattern: str) -> float:
    from flax import nnx
    import jax
    import optax
    from openpi.shared import nnx_utils

    selected = grads.filter(nnx_utils.PathRegex(pattern))
    if not jax.tree.leaves(selected):
        return 0.0
    return float(jax.device_get(optax.global_norm(selected)))


def make_observation(*, tactile: bool, auxiliary: bool):
    import jax.numpy as jnp
    from gr00t.simulation.pi05_tactile_unit import TactileObservation
    from openpi.models import model as openpi_model

    cls = TactileObservation if tactile else openpi_model.Observation
    kwargs = dict(
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
    )
    if tactile:
        kwargs.update(
            contact_state=jnp.linspace(-1, 1, 256).reshape(1, 256),
            contact_shared_target=(jnp.linspace(-0.25, 0.25, 256).reshape(1, 8, 32) if auxiliary else None),
            physical_aux_valid=(jnp.ones((1,), dtype=jnp.bool_) if auxiliary else None),
        )
    return cls(**kwargs)


def compare_official_data_fields():
    import jax
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config, data_loader
    from scripts.simulation.train_s4_3_pi1 import build_config

    b1 = dataclasses.replace(build_config("CONTACT_STATE_TOKENS", "validation", 0.0), num_workers=0)
    official_model = Pi0Config(
        pi05=True,
        action_horizon=30,
        max_token_len=250,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    official = dataclasses.replace(
        config.get_config("pinch_tongs"),
        model=official_model,
        data=config.SingleArmDataConfig(
            root=b1.data.root,
            repo_id="local_repo",
            base_config=config.DataConfig(prompt_from_task=True),
        ),
        assets_base_dir=b1.assets_base_dir,
        batch_size=32,
        num_workers=0,
        freeze_filter=official_model.get_freeze_filter(),
    )
    left = next(iter(data_loader.create_data_loader(official, shuffle=False, num_batches=1)))
    right = next(iter(data_loader.create_data_loader(b1, shuffle=False, num_batches=1)))
    comparisons = {
        "state": np.array_equal(np.asarray(left[0].state), np.asarray(right[0].state)),
        "actions": np.array_equal(np.asarray(left[1]), np.asarray(right[1])),
        "prompt_tokens": np.array_equal(
            np.asarray(left[0].tokenized_prompt), np.asarray(right[0].tokenized_prompt)
        ),
        "prompt_mask": np.array_equal(
            np.asarray(left[0].tokenized_prompt_mask), np.asarray(right[0].tokenized_prompt_mask)
        ),
    }
    for name in left[0].images:
        comparisons[f"image_{name}"] = np.array_equal(
            np.asarray(left[0].images[name]), np.asarray(right[0].images[name])
        )
    contact = np.asarray(right[0].contact_state)
    comparisons["contact_state_shape"] = contact.shape == (32, 256)
    comparisons["contact_state_finite"] = bool(np.isfinite(contact).all())
    jax.clear_caches()
    return comparisons


def main() -> None:
    setup_imports()
    import jax
    import jax.numpy as jnp
    from flax import nnx
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
    data_gates = compare_official_data_fields()
    common = dict(
        pi05=True,
        action_dim=32,
        action_horizon=30,
        max_token_len=8,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        dtype="bfloat16",
    )
    actions = jnp.linspace(-0.75, 0.75, 30 * 32).reshape(1, 30, 32)
    noise = jnp.linspace(0.9, -0.9, 30 * 32).reshape(1, 30, 32)
    key = jax.random.key(4301)
    init_key = jax.random.key(42)

    # Official path fixture.
    official_config = Pi0Config(**common)
    official_model = official_config.create(init_key)
    official_observation = make_observation(tactile=False, auxiliary=False)
    official_loss = np.asarray(jax.device_get(official_model.compute_loss(key, official_observation, actions)))
    official_action = np.asarray(
        jax.device_get(official_model.sample_actions(key, official_observation, num_steps=1, noise=noise))
    )
    official_prefix = prefix_arrays(official_model, official_observation)
    official_parameter_hash = array_hash(official_model, contains="PaliGemma")
    del official_model
    gc.collect()

    # NONE must instantiate the exact official class and reproduce the fixed fixture bit-for-bit.
    none_config = TactilePi0Config(**common, tactile_unit_mode=TactileUnitMode.NONE)
    none_model = none_config.create(init_key)
    none_loss = np.asarray(jax.device_get(none_model.compute_loss(key, official_observation, actions)))
    none_action = np.asarray(
        jax.device_get(none_model.sample_actions(key, official_observation, num_steps=1, noise=noise))
    )
    none_prefix = prefix_arrays(none_model, official_observation)
    none_parameter_hash = array_hash(none_model, contains="PaliGemma")
    parity_gates = {
        **data_gates,
        "exact_official_model_class": type(none_model) is official_pi0.Pi0,
        "same_initialized_official_parameters": official_parameter_hash == none_parameter_hash,
        "same_prefix_tokens": np.array_equal(official_prefix[0], none_prefix[0]),
        "same_prefix_input_mask": np.array_equal(official_prefix[1], none_prefix[1]),
        "same_prefix_attention_semantics": np.array_equal(official_prefix[2], none_prefix[2]),
        "same_pi05_loss": np.array_equal(official_loss, none_loss),
        "same_action_inference": np.array_equal(official_action, none_action),
        "action_shape": none_action.shape == (1, 30, 32),
    }
    del none_model
    gc.collect()

    # B1 adapter structure, prefix insertion, and isolated adapter gradient.
    b1_config = TactilePi0Config(**common, tactile_unit_mode=TactileUnitMode.CONTACT_STATE_TOKENS)
    b1_model = b1_config.create(init_key)
    b1_observation = make_observation(tactile=True, auxiliary=False)
    b1_prefix = prefix_arrays(b1_model, b1_observation)
    b1_action = np.asarray(jax.device_get(b1_model.sample_actions(key, b1_observation, num_steps=1, noise=noise)))
    b1_adapter_hash = array_hash(b1_model, contains="contact_adapter")

    b1_loss = b1_model.compute_loss_with_info(key, b1_observation, actions, train=False)[0]

    def b1_adapter_probe(model):
        # pi0.5's random-init adaRMS gates are exactly zero, so its action loss
        # cannot test prefix conditioning until pi05_base is loaded.  This local
        # gate checks the adapter itself; a separate mandatory loaded-base gate
        # checks the end-to-end official-loss gradient.
        tokens = model.contact_adapter(b1_observation.contact_state)
        return jnp.mean(jnp.square(tokens))

    b1_diff = nnx.DiffState(0, nnx.All(nnx.Param, nnx_utils.PathRegex(".*contact_adapter.*")))
    _, b1_grads = nnx.value_and_grad(b1_adapter_probe, argnums=b1_diff)(b1_model)
    b1_grad_norm = norm_matching(b1_grads, ".*contact_adapter.*")
    b1_gates = {
        "input_256": b1_observation.contact_state.shape == (1, 256),
        "eight_prefix_tokens": b1_prefix[0].shape[1] == official_prefix[0].shape[1] + 8,
        "prefix_tokens_finite": bool(np.isfinite(b1_prefix[0]).all()),
        "prefix_tokens_valid": bool(b1_prefix[1][0, -8:].all()),
        "prefix_tokens_non_autoregressive": not bool(b1_prefix[2][-8:].any()),
        "official_prefix_token_identities_unchanged": np.array_equal(
            b1_prefix[0][:, : official_prefix[0].shape[1]], official_prefix[0]
        ),
        "official_prefix_masks_unchanged": np.array_equal(
            b1_prefix[1][:, : official_prefix[1].shape[1]], official_prefix[1]
        ),
        "adapter_internal_gradient_finite_nonzero": np.isfinite(b1_grad_norm) and b1_grad_norm > 0,
        "loss_finite": bool(np.isfinite(float(jax.device_get(b1_loss)))),
        "action_shape_unchanged": b1_action.shape == (1, 30, 32),
        "no_E_T_parameters": not any("E_T" in str(path) for path, _ in nnx.iter_graph(b1_model)),
    }
    del b1_model, b1_grads
    gc.collect()

    # B2 must share B1 inference initialization and expose only a masked training auxiliary.
    b2_config = TactilePi0Config(
        **common,
        tactile_unit_mode=TactileUnitMode.CONTACT_STATE_TOKENS_PHYSICAL_AUX,
        lambda_phys=0.01,
    )
    b2_model = b2_config.create(init_key)
    b2_inference_observation = make_observation(tactile=True, auxiliary=False)
    b2_training_observation = make_observation(tactile=True, auxiliary=True)
    b2_action = np.asarray(
        jax.device_get(b2_model.sample_actions(key, b2_inference_observation, num_steps=1, noise=noise))
    )
    b2_adapter_hash = array_hash(b2_model, contains="contact_adapter")

    def physical_loss_fn(model):
        return model.compute_physical_auxiliary_loss(key, b2_training_observation, actions, train=False)

    b2_filter = nnx.All(
        nnx.Param,
        nnx.Any(
            nnx_utils.PathRegex(".*contact_adapter.*"),
            nnx_utils.PathRegex(".*physical_auxiliary.*"),
            nnx_utils.PathRegex(".*PaliGemma/llm.*"),
        ),
    )
    physical_loss, physical_grads = nnx.value_and_grad(
        physical_loss_fn, argnums=nnx.DiffState(0, b2_filter)
    )(b2_model)
    total_loss, loss_info = b2_model.compute_loss_with_info(
        key, b2_training_observation, actions, train=False
    )
    hard_inference_guard = False
    try:
        b2_model.sample_actions(key, b2_training_observation, num_steps=1, noise=noise)
    except ValueError as error:
        hard_inference_guard = "training-only" in str(error)
    b2_gates = {
        "same_contact_adapter_initialization_as_B1": b1_adapter_hash == b2_adapter_hash,
        "same_inference_action_as_B1": np.array_equal(b1_action, b2_action),
        "inference_action_shape": b2_action.shape == (1, 30, 32),
        "training_target_shape": b2_training_observation.contact_shared_target.shape == (1, 8, 32),
        "physical_loss_finite": bool(np.isfinite(float(jax.device_get(physical_loss)))),
        "total_loss_finite": bool(np.isfinite(float(jax.device_get(total_loss)))),
        "official_loss_finite": bool(np.isfinite(float(jax.device_get(loss_info["official_loss"])))),
        "auxiliary_gradient_finite_nonzero": (
            np.isfinite(norm_matching(physical_grads, ".*physical_auxiliary.*"))
            and norm_matching(physical_grads, ".*physical_auxiliary.*") > 0
        ),
        "action_expert_gradient_from_physical_loss_finite_nonzero": (
            np.isfinite(norm_matching(physical_grads, ".*PaliGemma/llm.*"))
            and norm_matching(physical_grads, ".*PaliGemma/llm.*") > 0
        ),
        "hard_training_target_inference_guard": hard_inference_guard,
        "no_S4_2_parameters": not any(
            any(label in str(path) for label in ("E_T", "A0", "C3", "B3", "A_H"))
            for path, _ in nnx.iter_graph(b2_model)
        ),
    }

    # Official-width parameter budgets are evaluated without allocating arrays.
    official_counts = {}
    for mode in (
        TactileUnitMode.CONTACT_STATE_TOKENS,
        TactileUnitMode.CONTACT_STATE_TOKENS_PHYSICAL_AUX,
    ):
        config = TactilePi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            max_token_len=250,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            tactile_unit_mode=mode,
            lambda_phys=0.01 if mode is TactileUnitMode.CONTACT_STATE_TOKENS_PHYSICAL_AUX else 0,
        )
        official_counts[mode.value] = count_named_parameters(nnx.eval_shape(config.create, jax.random.key(42)))
    b1_gates["official_adapter_parameter_count"] = official_counts[
        TactileUnitMode.CONTACT_STATE_TOKENS.value
    ]["contact_adapter"] == 199680
    b2_gates["official_auxiliary_parameter_budget"] = (
        official_counts[TactileUnitMode.CONTACT_STATE_TOKENS_PHYSICAL_AUX.value]["physical_auxiliary"]
        == 658176
        < 1_000_000
    )

    patch_path = ROOT / "patches/simulation/s4_3_pi1_openpi_hidden_loss_hook.patch"
    source_patch_check = subprocess.run(
        ["patch", "--dry-run", "-R", "-d", str(OPENPI), "-p1"],
        stdin=patch_path.open("rb"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    mode_contract = {
        "schema": "tactile3d-unit.s4-3-pi1-mode-contract.v1",
        "stage": "PI1M",
        "official_commit": "8d23b0fab23b17a58c4b55f3942e17013aaf8267",
        "modes": [mode.value for mode in TactileUnitMode],
        "official_parameter_counts": official_counts,
        "tracked_hashes": {
            symbolic(path): sha256(path)
            for path in (
                ROOT / "gr00t/simulation/pi05_tactile_unit.py",
                ROOT / "gr00t/simulation/s4_3_pi1.py",
                ROOT / "scripts/simulation/train_s4_3_pi1.py",
                patch_path,
                ROOT / "configs/simulation/s4_3_pi1_modes.json",
                ROOT / "configs/simulation/s4_3_pi1_training_protocol.json",
                ROOT / "configs/simulation/s4_3_pi1d_eval_protocol.json",
            )
        },
        "patch_reverse_dry_run": "PASS" if source_patch_check.returncode == 0 else "FAIL",
        "status": "PASS" if source_patch_check.returncode == 0 else "FAIL",
    }
    parity = {
        "schema": "tactile3d-unit.s4-3-pi1-mode-none-parity.v1",
        "tolerance": {"absolute": 0.0, "relative": 0.0},
        "gates": {name: "PASS" if value else "FAIL" for name, value in parity_gates.items()},
        "status": "PASS" if all(parity_gates.values()) else "FAIL",
    }
    b1_validation = {
        "schema": "tactile3d-unit.s4-3-pi1b-mode-validation.v1",
        "adapter_gradient_norm": b1_grad_norm,
        "adapter_parameter_count": 199680,
        "gates": {name: "PASS" if value else "FAIL" for name, value in b1_gates.items()},
        "status": "PASS" if all(b1_gates.values()) else "FAIL",
    }
    b2_validation = {
        "schema": "tactile3d-unit.s4-3-pi1c-mode-validation.v1",
        "physical_loss": float(jax.device_get(physical_loss)),
        "auxiliary_gradient_norm": norm_matching(physical_grads, ".*physical_auxiliary.*"),
        "action_expert_gradient_norm": norm_matching(physical_grads, ".*PaliGemma/llm.*"),
        "auxiliary_parameter_count": 658176,
        "gates": {name: "PASS" if value else "FAIL" for name, value in b2_gates.items()},
        "status": "PASS" if all(b2_gates.values()) else "FAIL",
    }
    write_json("mode_contract.json", mode_contract)
    write_json("mode_none_parity.json", parity)
    write_json("pi1b_mode_validation.json", b1_validation)
    write_json("pi1c_mode_validation.json", b2_validation)
    write_json(
        "physical_aux_calibration_protocol.json",
        {
            "schema": "tactile3d-unit.s4-3-pi1-physical-aux-calibration-protocol.v1",
            "status": "FROZEN_NOT_YET_EXECUTED",
            "batches": 4,
            "batch_size": 32,
            "iterator": "seed42 deterministically shuffled training iterator; first four batches",
            "model": "B2 initialization from exact pi05_base before optimizer updates",
            "formula": "0.1 * mean(L_official_pi05) / mean(L_phys)",
            "clamp": [0.001, 0.1],
            "rollout_performance_used": False,
        },
    )
    write_json(
        "training_protocol_freeze.json",
        {
            "schema": "tactile3d-unit.s4-3-pi1-training-protocol-freeze.v1",
            "status": (
                "PENDING_LOADED_BASE_ADAPTER_GRADIENT_GATE"
                if all(x["status"] == "PASS" for x in (mode_contract, parity, b1_validation, b2_validation))
                else "FAIL"
            ),
            "config": "$REPO_ROOT/configs/simulation/s4_3_pi1_training_protocol.json",
            "config_sha256": sha256(ROOT / "configs/simulation/s4_3_pi1_training_protocol.json"),
            "B1_and_B2_complete_before_training": True,
            "loaded_base_adapter_gradient_gate": "PENDING",
        },
    )
    write_json(
        "pi1d_eval_protocol_freeze.json",
        {
            "schema": "tactile3d-unit.s4-3-pi1d-eval-protocol-freeze.v1",
            "status": "PASS",
            "config": "$REPO_ROOT/configs/simulation/s4_3_pi1d_eval_protocol.json",
            "config_sha256": sha256(ROOT / "configs/simulation/s4_3_pi1d_eval_protocol.json"),
            "fresh_evaluator_seed": 1,
            "episodes": 50,
            "evaluation_performance_seen": False,
        },
    )
    statuses = {
        "mode_contract": mode_contract["status"],
        "mode_none_parity": parity["status"],
        "pi1b": b1_validation["status"],
        "pi1c": b2_validation["status"],
    }
    print(json.dumps(statuses, sort_keys=True))
    if any(value != "PASS" for value in statuses.values()):
        raise SystemExit("S4_3_PI1_MODE_IMPLEMENTATION_FAIL")


if __name__ == "__main__":
    main()
