#!/usr/bin/env python3
"""Validate the B_VAC_V mode, old-mode parity, sidecars and runtime guards."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OPENPI = ROOT / ".local/external/simulation/s4_3_pi1/openpi"
ARTIFACT = ROOT / ".local/artifacts/simulation/s4_3_pi2n/b_vac_v_mode_parity.json"
B2_SIDECAR = ROOT / ".local/datasets/simulation/s4_3_pi1/pinch_tongs_official_tactile/sidecar.npz"
VA27_TARGET = ROOT / ".local/datasets/simulation/s4_3_pi2m/pinch_tongs_va_t27/sidecar.npz"


def atomic_json(path: Path, payload: dict) -> None:
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


def observation(*, target: float | None = None, valid: bool = True):
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
        vac_vision_target=None if target is None else jnp.full((1, 8, 32), target, dtype=jnp.float32),
        vac_aux_valid=None if target is None else jnp.asarray([valid], dtype=jnp.bool_),
    )


def fixture_sidecar_gate() -> dict[str, bool]:
    from gr00t.simulation.pi05_tactile_unit import _SidecarDataset
    from gr00t.simulation.s4_3_pi1 import TactileUnitMode

    class Fixture:
        def __len__(self):
            return 3

        def __getitem__(self, index):
            return {"index": np.asarray(index, dtype=np.int64), "official": np.asarray(index + 10)}

    with tempfile.TemporaryDirectory(prefix="s43_pi2n_vac_v_") as directory:
        directory = Path(directory)
        contact = directory / "contact.npz"
        target = directory / "target.npz"
        np.savez(contact, index=np.arange(3), contact_state=np.ones((3, 256), dtype=np.float32))
        np.savez(target, index=np.arange(3), vac_vision_target=np.ones((3, 8, 32), dtype=np.float32), vac_aux_valid=np.asarray([True, False, True]))
        wrapped = _SidecarDataset(Fixture(), contact, TactileUnitMode.CONTACT_STATE_TOKENS_VAC_V_PHYSICAL_AUX, target)
        item = wrapped[1]
        return {
            "dual_sidecar_index_identity": int(item["index"]) == 1,
            "official_fields_preserved": int(item["official"]) == 11,
            "contact_state_shape": item["contact_state"].shape == (256,),
            "VAC_V_target_shape": item["vac_vision_target"].shape == (8, 32),
            "VAC_V_valid_boolean": np.asarray(item["vac_aux_valid"]).dtype == np.bool_,
            "contact_target_not_emitted": "contact_shared_target" not in item and "physical_aux_valid" not in item,
        }


def main() -> None:
    if ARTIFACT.exists():
        raise SystemExit(f"refusing to overwrite mode parity artifact: {ARTIFACT}")
    os.chdir(OPENPI)
    sys.path[:0] = [str(ROOT), str(OPENPI / "src")]
    from flax import nnx
    import flax.traverse_util
    import jax
    import jax.numpy as jnp
    from gr00t.simulation.pi05_tactile_unit import TactilePi0Config, count_named_parameters, install_openpi_runtime_hooks
    from gr00t.simulation.s4_3_pi1 import TactileUnitMode
    from openpi.models import pi0 as official_pi0
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import data_loader
    from scripts.simulation.train_s4_3_pi1 import build_config as build_b2_config
    from scripts.simulation.train_s4_3_pi2n import build_config as build_pi2n_config, vac_v_target

    install_openpi_runtime_hooks()
    common = dict(pi05=True, action_dim=32, action_horizon=30, max_token_len=8, paligemma_variant="dummy", action_expert_variant="dummy", dtype="bfloat16")
    init_key = jax.random.key(42)
    loss_key = jax.random.key(432)
    actions = jnp.linspace(-0.75, 0.75, 30 * 32).reshape(1, 30, 32)
    noise = jnp.linspace(0.9, -0.9, 30 * 32).reshape(1, 30, 32)
    infer = observation()
    official_obs = dataclasses.replace(infer, contact_state=None)
    low = observation(target=0.0)
    high = observation(target=1.0)
    invalid = observation(target=1.0, valid=False)

    official = Pi0Config(**common).create(init_key)
    none = TactilePi0Config(**common, tactile_unit_mode=TactileUnitMode.NONE).create(init_key)
    b1 = TactilePi0Config(**common, tactile_unit_mode=TactileUnitMode.CONTACT_STATE_TOKENS).create(init_key)
    bva = TactilePi0Config(**common, tactile_unit_mode=TactileUnitMode.VA_PHYSICAL_AUX, lambda_phys=0.01).create(init_key)
    b2 = TactilePi0Config(**common, tactile_unit_mode=TactileUnitMode.CONTACT_STATE_TOKENS_PHYSICAL_AUX, lambda_phys=0.01).create(init_key)
    bhva = TactilePi0Config(**common, tactile_unit_mode=TactileUnitMode.CONTACT_STATE_TOKENS_VA_PHYSICAL_AUX, lambda_phys=0.01).create(init_key)
    vac_config = TactilePi0Config(**common, tactile_unit_mode=TactileUnitMode.CONTACT_STATE_TOKENS_VAC_V_PHYSICAL_AUX, lambda_phys=0.01)
    vac = vac_config.create(init_key)

    b2_prefix = tuple(np.asarray(jax.device_get(value)) for value in b2.embed_prefix(infer))
    vac_prefix = tuple(np.asarray(jax.device_get(value)) for value in vac.embed_prefix(infer))
    official_action = np.asarray(jax.device_get(official.sample_actions(loss_key, official_obs, num_steps=1, noise=noise)))
    none_action = np.asarray(jax.device_get(none.sample_actions(loss_key, official_obs, num_steps=1, noise=noise)))
    b1_action = np.asarray(jax.device_get(b1.sample_actions(loss_key, infer, num_steps=1, noise=noise)))
    bva_action = np.asarray(jax.device_get(bva.sample_actions(loss_key, official_obs, num_steps=1, noise=noise)))
    b2_action = np.asarray(jax.device_get(b2.sample_actions(loss_key, infer, num_steps=1, noise=noise)))
    bhva_action = np.asarray(jax.device_get(bhva.sample_actions(loss_key, infer, num_steps=1, noise=noise)))
    vac_action = np.asarray(jax.device_get(vac.sample_actions(loss_key, infer, num_steps=1, noise=noise)))
    low_total, low_info = vac.compute_loss_with_info(loss_key, low, actions, train=False)
    high_total, high_info = vac.compute_loss_with_info(loss_key, high, actions, train=False)
    invalid_total, invalid_info = vac.compute_loss_with_info(loss_key, invalid, actions, train=False)
    low_total, low_info, high_total, high_info, invalid_total, invalid_info = jax.device_get((low_total, low_info, high_total, high_info, invalid_total, invalid_info))

    guard = False
    try:
        vac.sample_actions(loss_key, high, num_steps=1, noise=noise)
    except ValueError as error:
        guard = "training-only" in str(error)
    spec = vac_config.inputs_spec(batch_size=2)[0]
    b2_counts = count_named_parameters(b2)
    vac_counts = count_named_parameters(vac)
    flat_names = list(flax.traverse_util.flatten_dict(nnx.state(vac).to_pure_dict(), sep="/"))

    b2_loader_config = dataclasses.replace(build_b2_config("CONTACT_STATE_TOKENS_PHYSICAL_AUX", "pi2n_b2_loader_fixture", 0.01), batch_size=4, num_workers=0)
    vac_loader_config = dataclasses.replace(build_pi2n_config("B_VAC_V", "pi2n_vac_v_loader_fixture", 0.01, 1), batch_size=4, num_workers=0)
    b2_observation, b2_actions = next(iter(data_loader.create_data_loader(b2_loader_config, shuffle=False, num_batches=1)))
    vac_observation, vac_actions = next(iter(data_loader.create_data_loader(vac_loader_config, shuffle=False, num_batches=1)))

    target_path = vac_v_target()
    with np.load(target_path, allow_pickle=False) as target, np.load(B2_SIDECAR, allow_pickle=False) as contact, np.load(VA27_TARGET, allow_pickle=False) as clean:
        target_values = np.asarray(target["vac_vision_target"])
        clean_values = np.asarray(clean["va_shared_target"])
        valid = np.asarray(target["vac_aux_valid"])
        data_gates = {
            "target_fields_exact": set(target.files) == {"index", "vac_vision_target", "vac_aux_valid"},
            "target_index_equal_B2": np.array_equal(target["index"], contact["index"]),
            "target_valid_equal_B2": np.array_equal(valid, contact["physical_aux_valid"]),
            "target_finite_shape": target_values.shape == (40065, 8, 32) and np.isfinite(target_values).all(),
            "VAC_target_not_duplicate_clean_VA": not np.array_equal(target_values[valid], clean_values[valid]),
        }
        target_difference = {
            "mean_absolute": float(np.mean(np.abs(target_values[valid] - clean_values[valid]))),
            "maximum_absolute": float(np.max(np.abs(target_values[valid] - clean_values[valid]))),
        }

    gates = {
        **fixture_sidecar_gate(),
        **data_gates,
        "NONE_exact_official_class": type(none) is official_pi0.Pi0,
        "NONE_inference_bit_identical": np.array_equal(none_action, official_action),
        "BVA_inference_bit_identical_official": np.array_equal(bva_action, official_action),
        "B1_inference_bit_identical_B2": np.array_equal(b1_action, b2_action),
        "B_HVA_inference_bit_identical_B2": np.array_equal(bhva_action, b2_action),
        "B_VAC_V_prefix_bit_identical_B2": all(np.array_equal(left, right) for left, right in zip(vac_prefix, b2_prefix, strict=True)),
        "B_VAC_V_inference_bit_identical_B2": np.array_equal(vac_action, b2_action),
        "contact_adapter_initialization_identical_B2": state_hash(vac, "contact_adapter") == state_hash(b2, "contact_adapter"),
        "auxiliary_head_initialization_identical_B2": state_hash(vac, "physical_auxiliary") == state_hash(b2, "physical_auxiliary"),
        "parameter_counts_identical_B2": vac_counts == b2_counts,
        "target_changes_only_auxiliary_loss": np.array_equal(np.asarray(low_info["official_loss"]), np.asarray(high_info["official_loss"])) and not np.array_equal(np.asarray(low_info["physical_loss"]), np.asarray(high_info["physical_loss"])),
        "target_changes_total_loss": not np.array_equal(np.asarray(low_total), np.asarray(high_total)),
        "invalid_mask_zeros_auxiliary": float(invalid_info["physical_loss"]) == 0.0,
        "invalid_mask_keeps_BC": np.array_equal(np.asarray(invalid_total), np.asarray(invalid_info["official_loss"])),
        "hard_runtime_target_guard": guard,
        "runtime_observation_has_no_training_target": infer.vac_vision_target is None and infer.vac_aux_valid is None,
        "input_spec_exact": spec.contact_state.shape == (2, 256) and spec.vac_vision_target.shape == (2, 8, 32) and spec.vac_aux_valid.shape == (2,) and spec.va_shared_target is None and spec.contact_shared_target is None,
        "real_loader_same_images": set(b2_observation.images) == set(vac_observation.images) and all(np.array_equal(np.asarray(b2_observation.images[key]), np.asarray(vac_observation.images[key])) for key in b2_observation.images),
        "real_loader_same_state": np.array_equal(np.asarray(b2_observation.state), np.asarray(vac_observation.state)),
        "real_loader_same_actions": np.array_equal(np.asarray(b2_actions), np.asarray(vac_actions)),
        "real_loader_same_contact_state": np.array_equal(np.asarray(b2_observation.contact_state), np.asarray(vac_observation.contact_state)),
        "real_loader_same_valid_mask": np.array_equal(np.asarray(b2_observation.physical_aux_valid), np.asarray(vac_observation.vac_aux_valid)),
        "real_loader_no_contact_target": vac_observation.contact_shared_target is None and vac_observation.physical_aux_valid is None,
        "no_S4_2_teacher_parameters": not any(any(label in name for label in ("E_T", "A0", "C3", "B3", "A_H")) for name in flat_names),
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2n-vac-v-mode-parity.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "mode": TactileUnitMode.CONTACT_STATE_TOKENS_VAC_V_PHYSICAL_AUX.value,
        "parameter_counts": {"B2": b2_counts, "B_VAC_V": vac_counts},
        "VAC_vs_clean_VA_target_difference": target_difference,
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
    }
    atomic_json(ARTIFACT, payload)
    print(json.dumps({"status": payload["status"], "failed": [name for name, value in gates.items() if not value]}, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit("PI2N_B_VAC_V_MODE_PARITY_FAIL")


if __name__ == "__main__":
    main()
