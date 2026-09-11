#!/usr/bin/env python3
"""Run the frozen four-batch TRAIN-only BVA loss-scale calibration."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OPENPI = ROOT / ".local/external/simulation/s4_3_pi1/openpi"
PROTOCOL = ROOT / "configs/simulation/s4_3_pi2u_bva_protocol.json"
TARGET_MANIFEST = ROOT / ".local/artifacts/simulation/s4_3_pi2u/bva_target_manifest.json"
OUTPUT = ROOT / ".local/artifacts/simulation/s4_3_pi2u/bva_lambda_calibration.json"
BATCHES = 4
LOWER = 1e-3
UPPER = 1e-1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def batch_sha256(tree: Any) -> str:
    import jax
    import numpy as np

    digest = hashlib.sha256()
    for index, value in enumerate(jax.tree.leaves(jax.device_get(tree))):
        array = np.asarray(value)
        digest.update(f"{index}\0{array.dtype}\0{array.shape}\0".encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit("refusing to overwrite frozen BVA lambda calibration")
    protocol = json.loads(PROTOCOL.read_text())
    target_manifest = json.loads(TARGET_MANIFEST.read_text())
    if protocol.get("status") != "FROZEN_BEFORE_TRAINING" or target_manifest.get("status") != "PASS":
        raise SystemExit("BVA protocol and target manifest must pass before calibration")
    os.chdir(OPENPI)
    sys.path[:0] = [str(ROOT), str(OPENPI / "src")]
    from flax import nnx
    import jax
    import numpy as np
    from gr00t.simulation.pi05_tactile_unit import TactilePi0
    from gr00t.simulation.s4_3_pi1 import TactileUnitMode
    from openpi.training import data_loader, sharding
    from scripts.simulation.train_s4_3_pi2u_bva import build_config, load_official_train_module

    config = build_config("s43_pi2u_bva_lambda_calibration_seed42", LOWER, 1)
    loader = data_loader.create_data_loader(config, shuffle=True, num_batches=BATCHES)
    iterator = iter(loader)
    train_rng, init_rng = jax.random.split(jax.random.key(config.seed))
    mesh = sharding.make_mesh(config.fsdp_devices)
    train_state, _ = load_official_train_module().init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(train_state)
    model = nnx.merge(train_state.model_def, train_state.params)
    model.train()
    rows = []
    for batch_index in range(BATCHES):
        observation, actions = next(iterator)
        loss_rng = jax.random.fold_in(train_rng, batch_index)
        with sharding.set_mesh(mesh):
            total, info = model.compute_loss_with_info(loss_rng, observation, actions, train=True)
        total, info = jax.device_get((total, info))
        valid = np.asarray(jax.device_get(observation.va_aux_valid), dtype=bool)
        rows.append(
            {
                "iterator_batch_index": batch_index,
                "batch_tree_sha256": batch_sha256((observation, actions)),
                "loss_rng_key_data": [int(value) for value in np.asarray(jax.random.key_data(loss_rng))],
                "va_aux_valid_samples": int(valid.sum()),
                "official_pi05_loss": float(info["official_loss"]),
                "va_physical_loss": float(info["physical_loss"]),
                "provisional_total_loss_not_used": float(total),
            }
        )
    official_values = [row["official_pi05_loss"] for row in rows]
    physical_values = [row["va_physical_loss"] for row in rows]
    mean_official = float(np.mean(official_values))
    mean_physical = float(np.mean(physical_values))
    raw_lambda = 0.1 * mean_official / mean_physical
    final_lambda = min(UPPER, max(LOWER, raw_lambda))
    gates = {
        "exactly_four_training_batches": len(rows) == BATCHES,
        "seed42": config.seed == 42,
        "global_batch32": config.batch_size == 32,
        "BVA_initialization_before_updates": int(train_state.step) == 0,
        "BVA_mode": type(model) is TactilePi0 and model.tactile_unit_mode is TactileUnitMode.VA_PHYSICAL_AUX,
        "some_valid_target_in_every_batch": all(row["va_aux_valid_samples"] > 0 for row in rows),
        "official_losses_finite": all(math.isfinite(value) for value in official_values),
        "va_losses_finite_positive": all(math.isfinite(value) and value > 0 for value in physical_values),
        "lambda_finite_in_range": math.isfinite(final_lambda) and LOWER <= final_lambda <= UPPER,
        "no_dev_rollout_or_contact_selection": True,
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2u-bva-lambda-calibration.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "seed": 42,
        "batch_size": 32,
        "calibration_batches": rows,
        "mean_official_pi05_loss": mean_official,
        "mean_va_physical_loss": mean_physical,
        "formula": "clamp(0.1 * mean(L_official_pi05) / mean(L_va), 0.001, 0.1)",
        "lambda_phys_raw": raw_lambda,
        "lambda_phys": final_lambda,
        "clamp": [LOWER, UPPER],
        "clamp_applied": final_lambda != raw_lambda,
        "protocol_sha256": sha256_file(PROTOCOL),
        "target_manifest_sha256": sha256_file(TARGET_MANIFEST),
        "frozen_before_training": all(gates.values()),
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUTPUT)
    print(json.dumps({"status": payload["status"], "lambda_phys": final_lambda, "raw": raw_lambda}, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit("BVA lambda calibration failed")


if __name__ == "__main__":
    main()
