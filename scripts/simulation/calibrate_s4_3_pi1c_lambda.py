#!/usr/bin/env python3
"""Execute the preregistered four-batch TRAIN-only PI1C loss-scale calibration."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OPENPI = ROOT / ".local/external/simulation/s4_3_pi1/openpi"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi1"
OUTPUT = ARTIFACTS / "lambda_phys_calibration.json"
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
        digest.update(str(index).encode())
        digest.update(b"\0")
        digest.update(str(array.dtype).encode())
        digest.update(b"\0")
        digest.update(str(array.shape).encode())
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_json(payload: Any) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUTPUT)


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit("Refusing to overwrite the frozen PI1C lambda calibration")
    b1_freeze_path = ARTIFACTS / "pi1b_checkpoint_freeze.json"
    protocol_path = ARTIFACTS / "physical_aux_calibration_protocol.json"
    b1_freeze = json.loads(b1_freeze_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    if b1_freeze.get("status") != "PASS" or not b1_freeze.get("frozen"):
        raise SystemExit("PI1B checkpoint must be frozen before PI1C calibration")
    if protocol.get("status") != "FROZEN_NOT_YET_EXECUTED":
        raise SystemExit("PI1C calibration protocol was not frozen in its pre-execution state")
    if protocol.get("batches") != BATCHES or protocol.get("batch_size") != 32:
        raise SystemExit("PI1C calibration protocol batch contract mismatch")

    os.chdir(OPENPI)
    sys.path[:0] = [str(ROOT), str(OPENPI / "src")]
    from flax import nnx
    import jax
    import numpy as np
    from gr00t.simulation.pi05_tactile_unit import TactilePi0
    from gr00t.simulation.s4_3_pi1 import TactileUnitMode
    from openpi.training import data_loader, sharding
    from scripts.simulation.train_s4_3_pi1 import build_config, load_official_train_module

    # The provisional scalar is not used in the ratio. Architecture, initialization,
    # loader, RNG folding, and loss components exactly match the frozen B2 train path.
    config = build_config(
        "CONTACT_STATE_TOKENS_PHYSICAL_AUX",
        "s43_pi1c_lambda_calibration_seed42",
        LOWER,
    )
    loader = data_loader.create_data_loader(config, shuffle=True, num_batches=BATCHES)
    iterator = iter(loader)
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)
    official_train = load_official_train_module()
    train_state, _ = official_train.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(train_state)
    model = nnx.merge(train_state.model_def, train_state.params)
    model.train()

    rows = []
    for batch_index in range(BATCHES):
        observation, actions = next(iterator)
        loss_rng = jax.random.fold_in(train_rng, batch_index)
        with sharding.set_mesh(mesh):
            total_loss, info = model.compute_loss_with_info(
                loss_rng, observation, actions, train=True
            )
        total_loss, info = jax.device_get((total_loss, info))
        official_loss = float(info["official_loss"])
        physical_loss = float(info["physical_loss"])
        valid = np.asarray(jax.device_get(observation.physical_aux_valid), dtype=bool)
        rows.append(
            {
                "iterator_batch_index": batch_index,
                "batch_tree_sha256": batch_sha256((observation, actions)),
                "loss_rng_key_data": [int(value) for value in np.asarray(jax.random.key_data(loss_rng))],
                "physical_aux_valid_samples": int(valid.sum()),
                "official_pi05_loss": official_loss,
                "physical_loss": physical_loss,
                "provisional_total_loss_not_used_for_calibration": float(total_loss),
            }
        )

    official_values = [row["official_pi05_loss"] for row in rows]
    physical_values = [row["physical_loss"] for row in rows]
    mean_official = float(np.mean(official_values))
    mean_physical = float(np.mean(physical_values))
    raw_lambda = 0.1 * mean_official / mean_physical
    final_lambda = min(UPPER, max(LOWER, raw_lambda))
    gates = {
        "pi1b_frozen_before_calibration": b1_freeze["status"] == "PASS" and b1_freeze["frozen"],
        "exactly_four_batches": len(rows) == BATCHES,
        "global_batch_size_32": config.batch_size == 32,
        "seed42_deterministic_shuffle": config.seed == 42,
        "training_iterator_shuffle": True,
        "B2_initialization_before_optimizer_updates": int(train_state.step) == 0,
        "exact_pi05_base_initialization": "DexJoCo-Pi05/pi05_base/params" in str(config.weight_loader.params_path),
        "B2_mode_identity": (
            type(model) is TactilePi0
            and model.tactile_unit_mode is TactileUnitMode.CONTACT_STATE_TOKENS_PHYSICAL_AUX
        ),
        "all_batches_have_valid_physical_targets": all(
            row["physical_aux_valid_samples"] > 0 for row in rows
        ),
        "official_losses_finite": all(math.isfinite(value) for value in official_values),
        "physical_losses_finite_positive": all(
            math.isfinite(value) and value > 0 for value in physical_values
        ),
        "raw_lambda_finite_positive": math.isfinite(raw_lambda) and raw_lambda > 0,
        "final_lambda_in_frozen_range": LOWER <= final_lambda <= UPPER,
        "no_rollout_or_dev_selection": True,
        "pi1d_not_run": not (ARTIFACTS / "pi1d_results.json").exists(),
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi1c-lambda-phys-calibration.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "mode": "CONTACT_STATE_TOKENS_PHYSICAL_AUX",
        "model_state": "B2 initialization from exact pi05_base before optimizer updates",
        "seed": 42,
        "iterator": "seed42 deterministically shuffled training iterator; first four batches",
        "batch_size": 32,
        "calibration_batches": rows,
        "mean_official_pi05_loss": mean_official,
        "mean_physical_loss": mean_physical,
        "formula": "0.1 * mean(L_official_pi05) / mean(L_phys)",
        "lambda_phys_raw": raw_lambda,
        "clamp": [LOWER, UPPER],
        "lambda_phys": final_lambda,
        "clamp_applied": final_lambda != raw_lambda,
        "frozen_before_PI1C_training": all(gates.values()),
        "recalculate_during_training": False,
        "rollout_performance_used": False,
        "dev_or_pi1d_data_used": False,
        "protocol": "$REPO_ROOT/.local/artifacts/simulation/s4_3_pi1/physical_aux_calibration_protocol.json",
        "protocol_sha256": sha256_file(protocol_path),
        "pi1b_freeze": "$REPO_ROOT/.local/artifacts/simulation/s4_3_pi1/pi1b_checkpoint_freeze.json",
        "pi1b_freeze_sha256": sha256_file(b1_freeze_path),
        "training_config": "$REPO_ROOT/configs/simulation/s4_3_pi1_training_protocol.json",
        "training_config_sha256": sha256_file(ROOT / "configs/simulation/s4_3_pi1_training_protocol.json"),
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
    }
    atomic_json(payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "mean_official_pi05_loss": mean_official,
                "mean_physical_loss": mean_physical,
                "lambda_phys_raw": raw_lambda,
                "lambda_phys": final_lambda,
            },
            sort_keys=True,
        )
    )
    if payload["status"] != "PASS":
        failed = [name for name, value in gates.items() if not value]
        raise SystemExit("S4_3_PI1C_LAMBDA_CALIBRATION_FAIL: " + ",".join(failed))


if __name__ == "__main__":
    main()
