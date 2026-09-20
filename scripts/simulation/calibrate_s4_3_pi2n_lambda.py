#!/usr/bin/env python3
"""Calibrate a PI2N candidate on four frozen TRAIN batches before any update."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OPENPI = ROOT / ".local/external/simulation/s4_3_pi1/openpi"
PROTOCOL = ROOT / "configs/simulation/s4_3_pi2n_candidate_protocol.json"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2n"
B2_SIDECAR = ROOT / ".local/datasets/simulation/s4_3_pi1/pinch_tongs_official_tactile/sidecar.npz"
VA27_TARGET = ROOT / ".local/datasets/simulation/s4_3_pi2m/pinch_tongs_va_t27/sidecar.npz"
BATCHES = 4
LOWER = 1e-3
UPPER = 1e-1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(32 * 1024 * 1024), b""):
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


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def target_audit(model_id: str, candidate: dict[str, Any], vac_target: Path) -> dict[str, Any]:
    import numpy as np

    target_path = VA27_TARGET if model_id == "B_VA27" else vac_target
    target_key = "va_shared_target" if model_id == "B_VA27" else "vac_vision_target"
    valid_key = "va_aux_valid" if model_id == "B_VA27" else "vac_aux_valid"
    expected_fields = {"index", target_key, valid_key}
    with np.load(target_path, allow_pickle=False) as target, np.load(B2_SIDECAR, allow_pickle=False) as b2:
        values = np.asarray(target[target_key])
        valid = np.asarray(target[valid_key])
        gates = {
            "fields_exact": set(target.files) == expected_fields,
            "rows_40065": len(valid) == 40065,
            "shape_40065x8x32": values.shape == (40065, 8, 32),
            "finite": bool(np.isfinite(values).all()),
            "invalid_placeholder_zero": bool(np.all(values[~valid] == 0)),
            "index_equal_B2": bool(np.array_equal(target["index"], b2["index"])),
            "valid_equal_B2": bool(np.array_equal(valid, b2["physical_aux_valid"])),
            "valid_rows_37365": int(valid.sum()) == 37365,
            "invalid_rows_2700": int((~valid).sum()) == 2700,
        }
    if model_id == "B_VA27":
        gates["frozen_target_hash"] = sha256_file(target_path) == candidate["target_sha256"]
    return {
        "path": str(target_path),
        "sha256": sha256_file(target_path),
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", choices=("B_VA27", "B_VAC_V"), required=True)
    args = parser.parse_args()
    output = ARTIFACTS / f"{args.model_id.lower()}_lambda_calibration.json"
    if output.exists():
        raise SystemExit(f"refusing to overwrite frozen calibration: {output}")
    protocol = json.loads(PROTOCOL.read_text())
    if protocol.get("status") != "FROZEN_BEFORE_NEW_POLICY_TRAINING":
        raise SystemExit("PI2N candidate protocol is not frozen")

    os.chdir(OPENPI)
    sys.path[:0] = [str(ROOT), str(OPENPI / "src")]
    from flax import nnx
    import jax
    import numpy as np
    from gr00t.simulation.pi05_tactile_unit import TactilePi0, count_named_parameters
    from gr00t.simulation.s4_3_pi1 import TactileUnitMode
    from openpi.training import data_loader, sharding
    from scripts.simulation.train_s4_3_pi2n import build_config, load_official_train_module, vac_v_target

    candidate = protocol[args.model_id]
    audit = target_audit(args.model_id, candidate, vac_v_target())
    if audit["status"] != "PASS":
        atomic_json(ARTIFACTS / f"{args.model_id.lower()}_target_audit.json", audit)
        raise SystemExit(f"{args.model_id} target audit failed")
    config = build_config(args.model_id, f"s43_pi2n_{args.model_id.lower()}_lambda_calibration_seed42", LOWER, 1)
    loader = data_loader.create_data_loader(config, shuffle=True, num_batches=BATCHES)
    iterator = iter(loader)
    train_rng, init_rng = jax.random.split(jax.random.key(config.seed))
    mesh = sharding.make_mesh(config.fsdp_devices)
    train_state, _ = load_official_train_module().init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(train_state)
    model = nnx.merge(train_state.model_def, train_state.params)
    model.train()
    rows = []
    observations = []
    actions_values = []
    for batch_index in range(BATCHES):
        observation, actions = next(iterator)
        observations.append(observation)
        actions_values.append(actions)
        loss_rng = jax.random.fold_in(train_rng, batch_index)
        with sharding.set_mesh(mesh):
            total, info = model.compute_loss_with_info(loss_rng, observation, actions, train=True)
        total, info = jax.device_get((total, info))
        valid_value = observation.va_aux_valid if args.model_id == "B_VA27" else observation.vac_aux_valid
        contact_absent = observation.contact_state is None
        rows.append({
            "iterator_batch_index": batch_index,
            "batch_tree_sha256": batch_sha256((observation, actions)),
            "loss_rng_key_data": [int(value) for value in np.asarray(jax.random.key_data(loss_rng))],
            "aux_valid_samples": int(np.asarray(jax.device_get(valid_value), dtype=bool).sum()),
            "contact_state_absent": contact_absent,
            "official_pi05_loss": float(info["official_loss"]),
            "physical_loss": float(info["physical_loss"]),
            "provisional_total_loss_not_used": float(total),
        })
    official_values = [row["official_pi05_loss"] for row in rows]
    physical_values = [row["physical_loss"] for row in rows]
    mean_official = float(np.mean(official_values))
    mean_physical = float(np.mean(physical_values))
    raw_lambda = 0.1 * mean_official / mean_physical
    final_lambda = min(UPPER, max(LOWER, raw_lambda))
    expected_mode = TactileUnitMode.VA_PHYSICAL_AUX if args.model_id == "B_VA27" else TactileUnitMode.CONTACT_STATE_TOKENS_VAC_V_PHYSICAL_AUX
    counts = count_named_parameters(model)
    gates = {
        "target_audit": audit["status"] == "PASS",
        "exactly_four_training_batches": len(rows) == BATCHES,
        "seed42": config.seed == 42,
        "global_batch32": config.batch_size == 32,
        "initialization_before_updates": int(train_state.step) == 0,
        "mode_exact": type(model) is TactilePi0 and model.tactile_unit_mode is expected_mode,
        "some_valid_target_in_every_batch": all(row["aux_valid_samples"] > 0 for row in rows),
        "official_losses_finite": all(math.isfinite(value) for value in official_values),
        "aux_losses_finite_positive": all(math.isfinite(value) and value > 0 for value in physical_values),
        "lambda_finite_in_range": math.isfinite(final_lambda) and LOWER <= final_lambda <= UPPER,
        "B_VA27_NO_H": args.model_id != "B_VA27" or all(row["contact_state_absent"] for row in rows),
        "B_VA27_no_contact_adapter": args.model_id != "B_VA27" or counts["contact_adapter"] == 0,
        "B_VAC_V_contact_adapter_present": args.model_id != "B_VAC_V" or counts["contact_adapter"] > 0,
        "no_DEV_rollout_or_performance_selection": True,
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2n-lambda-calibration.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "model_id": args.model_id,
        "seed": 42,
        "batch_size": 32,
        "split": "TRAIN_ONLY",
        "calibration_batches": rows,
        "mean_official_pi05_loss": mean_official,
        "mean_physical_loss": mean_physical,
        "formula": "clamp(0.1 * mean(L_official_pi05) / mean(L_aux), 0.001, 0.1)",
        "lambda_phys_raw": raw_lambda,
        "lambda_phys": final_lambda,
        "clamp": [LOWER, UPPER],
        "clamp_applied": final_lambda != raw_lambda,
        "parameter_counts": counts,
        "protocol_sha256": sha256_file(PROTOCOL),
        "target_audit": audit,
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
    }
    atomic_json(ARTIFACTS / f"{args.model_id.lower()}_target_audit.json", audit)
    atomic_json(output, payload)
    print(json.dumps({"status": payload["status"], "model_id": args.model_id, "lambda_phys": final_lambda, "raw": raw_lambda}, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(f"{args.model_id}_LAMBDA_CALIBRATION_FAIL")


if __name__ == "__main__":
    main()
