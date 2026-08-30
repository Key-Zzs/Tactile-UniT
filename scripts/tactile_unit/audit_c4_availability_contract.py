#!/usr/bin/env python3
"""Freeze the C4 availability protocol and reproduce invalid-H degradation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.c4_availability_conditioning import (  # noqa: E402
    AvailabilityMode, ModalityAvailability, route_availability, sha256_file,
)
from scripts.tactile_unit.c4_runtime import (  # noqa: E402
    DEFAULT_CONFIG, atomic_json, identity_snapshot, load_config, load_full,
    load_parent_config, load_split,
)
from scripts.tactile_unit.c3mscc_runtime import load_frozen_shared_space  # noqa: E402
from scripts.tactile_unit.continuous_contact_bridge_common import load_s2_model  # noqa: E402
from scripts.tactile_unit.evaluate_c3mscc_contact_prediction import (  # noqa: E402
    oracle_probe, wrong_time_indices,
)
from scripts.tactile_unit.train_c3mscc_contact_prediction import (  # noqa: E402
    fit_probe, majority, predict_numpy, row_mse,
)
from gr00t.tactile_unit.c3r0_conditional_sufficiency import evaluate_prediction  # noqa: E402
from gr00t.tactile_unit.continuous_vac_shared_space import different_episode_permutation  # noqa: E402
from scripts.tactile_unit.vac_runtime_common import resolve_device, set_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    artifacts = ROOT / config["runtime"]["artifact_root"]
    identities = identity_snapshot(config)
    if not identities["pass"]:
        if not identities["equality"].get("full", False):
            raise RuntimeError("C4_FULL_PATH_CHECKPOINT_INVALID")
        raise RuntimeError("STRUCTURAL_FAIL: accepted frozen identity mismatch")
    truth = []
    for action in (True, False):
        for contact in (True, False):
            for vision in (True, False):
                availability = ModalityAvailability(vision, action, contact)
                truth.append({
                    "vision_available": vision, "action_available": action,
                    "contact_context_available": contact,
                    "mode": route_availability(availability).value,
                })
    expected = {
        (True, True, True): AvailabilityMode.FULL_AH,
        (False, True, True): AvailabilityMode.FULL_AH,
        (True, True, False): AvailabilityMode.FALLBACK_VA,
        (False, True, False): AvailabilityMode.FALLBACK_A,
    }
    router_pass = all(
        (AvailabilityMode(row["mode"]) is expected.get(
            (row["vision_available"], row["action_available"], row["contact_context_available"]),
            AvailabilityMode.ABSTAIN_NO_ACTION,
        )) for row in truth
    )
    protocol = {
        "schema": "tactile3d-unit.vac-c4-availability-protocol.v1",
        "evaluation": "LOCKED POST-HOC BENCHMARK RE-EVALUATION",
        "explicit_metadata_only": True, "zero_tensor_missingness_detection": False,
        "prediction_target": "u_c shared [B,8,32]", "private_residual": "ACTUAL-CONTACT-ONLY",
        "fallback_va_online_ready": False, "selection_split": "train+validation only",
        "test_loaded": False, "identity": identities,
    }
    router = {
        "schema": "tactile3d-unit.vac-c4-router-contract.v1",
        "deterministic": True, "neural_missingness_router": False,
        "truth_table": truth, "pass": router_pass, "test_loaded": False,
    }
    atomic_json(artifacts / "availability_protocol.json", protocol)
    atomic_json(artifacts / "router_contract.json", router)
    for name in ("availability_protocol.json", "router_contract.json"):
        (artifacts / f"{name}.sha256").write_text(
            sha256_file(artifacts / name) + f"  {name}\n"
        )
    atomic_json(artifacts / "fallback_selection.json.preregistered", {
        "architecture": config["architecture"], "trials": config["training"]["trials"],
        "simplicity_tolerance": config["validation"]["simplicity_tolerance"],
        "test_loaded": False,
    })
    atomic_json(artifacts / "uncertainty_selection.json.preregistered", {
        "model": config["uncertainty"], "high_error_threshold": "validation error 75th percentile",
        "common_scale_across_modes": True, "test_loaded": False,
    })

    device, lock_handle, gpu = resolve_device(args.device, allowed_physical=("0", "1", "2", "3"))
    try:
        set_seed(int(config["seed"]))
        train = load_split(config, "train")
        validation = load_split(config, "validation")
        full, metadata = load_full(config, device)
        parent = load_parent_config(config)
        correct = predict_numpy(full, validation, device, args.batch_size)
        h_mean = np.asarray(train["h_current"], dtype=np.float64).mean(0).astype(np.float32)
        different = different_episode_permutation(validation["episode_id"], int(config["seed"]) + 1)
        wrong = wrong_time_indices(validation["episode_id"], validation["t"])
        rng = np.random.default_rng(int(config["seed"]) + 2)
        noisy_h = np.asarray(validation["h_current"]) + rng.normal(
            0.0, float(np.asarray(train["h_current"]).std()) * 0.1,
            size=np.asarray(validation["h_current"]).shape,
        ).astype(np.float32)
        variants = {
            "correct": correct,
            "zero": predict_numpy(full, validation, device, args.batch_size, h_current=np.zeros_like(validation["h_current"])),
            "mean": predict_numpy(full, validation, device, args.batch_size, h_current=np.broadcast_to(h_mean, np.asarray(validation["h_current"]).shape)),
            "wrong_time": predict_numpy(full, validation, device, args.batch_size, h_current=np.asarray(validation["h_current"])[wrong]),
            "different_episode": predict_numpy(full, validation, device, args.batch_size, h_current=np.asarray(validation["h_current"])[different]),
            "noisy": predict_numpy(full, validation, device, args.batch_size, h_current=noisy_h),
        }
        train_prediction = predict_numpy(full, train, device, args.batch_size)
        probe = fit_probe(train_prediction, train["contact_transition"])
        baseline = {}
        for name, prediction in variants.items():
            labels = probe.predict(prediction.reshape(len(prediction), -1))
            semantics = evaluate_prediction(
                validation["contact_transition"], labels,
                majority(train["contact_transition"], len(labels), 4), 4,
            )
            baseline[name] = {
                "shared_mse": float(row_mse(prediction, validation["u_c"]).mean()),
                "contact_macro_f1": float(semantics["macro_f1"]),
                "future_change_macro_f1": float(semantics["future_change"]["macro_f1"]),
            }
        result = {
            "schema": "tactile3d-unit.vac-c4-availability-baseline.v1",
            "split": "validation", "test_loaded": False,
            "full_checkpoint": config["runtime"]["full_checkpoint"],
            "full_checkpoint_sha256": sha256_file(ROOT / config["runtime"]["full_checkpoint"]),
            "full_metadata_test_loaded": metadata.get("test_loaded"),
            "variants": baseline,
            "correct_beats_zero_and_mean_mse": baseline["correct"]["shared_mse"] < min(baseline["zero"]["shared_mse"], baseline["mean"]["shared_mse"]),
            "gpu": {**gpu, "preferred_physical": 1, "fallback": gpu.get("actual_physical") != 1},
            "identity": identities,
        }
        atomic_json(artifacts / "availability_baseline.json", result)
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    main()
