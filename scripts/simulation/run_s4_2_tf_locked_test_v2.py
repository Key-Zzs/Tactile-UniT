#!/usr/bin/env python3
"""Run the frozen FORMAL_TEST_V2 once and one equality-only repeat."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_2_dataset import sha256_file  # noqa: E402
from scripts.simulation.run_s4_2_8_locked_test import (  # noqa: E402
    CACHE_ROOT,
    atomic_json,
    build_locked_cache,
    evaluate_locked,
    load_json,
    load_npz,
)

ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_2_tf"
DATASET_ROOT = ROOT / ".local/datasets/simulation/s4_2_test_v2"
PRETEST = ARTIFACT_ROOT / "pretest_v2_freeze.json"
ACCESS = ARTIFACT_ROOT / "locked_test_v2_access.json"
TEST_CACHE = ROOT / ".local/cache/simulation/s4_2_tf/paired_test_v2.npz"
FIRST_RESULT = ARTIFACT_ROOT / "locked_test_v2.json"
REPEAT_RESULT = ARTIFACT_ROOT / "locked_test_v2_repeat.json"
V1_CACHE = CACHE_ROOT / "paired_test.npz"

CHECKPOINT_PATHS = {
    "contact_state": ROOT / ".local/experiments/simulation/s4_2r/contact_state/accepted.pt",
    "contact_C3": ROOT / ".local/experiments/simulation/s4_2r/contact_dynamics/selected.pt",
    "action": ROOT / ".local/experiments/simulation/s4_2_formal/action/selected.pt",
    "bridge": ROOT / ".local/experiments/simulation/s4_2_formal/bridge/selected.pt",
    "shared_private": ROOT / ".local/experiments/simulation/s4_2_formal/s4_2_6/shared_private.pt",
    "rejected_contact_rq": ROOT / ".local/experiments/simulation/s4_2_formal/s4_2_6/contact_rq_selected.pt",
    "conditional_A_plus_H": ROOT / ".local/experiments/simulation/s4_2_formal/s4_2_7/A_plus_H.pt",
    "conditional_V_plus_A_plus_H": ROOT / ".local/experiments/simulation/s4_2_formal/s4_2_7/V_plus_A_plus_H.pt",
    "conditional_V_plus_A_missing_H": ROOT / ".local/experiments/simulation/s4_2_formal/s4_2_7/V_plus_A_missing_H.pt",
    "conditional_A_only_missing_H": ROOT / ".local/experiments/simulation/s4_2_formal/s4_2_7/A_only_missing_H.pt",
    "uncertainty_full": ROOT / ".local/experiments/simulation/s4_2_formal/s4_2_7/uncertainty_full.pt",
    "uncertainty_missing_H": ROOT / ".local/experiments/simulation/s4_2_formal/s4_2_7/uncertainty_missing_H.pt",
}


def digest_strings(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values.tolist():
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def verify_pretest(*, repeat: bool) -> dict[str, Any]:
    pretest = load_json(PRETEST)
    if (
        pretest["status"] != "PASS"
        or pretest["test_loaded"]
        or pretest["formal_test_v2_model_metrics_loaded"]
    ):
        raise RuntimeError("valid pretest_v2_freeze is required before FORMAL_TEST_V2")
    for relative, expected in pretest["implementation_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"implementation changed after pretest_v2_freeze: {relative}")
    for name, path in CHECKPOINT_PATHS.items():
        if sha256_file(path) != pretest["checkpoint_sha256"][name]:
            raise RuntimeError(f"checkpoint changed after pretest_v2_freeze: {name}")
    selection_paths = {
        "paired_train": CACHE_ROOT / "paired_train.npz",
        "paired_validation": CACHE_ROOT / "paired_validation.npz",
        "shared_train": CACHE_ROOT / "shared_train.npz",
        "shared_validation": CACHE_ROOT / "shared_validation.npz",
    }
    for name, path in selection_paths.items():
        if sha256_file(path) != pretest["selection_cache_sha256"][name]:
            raise RuntimeError(f"selection cache changed after pretest_v2_freeze: {name}")
    if repeat:
        if not FIRST_RESULT.is_file() or not TEST_CACHE.is_file() or REPEAT_RESULT.exists():
            raise RuntimeError("FORMAL_TEST_V2 deterministic repeat is not authorized")
    elif FIRST_RESULT.exists() or TEST_CACHE.exists() or ACCESS.exists():
        raise RuntimeError("FORMAL_TEST_V2 first locked evaluation already started")
    return pretest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("first", "repeat"), required=True)
    parser.add_argument("--unit-checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vision-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()
    repeat = args.mode == "repeat"
    pretest = verify_pretest(repeat=repeat)
    torch.manual_seed(4242)
    np.random.seed(4242)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if not repeat:
        if args.unit_checkpoint is None:
            raise RuntimeError("--unit-checkpoint is required for FORMAL_TEST_V2 first access")
        build_locked_cache(
            args.unit_checkpoint,
            device,
            args.vision_batch_size,
            args.workers,
            dataset_root=DATASET_ROOT,
            pretest_path=PRETEST,
            test_cache=TEST_CACHE,
            access_path=ACCESS,
            prior_cache_paths=(
                CACHE_ROOT / "paired_train.npz",
                CACHE_ROOT / "paired_validation.npz",
                V1_CACHE,
            ),
            access_schema="tactile3d-unit.s4-2-tf-locked-test-v2-access.v1",
            test_name="FORMAL_TEST_V2",
        )
        test = load_npz(TEST_CACHE)
        if digest_strings(test["pair_id"]) != pretest["dataset_pair_identity_sha256"]:
            raise RuntimeError("FORMAL_TEST_V2 pair identity changed during feature extraction")
    result = evaluate_locked(
        device,
        args.bootstrap_samples,
        test_cache=TEST_CACHE,
        result_schema="tactile3d-unit.s4-2-tf-locked-test-v2.v1",
        test_name="FORMAL_TEST_V2",
    )
    if repeat:
        first = load_json(FIRST_RESULT)
        result["deterministic_repeat_of"] = first["metric_digest"]
        result["deterministic_equal"] = result["metric_digest"] == first["metric_digest"]
        atomic_json(REPEAT_RESULT, result)
        if not result["deterministic_equal"]:
            raise SystemExit("S4_2_TF_TEST_V2_DETERMINISTIC_REPEAT_FAIL")
    else:
        atomic_json(FIRST_RESULT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
