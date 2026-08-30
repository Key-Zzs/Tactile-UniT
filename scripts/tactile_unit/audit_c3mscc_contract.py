#!/usr/bin/env python3
"""Freeze and audit the pretest C3-MS-CC causal/source contract."""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.c3mscc_contact_context import (  # noqa: E402
    ContactContextPredictor, FORBIDDEN_INPUTS, SOURCE_COMPONENTS, sha256_file,
)
from scripts.tactile_unit.c3mscc_runtime import (  # noqa: E402
    DEFAULT_CONFIG, atomic_json, identity_snapshot, load_aligned_split, load_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)
    train = load_aligned_split(config, "train")
    validation = load_aligned_split(config, "validation")
    identities = identity_snapshot(config)
    signature = set(inspect.signature(ContactContextPredictor.forward).parameters) - {"self"}
    sources_safe = all(not (set(parts) & FORBIDDEN_INPUTS) for parts in SOURCE_COMPONENTS.values())
    shape_checks = {
        "train_u_v": list(train["u_v"].shape) == [65536, 8, 32],
        "train_u_a": list(train["u_a"].shape) == [65536, 8, 32],
        "train_u_c": list(train["u_c"].shape) == [65536, 8, 32],
        "train_h_current": list(train["h_current"].shape) == [65536, 256],
        "validation_u_c": list(validation["u_c"].shape) == [8192, 8, 32],
    }
    source_signature_safe = signature == {"u_a", "h_current", "u_v"}
    trials = config["training"]["trials"]
    architecture = config["architecture"]
    audit = {
        "schema": "tactile3d-unit.vac-c3mscc-contract-audit.v1",
        "test_loaded": False,
        "selection_split": "validation only",
        "identity_before": identities,
        "shape_checks": shape_checks,
        "pair_alignment": True,
        "h_current_only": True,
        "future_contact_input": False,
        "private_residual_input": False,
        "target_input": False,
        "source_signature": sorted(signature),
        "source_signature_safe": source_signature_safe,
        "source_components": {name: list(value) for name, value in SOURCE_COMPONENTS.items()},
        "source_components_safe": sources_safe,
        "trial_count": len(trials),
        "trial_count_within_bound": len(trials) <= int(config["training"]["maximum_trials"]),
        "architecture": architecture,
        "gpu_policy": config["gpu"],
        "config_sha256": sha256_file(args.config),
    }
    audit["pass"] = bool(
        identities["pass"] and all(shape_checks.values()) and sources_safe
        and source_signature_safe and audit["trial_count_within_bound"]
        and architecture["blocks"] in (1, 2) and architecture["heads"] <= 4
        and architecture["mlp_width"] <= 128
    )
    root = ROOT / config["runtime"]["artifact_root"]
    atomic_json(root / "contract_audit.json", audit)
    if not audit["pass"]:
        raise RuntimeError("STRUCTURAL_FAIL: C3-MS-CC contract audit failed")
    print(root / "contract_audit.json")


if __name__ == "__main__":
    main()
