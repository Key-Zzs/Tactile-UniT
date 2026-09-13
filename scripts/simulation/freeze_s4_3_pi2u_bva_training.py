#!/usr/bin/env python3
"""Freeze the executable BVA training identity after calibration and before launch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2u"
OUTPUT = ARTIFACTS / "bva_training_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit("refusing to overwrite frozen BVA training protocol")
    tracked = ROOT / "configs/simulation/s4_3_pi2u_bva_protocol.json"
    inputs = {
        "tracked_protocol": tracked,
        "temporal_remediation": ROOT / "configs/simulation/s4_3_pi2u_bva_temporal_remediation.json",
        "target_manifest": ARTIFACTS / "bva_target_manifest.json",
        "contact_leakage_audit": ARTIFACTS / "contact_leakage_audit.json",
        "mode_contract": ARTIFACTS / "bva_mode_contract.json",
        "none_parity": ARTIFACTS / "bva_none_parity.json",
        "lambda_calibration": ARTIFACTS / "bva_lambda_calibration.json",
        "va_bridge_manifest": ARTIFACTS / "va_bridge_checkpoint_manifest.json",
    }
    values = {name: json.loads(path.read_text()) for name, path in inputs.items()}
    if values["tracked_protocol"]["status"] != "FROZEN_BEFORE_TRAINING":
        raise SystemExit("tracked protocol is not frozen")
    if values["temporal_remediation"].get("status") != "FROZEN_BEFORE_REMEDIATION_TRAINING":
        raise SystemExit("temporal remediation is not frozen")
    pass_prerequisites = set(inputs) - {"tracked_protocol", "temporal_remediation"}
    if any(values[name].get("status") != "PASS" for name in pass_prerequisites):
        raise SystemExit("one or more BVA prerequisites failed")
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2u-bva-training-protocol.v1",
        "status": "FROZEN_BEFORE_TRAINING",
        "intervention": "BVA",
        "seed": 42,
        "steps": 30000,
        "global_batch_size": 32,
        "lambda_phys": values["lambda_calibration"]["lambda_phys"],
        "checkpoint_steps": [10000, 20000, 29999],
        "inputs_sha256": {name: sha256(path) for name, path in inputs.items()},
        "implementation_sha256": {
            "target_builder": sha256(ROOT / "scripts/simulation/build_s4_3_pi2u_bva_targets.py"),
            "model": sha256(ROOT / "gr00t/simulation/pi05_tactile_unit.py"),
            "mode": sha256(ROOT / "gr00t/simulation/s4_3_pi1.py"),
            "entrypoint": sha256(ROOT / "scripts/simulation/train_s4_3_pi2u_bva.py"),
            "launcher": sha256(ROOT / "scripts/simulation/launch_s4_3_pi2u_bva.sh"),
        },
        "gpu_selection": "launch-time scan of physical GPUs 0..3; take two idle devices when possible because 32 is not divisible by three, otherwise one",
        "evaluation_before_user_resume": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUTPUT)
    print(json.dumps({"status": payload["status"], "lambda_phys": payload["lambda_phys"]}, sort_keys=True))


if __name__ == "__main__":
    main()
