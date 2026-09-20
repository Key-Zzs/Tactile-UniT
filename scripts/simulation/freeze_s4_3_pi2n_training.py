#!/usr/bin/env python3
"""Freeze one PI2N training identity and a NAS-resident source snapshot."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2n"
BASE_MANIFEST = ROOT / ".local/artifacts/simulation/s4_3_pi0/official_base_model_manifest.json"
COMMON_FILES = (
    "gr00t/simulation/s4_3_pi1.py",
    "gr00t/simulation/pi05_tactile_unit.py",
    "scripts/simulation/train_s4_3_pi2n.py",
    "scripts/simulation/calibrate_s4_3_pi2n_lambda.py",
    "scripts/simulation/validate_s4_3_pi2n_loaded_base_gradients.py",
    "scripts/simulation/freeze_s4_3_pi2n_training.py",
    "configs/simulation/s4_3_pi2n_candidate_protocol.json",
    "configs/simulation/s4_3_pi2n_dag.json",
    "configs/simulation/s4_3_pi2n_diagnostics.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(32 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", choices=("B_VA27", "B_VAC_V"), required=True)
    args = parser.parse_args()
    run_root = Path(os.environ.get("PI2N_RUN_ROOT", ROOT / ".local/experiments/simulation/s4_3_pi2n")).resolve()
    expected_root = Path("/mnt/ugreen_nas/storage/UniT_storage/experiments/simulation/s4_3_pi2n")
    if run_root != expected_root:
        raise SystemExit(f"run root differs from audited NAS root: {run_root}")
    name = args.model_id.lower()
    output = ARTIFACTS / f"{name}_training_freeze.json"
    snapshot_dir = run_root / "code_snapshots" / args.model_id
    archive_path = snapshot_dir / "prelaunch_source.tar.gz"
    nas_freeze_path = snapshot_dir / "training_freeze.json"
    if output.exists() or archive_path.exists() or nas_freeze_path.exists():
        raise SystemExit(f"refusing to overwrite an existing {args.model_id} training freeze")

    calibration_path = ARTIFACTS / f"{name}_lambda_calibration.json"
    gradient_path = ARTIFACTS / f"{name}_loaded_base_gradient_gate.json"
    target_audit_path = ARTIFACTS / f"{name}_target_audit.json"
    prerequisites = {path.name: json.loads(path.read_text()) for path in (calibration_path, gradient_path, target_audit_path)}
    if any(payload.get("status") != "PASS" for payload in prerequisites.values()):
        raise SystemExit("calibration, target audit and loaded-base gradient gate must all PASS")
    if args.model_id == "B_VA27":
        historical_paths = (
            ROOT / ".local/artifacts/simulation/s4_3_pi2u/bva_mode_contract.json",
            ROOT / ".local/artifacts/simulation/s4_3_pi2u/bva_none_parity.json",
            ROOT / ".local/artifacts/simulation/s4_3_pi2m/corrected_target_audit.json",
        )
        historical = {path.name: json.loads(path.read_text()) for path in historical_paths}
        if any(payload.get("status") != "PASS" for payload in historical.values()):
            raise SystemExit("accepted BVA mode parity and corrected target audits must PASS")
        target_path = ROOT / ".local/datasets/simulation/s4_3_pi2m/pinch_tongs_va_t27/sidecar.npz"
        extra_files: tuple[str, ...] = ()
    else:
        mode_path = ARTIFACTS / "b_vac_v_mode_parity.json"
        historical = {mode_path.name: json.loads(mode_path.read_text())}
        if historical[mode_path.name].get("status") != "PASS":
            raise SystemExit("B_VAC_V mode parity must PASS")
        target_path = run_root / "caches/pinch_tongs_vac_v_t27/sidecar.npz"
        extra_files = (
            "scripts/simulation/build_s4_3_pi2n_vac_v_targets.py",
            "configs/simulation/s4_3_pi2n_vac_v_target_protocol.json",
        )

    source_files = tuple(dict.fromkeys((*COMMON_FILES, *extra_files)))
    source_hashes = {relative: sha256_file(ROOT / relative) for relative in source_files}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--", *source_files], cwd=ROOT)
    base_manifest = json.loads(BASE_MANIFEST.read_text())
    calibration = prerequisites[calibration_path.name]
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2n-training-freeze.v1",
        "status": "FROZEN_BEFORE_TRAINING",
        "model_id": args.model_id,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_head": head,
        "git_branch": branch,
        "source_files_sha256": source_hashes,
        "source_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_snapshot": "$PI2N_RUN_ROOT/code_snapshots/" + args.model_id + "/prelaunch_source.tar.gz",
        "source_snapshot_sha256": None,
        "initialization": "exact pi05_base independent initialization",
        "base_manifest_sha256": sha256_file(BASE_MANIFEST),
        "base_manifest_status": base_manifest.get("status"),
        "target_path": str(target_path),
        "target_sha256": sha256_file(target_path),
        "lambda_phys": calibration["lambda_phys"],
        "lambda_calibration_sha256": sha256_file(calibration_path),
        "loaded_base_gradient_gate_sha256": sha256_file(gradient_path),
        "target_audit_sha256": sha256_file(target_audit_path),
        "mode_prerequisites_sha256": {path.name: sha256_file(path) for path in historical_paths} if args.model_id == "B_VA27" else {"b_vac_v_mode_parity.json": sha256_file(ARTIFACTS / "b_vac_v_mode_parity.json")},
        "seed": 42,
        "steps": 30000,
        "global_batch_size": 32,
        "save_interval": 10000,
        "final_checkpoint_step": 29999,
        "checkpoint_selection": "FINAL_ONLY",
        "resume": False,
        "overwrite": False,
        "PI2B": "NOT_AUTHORIZED",
    }
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        for relative in source_files:
            archive.add(ROOT / relative, arcname=relative, recursive=False)
        diff_info = tarfile.TarInfo("working_tree.patch")
        diff_info.size = len(diff)
        diff_info.mtime = 0
        archive.addfile(diff_info, io.BytesIO(diff))
    payload["source_snapshot_sha256"] = sha256_file(archive_path)
    atomic_json(output, payload)
    atomic_json(nas_freeze_path, payload)
    print(json.dumps({"status": payload["status"], "model_id": args.model_id, "lambda_phys": payload["lambda_phys"], "snapshot_sha256": payload["source_snapshot_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
