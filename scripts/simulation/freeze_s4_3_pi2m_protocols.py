#!/usr/bin/env python3
"""Freeze PI2M training/evaluation/statistics inputs before B_HVA training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2m"
TRAINING_CONFIG = ROOT / "configs/simulation/s4_3_pi2m_bhva_protocol.json"
EVAL_CONFIG = ROOT / "configs/simulation/s4_3_pi2m_evaluation_protocol.json"
STAT_CONFIG = ROOT / "configs/simulation/s4_3_pi2m_statistical_protocol.json"
RESET_MANIFEST = ARTIFACTS / "fresh_reset_manifest.json"
CHECKPOINTS = {
    "B1": ROOT
    / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/"
    "s43_pi1b_contact_tokens_seed42/29999",
    "B2": ROOT
    / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/"
    "s43_pi1c_contact_tokens_physical_aux_seed42/29999",
}
EXPECTED_CHECKPOINTS = {
    "B1": "04b59cbc3491bf4e88dc75558a08a94e5588e2fbe398d413e1766a87c7ab6f4f",
    "B2": "86c4908533ca24da06ae66f943e3605b5ccbae455441290ca8fb35514359e949",
}
SOURCES = (
    "gr00t/simulation/pi05_tactile_unit.py",
    "gr00t/simulation/s4_3_pi1.py",
    "scripts/simulation/train_s4_3_pi2m_bhva.py",
    "scripts/simulation/launch_s4_3_pi2m_bhva.sh",
    "scripts/simulation/supervise_s4_3_pi2m_bhva_training.sh",
    "scripts/simulation/audit_s4_3_pi2m_bhva_launch.py",
    "scripts/simulation/serve_s4_3_pi2m_policy.py",
    "scripts/simulation/run_s4_3_pi2m_eval.py",
    "scripts/simulation/analyze_s4_3_pi2m.py",
    "scripts/simulation/freeze_s4_3_pi2m_protocols.py",
    "scripts/simulation/evaluate_s4_3_pi1d_augmented.py",
    "scripts/simulation/serve_s4_3_pi1_contact_state.py",
    "third_party/dexjoco/configs/rand_obj/pinch_tongs.yaml",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(32 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for candidate in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = candidate.relative_to(path).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(sha256_file(candidate).encode())
        digest.update(b"\0")
        digest.update(str(candidate.stat().st_size).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()


def reset_manifest() -> None:
    if RESET_MANIFEST.exists():
        raise SystemExit("refusing to overwrite PI2M fresh reset manifest")
    import mujoco
    import yaml

    sys.path.insert(0, str(ROOT / "third_party/dexjoco/dexjoco"))
    from dexjoco_openpi_client import eval_dexjoco_openpi as official
    from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv

    config_path = ROOT / "third_party/dexjoco/configs/rand_obj/pinch_tongs.yaml"
    config = yaml.safe_load(config_path.read_text())
    official._set_seed(7)
    env = DexJoCoOpenPIEnv(
        env_name=config["env_name"],
        camera_mapping=config["camera_mapping"],
        seed=7,
        rand_full=False,
        randomize_dynamics=False,
        dual_arm=False,
        prompt=config["prompt"],
        render_mode="rgb_array",
        pad_state_dim46=False,
    )
    identities = []
    try:
        env.start()
        raw = env.env.unwrapped
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        for _ in range(200):
            env.reset()
            state = np.empty(mujoco.mj_stateSize(raw.model, state_spec), dtype=np.float64)
            mujoco.mj_getState(raw.model, raw.data, state, state_spec)
            digest = hashlib.sha256()
            for array in (state, np.asarray(env.obs["state"], dtype=np.float64)):
                value = np.ascontiguousarray(array)
                digest.update(str(value.dtype).encode())
                digest.update(str(value.shape).encode())
                digest.update(value.tobytes())
            identities.append(digest.hexdigest())
    finally:
        env.close()
    sequence_hash = hashlib.sha256("\n".join(identities).encode()).hexdigest()
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2m-fresh-reset-manifest.v1",
        "status": "PASS",
        "scientific_policy_inference_performed": False,
        "policy_performance_seen": False,
        "task": "pinch_tongs",
        "regime": "rand_obj",
        "evaluator_seed": 7,
        "episodes": 200,
        "ordered_reset_identities": identities,
        "reset_sequence_sha256": sequence_hash,
        "identity_definition": "sha256(mjSTATE_INTEGRATION float64 bytes + official processed state float64 bytes)",
        "rand_full": False,
        "randomize_dynamics": False,
        "config_sha256": sha256_file(config_path),
    }
    atomic_json(RESET_MANIFEST, payload)
    print(json.dumps({"status": "PASS", "episodes": 200, "reset_sequence_sha256": sequence_hash}))


def freeze() -> None:
    outputs = (
        ARTIFACTS / "fresh_seed_audit.json",
        ARTIFACTS / "training_protocol_freeze.json",
        ARTIFACTS / "evaluation_protocol_freeze.json",
        ARTIFACTS / "statistical_protocol.json",
        ARTIFACTS / "regression_tests.json",
    )
    if any(path.exists() for path in outputs):
        raise SystemExit("refusing to overwrite PI2M frozen protocol artifacts")
    prerequisites = {
        "target": ARTIFACTS / "corrected_target_audit.json",
        "mode": ARTIFACTS / "mode_parity.json",
        "gradient": ARTIFACTS / "loaded_base_gradient_gate.json",
        "lambda": ARTIFACTS / "lambda_calibration.json",
        "reset": RESET_MANIFEST,
    }
    payloads = {name: json.loads(path.read_text()) for name, path in prerequisites.items()}
    if any(payload.get("status") != "PASS" for payload in payloads.values()):
        raise SystemExit("PI2M prerequisites must all PASS before protocol freeze")
    if command("git", "status", "--short"):
        raise SystemExit("tracked worktree must be clean before PI2M protocol freeze")
    if command("git", "branch", "--show-current") != "develop/sim-benchmark":
        raise SystemExit("PI2M protocol freeze requires develop/sim-benchmark")
    head = command("git", "rev-parse", "HEAD")
    checkpoint_hashes = {name: tree_hash(path) for name, path in CHECKPOINTS.items()}
    if checkpoint_hashes != EXPECTED_CHECKPOINTS:
        raise SystemExit("B1/B2 protected checkpoint identity changed")
    code_hashes = {relative: sha256_file(ROOT / relative) for relative in SOURCES}
    calibration = payloads["lambda"]
    reset = payloads["reset"]
    evidence_paths = {
        0: ROOT / ".local/artifacts/simulation/s4_3_pi0/official_checkpoint_eval.json",
        1: ROOT / ".local/artifacts/simulation/s4_3_pi1/pi1d_b1_eval.json",
        2: ROOT / ".local/artifacts/simulation/s4_3_pi2a/b1_raw_rollouts.json",
        3: ROOT
        / ".local/artifacts/simulation/s4_3_pi2u_superseded_temporal_0p9_v1/aborted_seed3/b1_raw_rollouts.json",
        4: ROOT
        / ".local/artifacts/simulation/s4_3_pi2u_superseded_temporal_0p9_v1/fresh_seed_retry_seed4.json",
        5: ROOT
        / ".local/artifacts/simulation/s4_3_pi2u_superseded_temporal_0p9_v1/b1_raw_rollouts.json",
        6: ROOT / ".local/artifacts/simulation/s4_3_pi2u/b1_raw_rollouts.json",
    }
    if any(not path.is_file() for path in evidence_paths.values()):
        raise SystemExit("prior evaluator-seed exposure evidence is incomplete")
    exposure = [
        {
            "seed": seed,
            "performance_exposed": True,
            "evidence": "$REPO_ROOT/" + path.relative_to(ROOT).as_posix(),
            "evidence_sha256": sha256_file(path),
        }
        for seed, path in evidence_paths.items()
    ]
    fresh_seed = {
        "schema": "tactile3d-unit.s4-3-pi2m-fresh-seed-audit.v1",
        "status": "PASS",
        "selection_rule": "smallest nonnegative unexposed pi0.5 policy-performance evaluator seed",
        "exposure_ledger": exposure,
        "selected_seed": 7,
        "selected_seed_performance_inspected_before_freeze": False,
        "reset_manifest_sha256": sha256_file(RESET_MANIFEST),
    }
    atomic_json(outputs[0], fresh_seed)
    training = {
        "schema": "tactile3d-unit.s4-3-pi2m-training-protocol-freeze.v1",
        "status": "FROZEN_BEFORE_TRAINING",
        "git_head": head,
        "model_id": "B_HVA",
        "mode": "CONTACT_STATE_TOKENS_VA_PHYSICAL_AUX",
        "seed": 42,
        "steps": 30000,
        "global_batch_size": 32,
        "allowed_device_counts": [1, 2, 4],
        "initialization": "exact pi05_base",
        "lambda_phys": float(calibration["lambda_phys"]),
        "lambda_raw": float(calibration["lambda_phys_raw"]),
        "lambda_clamp_applied": bool(calibration["clamp_applied"]),
        "lambda_calibration_sha256": sha256_file(prerequisites["lambda"]),
        "target_sidecar_sha256": payloads["target"]["target"]["sidecar_sha256"],
        "target_valid_rows": 37365,
        "target_invalid_rows": 2700,
        "target_physical_horizon_seconds": 0.54,
        "contact_sidecar_sha256": "833db9ddb4d37534bf38a7ed0b214fee2f000bb4e489d3fa9507b4e7bca5bd8e",
        "checkpoint_selection": "final restored train_state step30000 only",
        "optimizer_LR_augmentation_precision_save_recipe": "identical to frozen B2 recipe",
        "new_runs": 1,
        "existing_models_retrained": False,
        "prerequisite_sha256": {name: sha256_file(path) for name, path in prerequisites.items()},
        "config_sha256": sha256_file(TRAINING_CONFIG),
        "code_sha256": code_hashes,
        "PI2B_started": False,
    }
    atomic_json(outputs[1], training)
    evaluation = {
        "schema": "tactile3d-unit.s4-3-pi2m-evaluation-protocol-freeze.v1",
        "status": "FROZEN_BEFORE_TRAINING",
        "git_head": head,
        "config_sha256": sha256_file(EVAL_CONFIG),
        "models": ["B1", "B_HVA", "B2"],
        "episodes_per_model": 200,
        "total_rollouts": 600,
        "evaluator_seed": 7,
        "reset_sequence_sha256": reset["reset_sequence_sha256"],
        "reset_manifest_sha256": sha256_file(RESET_MANIFEST),
        "checkpoint_tree_sha256": {
            **checkpoint_hashes,
            "B_HVA": "PENDING_UNIQUE_SEED42_TRAINING",
        },
        "runtime_mode_all_models": "CONTACT_STATE_TOKENS",
        "performance_seen": False,
        "code_sha256": code_hashes,
    }
    atomic_json(outputs[2], evaluation)
    statistical = {
        **json.loads(STAT_CONFIG.read_text()),
        "schema": "tactile3d-unit.s4-3-pi2m-statistical-protocol-freeze.v1",
        "source_config_sha256": sha256_file(STAT_CONFIG),
        "analysis_source_sha256": sha256_file(ROOT / "scripts/simulation/analyze_s4_3_pi2m.py"),
        "git_head": head,
    }
    atomic_json(outputs[3], statistical)
    conda_root = Path(sys.executable).resolve().parents[3]
    unit_python = conda_root / "envs/unit/bin/python"
    test = subprocess.run(
        [
            str(unit_python),
            "-m",
            "pytest",
            "-q",
            "tests/simulation/test_s4_3_pi2m.py",
            "tests/simulation/test_s4_3_pi2u.py",
            "tests/simulation/test_s4_3_pi1.py",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    regression = {
        "schema": "tactile3d-unit.s4-3-pi2m-regression-tests.v1",
        "status": "PASS"
        if test.returncode == 0 and payloads["mode"]["status"] == "PASS"
        else "FAIL",
        "pytest_command": "unit/bin/python -m pytest -q tests/simulation/test_s4_3_pi2m.py tests/simulation/test_s4_3_pi2u.py tests/simulation/test_s4_3_pi1.py",
        "pytest_exit_code": test.returncode,
        "pytest_output": (test.stdout + test.stderr).strip(),
        "mode_parity_sha256": sha256_file(prerequisites["mode"]),
        "historical_artifacts_written_by_regression": False,
    }
    atomic_json(outputs[4], regression)
    print(
        json.dumps(
            {
                "status": "PASS"
                if regression["status"] == "PASS"
                else "FAIL",
                "lambda_phys": training["lambda_phys"],
                "evaluator_seed": 7,
                "reset_sequence_sha256": reset["reset_sequence_sha256"],
            },
            sort_keys=True,
        )
    )
    if regression["status"] != "PASS":
        raise SystemExit("PI2M_REGRESSION_FREEZE_FAIL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("reset-manifest")
    sub.add_parser("freeze")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "reset-manifest":
        reset_manifest()
    else:
        freeze()


if __name__ == "__main__":
    main()
