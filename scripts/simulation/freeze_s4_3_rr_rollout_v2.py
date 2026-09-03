#!/usr/bin/env python3
"""Freeze the RR5 rollout harness after the production-path RR4 smoke gate."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_3_training import atomic_json, read_json, sha256_file  # noqa: E402
from scripts.simulation.freeze_s4_3_restart_protocol import (  # noqa: E402
    CHECKPOINTS,
    package_audit,
    tracked_s4_2_hashes,
)

ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_rr"
OLD_ROOT = ROOT / ".local/artifacts/simulation/s4_3_restart"
FREEZE = ARTIFACT_ROOT / "pre_rollout_freeze_v2.json"
WORKER_MANIFEST = ARTIFACT_ROOT / "rollout_worker_manifest.json"
TRAINING = OLD_ROOT / "act_training_manifest.json"
OLD_FREEZE = OLD_ROOT / "pre_rollout_freeze.json"
RUNTIME_CONFIG = ROOT / "configs/simulation/s4_3_rr_rollout_runtime_v2.json"
BENCHMARK_CONFIG = ROOT / "configs/simulation/s4_3_rr_final_benchmark_contract.json"
EVAL_CONFIG = ROOT / "configs/simulation/s4_3_policy_eval_v1.json"
STATISTICS = ROOT / "scripts/simulation/analyze_s4_3_closed_loop.py"
ROLLOUT_CODE = (
    ROOT / "gr00t/simulation/s4_3_transport.py",
    ROOT / "scripts/simulation/run_s4_3_policy_rollouts_dex.py",
    ROOT / "scripts/simulation/serve_s4_3_act_policy.py",
    ROOT / "scripts/simulation/run_s4_3_rollout_job.py",
    ROOT / "scripts/simulation/run_s4_3_rollout_queue.py",
)
ALLOWED_CHANGED = {
    "configs/simulation/s4_3_rr_final_benchmark_contract.json",
    "configs/simulation/s4_3_rr_rollout_runtime_v2.json",
    "docs/research/s4_3_rr_act_rollout_runtime_remediation.md",
    "gr00t/simulation/s4_3_transport.py",
    "scripts/simulation/audit_s4_3_rr_transport.py",
    "scripts/simulation/freeze_s4_3_rr_rollout_v2.py",
    "scripts/simulation/run_s4_3_policy_rollouts_dex.py",
    "scripts/simulation/run_s4_3_rollout_job.py",
    "scripts/simulation/run_s4_3_rollout_queue.py",
    "scripts/simulation/run_s4_3_rr_production_smoke.py",
    "scripts/simulation/serve_s4_3_act_policy.py",
    "tests/simulation/test_s4_3_rollout_runtime.py",
}


def command(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        args, cwd=cwd, check=True, text=True, capture_output=True
    ).stdout.strip()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def changed_files() -> list[str]:
    rows = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    return sorted(row[3:] for row in rows if row)


def main() -> None:
    smoke = read_json(ARTIFACT_ROOT / "production_smoke_results.json")
    smoke_manifest = read_json(ARTIFACT_ROOT / "production_smoke_manifest.json")
    if (
        smoke["status"] != "PASS"
        or smoke["smokes"] != 6
        or smoke["scientific_results_produced"] != 0
        or smoke_manifest["status"] != "PASS"
    ):
        raise SystemExit("S4_3_RR_PRODUCTION_SMOKE_FAIL")

    old = read_json(OLD_FREEZE)
    training = read_json(TRAINING)
    checkpoints = {
        f"{row['task']}/{row['variant']}/{row['training_seed']}": sha256_file(
            ROOT / row["checkpoint"]
        )
        for row in training["jobs"]
    }
    expected_checkpoints = {
        f"{row['task']}/{row['variant']}/{row['training_seed']}": row["checkpoint_sha256"]
        for row in training["jobs"]
    }
    s4_2 = {name: sha256_file(ROOT / path) for name, path in CHECKPOINTS.items()}
    vision_identity_path = ROOT / ".local/artifacts/simulation/s4_2_formal/vision_identity.json"
    vision_identity = read_json(vision_identity_path)["checkpoint_file_sha256"]
    checkpoint_root = Path(os.environ["UNIT_FULLDATA_CKPT"]) / "tokenizer"
    vision_actual = {name: sha256_file(checkpoint_root / name) for name in vision_identity}
    policy_dependencies = {
        path: sha256_file(ROOT / path) for path in old["policy_dependency_code_sha256"]
    }
    packages, versions = package_audit()
    changes = changed_files()
    unexpected_changes = sorted(set(changes) - ALLOWED_CHANGED)
    runtime_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path) for path in ROLLOUT_CODE
    }
    history_files = {
        name: sha256_file(OLD_ROOT / name)
        for name in (
            "r12_structural_failure.json",
            "r12_failure_closeout.json",
            "closed_loop_rollouts.json",
            "final_decision.json",
        )
    }
    gates = {
        "checkpoint_hashes": checkpoints == expected_checkpoints and len(checkpoints) == 36,
        "s4_2_checkpoints": s4_2 == old["s4_2_checkpoint_sha256"],
        "s4_2_configs": tracked_s4_2_hashes() == old["s4_2_tracked_config_sha256"],
        "vision": vision_actual == vision_identity
        and sha256_file(vision_identity_path) == old["s4_2_vision_identity_sha256"],
        "policy_dependencies": policy_dependencies == old["policy_dependency_code_sha256"],
        "statistics": sha256_file(STATISTICS) == old["statistical_code_sha256"],
        "policy_protocol": sha256_file(
            ROOT / "configs/simulation/s4_3_restart_policy_protocol.json"
        )
        == old["policy_protocol_sha256"],
        "act_protocol": sha256_file(ROOT / "configs/simulation/s4_3_restart_act_protocol.json")
        == old["act_protocol_sha256"],
        "evaluation_resets": sha256_file(EVAL_CONFIG) == old["evaluation_reset_config_sha256"],
        "success_contract": sha256_file(OLD_ROOT / "task_success_contract.json")
        == old["success_contract_sha256"],
        "action_adapter": sha256_file(
            ROOT / "gr00t/simulation/dexjoco_adapter.py"
        )
        == old["action_adapter_source_sha256"],
        "environment_packages": packages
        == read_json(ARTIFACT_ROOT / "starting_integrity.json")["package_hashes_before"],
        "changed_files_scoped": not unexpected_changes,
        "smoke": smoke["status"] == "PASS",
    }
    freeze = {
        "schema": "tactile3d-unit.s4-3-rr-pre-rollout-freeze-v2.v1",
        "stage": "RR5",
        "previous_s4_3_2_decision": "S4_3_2_ENVIRONMENT_FAIL",
        "previous_failure": "AF_UNIX_PATH_TOO_LONG",
        "previous_scientific_rollouts_started": 0,
        "previous_policy_performance_seen": False,
        "historical_failure_modified": False,
        "historical_artifact_sha256": history_files,
        "runtime_config": str(RUNTIME_CONFIG.relative_to(ROOT)),
        "runtime_config_sha256": sha256_file(RUNTIME_CONFIG),
        "benchmark_contract": str(BENCHMARK_CONFIG.relative_to(ROOT)),
        "benchmark_contract_sha256": sha256_file(BENCHMARK_CONFIG),
        "rollout_code_sha256": runtime_hashes,
        "rollout_harness_sha256": canonical_sha256(runtime_hashes),
        "old_rollout_code_sha256": old["rollout_code_sha256"],
        "allowed_production_difference": "IPC endpoint construction/cleanup plus smoke/logging/audit scaffolding",
        "changed_files": changes,
        "unexpected_changed_files": unexpected_changes,
        "policy_dependency_code_sha256": policy_dependencies,
        "statistical_code": str(STATISTICS.relative_to(ROOT)),
        "statistical_code_sha256": sha256_file(STATISTICS),
        "checkpoint_count": 36,
        "checkpoint_selection_complete": True,
        "checkpoint_sha256": checkpoints,
        "checkpoint_set_sha256": read_json(OLD_ROOT / "act_checkpoint_manifest.json")[
            "checkpoint_set_sha256"
        ],
        "s4_2_checkpoint_sha256": s4_2,
        "s4_2_tracked_config_sha256": tracked_s4_2_hashes(),
        "s4_2_vision_identity_artifact_sha256": sha256_file(vision_identity_path),
        "s4_2_vision_checkpoint_files_sha256": vision_actual,
        "evaluation_reset_config": str(EVAL_CONFIG.relative_to(ROOT)),
        "evaluation_reset_config_sha256": sha256_file(EVAL_CONFIG),
        "evaluation_reset_count": 90,
        "success_contract_sha256": sha256_file(OLD_ROOT / "task_success_contract.json"),
        "timeouts": old["timeouts"],
        "warmup": old["warmup"],
        "replan_stride": old["replan_stride"],
        "action_adapter": old["action_adapter"],
        "action_adapter_source_sha256": old["action_adapter_source_sha256"],
        "statistics": read_json(BENCHMARK_CONFIG)["statistics"],
        "material_effect": read_json(BENCHMARK_CONFIG)["material_effect"],
        "runtime_uncertainty": "NOT_AVAILABLE_CAUSALLY_FOR_S4_3_2",
        "training_allowed": False,
        "checkpoint_selection_allowed": False,
        "policy_hyperparameter_change_allowed": False,
        "evaluation_seed_change_allowed": False,
        "scientific_rollout_performance_seen": False,
        "rollout_performance_seen": False,
        "scientific_protocol_immutable_after_first_rollout": True,
        "package_hashes": packages,
        "python_versions": versions,
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "FAIL",
    }
    atomic_json(FREEZE, freeze)

    workers = {
        "schema": "tactile3d-unit.s4-3-rr-rollout-worker-manifest.v1",
        "stage": "RR5",
        "worker": "scripts/simulation/run_s4_3_rollout_job.py",
        "worker_sha256": sha256_file(ROOT / "scripts/simulation/run_s4_3_rollout_job.py"),
        "endpoint_algorithm": "$RUNTIME_TMP/tu3d_<short_hash>_<pid>_<nonce>.sock",
        "jobs": [
            {
                "task": row["task"],
                "variant": row["variant"],
                "training_seed": row["training_seed"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "expected_resets": 30,
            }
            for row in training["jobs"]
        ],
        "expected_jobs": 36,
        "expected_rollouts": 1080,
        "status": "PASS",
    }
    atomic_json(WORKER_MANIFEST, workers)

    immutability = {
        "schema": "tactile3d-unit.s4-3-rr-s4-2-immutability.v1",
        "stage": "RR5",
        "checkpoint_sha256_before": old["s4_2_checkpoint_sha256"],
        "checkpoint_sha256_at_v2_freeze": s4_2,
        "checkpoint_byte_identical": gates["s4_2_checkpoints"],
        "vision_identity_artifact_sha256_before": old["s4_2_vision_identity_sha256"],
        "vision_checkpoint_files_sha256_before": vision_identity,
        "vision_sha256_at_v2_freeze": vision_actual,
        "vision_byte_identical": gates["vision"],
        "tracked_config_sha256_before": old["s4_2_tracked_config_sha256"],
        "tracked_config_sha256_at_v2_freeze": tracked_s4_2_hashes(),
        "tracked_configs_byte_identical": gates["s4_2_configs"],
        "status": (
            "PASS"
            if gates["s4_2_checkpoints"] and gates["vision"] and gates["s4_2_configs"]
            else "FAIL"
        ),
    }
    atomic_json(ARTIFACT_ROOT / "s4_2_immutability.json", immutability)

    figure, axis = plt.subplots(figsize=(7, 4))
    matrix = [[1, 1], [1, 1], [1, 1]]
    axis.imshow(matrix, cmap="Greens", vmin=0, vmax=1)
    axis.set_xticks(range(2), ["P0", "P3"])
    axis.set_yticks(range(3), ["pinch_tongs", "hammer_nail", "click_mouse"])
    axis.set_title("RR4 exact production-path smoke matrix: 6/6 PASS")
    for task in range(3):
        for variant in range(2):
            axis.text(variant, task, "PASS", ha="center", va="center")
    figure.tight_layout()
    figure.savefig(ARTIFACT_ROOT / "plots/04_production_smoke_matrix.png", dpi=180)
    plt.close(figure)

    print(
        json.dumps(
            {
                "checkpoint_count": len(checkpoints),
                "rollout_harness_sha256": freeze["rollout_harness_sha256"],
                "status": freeze["status"],
            },
            sort_keys=True,
        )
    )
    if freeze["status"] != "PASS":
        raise SystemExit("S4_3_RR_ROLLOUT_HARNESS_FAIL")


if __name__ == "__main__":
    main()
