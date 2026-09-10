#!/usr/bin/env python3
"""Freeze every model, source, and statistical choice before PI1D outcomes exist."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi1"
OUTPUT = ARTIFACTS / "pre_pi1d_freeze.json"
EVAL_ROOT = ROOT / ".local/experiments/simulation/s4_3_pi1/evaluation/seed1"
EVAL_LOG_ROOT = ROOT / ".local/logs/simulation/s4_3_pi1/pi1d"
MODELS = {
    "R0": ROOT / ".local/external/s4_3_pi0/models/DexJoCo-Pi05/pi05_dexjoco_ckpt/pinch_tongs",
    "B0": ROOT / ".local/experiments/simulation/s4_3_pi0/training/pinch_tongs/s43_pi0_official_seed42/29999",
    "B1": ROOT / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/s43_pi1b_contact_tokens_seed42/29999",
    "B2": ROOT / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/s43_pi1c_contact_tokens_physical_aux_seed42/29999",
}
OFFICIAL_EVALUATOR = ROOT / "third_party/dexjoco/dexjoco/dexjoco_openpi_client/eval_dexjoco_openpi.py"
OFFICIAL_WRAPPER = ROOT / "third_party/dexjoco/dexjoco/dexjoco_openpi_client/dexjoco_openpi_env.py"
SUCCESS_SOURCE = ROOT / "third_party/dexjoco/dexjoco/dexjoco/sim/envs/panda_pinch_tongs_env.py"
RAND_OBJ_CONFIG = ROOT / "third_party/dexjoco/configs/rand_obj/pinch_tongs.yaml"
SIDECAR = ROOT / ".local/datasets/simulation/s4_3_pi1/pinch_tongs_official_tactile/sidecar.npz"
CHECKPOINT_MANIFESTS = {
    "B1": ARTIFACTS / "pi1b_checkpoint_manifest.json",
    "B2": ARTIFACTS / "pi1c_checkpoint_manifest.json",
}
RUNTIME_SOURCES = (
    ROOT / "gr00t/simulation/pi1d_runtime.py",
    ROOT / "scripts/simulation/evaluate_s4_3_pi1d_augmented.py",
    ROOT / "scripts/simulation/serve_s4_3_pi1_contact_state.py",
    ROOT / "scripts/simulation/serve_s4_3_pi1_policy.py",
    ROOT / "scripts/simulation/run_s4_3_pi1d_evaluation.sh",
)
STATISTICS_SOURCE = ROOT / "scripts/simulation/summarize_s4_3_pi1d.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_manifest(path: Path) -> tuple[str, list[dict[str, Any]]]:
    rows = []
    for file_path in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        rows.append(
            {
                "path": file_path.relative_to(path).as_posix(),
                "bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["path"].encode())
        digest.update(b"\0")
        digest.update(row["sha256"].encode())
        digest.update(b"\0")
        digest.update(str(row["bytes"]).encode())
        digest.update(b"\n")
    return digest.hexdigest(), rows


def aggregate_source_hash(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def repo_path(path: Path) -> str:
    return "$REPO_ROOT/" + path.relative_to(ROOT).as_posix()


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite frozen PI1D protocol: {OUTPUT}")
    result_files = tuple(ARTIFACTS / f"pi1d_{name}_eval.json" for name in ("r0", "b0", "b1", "b2"))
    if EVAL_ROOT.exists() or EVAL_LOG_ROOT.exists() or any(path.exists() for path in result_files):
        raise SystemExit("PI1D evaluation output already exists; pre-performance freeze is no longer possible")

    fixture_path = ARTIFACTS / "runtime_none_parity.json"
    fixture = json.loads(fixture_path.read_text())
    protocol_path = ARTIFACTS / "pi1d_eval_protocol_freeze.json"
    protocol = json.loads(protocol_path.read_text())
    checkpoint_rows = {}
    checkpoint_hashes = {}
    for model, checkpoint in MODELS.items():
        tree_hash, rows = tree_manifest(checkpoint)
        checkpoint_hashes[model] = tree_hash
        checkpoint_rows[model] = {
            "checkpoint": repo_path(checkpoint),
            "checkpoint_tree_sha256": tree_hash,
            "files": len(rows),
            "bytes": sum(row["bytes"] for row in rows),
        }

    frozen_manifests = {model: json.loads(path.read_text()) for model, path in CHECKPOINT_MANIFESTS.items()}
    gates = {
        "no_evaluation_performance_seen": protocol["evaluation_performance_seen"] is False,
        "protocol_freeze_pass": protocol["status"] == "PASS",
        "runtime_none_parity_pass": fixture["status"] == "PASS",
        "all_four_checkpoints_present": all(path.is_dir() for path in MODELS.values()),
        "B1_checkpoint_matches_frozen_manifest": checkpoint_hashes["B1"]
        == frozen_manifests["B1"]["checkpoint_tree_sha256"],
        "B2_checkpoint_matches_frozen_manifest": checkpoint_hashes["B2"]
        == frozen_manifests["B2"]["checkpoint_tree_sha256"],
        "B1_B2_manifests_pass_and_frozen": all(
            row["status"] == "PASS" and row["frozen"] is True for row in frozen_manifests.values()
        ),
        "all_runtime_sources_present": all(path.is_file() for path in RUNTIME_SOURCES),
        "sidecar_present": SIDECAR.is_file(),
        "official_sources_present": OFFICIAL_EVALUATOR.is_file() and OFFICIAL_WRAPPER.is_file(),
        "success_source_present": SUCCESS_SOURCE.is_file(),
        "statistics_source_present": STATISTICS_SOURCE.is_file(),
        "evaluation_outputs_absent": not EVAL_ROOT.exists()
        and not EVAL_LOG_ROOT.exists()
        and not any(path.exists() for path in result_files),
        "no_model_selection_remains": True,
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi1d-pre-evaluation-freeze.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "stage": "S4.3-PI1D",
        "created_before_any_pi1d_rollout": True,
        "evaluation_performance_seen": False,
        "no_model_selection_remains": True,
        "fresh_evaluator_seed": 1,
        "episodes_per_model": 50,
        "evaluation_order": ["R0", "B0", "B1", "B2"],
        "ordered_reset_contract": {
            "same_seed": 1,
            "same_exact_50_sequential_resets": True,
            "fresh_environment_process_per_model": True,
            "reset_replacement": False,
            "reset_identity": "sha256(mjSTATE_INTEGRATION float64 bytes + official processed state float64 bytes)",
        },
        "models": checkpoint_rows,
        "checkpoint_tree_hash_algorithm": "sha256(sorted(relative_path NUL file_sha256 NUL bytes newline))",
        "official_evaluator": {
            "entrypoint": "dexjoco_openpi_client.eval_dexjoco_openpi.main",
            "source": repo_path(OFFICIAL_EVALUATOR),
            "source_sha256": sha256_file(OFFICIAL_EVALUATOR),
            "wrapper": repo_path(OFFICIAL_WRAPPER),
            "wrapper_sha256": sha256_file(OFFICIAL_WRAPPER),
            "aggregate_sha256": aggregate_source_hash((OFFICIAL_EVALUATOR, OFFICIAL_WRAPPER)),
        },
        "augmented_client": {
            "sources": {repo_path(path): sha256_file(path) for path in RUNTIME_SOURCES},
            "aggregate_sha256": aggregate_source_hash(RUNTIME_SOURCES),
            "read_only_contact_query": True,
            "physics_action_success_reset_camera_prompt_modified": False,
            "NONE_runtime_fixture": repo_path(fixture_path),
            "NONE_runtime_fixture_sha256": sha256_file(fixture_path),
        },
        "contact_state_sidecar": {
            "checkpoint": "$REPO_ROOT/.local/experiments/simulation/s4_2r/contact_state/accepted.pt",
            "checkpoint_sha256": json.loads((ARTIFACTS / "contact_state_cache_manifest.json").read_text())[
                "checkpoint_sha256"
            ],
            "client_service_source": repo_path(ROOT / "scripts/simulation/serve_s4_3_pi1_contact_state.py"),
            "client_service_sha256": sha256_file(ROOT / "scripts/simulation/serve_s4_3_pi1_contact_state.py"),
            "input": [26, 30],
            "output": [256],
            "current_only": True,
            "future_contact_used": False,
        },
        "tactile_sidecar": {
            "path": repo_path(SIDECAR),
            "sha256": sha256_file(SIDECAR),
            "live_contract": [30],
        },
        "history": {"shape": [26, 30], "bootstrap": "LEFT_REPEAT_FIRST", "causal": True},
        "success_criterion": {
            "source": repo_path(SUCCESS_SOURCE),
            "source_sha256": sha256_file(SUCCESS_SOURCE),
            "definition": "tongs lifted to task threshold and >=3 pinches, sustained for 30 control steps",
            "episode_horizon": 1000,
        },
        "rand_obj": {
            "config": repo_path(RAND_OBJ_CONFIG),
            "config_sha256": sha256_file(RAND_OBJ_CONFIG),
            "rand_full": False,
            "randomize_dynamics": False,
        },
        "replan_ratio": 0.8,
        "state_action_contract": {
            "policy_state": [23],
            "policy_action_chunk": [30, 22],
            "environment_action_after_rotvec_to_quaternion": [23],
            "pad_state_dim46": False,
            "camera_mapping": {"base": "front", "wrist": "wrist"},
            "prompt_from_frozen_rand_obj_config": True,
        },
        "statistics": {
            "source": repo_path(STATISTICS_SOURCE),
            "source_sha256": sha256_file(STATISTICS_SOURCE),
            "wilson_95ci": True,
            "paired_bootstrap_resamples": 100000,
            "paired_bootstrap_seed": 4301,
            "exact_mcnemar_two_sided": True,
            "primary_contrasts": ["B1-B0", "B2-B1", "B2-B0"],
            "material_improvement": "delta >= +0.10 and paired CI lower > 0",
            "material_hurt": "delta <= -0.10 and paired CI upper < 0",
        },
        "protocol_freeze": repo_path(protocol_path),
        "protocol_freeze_sha256": sha256_file(protocol_path),
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUTPUT)
    print(json.dumps({"status": payload["status"], "checkpoint_sha256": checkpoint_hashes}, sort_keys=True))
    if payload["status"] != "PASS":
        failed = [name for name, value in gates.items() if not value]
        raise SystemExit("S4_3_PI1D_PRE_FREEZE_FAIL: " + ",".join(failed))


if __name__ == "__main__":
    main()
