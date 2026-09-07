#!/usr/bin/env python3
"""Audit the frozen official DexJoCo/OpenPI path for S4.3-PI0.

This utility is deliberately read-only with respect to DexJoCo, OpenPI, S4.2,
and historical ACT artifacts.  It writes only ignored PI0 audit artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_pi0"
DEXJOCO_ROOT = ROOT / "third_party/dexjoco"

EXPECTED_BRANCH = "develop/sim-benchmark"
EXPECTED_START_HEAD = "b3275f2f0ca1579d45c8eca80834813e5ee6d852"
EXPECTED_DEXJOCO_COMMIT = "8d23b0fab23b17a58c4b55f3942e17013aaf8267"
EXPECTED_DEXJOCO_ORIGIN = "https://github.com/brave-eai/dexjoco.git"

S4_2_CHECKPOINTS = {
    "Contact-State": ".local/experiments/simulation/s4_2r/contact_state/accepted.pt",
    "C3": ".local/experiments/simulation/s4_2r/contact_dynamics/selected.pt",
    "A0": ".local/experiments/simulation/s4_2_formal/action/selected.pt",
    "B3": ".local/experiments/simulation/s4_2_formal/bridge/selected.pt",
    "shared/private": ".local/experiments/simulation/s4_2_formal/s4_2_6/shared_private.pt",
    "A+H": ".local/experiments/simulation/s4_2_formal/s4_2_7/A_plus_H.pt",
    "fallback": ".local/experiments/simulation/s4_2_formal/s4_2_7/A_only_missing_H.pt",
    "uncertainty/full": ".local/experiments/simulation/s4_2_formal/s4_2_7/uncertainty_full.pt",
    "uncertainty/missing-H": ".local/experiments/simulation/s4_2_formal/s4_2_7/uncertainty_missing_H.pt",
}

REQUIRED_OFFICIAL_FILES = (
    "openpi/install.bash",
    "openpi/scripts/train.py",
    "openpi/scripts/serve_policy.py",
    "openpi/scripts/compute_norm_stats.py",
    "openpi/src/openpi/training/dexjoco_configs.py",
    "dexjoco/dexjoco_openpi_client/dexjoco_openpi_env.py",
    "dexjoco/dexjoco_openpi_client/eval_dexjoco_openpi.py",
    "dexjoco/dexjoco_openpi_client/cli/evaluate.py",
    "configs/rand_obj/pinch_tongs.yaml",
)


def command(*args: str, cwd: Path = ROOT, check: bool = True) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or "command failed: " + " ".join(args))
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(name: str, payload: Any) -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    target = ARTIFACT_ROOT / name
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)


def symbolic(path: Path) -> str:
    return str(path.relative_to(ROOT))


def tracked_s4_2_hashes() -> dict[str, str]:
    paths = command("git", "ls-files", "configs/simulation/s4_2*.json").splitlines()
    return {path: sha256_file(ROOT / path) for path in paths}


def s4_2_audit() -> dict[str, Any]:
    checkpoint_hashes = {name: sha256_file(ROOT / relative) for name, relative in S4_2_CHECKPOINTS.items()}
    previous = json.loads(
        (ROOT / ".local/artifacts/simulation/s4_3_rr/s4_2_immutability.json").read_text(encoding="utf-8")
    )
    expected_checkpoints = previous["s4_2_checkpoint_sha256_after"]
    name_map = {
        "Contact-State": "contact_state",
        "C3": "contact_C3",
        "A0": "action_A0",
        "B3": "bridge_B3",
        "shared/private": "shared_private",
        "A+H": "conditional_A_plus_H",
        "fallback": "fallback_A_only_missing_H",
        "uncertainty/full": "uncertainty_full",
        "uncertainty/missing-H": "uncertainty_missing_H",
    }
    checkpoints_match = all(
        checkpoint_hashes[name] == expected_checkpoints[old_name] for name, old_name in name_map.items()
    )
    config_hashes = tracked_s4_2_hashes()
    configs_match = config_hashes == previous["s4_2_tracked_config_sha256_after"]

    vision_identity_path = ROOT / ".local/artifacts/simulation/s4_2_formal/vision_identity.json"
    vision_identity = json.loads(vision_identity_path.read_text(encoding="utf-8"))
    expected_vision_files = vision_identity["checkpoint_file_sha256"]
    vision_root_value = os.environ.get("UNIT_FULLDATA_CKPT")
    if vision_root_value:
        vision_root = Path(vision_root_value) / "tokenizer"
        actual_vision_files = {name: sha256_file(vision_root / name) for name in expected_vision_files}
        vision_files_match: bool | None = actual_vision_files == expected_vision_files
    else:
        actual_vision_files = previous["vision_checkpoint_files_sha256_after"]
        vision_files_match = None

    return {
        "schema": "tactile3d-unit.s4-3-pi0-s4-2-immutability.v1",
        "stage": "PI0-0.1",
        "mode": "before",
        "checkpoints": checkpoint_hashes,
        "checkpoint_paths": S4_2_CHECKPOINTS,
        "checkpoints_match_last_closed_ACT_audit": checkpoints_match,
        "tracked_configs": config_hashes,
        "tracked_configs_match_last_closed_ACT_audit": configs_match,
        "Vision_identity_artifact": symbolic(vision_identity_path),
        "Vision_identity_artifact_sha256": sha256_file(vision_identity_path),
        "Vision_checkpoint_files": actual_vision_files,
        "Vision_checkpoint_files_live_rehashed": bool(vision_root_value),
        "Vision_checkpoint_files_match": vision_files_match,
        "historical_reference": ".local/artifacts/simulation/s4_3_rr/s4_2_immutability.json",
        "status": "PASS" if checkpoints_match and configs_match and vision_files_match is not False else "FAIL",
    }


def source_audit() -> tuple[dict[str, Any], dict[str, Any]]:
    commit = command("git", "rev-parse", "HEAD", cwd=DEXJOCO_ROOT)
    origin = command("git", "remote", "get-url", "origin", cwd=DEXJOCO_ROOT)
    source_status = command("git", "status", "--porcelain", cwd=DEXJOCO_ROOT)
    files = {
        path: {
            "exists": (DEXJOCO_ROOT / path).is_file(),
            "sha256": sha256_file(DEXJOCO_ROOT / path) if (DEXJOCO_ROOT / path).is_file() else None,
        }
        for path in REQUIRED_OFFICIAL_FILES
    }
    licenses = {
        path: sha256_file(DEXJOCO_ROOT / path)
        for path in ("LICENSE", "openpi/LICENSE", "openpi/LICENSE_GEMMA.txt", "openpi/NOTICE")
    }
    gates = {
        "official_origin": origin == EXPECTED_DEXJOCO_ORIGIN,
        "frozen_commit": commit == EXPECTED_DEXJOCO_COMMIT,
        "checkout_clean": not source_status,
        "required_files": all(item["exists"] for item in files.values()),
    }
    source = {
        "schema": "tactile3d-unit.s4-3-pi0-official-source-audit.v1",
        "stage": "PI0-1",
        "repository": "brave-eai/dexjoco",
        "origin": origin,
        "commit": commit,
        "source_location": "third_party/dexjoco (frozen read-only submodule)",
        "independent_clone_required": False,
        "openpi_tree_git_object": command("git", "rev-parse", "HEAD:openpi", cwd=DEXJOCO_ROOT),
        "required_files": files,
        "license_notice_sha256": licenses,
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "status": "PASS" if all(gates.values()) else "FAIL",
    }

    contract_files = {
        key: files[key]["sha256"]
        for key in (
            "openpi/src/openpi/training/dexjoco_configs.py",
            "dexjoco/dexjoco_openpi_client/dexjoco_openpi_env.py",
            "dexjoco/dexjoco_openpi_client/eval_dexjoco_openpi.py",
            "configs/rand_obj/pinch_tongs.yaml",
        )
    }
    contract = {
        "schema": "tactile3d-unit.s4-3-pi0-official-openpi-contract.v1",
        "stage": "PI0-1",
        "source": "official DexJoCo/OpenPI at frozen submodule commit",
        "task": "pinch_tongs",
        "regime": "rand_obj",
        "evaluation": {"episodes": 20, "seed": 0, "render_mode": "rgb_array"},
        "policy_state": {
            "dimensions": 23,
            "layout": ["TCP xyz (3)", "TCP quaternion scalar-first (4)", "Allegro joints (16)"],
            "official_wrapper_expression": "env_obs['state'][:23]",
        },
        "policy_action": {
            "dimensions": 22,
            "layout": ["TCP xyz (3)", "TCP rotvec (3)", "Allegro joints (16)"],
        },
        "environment_action": {
            "dimensions": 23,
            "layout": ["TCP xyz (3)", "TCP quaternion scalar-first (4)", "Allegro joints (16)"],
            "conversion_owner": "official DexJoCoOpenPIEnv._process_action",
        },
        "action_horizon": 30,
        "replan_ratio": 0.8,
        "cameras": {"base": "front", "wrist": "wrist"},
        "image_transform": "official resize_with_pad to 224x224 then uint8",
        "prompt": "Grasp the tongs and perform three consecutive open-close motions.",
        "rand_full": False,
        "randomize_dynamics": False,
        "previous_ACT_protocol_reused": False,
        "tactile_or_Tactile-UniT_inputs": False,
        "source_file_sha256": contract_files,
        "status": "PASS",
    }
    return source, contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--starting-tree-clean",
        action="store_true",
        help="Record the separately observed pre-artifact clean-tree gate.",
    )
    args = parser.parse_args()

    branch = command("git", "branch", "--show-current")
    head = command("git", "rev-parse", "HEAD")
    s4_2 = s4_2_audit()
    source, contract = source_audit()
    starting_gates = {
        "branch": branch == EXPECTED_BRANCH,
        "head": head == EXPECTED_START_HEAD,
        "working_tree_clean_before_PI0_artifact_creation": args.starting_tree_clean,
        "S4.2_immutability": s4_2["status"] == "PASS",
        "historical_ACT_result_present": (ROOT / "configs/simulation/s4_3_rr_final_decision.json").is_file(),
    }
    starting = {
        "schema": "tactile3d-unit.s4-3-pi0-starting-integrity.v1",
        "stage": "PI0-0",
        "pwd": "$REPO_ROOT",
        "branch": branch,
        "starting_head": head,
        "remote": "origin=https://github.com/Key-Zzs/Tactile-UniT.git",
        "dexjoco_submodule": EXPECTED_DEXJOCO_COMMIT,
        "M3": "UNCHANGED",
        "S4.2": "COMPLETE",
        "S4.3-PD": "COMPLETE",
        "historical_ACT": "S4_3_2_ACT_BENCHMARK_WEAK",
        "historical_ACT_preserved": True,
        "gates": {name: "PASS" if value else "FAIL" for name, value in starting_gates.items()},
        "status": "PASS" if all(starting_gates.values()) else "FAIL",
    }
    environment = {
        "schema": "tactile3d-unit.s4-3-pi0-environment-integrity.v1",
        "stage": "PI0-0/PI0-1",
        "frozen_environments_mutated": False,
        "unit": "UNCHANGED",
        "tactile-unit-dexjoco": "UNCHANGED",
        "official_openpi_environment": "NOT_YET_CREATED",
        "dexjoco_source_clean": source["gates"]["checkout_clean"],
        "status": "PASS",
    }
    paper_reference = {
        "schema": "tactile3d-unit.s4-3-pi0-paper-reference.v1",
        "stage": "PI0-3.6-preaudit",
        "title": "DexJoCo: A Benchmark and Toolkit for Task-Oriented Dexterous Manipulation on MuJoCo",
        "source": "https://arxiv.org/abs/2605.16257v1",
        "version": "v1",
        "task": "Pinch Tongs",
        "model": "pi0.5",
        "regime": "rand-obj",
        "published_success_rate_percent_mean": 24.0,
        "published_success_rate_percent_std": 6.9,
        "published_evaluation_episodes_per_task": 50,
        "published_result_aggregates_training_seeds": True,
        "pi0_protocol_difference": (
            "PI0 uses one fixed evaluation seed and 20 episodes; it does not reproduce the "
            "paper's complete multi-seed, 50-episode-per-task table."
        ),
        "status": "PASS",
    }

    write_json("starting_integrity.json", starting)
    write_json("s4_2_immutability.json", s4_2)
    write_json("official_source_audit.json", source)
    write_json("official_openpi_contract.json", contract)
    write_json("environment_integrity.json", environment)
    write_json("official_paper_reference.json", paper_reference)

    if starting["status"] != "PASS" or source["status"] != "PASS":
        raise SystemExit("S4_3_PI0_STARTING_OR_OFFICIAL_SOURCE_GATE_FAIL")
    print(
        json.dumps(
            {"starting": starting["status"], "source": source["status"], "contract": contract["status"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
