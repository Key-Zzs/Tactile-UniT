#!/usr/bin/env python3
"""Audit Track A inputs without importing or reading Track B research state."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.pi2b_policy.contract import (
    BASE_SHA,
    MODEL_ORDER,
    SEED42_CHECKPOINT_HASHES,
    SIDECAR_HASHES,
    Workspace,
)
from gr00t.simulation.pi2b_policy.integrity import file_manifest, sha256_file, tree_sha256


ARTIFACT_RELATIVE = Path(".local/artifacts/simulation/s4_3_pi2b_policy")
CORE_SOURCE_HASHES = {
    "gr00t/simulation/pi05_tactile_unit.py": "d5ab1affbfbc16f708e9bcdf74d5dadac8c3730abb61f590de356e1292a90632",
    "gr00t/simulation/s4_3_pi1.py": "11ace9ee717e859d6dd3a18691a9ef087e97cb4a47ad2d42fc43f4ef381017dc",
}
OPENPI_SOURCE_HASHES = {
    "scripts/train.py": "49fa5a3c0cdf29102bf177cf373753dd3014889684c7384f0ab7ab0abf8ec7ed",
    "src/openpi/training/config.py": "9e681fcd5fef6c75a46977b3064826033110be2e514fe4d78776408be16cd592",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def verify_dataset(workspace: Workspace) -> dict[str, Any]:
    manifest_path = workspace.history_artifacts / "simulation/s4_3_pi0/official_dataset_manifest.json"
    expected = json.loads(manifest_path.read_text())
    rows = []
    for reference in expected["files"]:
        path = workspace.dataset_root / reference["path"]
        actual = sha256_file(path)
        rows.append(
            {
                "path": reference["path"],
                "bytes": path.stat().st_size,
                "sha256": actual,
                "expected_sha256": reference["sha256"],
                "match": actual == reference["sha256"] and path.stat().st_size == reference["bytes"],
            }
        )
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "files": rows,
        "episodes": expected["metadata"]["episodes"],
        "frames": expected["metadata"]["frames"],
        "state_dim": expected["metadata"]["state_dimensions"],
        "action_dim": expected["metadata"]["action_dimensions"],
        "status": "PASS" if all(row["match"] for row in rows) else "FAIL",
    }


def verify_sidecars(workspace: Workspace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, path in (("contact", workspace.contact_sidecar), ("va27", workspace.va27_sidecar)):
        digest = sha256_file(path)
        with np.load(path, allow_pickle=False) as source:
            index = np.asarray(source["index"])
            if name == "contact":
                valid = np.asarray(source["physical_aux_valid"], dtype=np.bool_)
                fields_ok = {"index", "contact_state", "contact_shared_target", "physical_aux_valid"}.issubset(
                    source.files
                )
            else:
                valid = np.asarray(source["va_aux_valid"], dtype=np.bool_)
                fields_ok = set(source.files) == {"index", "va_shared_target", "va_aux_valid"}
            result[name] = {
                "path": str(path),
                "sha256": digest,
                "expected_sha256": SIDECAR_HASHES[name],
                "rows": int(len(index)),
                "index_exact": bool(np.array_equal(index, np.arange(40065))),
                "valid_rows": int(valid.sum()),
                "invalid_rows": int((~valid).sum()),
                "fields_ok": fields_ok,
                "status": "PASS"
                if digest == SIDECAR_HASHES[name]
                and len(index) == 40065
                and np.array_equal(index, np.arange(40065))
                and int(valid.sum()) == 37365
                and int((~valid).sum()) == 2700
                and fields_ok
                else "FAIL",
            }
    with np.load(workspace.contact_sidecar, allow_pickle=False) as contact, np.load(
        workspace.va27_sidecar, allow_pickle=False
    ) as va:
        result["index_and_mask_equal"] = bool(
            np.array_equal(contact["index"], va["index"])
            and np.array_equal(contact["physical_aux_valid"], va["va_aux_valid"])
        )
    return result


def verify_base(workspace: Workspace) -> dict[str, Any]:
    root = workspace.base_params.parent
    reference_path = workspace.history_artifacts / "simulation/s4_3_pi0/official_base_model_manifest.json"
    reference = json.loads(reference_path.read_text())
    rows = []
    for expected in reference["files"]:
        path = root / expected["path"]
        actual = sha256_file(path)
        rows.append(
            {
                "path": expected["path"],
                "bytes": path.stat().st_size,
                "sha256": actual,
                "expected_sha256": expected["sha256"],
                "match": actual == expected["sha256"] and path.stat().st_size == expected["bytes"],
            }
        )
    return {
        "root": str(root),
        "files": rows,
        "file_count": len(rows),
        "status": "PASS" if all(row["match"] for row in rows) else "FAIL",
    }


def verify_seed42(workspace: Workspace) -> dict[str, Any]:
    result = {}
    for model_id in MODEL_ORDER:
        path = workspace.checkpoint_seed42(model_id)
        rows = file_manifest(path)
        digest = tree_sha256(rows)
        result[model_id] = {
            "path": str(path),
            "files": len(rows),
            "bytes": sum(int(row["bytes"]) for row in rows),
            "tree_sha256": digest,
            "expected_tree_sha256": SEED42_CHECKPOINT_HASHES[model_id],
            "status": "PASS" if digest == SEED42_CHECKPOINT_HASHES[model_id] else "FAIL",
        }
    return result


def audit() -> dict[str, Any]:
    workspace = Workspace.load(ROOT)
    artifact_root = ROOT / ARTIFACT_RELATIVE
    starting = artifact_root / "starting_integrity.json"
    if starting.exists():
        raise SystemExit("refusing to overwrite existing starting_integrity.json")
    tracked_local = git("ls-files", ".local").splitlines()
    status = git("status", "--short").splitlines()
    source_hashes = {
        relative: sha256_file(ROOT / relative) for relative in CORE_SOURCE_HASHES
    }
    openpi_hashes = {
        relative: sha256_file(workspace.openpi_root / relative) for relative in OPENPI_SOURCE_HASHES
    }
    coordination = json.loads(workspace.coordination_contract.read_text())
    dexjoco = workspace.main_root / "third_party/dexjoco"
    payload: dict[str, Any] = {
        "schema": "tactile3d-unit.s4-3-pi2b-policy-starting-integrity.v1",
        "created_at_utc": now(),
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
        "base_sha": BASE_SHA,
        "git_status_short_before_track_changes": status,
        "tracked_local_files": tracked_local,
        "workspace": {
            "root": str(workspace.root),
            "main_root": str(workspace.main_root),
            "common_git_dir": str(workspace.common_git_dir),
            "nas_experiment_root": str(workspace.nas_experiment_root),
            "write_root": str(workspace.write_root),
        },
        "coordination_contract": {
            "path": str(workspace.coordination_contract),
            "sha256": sha256_file(workspace.coordination_contract),
            "version": coordination["version"],
            "max_aggregate_gpus": coordination["rules"]["max_aggregate_gpus"],
            "heavy_jobs_runtime_barrier": coordination["rules"]["heavy_jobs_runtime_barrier"],
            "formal_policy_eval_runtime_barrier": coordination["rules"]["formal_policy_eval_runtime_barrier"],
            "policy_reads_new_teacher_weights": coordination["rules"]["policy_reads_new_teacher_weights"],
        },
        "source": {
            "core_hashes": source_hashes,
            "core_expected_hashes": CORE_SOURCE_HASHES,
            "openpi_root": str(workspace.openpi_root),
            "openpi_hashes": openpi_hashes,
            "openpi_expected_hashes": OPENPI_SOURCE_HASHES,
            "dexjoco_commit": git("rev-parse", "HEAD", cwd=dexjoco),
            "gr00t_import_root": str(ROOT),
            "teacher_worktree_imported": False,
        },
        "dataset": verify_dataset(workspace),
        "sidecars": verify_sidecars(workspace),
        "base": verify_base(workspace),
        "seed42_checkpoints": verify_seed42(workspace),
        "protected_teacher": {
            "E_T_expected_sha256": "3d4519195e7a0d9f5af63399840a48121d5f614ea587c80b9461fc696a372a19",
            "new_teacher_read": False,
        },
    }
    gates = {
        "branch_exact": payload["branch"] == "develop/pi2b-policy",
        "base_is_ancestor": subprocess.run(
            ["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"], cwd=ROOT, check=False
        ).returncode
        == 0,
        "tracked_local_none": not tracked_local,
        "coordination_contract_exact": payload["coordination_contract"]["sha256"]
        == "45cb51c079401873c92434b8d874469f2be45a189ecf0b5a76e103b3672f8bc7",
        "coordination_disallows_new_teacher": coordination["rules"]["policy_reads_new_teacher_weights"]
        is False,
        "core_source_exact": source_hashes == CORE_SOURCE_HASHES,
        "openpi_source_exact": openpi_hashes == OPENPI_SOURCE_HASHES,
        "dexjoco_exact": payload["source"]["dexjoco_commit"]
        == "8d23b0fab23b17a58c4b55f3942e17013aaf8267",
        "dataset_exact": payload["dataset"]["status"] == "PASS",
        "sidecars_exact": all(payload["sidecars"][key]["status"] == "PASS" for key in ("contact", "va27")),
        "sidecar_identity_equal": payload["sidecars"]["index_and_mask_equal"],
        "official_base_exact": payload["base"]["status"] == "PASS",
        "seed42_5_of_5_exact": all(
            value["status"] == "PASS" for value in payload["seed42_checkpoints"].values()
        ),
        "new_teacher_not_read": payload["protected_teacher"]["new_teacher_read"] is False,
    }
    payload["gates"] = {name: "PASS" if value else "FAIL" for name, value in gates.items()}
    payload["status"] = "PASS" if all(gates.values()) else "FAIL"
    atomic_json(starting, payload)
    protected = {
        "schema": "tactile3d-unit.s4-3-pi2b-policy-protected-hashes.v1",
        "created_at_utc": now(),
        "status": payload["status"],
        "seed42_checkpoint_tree_sha256": {
            key: value["tree_sha256"] for key, value in payload["seed42_checkpoints"].items()
        },
        "sidecar_sha256": {
            key: payload["sidecars"][key]["sha256"] for key in ("contact", "va27")
        },
        "base_files": {row["path"]: row["sha256"] for row in payload["base"]["files"]},
        "core_source_sha256": source_hashes,
        "openpi_source_sha256": openpi_hashes,
        "teacher_track_inputs_read": False,
    }
    atomic_json(artifact_root / "protected_hashes_before.json", protected)
    print(json.dumps({"status": payload["status"], "gates": payload["gates"]}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    audit()
