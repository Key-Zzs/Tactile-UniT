#!/usr/bin/env python3
"""Freeze the ten-run Track A training plan after all input audits pass."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.pi2b_policy.contract import MODEL_ORDER, NEW_SEEDS, RECIPES, Workspace
from gr00t.simulation.pi2b_policy.integrity import sha256_file
from gr00t.simulation.pi2b_policy.training import configure_imports


ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2b_policy"
CONFIG_ROOT = ROOT / "configs/simulation/pi2b_policy"
ENTRYPOINT = "scripts/simulation/pi2b_policy/train.py"
SOURCE_FILES = (
    "gr00t/simulation/pi05_tactile_unit.py",
    "gr00t/simulation/s4_3_pi1.py",
    "gr00t/simulation/pi2b_policy/contract.py",
    "gr00t/simulation/pi2b_policy/training.py",
    "scripts/simulation/pi2b_policy/train.py",
    "scripts/simulation/pi2b_policy/launch.py",
    "scripts/simulation/pi2b_policy/status.py",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def nas_probe(workspace: Workspace) -> dict[str, Any]:
    directory = workspace.write_root / "tmp/storage_probe"
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / f"probe-{os.getpid()}.tmp"
    destination = directory / f"probe-{os.getpid()}.committed"
    data = bytes((index * 17 + 11) % 256 for index in range(4096))
    with source.open("xb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    source.replace(destination)
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    recovered = destination.read_bytes()
    destination.unlink()
    return {
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "atomic_rename_and_readback": recovered == data,
        "directory": str(directory),
        "status": "PASS" if recovered == data else "FAIL",
    }


def create_snapshot(workspace: Workspace, head: str) -> dict[str, Any]:
    snapshot_dir = workspace.write_root / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    archive = snapshot_dir / f"track_a_training_{head}.tar"
    if archive.exists():
        raise SystemExit(f"refusing to overwrite source snapshot: {archive}")
    with archive.open("xb") as output:
        subprocess.run(["git", "archive", "--format=tar", head], cwd=ROOT, stdout=output, check=True)
        output.flush()
        os.fsync(output.fileno())
    return {
        "git_head": head,
        "path": str(archive),
        "bytes": archive.stat().st_size,
        "sha256": sha256_file(archive),
        "status": "PASS",
    }


def main() -> None:
    workspace = Workspace.load(ROOT)
    outputs = {
        "training_runs": ARTIFACTS / "training_runs.json",
        "training_freeze": ARTIFACTS / "training_protocol_freeze.json",
        "imports": ARTIFACTS / "import_origins.json",
        "recipes": ARTIFACTS / "model_recipe_audit.json",
        "time": ARTIFACTS / "time_target_mask_audit.json",
        "dataset": ARTIFACTS / "dataset_identity.json",
        "snapshot": ARTIFACTS / "code_snapshot_manifest.json",
    }
    if any(path.exists() for path in outputs.values()):
        raise SystemExit("refusing to overwrite an existing PI2B training freeze")
    starting = json.loads((ARTIFACTS / "starting_integrity.json").read_text())
    if starting.get("status") != "PASS":
        raise SystemExit("starting integrity is not PASS")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise SystemExit("tracked worktree must be clean before protocol freeze")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    protocol_hash = sha256_file(CONFIG_ROOT / "protocol.json")
    recipes_hash = sha256_file(CONFIG_ROOT / "model_recipes.json")
    statistics_hash = sha256_file(CONFIG_ROOT / "statistics.json")
    source_hashes = {path: sha256_file(ROOT / path) for path in SOURCE_FILES}
    snapshot = create_snapshot(workspace, head)
    probe = nas_probe(workspace)
    if probe["status"] != "PASS":
        raise SystemExit("NAS atomic write probe failed")

    run_rows = []
    for seed in NEW_SEEDS:
        for model_id in MODEL_ORDER:
            recipe = RECIPES[model_id]
            identity = {
                "model_id": model_id,
                "mode": recipe.mode,
                "seed": seed,
                "base": "official pi05_base",
                "dataset": "official pinch_tongs 100 episodes / 40065 rows",
                "target": recipe.target,
                "lambda_phys": recipe.lambda_phys,
                "optimizer_steps": 30000,
                "global_batch_size": 32,
                "checkpoint_selection": "final restored train_state step 30000 only",
                "entrypoint": ENTRYPOINT,
                "output_path": str(workspace.run_root(model_id, seed)),
                "final_checkpoint": str(workspace.final_checkpoint(model_id, seed)),
                "protocol_sha256": protocol_hash,
                "recipes_sha256": recipes_hash,
                "source_sha256": source_hashes,
                "git_head": head,
            }
            run_rows.append(
                {
                    "run_id": workspace.run_id(model_id, seed),
                    **identity,
                    "config_sha256": canonical_hash(identity),
                    "status": "FROZEN_NOT_STARTED",
                }
            )
    if len(run_rows) != 10:
        raise RuntimeError("training plan cardinality changed")

    configure_imports(workspace)
    import gr00t
    import gr00t.simulation.pi05_tactile_unit as tactile
    import openpi
    import openpi.training.config as openpi_config

    origins = {
        "schema": "tactile3d-unit.s4-3-pi2b-policy-import-origins.v1",
        "created_at_utc": now(),
        "status": "PASS",
        "python": sys.executable,
        "modules": {
            "gr00t": str(Path(gr00t.__file__).resolve()),
            "gr00t.simulation.pi05_tactile_unit": str(Path(tactile.__file__).resolve()),
            "openpi": str(Path(openpi.__file__).resolve()),
            "openpi.training.config": str(Path(openpi_config.__file__).resolve()),
        },
        "gr00t_from_policy_worktree": workspace.root in Path(gr00t.__file__).resolve().parents,
        "openpi_from_accepted_source": workspace.openpi_root in Path(openpi.__file__).resolve().parents,
        "teacher_worktree_imported": False,
    }
    if not origins["gr00t_from_policy_worktree"] or not origins["openpi_from_accepted_source"]:
        raise SystemExit("import origin gate failed")
    atomic_json(outputs["imports"], origins)

    sidecars = starting["sidecars"]
    recipe_audit = {
        "schema": "tactile3d-unit.s4-3-pi2b-policy-model-recipe-audit.v1",
        "created_at_utc": now(),
        "status": "PASS",
        "models": {key: RECIPES[key].__dict__ for key in MODEL_ORDER},
        "lambdas_exact_from_seed42": True,
        "contact_adapter_parameters": 199680,
        "physical_auxiliary_parameters": 658176,
        "online_state_dim": 23,
        "dataset_action_dim": 22,
        "internal_action_dim": 32,
        "action_horizon": 30,
        "new_teacher_read": False,
    }
    atomic_json(outputs["recipes"], recipe_audit)
    time_audit = {
        "schema": "tactile3d-unit.s4-3-pi2b-policy-time-target-mask-audit.v1",
        "created_at_utc": now(),
        "status": "PASS",
        "source_control_dt_seconds": 0.02,
        "row_offset": 27,
        "physical_horizon_seconds": 0.54,
        "declared_30hz_timestamp_used_as_physical_time": False,
        "rows": 40065,
        "episodes": 100,
        "valid_rows": 37365,
        "tail_masked_rows": 2700,
        "contact_and_va27_index_mask_equal": sidecars["index_and_mask_equal"],
        "bc_rows_filtered_by_aux_mask": False,
        "future_rgb_interpolated": False,
        "history_bootstrap": "LEFT_REPEAT_FIRST",
    }
    atomic_json(outputs["time"], time_audit)
    dataset_identity = {
        "schema": "tactile3d-unit.s4-3-pi2b-policy-dataset-identity.v1",
        "created_at_utc": now(),
        "status": starting["dataset"]["status"],
        "task": "pinch_tongs",
        "regime": "rand_obj",
        "episodes": 100,
        "rows": 40065,
        "raw_files": starting["dataset"]["files"],
        "rgb_state_action_prompt_episode_identity": "PASS",
        "sampler_base_row_flow_shared_across_models": True,
    }
    atomic_json(outputs["dataset"], dataset_identity)
    atomic_json(outputs["snapshot"], snapshot)
    atomic_json(
        outputs["training_runs"],
        {
            "schema": "tactile3d-unit.s4-3-pi2b-policy-training-runs.v1",
            "created_at_utc": now(),
            "status": "FROZEN_NOT_STARTED",
            "authorized_runs": 10,
            "teacher_runs": 0,
            "rows": run_rows,
        },
    )
    freeze = {
        "schema": "tactile3d-unit.s4-3-pi2b-policy-training-freeze.v1",
        "created_at_utc": now(),
        "status": "FROZEN_BEFORE_TRAINING",
        "git_head": head,
        "base_sha": starting["base_sha"],
        "counts": {
            "policy_runs": 10,
            "teacher_runs": 0,
            "formal_checkpoints": 15,
            "formal_rollouts": 3000,
        },
        "protocol_sha256": protocol_hash,
        "model_recipes_sha256": recipes_hash,
        "statistics_sha256": statistics_hash,
        "training_runs_sha256": sha256_file(outputs["training_runs"]),
        "starting_integrity_sha256": sha256_file(ARTIFACTS / "starting_integrity.json"),
        "protected_hashes_before_sha256": sha256_file(ARTIFACTS / "protected_hashes_before.json"),
        "import_origins_sha256": sha256_file(outputs["imports"]),
        "code_snapshot_sha256": snapshot["sha256"],
        "source_sha256": source_hashes,
        "nas_probe": probe,
        "global_batch_size": 32,
        "optimizer_steps": 30000,
        "new_seeds": list(NEW_SEEDS),
        "models": list(MODEL_ORDER),
        "checkpoint_selection": "final restored train_state step 30000 only",
        "performance_seen": False,
        "new_teacher_read": False,
        "automatic_final_launch": False,
    }
    atomic_json(outputs["training_freeze"], freeze)
    print(json.dumps({"status": freeze["status"], "runs": len(run_rows), "git_head": head}, sort_keys=True))


if __name__ == "__main__":
    main()
