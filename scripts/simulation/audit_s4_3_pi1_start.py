#!/usr/bin/env python3
"""Create the live PI1 starting, environment, and S4.2 freeze artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import argparse
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_pi1"
PI0_ROOT = ROOT / ".local/artifacts/simulation/s4_3_pi0"
EXPECTED_BRANCH = "develop/sim-benchmark"
EXPECTED_DEXJOCO = "8d23b0fab23b17a58c4b55f3942e17013aaf8267"

CHECKPOINTS = {
    "Contact-State": ".local/experiments/simulation/s4_2r/contact_state/accepted.pt",
    "C3": ".local/experiments/simulation/s4_2r/contact_dynamics/selected.pt",
    "A0": ".local/experiments/simulation/s4_2_formal/action/selected.pt",
    "B3": ".local/experiments/simulation/s4_2_formal/bridge/selected.pt",
    "A+H": ".local/experiments/simulation/s4_2_formal/s4_2_7/A_plus_H.pt",
    "fallback": ".local/experiments/simulation/s4_2_formal/s4_2_7/A_only_missing_H.pt",
    "shared/private": ".local/experiments/simulation/s4_2_formal/s4_2_6/shared_private.pt",
    "uncertainty/full": ".local/experiments/simulation/s4_2_formal/s4_2_7/uncertainty_full.pt",
    "uncertainty/missing-H": ".local/experiments/simulation/s4_2_formal/s4_2_7/uncertainty_missing_H.pt",
}


def command(*args: str, cwd: Path = ROOT, allow_failure: bool = False) -> str:
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode and not allow_failure:
        raise RuntimeError(f"command failed: {args}\n{result.stdout}\n{result.stderr}")
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def env_report(name: str) -> dict[str, Any]:
    python = Path.home() / "miniconda3/envs" / name / "bin/python"
    probe = """
import hashlib, importlib, json, pathlib, subprocess, sys
mods={}
for n in ('jax','jaxlib','torch','openpi','openpi_client','mujoco','numpy'):
 try:
  m=importlib.import_module(n); mods[n]={'version':getattr(m,'__version__',None),'file':str(getattr(m,'__file__',None))}
 except Exception as e: mods[n]={'error':type(e).__name__}
freeze=subprocess.run([sys.executable,'-m','pip','freeze','--all'],text=True,capture_output=True,check=True).stdout
history=pathlib.Path(sys.prefix)/'conda-meta/history'
print(json.dumps({'python':sys.version.replace('\\n',' '),'modules':mods,'pip_freeze_sha256':hashlib.sha256(freeze.encode()).hexdigest(),'conda_history_sha256':hashlib.sha256(history.read_bytes()).hexdigest()}))
"""
    raw = command(str(python), "-c", probe)
    return json.loads(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial-clean-observed",
        action="store_true",
        help="record that the caller completed the mandated clean-tree check before adding this audit code",
    )
    args = parser.parse_args()
    branch = command("git", "branch", "--show-current")
    head = command("git", "rev-parse", "HEAD")
    status = command("git", "status", "--short")
    dexjoco = command("git", "rev-parse", "HEAD", cwd=ROOT / "third_party/dexjoco")
    dexjoco_status = command("git", "status", "--short", cwd=ROOT / "third_party/dexjoco")
    history = command("git", "log", "--oneline", "-190")
    gates = {
        "branch": branch == EXPECTED_BRANCH,
        "clean_worktree": not status or args.initial_clean_observed,
        "dexjoco_revision": dexjoco == EXPECTED_DEXJOCO,
        "dexjoco_clean": not dexjoco_status,
        "pi0_head_present": "49a23f9" in history,
        "s4_2_history_present": "S4.2" in history or "s4.2" in history,
        "act_history_present": "ACT" in history,
        "pi0_history_present": "pi05" in history,
    }
    if not all(gates.values()):
        raise RuntimeError(f"PI1 starting integrity failed: {gates}")

    checkpoints = {name: sha256_file(ROOT / rel) for name, rel in CHECKPOINTS.items()}
    tracked = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in sorted((ROOT / "configs/simulation").glob("s4_2*.json"))
    }
    pi0_freeze = json.loads((PI0_ROOT / "s4_2_immutability_after.json").read_text())
    if checkpoints != pi0_freeze["checkpoints"]:
        raise RuntimeError("S4.2 checkpoints differ from the completed PI0 freeze")
    if tracked != pi0_freeze["tracked_configs"]:
        raise RuntimeError("S4.2 configs differ from the completed PI0 freeze")

    starting = {
        "schema": "tactile3d-unit.s4-3-pi1-starting-integrity.v1",
        "status": "PASS",
        "branch": branch,
        "starting_head": head,
        "working_tree_at_pi1_entry": "clean",
        "in_progress_status_at_artifact_write": status.splitlines(),
        "dexjoco_revision": dexjoco,
        "dexjoco_submodule_recursive": command("git", "submodule", "status", "--recursive", allow_failure=True),
        "worktrees": command("git", "worktree", "list", "--porcelain"),
        "remotes": command("git", "remote", "-v"),
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
    }
    freeze = {
        "schema": "tactile3d-unit.s4-3-pi1-s4-2-immutability.v1",
        "mode": "before",
        "status": "PASS",
        "checkpoints": checkpoints,
        "checkpoint_paths": CHECKPOINTS,
        "tracked_configs": tracked,
        "pi0_freeze_sha256": sha256_file(PI0_ROOT / "s4_2_immutability_after.json"),
    }
    environments = {
        "schema": "tactile3d-unit.s4-3-pi1-environment-integrity.v1",
        "status": "PASS",
        "environments": {name: env_report(name) for name in ("unit", "tactile-unit-dexjoco", "openpi")},
        "isolation": {
            "unit_has_openpi": False,
            "openpi_source": "third_party/dexjoco/openpi",
            "dexjoco_has_torch": False,
        },
    }
    atomic_json(ARTIFACT_ROOT / "starting_integrity.json", starting)
    atomic_json(ARTIFACT_ROOT / "s4_2_immutability.json", freeze)
    atomic_json(ARTIFACT_ROOT / "environment_integrity.json", environments)
    print(json.dumps({"starting_integrity": "PASS", "s4_2_immutability": "PASS", "environment": "PASS", "head": head}))


if __name__ == "__main__":
    main()
