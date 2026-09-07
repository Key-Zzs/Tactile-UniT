#!/usr/bin/env python3
"""Run the final immutable-source and S4.2 integrity audit for S4.3-PI0."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_pi0"
BEFORE = ARTIFACT_ROOT / "s4_2_immutability.json"
AFTER = ARTIFACT_ROOT / "s4_2_immutability_after.json"
OUTPUT = ARTIFACT_ROOT / "final_integrity.json"
AUDIT_SOURCE = ROOT / "scripts/simulation/audit_s4_3_pi0.py"
EXPECTED_BRANCH = "develop/sim-benchmark"
EXPECTED_START_HEAD = "b3275f2f0ca1579d45c8eca80834813e5ee6d852"
EXPECTED_DEXJOCO = "8d23b0fab23b17a58c4b55f3942e17013aaf8267"


def command(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_audit_module():
    spec = importlib.util.spec_from_file_location("s4_3_pi0_audit", AUDIT_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load PI0 audit module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    before = json.loads(BEFORE.read_text(encoding="utf-8"))
    current = load_audit_module().s4_2_audit()
    current["stage"] = "PI0-12"
    current["mode"] = "after"
    current["matches_pi0_before"] = {
        "checkpoints": current["checkpoints"] == before["checkpoints"],
        "tracked_configs": current["tracked_configs"] == before["tracked_configs"],
        "vision_identity": current["Vision_identity_artifact_sha256"] == before["Vision_identity_artifact_sha256"],
        "vision_checkpoint_files": current["Vision_checkpoint_files"] == before["Vision_checkpoint_files"],
    }
    current["status"] = (
        "PASS" if current["status"] == "PASS" and all(current["matches_pi0_before"].values()) else "FAIL"
    )
    write_json(AFTER, current)

    dexjoco = ROOT / "third_party/dexjoco"
    branch = command("git", "branch", "--show-current")
    current_head = command("git", "rev-parse", "HEAD")
    dexjoco_head = command("git", "rev-parse", "HEAD", cwd=dexjoco)
    dexjoco_clean = not command("git", "status", "--porcelain", cwd=dexjoco)
    changed_tracked = command("git", "diff", "--name-only").splitlines()
    changed_s4_2 = [path for path in changed_tracked if path.startswith("configs/simulation/s4_2")]
    gates = {
        "branch_unchanged": branch == EXPECTED_BRANCH,
        "history_descends_from_start": subprocess.run(
            ["git", "merge-base", "--is-ancestor", EXPECTED_START_HEAD, current_head],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0,
        "dexjoco_revision_frozen": dexjoco_head == EXPECTED_DEXJOCO,
        "dexjoco_clean": dexjoco_clean,
        "s4_2_byte_identical": current["status"] == "PASS",
        "no_changed_s4_2_tracked_config": not changed_s4_2,
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi0-final-integrity.v1",
        "stage": "PI0-12",
        "branch": branch,
        "starting_head": EXPECTED_START_HEAD,
        "current_head": current_head,
        "dexjoco_revision": dexjoco_head,
        "dexjoco_clean": dexjoco_clean,
        "changed_s4_2_tracked_configs": changed_s4_2,
        "s4_2_after_artifact": "$ARTIFACT_ROOT/s4_2_immutability_after.json",
        "gates": {name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
        "status": "PASS" if all(gates.values()) else "FAIL",
    }
    write_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit("S4_3_PI0_FINAL_INTEGRITY_FAIL")


if __name__ == "__main__":
    main()
