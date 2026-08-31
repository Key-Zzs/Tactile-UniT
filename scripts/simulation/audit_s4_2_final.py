#!/usr/bin/env python3
"""Audit the S4.2 hard-stop decision and write ignored final evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_2"
CONFIG = ROOT / "configs/simulation/s4_2_final_decision.json"
DOWNSTREAM = (
    "dynamics_selection.json",
    "dynamics_evaluation.json",
    "action_selection.json",
    "action_evaluation.json",
    "vision_identity.json",
    "paired_latent_manifest.json",
    "bridge_selection.json",
    "bridge_evaluation.json",
    "discrete_evaluation.json",
    "shared_private_evaluation.json",
    "conditional_sufficiency.json",
    "uncertainty_evaluation.json",
    "pretest_freeze.json",
    "locked_test_evaluation.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(*args: str, cwd: Path = ROOT) -> str:
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "command failed")
    return result.stdout.strip()


def capture_environment(python: Path, destination: Path) -> str:
    version = command(str(python), "--version")
    freeze = command(str(python), "-m", "pip", "freeze")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(f"{version}\n{freeze}\n", encoding="utf-8")
    return sha256(destination)


def normalized_package_hash(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()[1:]
    lines = [line for line in lines if not line.startswith("-e ")]
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def pip_freeze_hash(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()[1:]
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-python", type=Path, required=True)
    parser.add_argument("--dexjoco-python", type=Path, required=True)
    return parser.parse_args()


def human_acceptance() -> str:
    rows = [
        ("Dataset", "dataset_quality.json", "python scripts/simulation/audit_s4_2_dataset.py --root $DEXJOCO_DATA_ROOT/s4_2 --artifacts .local/artifacts/simulation/s4_2", "Do all integrity/leakage gates pass and are warnings explicit?", "PASS_WITH_WARNINGS"),
        ("Teacher", "teacher_evaluation.json", "python scripts/simulation/train_sim_contact_teacher.py", "Is effective rank 11.255626 below the frozen 16.0 gate?", "FAIL"),
        ("Dynamics", "dynamics_evaluation.json", "NOT RUN", "Is this artifact absent because S4.2-2 blocked it?", "NOT_RUN_DEPENDENCY_BLOCKED"),
        ("Action", "action_evaluation.json", "NOT RUN", "Is this artifact absent because S4.2-2 blocked it?", "NOT_RUN_DEPENDENCY_BLOCKED"),
        ("Vision", "vision_identity.json", "NOT RUN", "Is this artifact absent because S4.2-2 blocked it?", "NOT_RUN_DEPENDENCY_BLOCKED"),
        ("Bridge", "bridge_evaluation.json", "NOT RUN", "Is this artifact absent because S4.2-2 blocked it?", "NOT_RUN_DEPENDENCY_BLOCKED"),
        ("Bottleneck", "discrete_evaluation.json", "NOT RUN", "Is this artifact absent because S4.2-2 blocked it?", "NOT_RUN_DEPENDENCY_BLOCKED"),
        ("Shared/private", "shared_private_evaluation.json", "NOT RUN", "Is this artifact absent because S4.2-2 blocked it?", "NOT_RUN_DEPENDENCY_BLOCKED"),
        ("Conditional sufficiency", "conditional_sufficiency.json", "NOT RUN", "Is this artifact absent because S4.2-2 blocked it?", "NOT_RUN_DEPENDENCY_BLOCKED"),
        ("Uncertainty", "uncertainty_evaluation.json", "NOT RUN", "Is this artifact absent because S4.2-2 blocked it?", "NOT_RUN_DEPENDENCY_BLOCKED"),
        ("Final decision", "final_decision.json", "python scripts/simulation/audit_s4_2_final.py --unit-python $UNIT_PYTHON --dexjoco-python $DEXJOCO_PYTHON", "Does the decision remain S4_2_CONTACT_STATE_FAIL with no locked-test access?", "PASS"),
    ]
    text = ["# S4.2 Human Acceptance", "", "All paths are relative to `.local/artifacts/simulation/s4_2/`.", ""]
    for stage, artifact, cmd, question, status in rows:
        text.extend([f"## {stage}", "", f"- Artifact: `{artifact}`", f"- Exact command: `{cmd}`", f"- Inspection: {question}", f"- Status: **{status}**", ""])
    text.extend(["## Plots", "", "Run `MPLBACKEND=Agg python scripts/simulation/visualize_s4_2.py`.", "Eight pre-stop evidence plots are present. Plot numbers 7-8 and 10-21 are intentionally unavailable because their stages were not run.", ""])
    return "\n".join(text)


def main() -> None:
    args = arguments()
    decision = json.loads(CONFIG.read_text(encoding="utf-8"))
    teacher = json.loads((ARTIFACTS / "teacher_evaluation.json").read_text(encoding="utf-8"))
    selection = json.loads((ARTIFACTS / "teacher_selection.json").read_text(encoding="utf-8"))
    protocol = json.loads((ARTIFACTS / "protocol_freeze.json").read_text(encoding="utf-8"))
    assert decision["decision"] == "S4_2_CONTACT_STATE_FAIL"
    assert teacher["status"] == decision["decision"]
    assert [name for name, passed in teacher["gates"].items() if not passed] == ["no_collapse"]
    assert teacher["collapse"]["effective_rank"] < decision["contact_state"]["effective_rank_min"]
    assert teacher["test_loaded"] is False
    assert selection["test_loaded"] is False and selection["selection_uses_test"] is False
    assert protocol["test_loaded"] is False
    absent = [name for name in DOWNSTREAM if not (ARTIFACTS / name).exists()]
    if len(absent) != len(DOWNSTREAM):
        raise RuntimeError(f"unexpected dependent artifacts exist: {sorted(set(DOWNSTREAM) - set(absent))}")

    environment = ARTIFACTS / "environment"
    unit_after = capture_environment(args.unit_python, environment / "unit_after.txt")
    dex_after = capture_environment(args.dexjoco_python, environment / "tactile-unit-dexjoco_after.txt")
    unit_before = sha256(environment / "unit_before.txt")
    dex_before = sha256(environment / "tactile-unit-dexjoco_before.txt")
    m3 = {
        "manifest_sha256": sha256(ROOT / "configs/tactile_unit/m3_system_manifest.json"),
        "c6_contract_sha256": sha256(ROOT / "configs/tactile_unit/c6_m3_system_evaluation.json"),
        "limitations_sha256": sha256(ROOT / "configs/tactile_unit/m3_limitations.json"),
    }
    expected_m3 = decision["regression"]
    assert m3["manifest_sha256"] == expected_m3["m3_manifest_sha256"]
    assert m3["c6_contract_sha256"] == expected_m3["m3_c6_contract_sha256"]
    assert m3["limitations_sha256"] == expected_m3["m3_limitations_sha256"]
    unit_packages_before = normalized_package_hash(environment / "unit_before.txt")
    unit_packages_after = normalized_package_hash(environment / "unit_after.txt")
    dex_freeze_before = pip_freeze_hash(environment / "tactile-unit-dexjoco_before.txt")
    dex_freeze_after = pip_freeze_hash(environment / "tactile-unit-dexjoco_after.txt")
    assert unit_packages_before == unit_packages_after
    assert unit_packages_after == decision["environment"]["unit_normalized_package_set_sha256"]
    assert dex_before == dex_after
    assert dex_freeze_before == dex_freeze_after
    assert command("git", "branch", "--show-current") == "develop/sim-benchmark"
    assert command("git", "-C", "third_party/dexjoco", "status", "--short") == ""
    assert command("git", "-C", "third_party/dexjoco", "rev-parse", "HEAD") == "8d23b0fab23b17a58c4b55f3942e17013aaf8267"
    assert command("git", "ls-files", ".local") == ""
    nested = command("git", "submodule", "status", "--recursive").splitlines()
    assert any(line.startswith("-5ba07ac") and "diffusion_policy" in line for line in nested)
    smoke = json.loads((ARTIFACTS / "robocasa_regression/smoke_test.json").read_text())
    assert smoke["status"] == "PASS" and smoke["backend"] == "EGL"

    result = {
        "schema": "tactile3d-unit.s4-2-final-audit.v1",
        "decision": decision["decision"],
        "status": "PASS_LIMITED_TO_FAILURE_AUDIT",
        "environment": {
            "unit_identity_before": unit_before,
            "unit_identity_after": unit_after,
            "unit_raw_identity_unchanged": unit_before == unit_after,
            "unit_raw_difference": "editable gr00t VCS revision only",
            "unit_normalized_packages_before": unit_packages_before,
            "unit_normalized_packages_after": unit_packages_after,
            "unit_packages_unchanged": True,
            "dexjoco_identity_before": dex_before,
            "dexjoco_identity_after": dex_after,
            "dexjoco_identity_unchanged": True,
            "dexjoco_pip_freeze_before": dex_freeze_before,
            "dexjoco_pip_freeze_after": dex_freeze_after,
        },
        "m3": {**m3, "unchanged": True},
        "dexjoco_submodule": "CLEAN_PIN_MATCH",
        "nested_diffusion_policy": "UNINITIALIZED",
        "dependent_artifacts_absent": absent,
        "locked_test": "NOT_RUN_DEPENDENCY_BLOCKED",
        "test_loaded_for_model_evaluation": False,
        "robocasa": "PASS",
        "unit_pytest": decision["regression"]["unit_pytest"],
        "dexjoco_scoped_tests": decision["regression"]["dexjoco_scoped_tests"],
    }
    (ARTIFACTS / "final_decision.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ARTIFACTS / "HUMAN_ACCEPTANCE.md").write_text(human_acceptance(), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
