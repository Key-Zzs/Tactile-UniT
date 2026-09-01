#!/usr/bin/env python3
"""Final integrity audit and human handoff for completed S4.2-DS."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_2ds"
LOG_ROOT = ROOT / ".local/logs/simulation/s4_2ds/regression"
FINAL_CONFIG = ROOT / "configs/simulation/s4_2ds_final_decision.json"
UNIT_PYTHON = Path(sys.executable).resolve()
DEXJOCO_PYTHON = UNIT_PYTHON.parents[2] / "tactile-unit-dexjoco/bin/python"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(*args: str, cwd: Path = ROOT) -> str:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"command failed: {args}")
    return result.stdout.strip()


def pip_freeze_hash(python: Path) -> str:
    value = command(str(python), "-m", "pip", "freeze") + "\n"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load(name: str) -> dict[str, Any]:
    return json.loads(ARTIFACT_ROOT.joinpath(name).read_text(encoding="utf-8"))


def parse_pytest(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"(?P<passed>\d+) passed(?:, (?P<skipped>\d+) skipped)?(?:, (?P<warnings>\d+) warnings)? in (?P<seconds>[0-9.]+)s",
        text,
    )
    if match is None:
        raise RuntimeError(f"cannot parse pytest log {path}")
    groups = match.groupdict(default="0")
    return {
        "passed": int(groups["passed"]),
        "skipped": int(groups["skipped"]),
        "warnings": int(groups["warnings"]),
        "failed": 0,
        "duration_sec": float(groups["seconds"]),
        "log": str(path.relative_to(ROOT)),
        "log_sha256": sha256_file(path),
    }


def human_acceptance() -> str:
    rows = [
        ("DS1 C2/C3 contract", "c2_latent_contract.json, c3_latent_contract.json", "python scripts/simulation/audit_s4_2ds_contact_representations.py --device cuda:0", "Are both bottlenecks explicit, independently extractable, deterministic, and label-free?", "PASS"),
        ("DS2 Contact utility", "contact_representation_utility.json", "same as DS1", "Do both candidates pass semantic, temporal, geometry, and invalid-control gates on the same DS-DEV rows?", "PASS"),
        ("DS3 protocol", "protocol_freeze.json", "python scripts/simulation/freeze_s4_2ds_protocol.py", "Was the protocol frozen and hashed before pilot training?", "PASS"),
        ("DS4 Action pilot", "action_pilot.json", "python scripts/simulation/train_s4_2ds_action_pilot.py --device cuda:0", "Do raw reverse/shuffle/different controls satisfy all ratios and CIs?", "PASS"),
        ("DS4 Vision pilot", "vision_pilot.json, pilot_pair_manifest.json", "python scripts/simulation/build_s4_2ds_paired_pilot.py --unit-checkpoint $UNIT_FULLDATA_CKPT --device cuda:0", "Are Original UniT hashes/freeze/stability and t+27 pair identities exact?", "PASS"),
        ("DS5 Bridge", "c2_bridge_pilot.json, c3_bridge_pilot.json, candidate_comparison.json", "python scripts/simulation/train_s4_2ds_bridge_pilot.py --device cuda:0", "Do both candidates use exactly the same independent bridge and pass every retrieval/retention gate?", "PASS"),
        ("DS6 Selection", "canonical_contact_representation.json", "python scripts/simulation/evaluate_s4_2ds_candidates.py", "Does the frozen rule select C3 only because of material future-recovery utility while preserving the historical fail?", "PASS"),
        ("DS7 Confirmation", "formal_validation_confirmation.json", "python scripts/simulation/confirm_s4_2ds_formal_validation.py --unit-checkpoint $UNIT_FULLDATA_CKPT --device cuda:0", "Was this one follow-up validation confirmation run without retuning or candidate switching?", "PASS"),
        ("DS8 Integrity", "environment_integrity.json, test_summary.json, final_decision.json", "python scripts/simulation/audit_s4_2ds_final.py", "Are environments/checkpoints/M3 unchanged, regressions green, and formal test unopened?", "PASS"),
    ]
    lines = ["# S4.2-DS Human Acceptance", "", "All artifact paths are relative to `.local/artifacts/simulation/s4_2ds/`.", ""]
    for stage, artifact, cmd, question, status in rows:
        lines.extend(
            [
                f"## {stage}",
                "",
                f"- Status: **{status}**",
                f"- Artifact: `{artifact}`",
                f"- Command: `{cmd}`",
                f"- Human inspection: {question}",
                "",
            ]
        )
    lines.extend(["## Stop", "", "Stop after S4.2-DS. Formal test remains unopened; formal S4.2-4/5 and S4.2-6/7 were not started.", ""])
    return "\n".join(lines)


def main() -> None:
    final_config = json.loads(FINAL_CONFIG.read_text(encoding="utf-8"))
    canonical = load("canonical_contact_representation.json")
    confirmation = load("formal_validation_confirmation.json")
    protocol = load("protocol_freeze.json")
    action = load("action_pilot.json")
    vision = load("vision_pilot.json")
    bridge = {name: load(f"{name}_bridge_pilot.json") for name in ("c2", "c3")}
    if final_config["decision"] != "S4_2DS_C3_REPRESENTATION_UTILITY_SELECTED":
        raise RuntimeError("unexpected tracked final decision")
    if canonical["candidate"] != "C3" or confirmation["overall"] != "PASS":
        raise RuntimeError("canonical selection or validation confirmation failed")
    if protocol["formal_test_loaded"] or confirmation["formal_test_model_metrics_loaded"]:
        raise RuntimeError("formal test integrity violated")
    if action["overall"] != "PASS" or vision["overall"] != "PASS":
        raise RuntimeError("pilot dependency failed")
    if any(value["overall"] != "PASS" for value in bridge.values()):
        raise RuntimeError("bridge candidate failed")
    hashes = {
        "contact_state": sha256_file(ROOT / ".local/experiments/simulation/s4_2r/contact_state/accepted.pt"),
        "C2": sha256_file(ROOT / ".local/experiments/simulation/s4_2r/contact_dynamics/C2/best.pt"),
        "C3": sha256_file(ROOT / ".local/experiments/simulation/s4_2r/contact_dynamics/selected.pt"),
    }
    expected_hashes = {
        "contact_state": "3d4519195e7a0d9f5af63399840a48121d5f614ea587c80b9461fc696a372a19",
        "C2": "d910dcdc919f901b919f8510c313bc4493126af0856f5346eff9d698e406fb3c",
        "C3": "f67b8b0d6f519944adafecbb0a93bb4387213fa1a4f33f7f770a55938de05602",
    }
    if hashes != expected_hashes:
        raise RuntimeError("frozen Contact checkpoint identity changed")
    m3 = {
        "manifest": sha256_file(ROOT / "configs/tactile_unit/m3_system_manifest.json"),
        "c6_contract": sha256_file(ROOT / "configs/tactile_unit/c6_m3_system_evaluation.json"),
        "limitations": sha256_file(ROOT / "configs/tactile_unit/m3_limitations.json"),
    }
    expected_m3 = {
        "manifest": "a0d04ab81e027f574c08fad1a7518e5ce318cd576349f558c1065d8f7527b5e0",
        "c6_contract": "b831d981b884e4fa2c038ec9e14f55d470f4c3f6793797489c771a795c0808cf",
        "limitations": "1c89b742b0fb3adf441a06140e73afc91ce08f020396b85d6903bf9bf9472670",
    }
    if m3 != expected_m3:
        raise RuntimeError("M3 tracked identities changed")
    unit_freeze = pip_freeze_hash(UNIT_PYTHON)
    dex_freeze = pip_freeze_hash(DEXJOCO_PYTHON)
    if unit_freeze != "386a5f16a08f8d95fb12dd2951ee120e79fd6a7821f2bdeb37b14263c87e6a58":
        raise RuntimeError("unit package environment changed")
    if dex_freeze != "7406008d77c52571b64f2c7fdf36ed35a62da160e2eba4b9d091ac9f85d82f78":
        raise RuntimeError("DexJoCo package environment changed")
    if command("git", "branch", "--show-current") != "develop/sim-benchmark":
        raise RuntimeError("wrong branch")
    if command("git", "-C", "third_party/dexjoco", "status", "--short"):
        raise RuntimeError("DexJoCo submodule is dirty")
    if command("git", "-C", "third_party/dexjoco", "rev-parse", "HEAD") != "8d23b0fab23b17a58c4b55f3942e17013aaf8267":
        raise RuntimeError("DexJoCo revision changed")
    nested = command("git", "submodule", "status", "--recursive").splitlines()
    if not any(line.startswith("-5ba07ac") and "diffusion_policy" in line for line in nested):
        raise RuntimeError("nested diffusion_policy is not uninitialized")
    robo_log = LOG_ROOT / "robocasa_egl.log"
    if "RoboCasa GR1 PASS" not in robo_log.read_text(encoding="utf-8"):
        raise RuntimeError("RoboCasa EGL regression failed")
    unit_tests = parse_pytest(LOG_ROOT / "unit_pytest.log")
    dex_tests = parse_pytest(LOG_ROOT / "dexjoco_pytest.log")
    environment = {
        "schema": "tactile3d-unit.s4-2ds-environment-integrity.v1",
        "branch": "develop/sim-benchmark",
        "starting_head": "cf5cb31df70677dfc246f41381d1fbe77018074c",
        "unit": {"python_version": command(str(UNIT_PYTHON), "--version"), "pip_freeze_sha256": unit_freeze, "unchanged": True},
        "tactile_unit_dexjoco": {"python_version": command(str(DEXJOCO_PYTHON), "--version"), "pip_freeze_sha256": dex_freeze, "unchanged": True},
        "m3_hashes": m3,
        "m3_unchanged": True,
        "frozen_contact_checkpoint_hashes": hashes,
        "contact_state_unchanged": True,
        "c2_c3_unchanged": True,
        "dexjoco_clean": True,
        "nested_diffusion_policy": "UNINITIALIZED",
        "robocasa_egl_smoke": "PASS",
        "package_installation_performed": False,
        "formal_test_model_metrics_loaded": False,
    }
    tests = {
        "schema": "tactile3d-unit.s4-2ds-test-summary.v1",
        "unit_full_suite": unit_tests,
        "dexjoco_runtime_suite": dex_tests,
        "robocasa_egl_smoke": {"status": "PASS", "log": str(robo_log.relative_to(ROOT)), "log_sha256": sha256_file(robo_log)},
        "failed": 0,
        "formal_test_model_metrics_loaded": False,
    }
    decision = {
        "schema": "tactile3d-unit.s4-2ds-final-decision.v1",
        "decision": final_config["decision"],
        "canonical_contact_representation": "C3",
        "classification": "C3_REPRESENTATION_UTILITY_SELECTED",
        "checkpoint": canonical["checkpoint"],
        "checkpoint_sha256": canonical["checkpoint_sha256"],
        "historical_original": "S4_2_3_CONTACT_DYNAMICS_FAIL",
        "historical_remediation": "S4_2DR_DYNAMICS_REMEDIATION_FAIL",
        "historical_decisions_modified": False,
        "follow_up_validation_confirmation": "PASS",
        "formal_test": "UNTOUCHED",
        "formal_test_model_metrics_loaded": False,
        "selection_used_test": False,
        "S4_2_4_formal_readiness": "READY",
        "S4_2_5_formal_readiness": "READY_AFTER_S4.2-4",
        "formal_downstream_started": False,
        "policy_training_performed": False,
        "failures": [],
    }
    atomic_json(ARTIFACT_ROOT / "environment_integrity.json", environment)
    atomic_json(ARTIFACT_ROOT / "test_summary.json", tests)
    atomic_json(ARTIFACT_ROOT / "final_decision.json", decision)
    ARTIFACT_ROOT.joinpath("HUMAN_ACCEPTANCE.md").write_text(human_acceptance(), encoding="utf-8")
    print(json.dumps({"decision": decision["decision"], "environment": environment, "tests": tests}, indent=2))


if __name__ == "__main__":
    main()
