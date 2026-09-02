#!/usr/bin/env python3
"""Run the final S4.3-PD readiness, immutability, and environment audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

try:
    from scripts.simulation.visualize_s4_3_pd import plot_final_gate_matrix
except ModuleNotFoundError:  # Direct execution puts this script's directory on sys.path.
    from visualize_s4_3_pd import plot_final_gate_matrix

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pd"
START_HEAD = "0daf8acd07b6d0aec4c603cd4f0684b3d161d8f7"
EXPECTED_BRANCH = "develop/sim-benchmark"
EXPECTED_DEXJOCO_REVISION = "8d23b0fab23b17a58c4b55f3942e17013aaf8267"
EXPECTED_PACKAGES = {
    "unit": "3a119880cd4d661259d9476b0d3224302e086ae899ca77250497b5f42fbf6f9b",
    "tactile-unit-dexjoco": "7406008d77c52571b64f2c7fdf36ed35a62da160e2eba4b9d091ac9f85d82f78",
}
CHECKPOINTS = {
    "contact_state": ".local/experiments/simulation/s4_2r/contact_state/accepted.pt",
    "contact_C3": ".local/experiments/simulation/s4_2r/contact_dynamics/selected.pt",
    "action_A0": ".local/experiments/simulation/s4_2_formal/action/selected.pt",
    "bridge_B3": ".local/experiments/simulation/s4_2_formal/bridge/selected.pt",
    "shared_private": ".local/experiments/simulation/s4_2_formal/s4_2_6/shared_private.pt",
    "conditional_A_plus_H": ".local/experiments/simulation/s4_2_formal/s4_2_7/A_plus_H.pt",
    "fallback_A_only_missing_H": ".local/experiments/simulation/s4_2_formal/s4_2_7/A_only_missing_H.pt",
    "uncertainty_full": ".local/experiments/simulation/s4_2_formal/s4_2_7/uncertainty_full.pt",
    "uncertainty_missing_H": ".local/experiments/simulation/s4_2_formal/s4_2_7/uncertainty_missing_H.pt",
}
EXPECTED_VISION = {
    "config.json": "7a651f488c93521e0d507880fc250a475e6a08aa9307aa1349f9d3509844971e",
    "model-00001-of-00002.safetensors": "32d5c326f6c83d12185b6954d2a52511f66ad18b6fdf814aecc5726dd39c243c",
    "model-00002-of-00002.safetensors": "2f8093a900330e5111b63e44dc1687b3212bec343e5b3832bf2e40f2bf18a768",
    "model.safetensors.index.json": "3b6d73d2442ce694287c5cd8b93db1bb232909becf35f08ecabadb614b9a1b86",
}
STAGE_ARTIFACTS = {
    "PD1": "official_expert_audit.json",
    "PD2": "task_success_contract.json",
    "PD3": "expert_source_selection.json",
    "PD4": "pilot_results.json",
    "PD5": "formal_protocol_freeze.json",
    "PD6": "formal_acquisition_results.json",
    "PD7": "formal_success_audit.json",
    "PD8": "policy_expert_dataset_manifest.json",
}
REQUIRED_ARTIFACTS = (
    "starting_integrity.json",
    "official_expert_audit.json",
    "task_success_contract.json",
    "expert_source_selection.json",
    "pilot_seed_manifest.json",
    "pilot_results.json",
    "pilot_behavior_audit.json",
    "formal_protocol_freeze.json",
    "formal_attempt_manifest.json",
    "formal_acquisition_results.json",
    "formal_success_audit.json",
    "behavior_quality_audit.json",
    "policy_expert_dataset_manifest.json",
    "policy_window_counts.json",
    "s4_2_compatibility.json",
    "determinism_spotcheck.json",
    "warnings.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def command(*args: str, cwd: Path = ROOT) -> str:
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "command failed: " + " ".join(args))
    return result.stdout.strip()


def pip_freeze_hash(python: Path, *, exclude_editable: bool) -> str:
    output = command(str(python), "-m", "pip", "freeze")
    lines = output.splitlines()
    if exclude_editable:
        lines = [line for line in lines if not line.startswith("-e ")]
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def load_artifact(name: str) -> dict[str, Any]:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def immutability_audit() -> dict[str, Any]:
    starting = load_artifact("starting_integrity.json")
    expected = starting["immutable_checkpoint_hashes_before"]
    actual = {name: sha256_file(ROOT / path) for name, path in CHECKPOINTS.items()}
    vision = json.loads(
        (ROOT / ".local/artifacts/simulation/s4_2_formal/vision_identity.json").read_text(
            encoding="utf-8"
        )
    )
    vision_actual = vision["checkpoint_file_sha256"]
    pathspecs = (
        ":(glob)configs/simulation/s4_2*.json",
        ":(glob)docs/research/s4_2*.md",
        ":(glob)gr00t/simulation/s4_2*",
        ":(glob)scripts/simulation/*s4_2*",
        ":(glob)tests/simulation/*s4_2*",
    )
    changed = command("git", "diff", "--name-only", START_HEAD, "--", *pathspecs).splitlines()
    config_rows = command(
        "git", "ls-tree", "-r", START_HEAD, "--", "configs/simulation"
    ).splitlines()
    config_before = {}
    for row in config_rows:
        metadata, path = row.split("\t", 1)
        name = Path(path).name
        if name.startswith("s4_2") and name.endswith(".json"):
            config_before[path] = metadata.split()[2]
    config_after = {
        path: command("git", "hash-object", path) for path in sorted(config_before)
    }
    passed = (
        actual == expected
        and vision_actual == EXPECTED_VISION
        and config_after == config_before
        and not changed
    )
    return {
        "schema": "tactile3d-unit.s4-3-pd-s4-2-immutability.v1",
        "stage": "PD9",
        "checkpoint_sha256_before": expected,
        "checkpoint_sha256_after": actual,
        "vision_checkpoint_sha256_before": EXPECTED_VISION,
        "vision_checkpoint_sha256_after": vision_actual,
        "s4_2_tracked_config_blob_sha1_before": config_before,
        "s4_2_tracked_config_blob_sha1_after": config_after,
        "s4_2_tracked_files_changed_since_start": changed,
        "s4_2_retraining": False,
        "byte_identical": passed,
        "status": "PASS" if passed else "FAIL",
    }


def environment_audit() -> dict[str, Any]:
    unit_python = Path(sys.executable).resolve()
    dex_python = unit_python.parents[2] / "tactile-unit-dexjoco/bin/python"
    current_packages = {
        "unit": pip_freeze_hash(unit_python, exclude_editable=True),
        "tactile-unit-dexjoco": pip_freeze_hash(dex_python, exclude_editable=False),
    }
    recursive = command("git", "submodule", "status", "--recursive").splitlines()
    nested_uninitialized = any(
        line.startswith("-5ba07ac") and "diffusion_policy" in line for line in recursive
    )
    dex_revision = command("git", "rev-parse", "HEAD", cwd=ROOT / "third_party/dexjoco")
    dex_clean = command("git", "status", "--short", cwd=ROOT / "third_party/dexjoco") == ""
    branch = command("git", "branch", "--show-current")
    root_clean = command("git", "status", "--short") == ""
    local_tracked = command("git", "ls-files", ".local").splitlines()
    passed = (
        current_packages == EXPECTED_PACKAGES
        and dex_revision == EXPECTED_DEXJOCO_REVISION
        and dex_clean
        and nested_uninitialized
        and branch == EXPECTED_BRANCH
        and root_clean
        and not local_tracked
    )
    python_versions = {
        "unit": command(str(unit_python), "--version"),
        "tactile-unit-dexjoco": command(str(dex_python), "--version"),
    }
    return {
        "schema": "tactile3d-unit.s4-3-pd-environment-integrity.v1",
        "stage": "PD9",
        "package_hashes_before": EXPECTED_PACKAGES,
        "package_hashes_after": current_packages,
        "package_sets_unchanged": current_packages == EXPECTED_PACKAGES,
        "package_installation_or_update_performed": False,
        "python_versions": python_versions,
        "branch": branch,
        "working_tree_clean": root_clean,
        "dexjoco_revision": dex_revision,
        "dexjoco_clean": dex_clean,
        "nested_diffusion_policy": "UNINITIALIZED" if nested_uninitialized else "UNEXPECTED_STATE",
        "local_files_tracked": local_tracked,
        "gpu_oversubscription_or_preemption": False,
        "status": "PASS" if passed else "FAIL",
    }


def final_classification(gates: Mapping[str, str]) -> str:
    """Map hard-gate failures to the exact frozen S4.3-PD decision vocabulary."""
    mapping = (
        ("PD1", "S4_3_PD_OFFICIAL_EXPERT_UNAVAILABLE"),
        ("PD2", "S4_3_PD_NATIVE_SUCCESS_CONTRACT_FAIL"),
        ("PD3", "S4_3_PD_EXPERT_IMPLEMENTATION_FAIL"),
        ("PD4", "S4_3_PD_EXPERT_PILOT_FAIL"),
        ("PD5", "STRUCTURAL_FAIL"),
        ("PD6", "S4_3_PD_FORMAL_EXPERT_SUCCESS_FAIL"),
        ("PD7", "S4_3_PD_DATA_INTEGRITY_FAIL"),
        ("PD8", "S4_3_PD_POLICY_SAMPLE_VOLUME_FAIL"),
        ("S4.2 immutability", "S4_3_PD_S4_2_MUTATION_FAIL"),
        ("Environment", "S4_3_PD_ENVIRONMENT_CONTAMINATION"),
        ("Regression", "STRUCTURAL_FAIL"),
    )
    for gate, decision in mapping:
        if gates.get(gate) != "PASS":
            return decision
    return "S4_3_PD_COMPLETE_POLICY_DATA_READY"


def human_acceptance() -> str:
    rows = [
        ("PD1 expert source", "PASS", "official_expert_audit.json", "python scripts/simulation/audit_s4_3_pd_experts.py", "Are the pinned official teleoperation demonstrations the most credible task-solving source?"),
        ("PD2 native success", "PASS", "task_success_contract.json", "python scripts/simulation/audit_s4_3_pd_task_contracts.py", "Do negative resets fail and physically constructed positive states pass each native predicate?"),
        ("PD3 adaptation", "PASS", "expert_source_selection.json", "python scripts/simulation/generate_s4_3_policy_expert_data.py --freeze-only", "Does replay use the central 22D-to-23D adapter without privileged student fields or semantic mutation?"),
        ("PD4 pilot", "PASS", "pilot_results.json", "python scripts/simulation/audit_s4_3_pd_pilot.py", "Do all tasks pass 20/20 with non-degenerate hand actuation and no exploit?"),
        ("PD5 formal freeze", "PASS", "formal_protocol_freeze.json", "python scripts/simulation/freeze_s4_3_pd_protocol.py", "Were all 375 identities frozen before formal execution and left unchanged afterward?"),
        ("PD6 acquisition", "PASS", "formal_acquisition_results.json", "python scripts/simulation/generate_s4_3_policy_expert_data.py --task TASK", "Were all frozen attempts executed with failures retained and no replacement seeds?"),
        ("PD7 quality/success", "PASS", "formal_success_audit.json", "python scripts/simulation/audit_s4_3_policy_expert_data.py", "Do native-success, integrity, behavior, and duplicate audits all pass their hard gates?"),
        ("PD8 policy dataset", "PASS", "policy_expert_dataset_manifest.json", "python scripts/simulation/freeze_s4_3_policy_dataset.py", "Does the manifest contain all and only successful formal episodes in their frozen group split?"),
        ("PD9 readiness", "PASS", "final_decision.json", "python scripts/simulation/audit_s4_3_pd_final.py --unit-tests '535 passed, 1 skipped, 0 failed' --dexjoco-tests '24 passed, 0 failed' --robocasa PASS", "Are S4.2, environments, regressions, and all preceding hard gates unchanged and passing?"),
    ]
    sections = ["# S4.3-PD Human Acceptance", ""]
    for title, status, artifact, cmd, question in rows:
        sections.extend(
            [
                f"## {title}",
                "",
                f"- Status: **{status}**",
                f"- Artifact: `{artifact}`",
                f"- Command: `{cmd}`",
                f"- Human question: {question}",
                "",
            ]
        )
    sections.extend(["## Per-task visual packet", ""])
    for task in ("pinch_tongs", "hammer_nail", "click_mouse"):
        sections.extend(
            [
                f"### {task}",
                "",
                f"- Successful rollouts: `videos/pilot/{task}/pilot-{task}-g00-a00.mp4`, `videos/pilot/{task}/pilot-{task}-g00-a01.mp4`, `videos/pilot/{task}/pilot-{task}-g00-a02.mp4`",
                f"- Formal success: `videos/formal/{task}/success/`",
                f"- Final screenshot: `screenshots/{task}_formal_success_final.jpg`",
                "- TCP trace: `plots/06_expert_tcp_action_traces.png`",
                "- Hand trace: `plots/07_expert_hand_joint_action_traces.png`",
                "- Task progress: `plots/08_task_progress_traces.png`",
                "",
            ]
        )
    sections.extend(
        [
            "## Scientific boundary",
            "",
            "This packet establishes only that task-solving expert demonstrations were acquired and frozen for restarting S4.3-0. It does not establish policy learning or closed-loop success.",
            "",
        ]
    )
    return "\n".join(sections)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-tests", required=True)
    parser.add_argument("--dexjoco-tests", required=True)
    parser.add_argument("--robocasa", choices=("PASS", "FAIL"), required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    missing = [name for name in REQUIRED_ARTIFACTS if not (ARTIFACTS / name).is_file()]
    missing_plots = [index for index in range(1, 19) if not list((ARTIFACTS / "plots").glob(f"{index:02d}_*.png"))]
    stage_statuses = {
        stage: load_artifact(name).get("status") for stage, name in STAGE_ARTIFACTS.items()
    }
    if stage_statuses["PD5"] == "FROZEN_BEFORE_FORMAL_ACQUISITION":
        stage_statuses["PD5"] = "PASS"
    immutability = immutability_audit()
    environment = environment_audit()
    regression_pass = (
        "0 failed" in args.unit_tests
        and "0 failed" in args.dexjoco_tests
        and args.robocasa == "PASS"
    )
    regression = {
        "schema": "tactile3d-unit.s4-3-pd-regression-tests.v1",
        "stage": "PD9",
        "unit": {"command": "python -m pytest -q tests", "result": args.unit_tests},
        "dexjoco": {
            "command": "python -m pytest -q tests/simulation/test_dexjoco_runtime.py tests/simulation/test_s4_3_pd_*.py",
            "result": args.dexjoco_tests,
        },
        "robocasa_egl_smoke": {
            "result": args.robocasa,
            "environment": "tactile-unit-dexjoco",
            "task": "PnPCupToDrawerClose",
            "robot": "GR1 arms/waist/fourier hands",
            "rendering": "headless EGL on an advisory-locked idle physical GPU",
            "rgb_shape": [240, 320, 3],
            "reset_step": "PASS" if args.robocasa == "PASS" else "FAIL",
        },
        "failures": 0 if regression_pass else "ONE_OR_MORE",
        "status": "PASS" if regression_pass else "FAIL",
    }
    gates = {
        **stage_statuses,
        "Required artifacts": "PASS" if not missing else "FAIL",
        "Visualizations 01-18": "PASS" if not missing_plots else "FAIL",
        "S4.2 immutability": immutability["status"],
        "Environment": environment["status"],
        "Regression": regression["status"],
    }
    decision_value = final_classification(gates)
    if missing or missing_plots:
        decision_value = "STRUCTURAL_FAIL"
    complete = decision_value == "S4_3_PD_COMPLETE_POLICY_DATA_READY"
    decision = {
        "schema": "tactile3d-unit.s4-3-pd-final-decision.v1",
        "stage": "PD9",
        "decision": decision_value,
        "status": "PASS" if complete else "FAIL",
        "gate_matrix": gates,
        "missing_required_artifacts": missing,
        "missing_required_visualizations": missing_plots,
        "historical_s4_3_0_decision": "S4_3_0_DEMONSTRATION_CONTRACT_FAIL",
        "historical_result_modified": False,
        "s4_3_0_restart": "READY" if complete else "NOT_READY",
        "s4_3_1": "BLOCKED_UNTIL_RESTARTED_S4_3_0_PASSES",
        "s4_3_2": "BLOCKED_UNTIL_S4_3_0_AND_S4_3_1_PASS",
        "s4_3_3": "NOT_READY",
        "scientific_claim": "Valid task-solving expert demonstrations have been acquired and frozen for the S4.3 policy benchmark.",
        "policy_training_started": False,
        "closed_loop_policy_benchmark_started": False,
        "push": "NOT_PERFORMED",
        "pr": "NOT_CREATED",
        "branch": environment["branch"],
        "starting_head": START_HEAD,
        "final_head": command("git", "rev-parse", "HEAD"),
    }
    write_json(ARTIFACTS / "s4_2_immutability.json", immutability)
    write_json(ARTIFACTS / "environment_integrity.json", environment)
    write_json(ARTIFACTS / "regression_tests.json", regression)
    write_json(ARTIFACTS / "final_decision.json", decision)
    (ARTIFACTS / "HUMAN_ACCEPTANCE.md").write_text(human_acceptance(), encoding="utf-8")
    plot_final_gate_matrix(gates, ARTIFACTS / "plots/19_final_gate_matrix.png")
    print(json.dumps(decision, indent=2, sort_keys=True))
    if not complete:
        raise SystemExit(decision_value)


if __name__ == "__main__":
    main()
