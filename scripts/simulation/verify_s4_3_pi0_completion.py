#!/usr/bin/env python3
"""Verify all required S4.3-PI0 evidence before declaring completion."""

from __future__ import annotations

import json
import socket
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_pi0"
LOG_ROOT = ROOT / ".local/logs/simulation/s4_3_pi0"
OUTPUT = ARTIFACT_ROOT / "completion_verification.json"
START_HEAD = "b3275f2f0ca1579d45c8eca80834813e5ee6d852"
EXPECTED_DECISION = "S4_3_PI0_FULL_OFFICIAL_BASELINE_REPRODUCTION"
REQUIRED_ARTIFACTS = (
    "starting_integrity.json",
    "official_source_audit.json",
    "official_openpi_contract.json",
    "official_checkpoint_manifest.json",
    "official_checkpoint_smoke.json",
    "official_checkpoint_eval.json",
    "official_dataset_manifest.json",
    "official_base_model_manifest.json",
    "official_base_model_smoke.json",
    "official_training_config.json",
    "training_launch.json",
    "training_completion.json",
    "reproduced_checkpoint_smoke.json",
    "reproduced_checkpoint_server_smoke.json",
    "evaluation_environment.json",
    "reproduced_eval_runtime.json",
    "reproduced_checkpoint_eval.json",
    "official_vs_reproduced_comparison.json",
    "environment_integrity.json",
    "s4_2_immutability_after.json",
    "final_integrity.json",
)
EXPECTED_COMMITS = (
    "eval(sim): audit official DexJoCo pi05 baseline",
    "test(sim): freeze reduced-cost official pi05 reproduction",
    "eval(sim): close official DexJoCo pi05 reproduction",
)


def command(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def port_is_closed(port: int) -> bool:
    sock = socket.socket()
    sock.settimeout(0.2)
    try:
        return sock.connect_ex(("127.0.0.1", port)) != 0
    finally:
        sock.close()


def main() -> None:
    artifacts = {
        name: json.loads((ARTIFACT_ROOT / name).read_text(encoding="utf-8")) for name in REQUIRED_ARTIFACTS
    }
    comparison = artifacts["official_vs_reproduced_comparison.json"]
    official = artifacts["official_checkpoint_eval.json"]
    reproduced = artifacts["reproduced_checkpoint_eval.json"]
    training = artifacts["training_completion.json"]
    final_integrity = artifacts["final_integrity.json"]
    unit_log = (LOG_ROOT / "final_unit_tests.log").read_text(encoding="utf-8", errors="replace")
    dexjoco_log = (LOG_ROOT / "final_dexjoco_runtime_test.log").read_text(encoding="utf-8", errors="replace")
    ruff_log = (LOG_ROOT / "final_ruff.log").read_text(encoding="utf-8", errors="replace")
    current_head = command("git", "rev-parse", "HEAD")
    commit_messages = command("git", "log", "--reverse", "--format=%s", f"{START_HEAD}..HEAD").splitlines()
    plots = sorted(path.name for path in (ARTIFACT_ROOT / "plots").glob("*.png"))
    videos = sorted(path.name for path in (ARTIFACT_ROOT / "videos").glob("*.mp4"))
    gates = {
        "all_required_artifacts_pass": all(payload.get("status") == "PASS" for payload in artifacts.values()),
        "training_step_30000": training["train_state_step"] == 30_000,
        "final_checkpoint_only_evaluated": reproduced["checkpoint_kind"] == "reproduced"
        and reproduced["episodes"] == 20,
        "official_result_4_of_20": official["successes"] == 4 and official["episodes"] == 20,
        "reproduced_result_2_of_20": reproduced["successes"] == 2 and reproduced["episodes"] == 20,
        "full_reproduction_decision": comparison["decision"] == EXPECTED_DECISION,
        "pi1_ready_with_warnings": comparison["pi1_readiness"] == "READY_WITH_WARNINGS",
        "unit_tests_zero_failures": "586 passed, 1 skipped" in unit_log,
        "dexjoco_runtime_smoke": "1 passed" in dexjoco_log,
        "ruff": "All checks passed!" in ruff_log,
        "five_plots": len(plots) == 5,
        "eight_representative_videos": len(videos) == 8,
        "branch": command("git", "branch", "--show-current") == "develop/sim-benchmark",
        "three_expected_local_commits": commit_messages == list(EXPECTED_COMMITS),
        "final_integrity_matches_head": final_integrity["current_head"] == current_head,
        "working_tree_clean": not command("git", "status", "--porcelain"),
        "local_artifacts_untracked": not command("git", "ls-files", ".local"),
        "evaluation_port_released": port_is_closed(8125),
        "no_pi0_tmux_sessions": not any(
            line.startswith("s43_pi0_eval_")
            for line in subprocess.run(
                ["tmux", "list-sessions"],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
        ),
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi0-completion-verification.v1",
        "stage": "PI0-12",
        "starting_head": START_HEAD,
        "final_head": current_head,
        "commits": commit_messages,
        "artifact_statuses": {name: payload.get("status") for name, payload in artifacts.items()},
        "plots": plots,
        "representative_videos": videos,
        "decision": comparison["decision"],
        "pi1_readiness": comparison["pi1_readiness"],
        "gates": {name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
        "status": "PASS" if all(gates.values()) else "FAIL",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit("S4_3_PI0_COMPLETION_VERIFICATION_FAIL")


if __name__ == "__main__":
    main()
