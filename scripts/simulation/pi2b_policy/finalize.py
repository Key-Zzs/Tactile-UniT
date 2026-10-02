#!/usr/bin/env python3
"""Close Track A from complete evidence without launching new experiments."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2b_policy"
WORKSPACE_MANIFEST = ROOT / ".local/artifacts/pi2b_workspace_setup/workspace_manifest.json"
STARTING = ARTIFACTS / "starting_integrity.json"
TRAINING = ARTIFACTS / "training_completion.json"
PREFINAL = ARTIFACTS / "pre_final_freeze.json"
COMPLETENESS = ARTIFACTS / "rollout_completeness.json"
EXECUTION = ARTIFACTS / "final_gpu_execution.json"
AMENDMENT = ARTIFACTS / "runtime_amendment_cross_device_publish.json"
PER_SEED = ARTIFACTS / "per_seed_statistics.json"
CROSSED = ARTIFACTS / "crossed_seed_reset_analysis.json"
NEW_ONLY = ARTIFACTS / "new_seeds_only_sensitivity.json"
INDEPENDENT = ARTIFACTS / "statistics_independent_audit.json"
PROCESS = ARTIFACTS / "contact_process_metrics.json"
DECISION = ARTIFACTS / "final_decision.json"
PLOTS = ARTIFACTS / "plots_manifest.json"
TRACKED_DECISION = ROOT / "configs/simulation/pi2b_policy/final_decision.json"
REPORT = ROOT / "docs/research/s4_3_pi2b_policy_confirmation.md"
PAPER_BASE = ROOT / ".local/paper/PAPER_CORE_BASE.md"
PROTECTED_AFTER = ARTIFACTS / "protected_hashes_after.json"
ENVIRONMENT = ARTIFACTS / "environment_integrity.json"
REGRESSION = ARTIFACTS / "regression_tests.json"
RESOURCES = ARTIFACTS / "training_resource_summary.json"
RESULT_SUMMARY = ARTIFACTS / "result_summary.json"
PAPER_DELTA_MD = ARTIFACTS / "paper_delta.md"
PAPER_DELTA_JSON = ARTIFACTS / "paper_delta.json"
HANDOFF = ARTIFACTS / "integration_handoff.json"
HUMAN = ARTIFACTS / "HUMAN_ACCEPTANCE.md"
RESUME = ARTIFACTS / "status/resume_state.json"
COMPLETION = ARTIFACTS / "final_completion_audit.json"
BASE_SHA = "8f39aed123adf0a8b7241e75472e678355444e5e"
MODELS = ("B0", "B_VA27", "B1", "B_HVA", "B2")
SEEDS = (42, 43, 44)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def command(*args: str, cwd: Path = ROOT, check: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=check, env=env)


def git(*args: str, check: bool = True) -> str:
    return command("git", *args, check=check).stdout.strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit("PI2B_POLICY_CLOSEOUT_FAIL: " + message)


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def status_pass(path: Path) -> bool:
    return path.is_file() and read_json(path).get("status") == "PASS"


def render_paper_delta(source_hash: str) -> str:
    return f"""# PAPER_CORE delta — S4.3-PI2B Policy Multi-Seed Confirmation

Source PAPER_CORE SHA256: `{source_hash}`

This is an integration proposal only. It has not modified the authoritative
`PAPER_CORE.md` and must be reconciled with any other track before application.

## Header and evidence ledger

- Change `PI2B_status: DRAFT_NOT_EXECUTED` to
  `PI2B_status: COMPLETE_VALID_THREE_SEEDS_LIMITED`.
- Append to `local_artifact_verification`:
  `PI2B_10_NEW_30K_RUNS_COMPLETE_COLD_LOAD_VERIFIED;_PI2B_FINAL_3000_OUTCOMES_COMPLETE_VALID;_PI2B_MULTI_SEED_STATISTICS_AND_INDEPENDENT_AUDIT_PASS`.
- Preserve the existing PI2M and PI2N evidence strings and all historical
  negative results.

## Current-evidence summary replacement

Replace any sentence that presents the fixed-seed42 target ordering or H
increment as stable with the following scoped statement:

> In S4.3-PI2B on `pinch_tongs`, the five frozen policy recipes were evaluated
> across training seeds 42/43/44 on a common 200-reset cohort. The H increment
> under VA27 changed from +13 pp at seed42 to -48/-18 pp at seeds43/44, while
> B2-B_HVA changed from -18.5 pp to +6.5/+9 pp. These sign reversals establish
> substantial recipe sensitivity to the three tested training seeds; they do
> not support a stable positive H increment or a stable ordering of Contact/VAC
> and clean-VA targets. The result is limited to one task and three seeds.

## Stage 2 section replacement

Replace the `DRAFT_NOT_EXECUTED` paragraph and the proposed `11`-new-run /
`seed8` design with the actual execution:

- Five exact models: B0, B_VA27, B1, B_HVA, B2.
- Reused five live-hash-verified seed42 checkpoints.
- Added seeds43/44 for every model: exactly ten new 30k runs, not eleven.
- Used reset blocks 16–19 (200 frozen identities), not seed8.
- Completed 15 × 200 = 3,000 canonical outcomes.
- Seed42 was development-exposed; seeds43/44 were preregistered additions.
- Engineering status is `COMPLETE_VALID`; all three core direction axes are
  `MIXED`; training-seed scope is `THREE_SEEDS_LIMITED`.

## Result table to add

| Contrast | Seed42 / 43 / 44 (pp) | Mean (pp) | Conditional reset 95% CI | Two-way sensitivity 95% CI |
|---|---|---:|---:|---:|
| B_HVA-B_VA27 | +13 / -48 / -18 | -17.67 | [-23.00,-12.33] | [-46.50,+11.00] |
| B2-B_HVA | -18.5 / +6.5 / +9 | -1.00 | [-5.17,+3.33] | [-17.00,+11.83] |
| B2-B1 | +11.5 / -13.5 / +13 | +3.67 | [-0.67,+8.17] | [-12.00,+16.67] |
| B_HVA-B0 | +17.5 / -40.5 / -10 | -11.00 | [-15.83,-6.33] | [-39.00,+15.50] |
| B2-B0 | -1 / -34 / -1 | -12.00 | [-17.00,-7.00] | [-32.00,+4.00] |
| B_VA27-B0 | +4.5 / +7.5 / +8 | +6.67 | [+1.33,+12.00] | [-0.33,+13.83] |
| B1-B0 | -12.5 / -20.5 / -14 | -15.67 | [-20.33,-11.00] | [-23.33,-8.50] |

Conditional reset intervals condition on the fifteen trained checkpoints. The
two-way intervals are finite-sample sensitivities with only three outer units.
The minimum exact two-sided seed sign-flip p-value is 0.25. Neither interval is
evidence of equivalence or a guarantee for future training runs.

## Claim ledger changes

- CL04 (stable full-method superiority): mark
  `NOT_SUPPORTED_BY_PI2B_THREE_SEEDS`; B_HVA-B0 and B2-B0 are not stable
  positive effects. Do not convert this into a universal negative claim.
- CL06/CL13 (Contact/VAC versus clean VA target): replace the fixed-seed
  directional summary with `MIXED_ACROSS_TRAINING_SEEDS`; retain PI2M/PI2N
  seed42 results as historical fixed-seed evidence.
- CL15 (positive H increment under VA27): replace `SUPPORTED_WITH_SCOPE` with
  `MIXED_ACROSS_TRAINING_SEEDS`; record +13/-48/-18 pp.
- CL05 remains `NOT_ESTABLISHED`; the consistent negative B1-B0 point
  direction over three seeds is not an equivalence or universal mechanism
  result.

## Figure and artifact ledger changes

- Mark the multi-seed portion of Fig4 complete and source it from the five
  frozen Track-A plots; do not combine absolute rates from the older PI2N
  cohort with these reset blocks.
- Add `.local/artifacts/simulation/s4_3_pi2b_policy/` and
  `docs/research/s4_3_pi2b_policy_confirmation.md` to the artifact ledger.

## Required retained limitations

- one task, three training seeds, weak outer-level variance estimation;
- seed42 selected/development-exposed, seeds43/44 preregistered;
- asynchronous policy sampling is controlled by the accepted runtime contract
  but not claimed bit-exact across arbitrary replays;
- no Track-B teacher entered this experiment;
- no real robot, second task, human-robot transfer or Original-UniT
  superiority result;
- 10 pp is descriptive only, not an engineering gate;
- no equivalence claim and no automatic experiment expansion.
"""


def render_human(head: str, per_seed: dict[str, Any], crossed: dict[str, Any], decision: dict[str, Any]) -> str:
    model_lines = []
    for model in MODELS:
        for seed in SEEDS:
            row = per_seed["checkpoint_statistics"][model][str(seed)]
            model_lines.append(
                f"| {model} | {seed} | {row['successes']}/200 | {row['success_percent']:.1f}% | "
                f"[{row['wilson_95ci_percent'][0]:.2f},{row['wilson_95ci_percent'][1]:.2f}]% |"
            )
    primary_lines = []
    for name in ("B_HVA-B_VA27", "B2-B_HVA"):
        row = crossed["contrasts"][name]
        points = row["per_seed_risk_difference_percentage_points"]
        conditional = row["conditional_shared_reset_bootstrap_percentage_points"]["interval"]
        two_way = row["two_way_seed_by_reset_sensitivity_percentage_points"]["interval"]
        primary_lines.append(
            f"| {name} | {points['42']:+.1f}/{points['43']:+.1f}/{points['44']:+.1f} | "
            f"{row['training_seed_summary_percentage_points']['mean']:+.2f} | "
            f"[{conditional[0]:+.2f},{conditional[1]:+.2f}] | [{two_way[0]:+.2f},{two_way[1]:+.2f}] |"
        )
    return "\n".join(
        [
            "# S4.3-PI2B Policy Track A — Human Acceptance",
            "",
            "Status: **COMPLETE_VALID — THREE_SEEDS_LIMITED**",
            "",
            f"Final tracked HEAD: `{head}` on `develop/pi2b-policy`.",
            "No push, PR, merge, tag, Track-B teacher substitution, paper-core mutation, or experiment expansion was performed.",
            "",
            "## Fifteen checkpoint results",
            "",
            "| Model | Seed | Success | Rate | Wilson 95% CI |",
            "|---|---:|---:|---:|---:|",
            *model_lines,
            "",
            "## Primary effects",
            "",
            "| Contrast | Seed42/43/44 (pp) | Mean (pp) | Conditional CI | Two-way CI |",
            "|---|---:|---:|---:|---:|",
            *primary_lines,
            "",
            "## Decision axes",
            "",
            f"- H_GIVEN_VA27: `{decision['H_GIVEN_VA27']}`",
            f"- VAC_C_VS_VA_TARGET: `{decision['VAC_C_VS_VA_TARGET']}`",
            f"- VAC_C_VS_H_ONLY: `{decision['VAC_C_VS_H_ONLY']}`",
            "- TRAINING_SEED_SCOPE: `THREE_SEEDS_LIMITED`",
            "",
            "The sign reversals are retained as canonical results. Three training seeds do not establish population-level significance; inconclusive is not equivalence.",
            "",
            "## Read-only acceptance commands",
            "",
            "```bash",
            "jq . .local/artifacts/simulation/s4_3_pi2b_policy/final_completion_audit.json",
            "jq . .local/artifacts/simulation/s4_3_pi2b_policy/final_decision.json",
            "jq . .local/artifacts/simulation/s4_3_pi2b_policy/statistics_independent_audit.json",
            "jq . .local/artifacts/simulation/s4_3_pi2b_policy/integration_handoff.json",
            "git status --short",
            "```",
            "",
            "## Stop",
            "",
            "No additional training, teacher work, push, PR, merge, or hardware experiment is authorized by this acceptance file.",
            "",
        ]
    )


def main() -> None:
    outputs = (
        PROTECTED_AFTER,
        ENVIRONMENT,
        REGRESSION,
        RESOURCES,
        RESULT_SUMMARY,
        PAPER_DELTA_MD,
        PAPER_DELTA_JSON,
        HANDOFF,
        HUMAN,
        RESUME,
        COMPLETION,
    )
    require(not any(path.exists() for path in outputs), "refusing to overwrite closeout outputs")
    required = (
        WORKSPACE_MANIFEST,
        STARTING,
        TRAINING,
        PREFINAL,
        COMPLETENESS,
        EXECUTION,
        AMENDMENT,
        PER_SEED,
        CROSSED,
        NEW_ONLY,
        INDEPENDENT,
        PROCESS,
        DECISION,
        PLOTS,
        TRACKED_DECISION,
        REPORT,
        PAPER_BASE,
    )
    require(all(path.is_file() for path in required), "required closeout input missing")
    require(git("branch", "--show-current") == "develop/pi2b-policy", "wrong branch")
    require(not git("status", "--short"), "tracked worktree must be clean")
    head = git("rev-parse", "HEAD")
    require(not git("rev-list", "--min-parents=2", f"{BASE_SHA}..HEAD"), "merge commit detected")
    upstream = git("rev-parse", "--abbrev-ref", "@{upstream}", check=False)
    require(not upstream, "policy branch unexpectedly has an upstream")

    workspace = read_json(WORKSPACE_MANIFEST)
    starting = read_json(STARTING)
    training = read_json(TRAINING)
    prefinal = read_json(PREFINAL)
    completeness = read_json(COMPLETENESS)
    execution = read_json(EXECUTION)
    amendment = read_json(AMENDMENT)
    per_seed = read_json(PER_SEED)
    crossed = read_json(CROSSED)
    new_only = read_json(NEW_ONLY)
    independent = read_json(INDEPENDENT)
    process = read_json(PROCESS)
    decision = read_json(DECISION)
    plots = read_json(PLOTS)
    tracked_decision = read_json(TRACKED_DECISION)
    status_inputs = (training, prefinal, completeness, execution, amendment, per_seed, crossed, new_only, independent, process, plots)
    require(all(item.get("status") == "PASS" for item in status_inputs), "one or more input status gates are not PASS")
    require(decision.get("status") == "COMPLETE_VALID", "local final decision is not complete")
    require(tracked_decision.get("status") == "COMPLETE_VALID", "tracked final decision is not complete")
    require(len(training["runs"]) == 10, "training run count is not ten")
    require(len(prefinal["checkpoints"]) == 15, "checkpoint count is not fifteen")
    require(completeness.get("total_canonical_outcomes") == 3000, "formal outcome count is not 3000")
    require(len(execution["workers"]) == 60 and execution.get("exclusive_barrier_held_for_full_wave") is True, "formal execution barrier/worker gate failed")
    require(amendment.get("scope") == "ARTIFACT_PUBLICATION_ONLY" and amendment.get("policy_or_evaluator_semantics_changed") is False, "runtime amendment scope drift")
    require(decision.get("TRACK_B_NEW_TEACHER_READ") is False and prefinal.get("track_b_new_teacher_read") is False, "Track-B isolation failure")

    main_paper = Path(workspace["paper_core_source"])
    source_paper_hash = workspace["paper_core_sha256"]
    require(sha256_file(main_paper) == source_paper_hash, "authoritative PAPER_CORE changed")
    require(sha256_file(PAPER_BASE) == source_paper_hash, "local PAPER_CORE base snapshot changed")

    current_core = {path: sha256_file(ROOT / path) for path in starting["source"]["core_hashes"]}
    require(current_core == starting["source"]["core_hashes"], "core source protected hash drift")
    openpi_root = Path(starting["source"]["openpi_root"])
    current_openpi = {path: sha256_file(openpi_root / path) for path in starting["source"]["openpi_hashes"]}
    require(current_openpi == starting["source"]["openpi_hashes"], "OpenPI protected hash drift")
    sidecars = {
        name: sha256_file(Path(starting["sidecars"][name]["path"])) for name in ("contact", "va27")
    }
    require(sidecars == {name: starting["sidecars"][name]["sha256"] for name in sidecars}, "sidecar protected hash drift")
    teacher_path = ROOT / ".local/experiments/simulation/s4_2r/contact_state/accepted.pt"
    teacher_hash = sha256_file(teacher_path)
    require(teacher_hash == starting["protected_teacher"]["E_T_expected_sha256"], "E_T protected hash drift")
    prompt_path = ROOT / ".local/prompt/PROMPT2_PI2B_POLICY_TRACK_A.md"
    require(sha256_file(prompt_path) == workspace["prompt_hashes"]["PROMPT2_PI2B_POLICY_TRACK_A.md"], "Track-A prompt drift")
    checkpoint_hashes = {
        f"{row['model']}/seed{row['training_seed']}": row["tree_sha256"] for row in prefinal["checkpoints"]
    }
    protected = {
        "schema": "tactile3d-unit.s4-3-pi2b-policy-protected-hashes-after.v1",
        "status": "PASS",
        "created_at_utc": now(),
        "live_file_sha256": {
            "E_T": teacher_hash,
            "contact_sidecar": sidecars["contact"],
            "va27_sidecar": sidecars["va27"],
            "authoritative_PAPER_CORE": source_paper_hash,
            "local_PAPER_CORE_BASE": sha256_file(PAPER_BASE),
            "Track_A_prompt": sha256_file(prompt_path),
            **{f"policy_source/{key}": value for key, value in current_core.items()},
            **{f"openpi_source/{key}": value for key, value in current_openpi.items()},
        },
        "checkpoint_tree_sha256": checkpoint_hashes,
        "checkpoint_verification": "15 trees were live-hashed and persisted in starting_integrity/training_completion, then rebound in pre_final_freeze; closeout deliberately does not repeat the documented 140+ GB NAS scan",
        "pi05_base_file_sha256": read_json(ARTIFACTS / "protected_hashes_before.json")["base_files"],
        "pi05_base_verification": "full live file hashes recorded before training; source checkpoint was read-only and rebound by every independent initialization",
        "paper_core_unchanged": True,
        "new_track_b_teacher_read": False,
    }
    atomic_json(PROTECTED_AFTER, protected)

    import_origins = read_json(ARTIFACTS / "import_origins.json")
    openpi_python = Path(import_origins["python"])
    require(openpi_python.is_file(), "frozen OpenPI interpreter missing")
    probe_env = os.environ.copy()
    probe_env["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(openpi_root / "src")))
    probe_code = (
        "import importlib.util,json;"
        "print(json.dumps({n:importlib.util.find_spec(n).origin for n in "
        "['gr00t','gr00t.simulation.pi05_tactile_unit','openpi','openpi.training.config']}))"
    )
    probe = command(str(openpi_python), "-c", probe_code, env=probe_env)
    observed_origins = json.loads(probe.stdout)
    require(observed_origins == import_origins["modules"], "actual import origins changed")
    environment = {
        "schema": "tactile3d-unit.s4-3-pi2b-policy-environment-integrity.v1",
        "status": "PASS",
        "created_at_utc": now(),
        "control_python": sys.executable,
        "control_python_version": platform.python_version(),
        "openpi_python": str(openpi_python),
        "openpi_python_version": command(str(openpi_python), "--version").stdout.strip() or command(str(openpi_python), "--version").stderr.strip(),
        "module_origins": observed_origins,
        "source_hashes_match_starting_integrity": True,
        "shared_environment_mutation_performed": False,
        "editable_install_performed": False,
        "teacher_worktree_imported": False,
    }
    atomic_json(ENVIRONMENT, environment)

    test_command = (
        str(openpi_python),
        "-m",
        "pytest",
        "-q",
        "tests/simulation/pi2b_policy",
        "tests/simulation/test_s4_3_pi1.py",
        "tests/simulation/test_s4_3_pi2m.py",
        "tests/simulation/test_s4_3_pi2n.py",
    )
    test_env = os.environ.copy()
    test_env["PYTHONPATH"] = str(ROOT)
    test_result = command(*test_command, env=test_env, check=False)
    require(test_result.returncode == 0 and "54 passed" in test_result.stdout, "final regression suite failed")
    regression = {
        "schema": "tactile3d-unit.s4-3-pi2b-policy-regression-tests.v1",
        "status": "PASS",
        "created_at_utc": now(),
        "command": list(test_command),
        "exit_code": test_result.returncode,
        "stdout": test_result.stdout.strip(),
        "stderr": test_result.stderr.strip(),
        "tests_passed": 54,
    }
    atomic_json(REGRESSION, regression)

    status_root = Path(workspace["worktrees"]["policy"]["write_root"]) / "status/training"
    resource_rows = []
    for launch_path in sorted((ARTIFACTS / "launches").glob("seed*.json")):
        launch = read_json(launch_path)
        status = read_json(status_root / f"{launch['run_id']}.json")
        start = datetime.fromisoformat(launch["created_at_utc"])
        stop = datetime.fromisoformat(status["updated_at_utc"])
        hours = (stop - start).total_seconds() / 3600.0
        resource_rows.append(
            {
                "run_id": launch["run_id"],
                "model": launch["model_id"],
                "training_seed": launch["seed"],
                "physical_gpu_index": status["physical_gpu_index"],
                "fsdp_devices": status["fsdp_devices"],
                "global_batch_size": status["global_batch_size"],
                "optimizer_steps": status["optimizer_steps"],
                "wall_hours": hours,
                "gpu_hours": hours * status["fsdp_devices"],
                "start_utc": launch["created_at_utc"],
                "stop_utc": status["updated_at_utc"],
            }
        )
    require(len(resource_rows) == 10, "training resource row count is not ten")
    resources = {
        "schema": "tactile3d-unit.s4-3-pi2b-policy-training-resources.v1",
        "status": "PASS",
        "created_at_utc": now(),
        "runs": resource_rows,
        "total_gpu_hours": sum(row["gpu_hours"] for row in resource_rows),
        "measurement": "launch manifest creation to terminal status update",
    }
    atomic_json(RESOURCES, resources)

    result_summary = {
        "schema": "tactile3d-unit.s4-3-pi2b-policy-result-summary.v1",
        "status": "COMPLETE_VALID",
        "created_at_utc": now(),
        "training": {"planned": 10, "completed": 10, "optimizer_steps_each": 30000, "total_gpu_hours": resources["total_gpu_hours"]},
        "formal_evaluation": {"checkpoints": 15, "shared_resets": 200, "canonical_outcomes": 3000, "successes": independent["total_successes"], "failures": independent["total_failures"], "timeouts": 0},
        "decision_axes": {
            "H_GIVEN_VA27": decision["H_GIVEN_VA27"],
            "VAC_C_VS_VA_TARGET": decision["VAC_C_VS_VA_TARGET"],
            "VAC_C_VS_H_ONLY": decision["VAC_C_VS_H_ONLY"],
            "TRAINING_SEED_SCOPE": decision["TRAINING_SEED_SCOPE"],
        },
        "primary_contrasts": {name: crossed["contrasts"][name] for name in ("B_HVA-B_VA27", "B2-B_HVA")},
        "seed42_disclosure": decision["SEED42_DISCLOSURE"],
        "new_seed_disclosure": decision["SEEDS43_44_DISCLOSURE"],
        "equivalence_claimed": False,
        "track_b_new_teacher_read": False,
    }
    atomic_json(RESULT_SUMMARY, result_summary)

    paper_delta_text = render_paper_delta(source_paper_hash)
    atomic_text(PAPER_DELTA_MD, paper_delta_text)
    paper_delta = {
        "schema": "tactile3d-unit.s4-3-pi2b-policy-paper-delta.v1",
        "status": "READY_FOR_INTEGRATION_NOT_APPLIED",
        "created_at_utc": now(),
        "source_paper_core_path": str(main_paper),
        "source_paper_core_sha256": source_paper_hash,
        "source_paper_core_unchanged": sha256_file(main_paper) == source_paper_hash,
        "paper_delta_markdown_path": relative(PAPER_DELTA_MD),
        "paper_delta_markdown_sha256": sha256_file(PAPER_DELTA_MD),
        "required_corrections": [
            "PI2B DRAFT_NOT_EXECUTED -> COMPLETE_VALID_THREE_SEEDS_LIMITED",
            "11 proposed new runs -> 10 actually completed new runs",
            "proposed evaluator seed8 -> actual frozen reset blocks 16-19",
            "fixed-seed H/target direction -> mixed across training seeds",
            "Fig4 MULTI_SEED_PENDING -> Track-A multi-seed complete",
        ],
        "claim_updates": {
            "CL04": "NOT_SUPPORTED_BY_PI2B_THREE_SEEDS",
            "CL06": "MIXED_ACROSS_TRAINING_SEEDS",
            "CL13": "MIXED_ACROSS_TRAINING_SEEDS",
            "CL15": "MIXED_ACROSS_TRAINING_SEEDS",
        },
        "historical_negative_results_preserved": True,
        "human_robot_transfer_claimed": False,
        "real_hardware_claimed": False,
        "original_unit_superiority_claimed": False,
        "applied_to_main_paper": False,
    }
    atomic_json(PAPER_DELTA_JSON, paper_delta)

    human_text = render_human(head, per_seed, crossed, decision)
    atomic_text(HUMAN, human_text)
    commits = [
        {"sha": line.split(maxsplit=1)[0], "subject": line.split(maxsplit=1)[1]}
        for line in git("log", "--reverse", "--format=%H %s", f"{BASE_SHA}..HEAD").splitlines()
    ]
    artifact_paths = (
        STARTING,
        TRAINING,
        PREFINAL,
        COMPLETENESS,
        EXECUTION,
        AMENDMENT,
        PER_SEED,
        CROSSED,
        NEW_ONLY,
        INDEPENDENT,
        PROCESS,
        DECISION,
        PLOTS,
        PROTECTED_AFTER,
        ENVIRONMENT,
        REGRESSION,
        RESOURCES,
        RESULT_SUMMARY,
        PAPER_DELTA_MD,
        PAPER_DELTA_JSON,
        HUMAN,
        TRACKED_DECISION,
        REPORT,
    )
    artifact_manifest = {relative(path): sha256_file(path) for path in artifact_paths}
    handoff = {
        "schema": "tactile3d-unit.s4-3-pi2b-policy-integration-handoff.v1",
        "status": "READY_FOR_USER_DIRECTED_INTEGRATION",
        "created_at_utc": now(),
        "base_sha": BASE_SHA,
        "branch": "develop/pi2b-policy",
        "final_head": head,
        "commits": commits,
        "tracked_worktree_clean": True,
        "merge_commits_since_base": [],
        "upstream_configured": False,
        "push_pr_merge_tag_release_performed": False,
        "source_paper_core_path": str(main_paper),
        "source_paper_core_sha256": source_paper_hash,
        "source_paper_core_unchanged": True,
        "paper_delta_path": relative(PAPER_DELTA_MD),
        "paper_delta_sha256": sha256_file(PAPER_DELTA_MD),
        "paper_delta_json_path": relative(PAPER_DELTA_JSON),
        "paper_delta_json_sha256": sha256_file(PAPER_DELTA_JSON),
        "protected_hashes_after_path": relative(PROTECTED_AFTER),
        "protected_hashes_after_sha256": sha256_file(PROTECTED_AFTER),
        "artifact_manifest": artifact_manifest,
        "authoritative_manifests": {
            "workspace": relative(WORKSPACE_MANIFEST),
            "training": relative(TRAINING),
            "pre_final": relative(PREFINAL),
            "rollout_completeness": relative(COMPLETENESS),
            "statistics": relative(CROSSED),
            "decision": relative(DECISION),
            "tracked_decision": relative(TRACKED_DECISION),
            "paper_delta": relative(PAPER_DELTA_JSON),
        },
        "integration_instructions": [
            "Review and reconcile paper_delta.md against the then-current authoritative PAPER_CORE; do not copy it blindly.",
            "Preserve PI2M/PI2N fixed-seed results as historical evidence while updating present-tense claims to mixed across seeds.",
            "Do not merge or push this branch without separate user authorization.",
        ],
        "retained_warnings": [
            "Only three training seeds and one task were tested.",
            "Seed42 was previously selected/development-exposed; seeds43/44 were preregistered additions.",
            "Conditional reset intervals do not quantify future-training-run uncertainty.",
            "Two-way intervals are weak finite-sample sensitivities with three outer units.",
            "No equivalence, real-hardware, human-robot-transfer, or Original-UniT superiority claim is supported.",
            "Track-B new teacher results and weights were not read or substituted.",
        ],
    }
    atomic_json(HANDOFF, handoff)

    resume = {
        "schema": "tactile3d-unit.s4-3-pi2b-policy-resume-state.v1",
        "status": "COMPLETE",
        "created_at_utc": now(),
        "stage": "S4.3-PI2B-Policy",
        "next_action": "STOP_AWAIT_USER_INTEGRATION_AUTHORIZATION",
        "training_runs_complete": 10,
        "formal_rollouts_complete": 3000,
        "long_running_processes_expected": 0,
        "automatic_successor": False,
    }
    atomic_json(RESUME, resume)

    coordination_request = Path(workspace["common_git_dir"]) / "pi2b_coordination/requests/policy.json"
    coordination = read_json(coordination_request)
    completion_gates = {
        "branch_and_head_exact": git("branch", "--show-current") == "develop/pi2b-policy" and git("rev-parse", "HEAD") == head,
        "tracked_worktree_clean": not git("status", "--short"),
        "ten_training_runs_complete": training.get("status") == "PASS" and len(training["runs"]) == 10,
        "fifteen_checkpoints_frozen": prefinal.get("status") == "PASS" and len(prefinal["checkpoints"]) == 15,
        "three_thousand_shared_reset_outcomes": completeness.get("status") == "PASS" and completeness.get("total_canonical_outcomes") == 3000,
        "exclusive_evaluation_execution_pass": execution.get("status") == "PASS" and execution.get("exclusive_barrier_held_for_full_wave") is True,
        "coordination_request_complete": coordination.get("status") == "COMPLETE" and coordination.get("completed_blocks") == 60,
        "runtime_amendment_scoped_and_pass": amendment.get("status") == "PASS" and amendment.get("scope") == "ARTIFACT_PUBLICATION_ONLY",
        "per_seed_statistics_pass": per_seed.get("status") == "PASS",
        "crossed_analysis_pass": crossed.get("status") == "PASS",
        "new_seeds_sensitivity_pass": new_only.get("status") == "PASS",
        "independent_statistics_audit_pass": independent.get("status") == "PASS" and independent.get("checks_performed") == 3319,
        "contact_process_metrics_pass": process.get("status") == "PASS",
        "plots_pass": plots.get("status") == "PASS" and len(plots.get("plots", [])) == 5,
        "decision_complete_valid": decision.get("status") == "COMPLETE_VALID" and tracked_decision.get("status") == "COMPLETE_VALID",
        "protected_hashes_after_pass": status_pass(PROTECTED_AFTER),
        "environment_integrity_pass": status_pass(ENVIRONMENT),
        "regression_54_pass": status_pass(REGRESSION) and read_json(REGRESSION).get("tests_passed") == 54,
        "paper_core_unchanged": sha256_file(main_paper) == source_paper_hash,
        "paper_delta_ready_not_applied": read_json(PAPER_DELTA_JSON).get("status") == "READY_FOR_INTEGRATION_NOT_APPLIED",
        "integration_handoff_ready": read_json(HANDOFF).get("status") == "READY_FOR_USER_DIRECTED_INTEGRATION",
        "no_merge_commits": not git("rev-list", "--min-parents=2", f"{BASE_SHA}..HEAD"),
        "no_upstream_push_target": not git("rev-parse", "--abbrev-ref", "@{upstream}", check=False),
        "track_b_new_teacher_not_read": decision.get("TRACK_B_NEW_TEACHER_READ") is False,
        "automatic_experiment_expansion_disabled": decision.get("AUTO_EXPANSION_AUTHORIZED") is False,
    }
    require(all(completion_gates.values()), "final completion gate failure: " + ",".join(name for name, passed in completion_gates.items() if not passed))
    completion = {
        "schema": "tactile3d-unit.s4-3-pi2b-policy-final-completion.v1",
        "status": "COMPLETE_VALID",
        "created_at_utc": now(),
        "base_sha": BASE_SHA,
        "branch": "develop/pi2b-policy",
        "final_head": head,
        "completion_gates": completion_gates,
        "paper_delta_sha256": sha256_file(PAPER_DELTA_MD),
        "integration_handoff_sha256": sha256_file(HANDOFF),
        "human_acceptance_sha256": sha256_file(HUMAN),
        "result_summary_sha256": sha256_file(RESULT_SUMMARY),
        "tracked_worktree_clean": True,
        "stop": "NO_PUSH_PR_MERGE_TAG_NEW_TEACHER_HARDWARE_OR_AUTOMATIC_EXPERIMENT",
    }
    atomic_json(COMPLETION, completion)
    print(
        json.dumps(
            {
                "status": "COMPLETE_VALID",
                "head": head,
                "training_runs": 10,
                "formal_outcomes": 3000,
                "regression_tests": 54,
                "paper_delta": str(PAPER_DELTA_MD),
                "integration_handoff": str(HANDOFF),
                "completion_audit": str(COMPLETION),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
