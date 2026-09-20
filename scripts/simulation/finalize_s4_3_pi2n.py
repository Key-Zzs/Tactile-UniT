#!/usr/bin/env python3
"""Close PI2N from complete frozen evidence without launching new experiments."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.simulation.freeze_s4_3_pi2n_final import environment_snapshot


ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2n"
PAPER = ROOT / ".local/paper/PAPER_CORE.md"
STARTING = ARTIFACTS / "starting_integrity.json"
PRE_FINAL = ARTIFACTS / "pre_final_freeze.json"
COMPLETENESS = ARTIFACTS / "rollout_completeness.json"
EXECUTION = ARTIFACTS / "final_gpu_execution.json"
JOB_STATUS = ARTIFACTS / "final_job_status.json"
STATISTICS = ARTIFACTS / "paired_statistics.json"
PROCESS = ARTIFACTS / "contact_process_metrics.json"
CLAIM = ARTIFACTS / "claim_freeze.json"
FINAL_DECISION = ARTIFACTS / "final_decision.json"
INDEPENDENT = ARTIFACTS / "statistics_independent_audit.json"
PLOTS = ARTIFACTS / "plots_manifest.json"
REGRESSION = ARTIFACTS / "regression_tests.json"
PAPER_AUDIT = ARTIFACTS / "paper_core_update_audit_final.json"
PROTECTED_AFTER = ARTIFACTS / "protected_integrity_after.json"
ENVIRONMENT_AFTER = ARTIFACTS / "environment_integrity.json"
RETRY_LEDGER = ARTIFACTS / "retry_ledger.json"
PI2B_DRAFT = ARTIFACTS / "pi2b_recommendation_draft.json"
COMPLETION = ARTIFACTS / "final_completion_audit.json"
HUMAN_ACCEPTANCE = ARTIFACTS / "HUMAN_ACCEPTANCE.md"
DEXJOCO_HEAD = "8d23b0fab23b17a58c4b55f3942e17013aaf8267"
BASE_MANIFEST = ROOT / ".local/artifacts/simulation/s4_3_pi0/official_base_model_manifest.json"
UNIT_PROTOCOL = ROOT / "configs/simulation/s4_3_pi2n_vac_v_target_protocol.json"
UNIT_TOKENIZER = ROOT / ".local/external/s4_3_pi2u/unit_fulldata/VLA-UniT-3B-fulldata/tokenizer"
TEACHERS = {
    "E_T": ROOT / ".local/experiments/simulation/s4_2r/contact_state/accepted.pt",
    "C3": ROOT / ".local/experiments/simulation/s4_2r/contact_dynamics/selected.pt",
    "A0": ROOT / ".local/experiments/simulation/s4_2_formal/action/selected.pt",
    "B3_VAC": ROOT / ".local/experiments/simulation/s4_2_formal/bridge/selected.pt",
    "A_plus_H": ROOT
    / ".local/experiments/simulation/s4_2_formal/s4_2_7/A_plus_H.pt",
    "VA_only_bridge": ROOT
    / ".local/experiments/simulation/s4_3_pi2u/va_bridge/frozen.pt",
}
SIDECARS = {
    "B2_contact": ROOT
    / ".local/datasets/simulation/s4_3_pi1/pinch_tongs_official_tactile/sidecar.npz",
    "clean_VA27": ROOT
    / ".local/datasets/simulation/s4_3_pi2m/pinch_tongs_va_t27/sidecar.npz",
    "VAC_V27": ROOT
    / ".local/experiments/simulation/s4_3_pi2n/caches/"
    "pinch_tongs_vac_v_t27/sidecar.npz",
}


def command(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256_file(file).encode())
        digest.update(b"\0")
        digest.update(str(file.stat().st_size).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def resolve_symbolic(value: str) -> Path:
    if not value.startswith("$REPO_ROOT/"):
        raise RuntimeError(f"non-repository symbolic path: {value}")
    return ROOT / value.removeprefix("$REPO_ROOT/")


def mount_is_true_nfs() -> tuple[bool, list[dict[str, str]]]:
    output = subprocess.check_output(
        [
            "findmnt",
            "-rn",
            "-T",
            str(ROOT / ".local/experiments"),
            "-o",
            "TARGET,SOURCE,FSTYPE",
        ],
        text=True,
    )
    rows = []
    for line in output.splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) == 3:
            rows.append(
                {"target": parts[0], "source": parts[1], "filesystem": parts[2]}
            )
    passed = any(
        row["filesystem"].startswith("nfs")
        and row["source"] == "192.168.110.40:/volume2/ugreen_nas"
        for row in rows
    )
    return passed, rows


def render_human_acceptance(
    *, head: str, statistics: dict[str, Any], claim: dict[str, Any]
) -> str:
    model_lines = []
    for model in ("B0", "B_VA27", "B1", "B_HVA", "B2", "B_VAC_V"):
        row = statistics["model_statistics"][model]
        label = "VAC* (B_VAC_V)" if model == "B_VAC_V" else model
        lower, upper = row["wilson_95ci_percent"]
        model_lines.append(
            f"| {label} | {row['successes']}/200 ({row['success_percent']:.1f}%) "
            f"| [{lower:.2f}, {upper:.2f}]% |"
        )
    pair_lines = []
    for name, row in statistics["paired_comparisons"].items():
        lower, upper = row["paired_bootstrap_95ci_percentage_points"]
        pair_lines.append(
            f"| {name} | {row['risk_difference_percentage_points']:+.1f} pp "
            f"| [{lower:.1f}, {upper:.1f}] pp | "
            f"{row['mcnemar_exact_two_sided_raw_p']:.6g} | "
            f"{row['holm_adjusted_p']:.6g} | {row['classification']} |"
        )
    return "\n".join(
        [
            "# S4.3-PI2N Human Acceptance",
            "",
            f"Status: **{claim['ENGINEERING_STATUS']} — {claim['CLAIM_LEVEL']}**",
            "",
            "## Identity and scope",
            "",
            f"- Git HEAD: `{head}` on `develop/sim-benchmark`; no push, PR, merge, tag, release, new branch, or worktree.",
            "- Formal cohort: six frozen seed42 checkpoints, 200 matched PI2N_FINAL resets per model, 1,200 canonical outcomes.",
            "- VAC* is the development-selected `B_VAC_V`; DEV performance was used only for selection and is not confirmatory evidence.",
            "- X remained `NOT_RUN_NOT_JUSTIFIED`; PI2B and real-robot work were not started.",
            "",
            "## Formal model results",
            "",
            "| Model | Native success | Wilson 95% CI |",
            "|---|---:|---:|",
            *model_lines,
            "",
            "## Frozen paired comparisons",
            "",
            "| Contrast | Difference | Paired 95% CI | Raw p | Holm p | Decision |",
            "|---|---:|---:|---:|---:|---|",
            *pair_lines,
            "",
            "## Decision axes",
            "",
            f"- VAC_READOUT_DIAGNOSIS: `{claim['VAC_READOUT_DIAGNOSIS']}`",
            f"- VAC_VS_STRONG_VA: `{claim['VAC_VS_STRONG_VA']}`",
            f"- VAC_REFINEMENT_VS_B2: `{claim['VAC_REFINEMENT_VS_B2']}`",
            f"- H_INCREMENT_GIVEN_VA: `{claim['H_INCREMENT_GIVEN_VA']}`",
            f"- FULL_METHOD_VS_B0: `{claim['FULL_METHOD_VS_B0']}`",
            "",
            "Inconclusive results are not equivalence. Results are fixed-training-seed only and do not establish cross-seed, cross-task, or real-hardware generality.",
            "",
            "## Read-only acceptance commands",
            "",
            "```bash",
            "/home/wbcd/miniconda3/envs/unit/bin/python scripts/simulation/status_s4_3_pi2n.py",
            "jq . .local/artifacts/simulation/s4_3_pi2n/final_completion_audit.json",
            "jq . .local/artifacts/simulation/s4_3_pi2n/paired_statistics.json",
            "jq . .local/artifacts/simulation/s4_3_pi2n/statistics_independent_audit.json",
            "jq . .local/artifacts/simulation/s4_3_pi2n/protected_integrity_after.json",
            "jq . .local/artifacts/simulation/s4_3_pi2n/environment_integrity.json",
            "git status --short",
            "```",
            "",
            "## Stop",
            "",
            "No further training, PI2B execution, push, or real-robot work is authorized by this file.",
            "",
        ]
    )


def audit_file_manifest(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    expected = {
        row["path"]: (row["sha256"], int(row["expected_bytes"]))
        for row in manifest["files"]
    }
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_paths != set(expected):
        raise SystemExit("PI2N closeout pi05_base file-set drift")
    observed = {
        relative: {
            "sha256": sha256_file(root / relative),
            "bytes": (root / relative).stat().st_size,
        }
        for relative in sorted(expected)
    }
    if any(
        observed[relative]["sha256"] != expected[relative][0]
        or observed[relative]["bytes"] != expected[relative][1]
        for relative in expected
    ):
        raise SystemExit("PI2N closeout pi05_base content drift")
    return {
        "path": "$REPO_ROOT/" + root.relative_to(ROOT).as_posix(),
        "files": len(observed),
        "bytes": sum(row["bytes"] for row in observed.values()),
        "manifest_sha256": sha256_file(BASE_MANIFEST),
        "all_files_exact": True,
    }


def main() -> None:
    outputs = (
        PROTECTED_AFTER,
        ENVIRONMENT_AFTER,
        RETRY_LEDGER,
        PI2B_DRAFT,
        COMPLETION,
        HUMAN_ACCEPTANCE,
    )
    if any(path.exists() for path in outputs):
        raise SystemExit("refusing to overwrite PI2N closeout artifacts")
    required = (
        STARTING,
        PRE_FINAL,
        COMPLETENESS,
        EXECUTION,
        JOB_STATUS,
        STATISTICS,
        PROCESS,
        CLAIM,
        FINAL_DECISION,
        INDEPENDENT,
        PLOTS,
        REGRESSION,
        PAPER_AUDIT,
        PAPER,
        BASE_MANIFEST,
        UNIT_PROTOCOL,
    )
    if any(not path.is_file() for path in required):
        raise SystemExit("PI2N_CLOSEOUT_REQUIRED_INPUT_MISSING")
    if command("git", "branch", "--show-current") != "develop/sim-benchmark":
        raise SystemExit("PI2N closeout requires develop/sim-benchmark")
    if command("git", "status", "--short"):
        raise SystemExit("tracked worktree must be clean for PI2N closeout")

    starting = read_json(STARTING)
    pre_final = read_json(PRE_FINAL)
    completeness = read_json(COMPLETENESS)
    execution = read_json(EXECUTION)
    job = read_json(JOB_STATUS)
    statistics = read_json(STATISTICS)
    process = read_json(PROCESS)
    claim = read_json(CLAIM)
    final_decision = read_json(FINAL_DECISION)
    independent = read_json(INDEPENDENT)
    plots = read_json(PLOTS)
    regression = read_json(REGRESSION)
    paper_audit = read_json(PAPER_AUDIT)
    status_gates = {
        "pre_final_PASS": pre_final.get("status") == "PASS",
        "rollout_completeness_PASS": completeness.get("status") == "PASS"
        and completeness.get("total_canonical_outcomes") == 1200,
        "execution_PASS": execution.get("status") == "PASS"
        and len(execution.get("workers", [])) == 24,
        "job_DONE_0": job.get("state") == "DONE" and job.get("exit_code") == 0,
        "statistics_PASS": statistics.get("status") == "PASS",
        "process_metrics_PASS": process.get("status") == "PASS",
        "claim_complete": claim.get("ENGINEERING_STATUS") == "COMPLETE_VALID"
        and claim.get("CLAIM_LEVEL") == "FIXED_SEED_ONLY",
        "final_decision_complete": final_decision.get("status")
        == "COMPLETE_VALID",
        "independent_statistics_PASS": independent.get("status") == "PASS",
        "plots_PASS": plots.get("status") == "PASS",
        "regression_PASS": regression.get("status") == "PASS",
        "paper_audit_PASS": paper_audit.get("status") == "PASS"
        and paper_audit.get("updated_sha256") == sha256_file(PAPER),
    }
    if not all(status_gates.values()):
        failed = [name for name, value in status_gates.items() if not value]
        raise SystemExit("PI2N_CLOSEOUT_STATUS_GATE_FAIL: " + ",".join(failed))

    source_hashes = {
        symbolic: sha256_file(resolve_symbolic(symbolic))
        for symbolic in pre_final["sources_sha256"]
    }
    if source_hashes != pre_final["sources_sha256"]:
        raise SystemExit("PI2N closeout frozen source drift")
    checkpoint_hashes = {
        model: tree_hash(resolve_symbolic(path))
        for model, path in pre_final["checkpoint_paths"].items()
    }
    if checkpoint_hashes != pre_final["checkpoint_tree_sha256"]:
        raise SystemExit("PI2N closeout checkpoint drift")
    teacher_before = read_json(
        ARTIFACTS / "checkpoint_immutability_before.json"
    )["frozen_teacher_files"]
    teacher_after = {name: sha256_file(path) for name, path in TEACHERS.items()}
    if teacher_after != teacher_before:
        raise SystemExit("PI2N closeout teacher drift")
    expected_sidecars = {
        "B2_contact": "833db9ddb4d37534bf38a7ed0b214fee2f000bb4e489d3fa9507b4e7bca5bd8e",
        "clean_VA27": "7d51a23672273ec3ea46f947080c0c3ea66db55332522b199caab00ad4fc2126",
        "VAC_V27": "8014842ac86bfb3f44cf7d01c4198b59a6cb35c5d7348cafea7adb851164d752",
    }
    sidecar_hashes = {name: sha256_file(path) for name, path in SIDECARS.items()}
    if sidecar_hashes != expected_sidecars:
        raise SystemExit("PI2N closeout target sidecar drift")
    base_manifest = read_json(BASE_MANIFEST)
    if base_manifest.get("status") != "PASS":
        raise SystemExit("PI2N closeout pi05_base manifest is not PASS")
    base_root = resolve_symbolic(base_manifest["mirror_path"])
    base_audit = audit_file_manifest(base_root, base_manifest)
    unit_protocol = read_json(UNIT_PROTOCOL)
    expected_unit = unit_protocol["teacher"]["official_unit_files_sha256"]
    unit_after = {
        name: sha256_file(UNIT_TOKENIZER / name) for name in expected_unit
    }
    if unit_after != expected_unit:
        raise SystemExit("PI2N closeout Original UniT tokenizer drift")
    raw_hashes = {}
    for name, row in completeness["raw_artifacts"].items():
        path = resolve_symbolic(row["path"])
        raw_hashes[name] = sha256_file(path)
        if raw_hashes[name] != row["sha256"]:
            raise SystemExit(f"PI2N closeout raw rollout drift: {name}")
    mount_pass, mount_rows = mount_is_true_nfs()
    if not mount_pass:
        raise SystemExit("PI2N closeout NAS mount is not the audited NFS")
    current_environment = environment_snapshot()
    if current_environment != pre_final["environment_snapshot"]:
        raise SystemExit("PI2N closeout package environment drift")
    dexjoco_head = command("git", "rev-parse", "HEAD", cwd=ROOT / "third_party/dexjoco")
    if dexjoco_head != DEXJOCO_HEAD:
        raise SystemExit("PI2N closeout DexJoCo revision drift")
    pi2b_root = ROOT / ".local/experiments/simulation/s4_3_pi2b"
    if pi2b_root.exists():
        raise SystemExit("PI2B experiment directory exists")

    now = datetime.now(timezone.utc).isoformat()
    protected = {
        "schema": "tactile3d-unit.s4-3-pi2n-protected-integrity-after.v1",
        "status": "PASS",
        "audited_at": now,
        "checkpoint_tree_sha256": checkpoint_hashes,
        "teacher_file_sha256": teacher_after,
        "pi05_base": base_audit,
        "original_unit_tokenizer_sha256": unit_after,
        "target_sidecar_sha256": sidecar_hashes,
        "raw_rollout_sha256": raw_hashes,
        "source_sha256": source_hashes,
        "historical_artifacts_modified": False,
        "PI2B_started": False,
        "real_robot_started": False,
    }
    environment = {
        "schema": "tactile3d-unit.s4-3-pi2n-environment-integrity.v1",
        "status": "PASS",
        "audited_at": now,
        "before": pre_final["environment_snapshot"],
        "after": current_environment,
        "package_sets_unchanged": True,
        "dexjoco_revision": dexjoco_head,
        "branch": "develop/sim-benchmark",
        "tracked_worktree_clean": True,
        "local_files_tracked": command("git", "ls-files", ".local") != "",
        "mount_rows": mount_rows,
        "package_installation_performed_by_PI2N": False,
    }
    if environment["local_files_tracked"]:
        raise SystemExit("PI2N closeout found tracked .local files")
    retry = {
        "schema": "tactile3d-unit.s4-3-pi2n-retry-ledger.v1",
        "status": "PASS",
        "created_at": now,
        "canonical_workers": [
            {
                "model": row["model"],
                "seed": row["seed"],
                "infrastructure_retries": row["infrastructure_retries"],
            }
            for row in execution["workers"]
        ],
        "total_infrastructure_retries": sum(
            int(row["infrastructure_retries"]) for row in execution["workers"]
        ),
        "native_failures_and_timeouts_in_denominator": True,
        "best_of_retry_splicing": False,
        "canonical_cohort": "first complete clean frozen FINAL cohort",
    }
    if not all(
        isinstance(row["infrastructure_retries"], int)
        and row["infrastructure_retries"] >= 0
        for row in execution["workers"]
    ):
        raise SystemExit("PI2N closeout retry count integrity failed")
    candidate_manifests = {
        model: read_json(ARTIFACTS / f"{model.lower()}_checkpoint_manifest.json")
        for model in ("B_VA27", "B_VAC_V")
    }
    measured_gpu_hours = [
        row.get("gpu_hours")
        for row in candidate_manifests.values()
        if isinstance(row.get("gpu_hours"), (int, float))
    ]
    pi2b = {
        "schema": "tactile3d-unit.s4-3-pi2n-pi2b-recommendation-draft.v1",
        "status": "DRAFT_NOT_EXECUTED",
        "created_at": now,
        "training_authorized": False,
        "started": False,
        "recommended_core_models": ["B0", "B_HVA", "B_VAC_V"],
        "conditional_controls": {
            "B_VA27": "include when testing H increment under VA supervision",
            "B1": "include when testing auxiliary value under the same H input",
            "B2": "include only when replication of the original VAC readout remains a stated question",
        },
        "candidate_training_seeds": [43, 44],
        "seed_unused_status": "MUST_BE_AUDITED_BEFORE_ANY_FUTURE_AUTHORIZATION",
        "minimum_new_runs_if_authorized": 6,
        "full_six_model_new_runs_if_authorized": 12,
        "measured_seed42_gpu_hours_per_run": measured_gpu_hours,
        "cost_estimate": "use measured seed42 GPU-hours; do not assume four-GPU linear speedup",
        "formal_PI2N_decision_axes": {
            key: claim[key]
            for key in (
                "VAC_VS_STRONG_VA",
                "VAC_REFINEMENT_VS_B2",
                "H_INCREMENT_GIVEN_VA",
                "FULL_METHOD_VS_B0",
            )
        },
        "second_task_started": False,
        "real_robot_started": False,
        "world_model_started": False,
    }
    head = command("git", "rev-parse", "HEAD")
    completion = {
        "schema": "tactile3d-unit.s4-3-pi2n-final-completion-audit.v1",
        "status": "PASS",
        "completed_at": now,
        "starting_head": starting["starting_head"],
        "final_head": head,
        "branch": "develop/sim-benchmark",
        "DAG": {
            "N0": "COMPLETE_VALID",
            "N1_D": "COMPLETE_VALID",
            "N1_R": "COMPLETE_VALID",
            "N2_T": "COMPLETE_VALID",
            "N2_O": "COMPLETE_VALID",
            "N3_C": "COMPLETE_VALID",
            "N4_V": "COMPLETE_VALID",
            "N4_X": "COMPLETE_NEGATIVE_NOT_RUN_NOT_JUSTIFIED",
            "N5": "COMPLETE_VALID",
            "N6": "COMPLETE_VALID",
            "N7": "COMPLETE_VALID",
            "N8": "COMPLETE_VALID",
        },
        "decision_axes": {
            key: claim[key]
            for key in (
                "ENGINEERING_STATUS",
                "VAC_READOUT_DIAGNOSIS",
                "VAC_VS_STRONG_VA",
                "VAC_REFINEMENT_VS_B2",
                "H_INCREMENT_GIVEN_VA",
                "FULL_METHOD_VS_B0",
                "CLAIM_LEVEL",
            )
        },
        "artifacts_sha256": {
            "paired_statistics": sha256_file(STATISTICS),
            "contact_process_metrics": sha256_file(PROCESS),
            "claim_freeze": sha256_file(CLAIM),
            "final_decision": sha256_file(FINAL_DECISION),
            "statistics_independent_audit": sha256_file(INDEPENDENT),
            "plots_manifest": sha256_file(PLOTS),
            "regression_tests": sha256_file(REGRESSION),
            "paper_core_update_audit": sha256_file(PAPER_AUDIT),
            "PAPER_CORE": sha256_file(PAPER),
        },
        "push_performed": False,
        "new_branch_or_worktree": False,
        "PI2B_started": False,
        "real_robot_started": False,
    }
    acceptance = render_human_acceptance(
        head=head, statistics=statistics, claim=claim
    )
    atomic_json(PROTECTED_AFTER, protected)
    atomic_json(ENVIRONMENT_AFTER, environment)
    atomic_json(RETRY_LEDGER, retry)
    atomic_json(PI2B_DRAFT, pi2b)
    atomic_text(HUMAN_ACCEPTANCE, acceptance)
    completion["artifacts_sha256"].update(
        {
            "protected_integrity_after": sha256_file(PROTECTED_AFTER),
            "environment_integrity": sha256_file(ENVIRONMENT_AFTER),
            "retry_ledger": sha256_file(RETRY_LEDGER),
            "pi2b_recommendation_draft": sha256_file(PI2B_DRAFT),
            "HUMAN_ACCEPTANCE": sha256_file(HUMAN_ACCEPTANCE),
        }
    )
    atomic_json(COMPLETION, completion)
    print(
        json.dumps(
            {
                "status": "PASS",
                "head": head,
                "claim_level": claim["CLAIM_LEVEL"],
                "PI2B_started": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
