#!/usr/bin/env python3
"""Bind PI2N FINAL to six frozen checkpoints before formal performance access."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2n"
OUTPUT = ARTIFACTS / "pre_final_freeze.json"
EVALUATION_PROTOCOL = ROOT / "configs/simulation/s4_3_pi2n_evaluation_protocol.json"
RUNTIME_V2 = ARTIFACTS / "runtime_protocol_v2.json"
FINAL_MANIFEST = ARTIFACTS / "final_reset_manifest.json"
DEVELOPMENT_RESULTS = ARTIFACTS / "development_results.json"
DEVELOPMENT_EXECUTION = ARTIFACTS / "development_gpu_execution.json"
DEVELOPMENT_MANIFEST = ARTIFACTS / "development_manifest.json"
PRE_DEV_FREEZE = ARTIFACTS / "pre_dev_freeze.json"
EXPOSURE_LEDGER = ARTIFACTS / "reset_exposure_ledger.json"
SELECTION = ARTIFACTS / "vac_star_selection.json"
ELIGIBILITY = ARTIFACTS / "candidate_eligibility.json"
REFINEMENT = ARTIFACTS / "refinement_decision.json"
PROTECTED = ARTIFACTS / "checkpoint_immutability_before.json"
MODELS = ("B0", "B_VA27", "B1", "B_HVA", "B2", "B_VAC_V")
CHECKPOINT_PATHS = {
    "B0": ROOT
    / ".local/experiments/simulation/s4_3_pi0/training/pinch_tongs/"
    "s43_pi0_official_seed42/29999",
    "B_VA27": ROOT
    / ".local/experiments/simulation/s4_3_pi2n/runs/B_VA27/pinch_tongs/"
    "s43_pi2n_b_va27_seed42/29999",
    "B1": ROOT
    / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/"
    "s43_pi1b_contact_tokens_seed42/29999",
    "B_HVA": ROOT
    / ".local/experiments/simulation/s4_3_pi2m/bhva/pinch_tongs/"
    "s43_pi2m_bhva_seed42/29999",
    "B2": ROOT
    / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/"
    "s43_pi1c_contact_tokens_physical_aux_seed42/29999",
    "B_VAC_V": ROOT
    / ".local/experiments/simulation/s4_3_pi2n/runs/B_VAC_V/pinch_tongs/"
    "s43_pi2n_b_vac_v_seed42/29999",
}
CHECKPOINT_MANIFESTS = {
    "B_VA27": ARTIFACTS / "b_va27_checkpoint_manifest.json",
    "B1": ROOT / ".local/artifacts/simulation/s4_3_pi1/pi1b_checkpoint_manifest.json",
    "B_HVA": ROOT
    / ".local/artifacts/simulation/s4_3_pi2m/bhva_checkpoint_manifest.json",
    "B2": ROOT / ".local/artifacts/simulation/s4_3_pi1/pi1c_checkpoint_manifest.json",
    "B_VAC_V": ARTIFACTS / "b_vac_v_checkpoint_manifest.json",
}
EXPECTED_PROTECTED = {
    "B0": "1e7a6ace5d69a988a8b258e1c56a24d88b077580a05b27be3f510df9ac3864f3",
    "B1": "04b59cbc3491bf4e88dc75558a08a94e5588e2fbe398d413e1766a87c7ab6f4f",
    "B_HVA": "82469f1d48b09f1c6152019512dba82f0f899f07bb0aab76bb58c0e0ba1e3a8b",
    "B2": "86c4908533ca24da06ae66f943e3605b5ccbae455441290ca8fb35514359e949",
}
SOURCE_FILES = (
    "scripts/simulation/freeze_s4_3_pi2n_final.py",
    "scripts/simulation/run_s4_3_pi2n_final.py",
    "scripts/simulation/launch_s4_3_pi2n_final.py",
    "scripts/simulation/analyze_s4_3_pi2n_final.py",
    "scripts/simulation/audit_s4_3_pi2n_final_statistics.py",
    "scripts/simulation/visualize_s4_3_pi2n_final.py",
    "scripts/simulation/finalize_s4_3_pi2n.py",
    "scripts/simulation/serve_s4_3_pi2n_policy.py",
    "scripts/simulation/run_s4_3_pi2u_eval.py",
    "scripts/simulation/evaluate_s4_3_pi1d_augmented.py",
    "scripts/simulation/serve_s4_3_pi1_contact_state.py",
    "gr00t/simulation/pi05_tactile_unit.py",
    "gr00t/simulation/s4_3_pi1.py",
    "configs/simulation/s4_3_pi2n_evaluation_protocol.json",
    "third_party/dexjoco/configs/rand_obj/pinch_tongs.yaml",
)
FROZEN_INPUTS = {
    "evaluation_protocol": EVALUATION_PROTOCOL,
    "runtime_protocol_v2": RUNTIME_V2,
    "final_reset_manifest": FINAL_MANIFEST,
    "development_results": DEVELOPMENT_RESULTS,
    "development_gpu_execution": DEVELOPMENT_EXECUTION,
    "development_manifest": DEVELOPMENT_MANIFEST,
    "pre_dev_freeze": PRE_DEV_FREEZE,
    "reset_exposure_ledger": EXPOSURE_LEDGER,
    "vac_star_selection": SELECTION,
    "candidate_eligibility": ELIGIBILITY,
    "refinement_decision": REFINEMENT,
    "checkpoint_immutability_before": PROTECTED,
    **{
        f"{model}_checkpoint_manifest": path
        for model, path in CHECKPOINT_MANIFESTS.items()
    },
}
CONDA_ROOT = Path(sys.executable).resolve().parents[3]
ENVIRONMENTS = {
    "unit": CONDA_ROOT / "envs/unit/bin/python",
    "openpi": CONDA_ROOT / "envs/openpi/bin/python",
    "tactile-unit-dexjoco": CONDA_ROOT / "envs/tactile-unit-dexjoco/bin/python",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(file.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256_file(file).encode())
        digest.update(b"\0")
        digest.update(str(file.stat().st_size).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def symbolic(path: Path) -> str:
    return "$REPO_ROOT/" + path.relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def normalized_pip_freeze_sha256(python: Path) -> str:
    output = subprocess.check_output(
        [str(python), "-m", "pip", "freeze"], cwd=ROOT, text=True
    ).splitlines()
    normalized = sorted(line for line in output if not line.startswith("-e "))
    return hashlib.sha256(("\n".join(normalized) + "\n").encode()).hexdigest()


def environment_snapshot() -> dict[str, Any]:
    snapshot = {}
    for name, python in ENVIRONMENTS.items():
        prefix = python.parents[1]
        history = prefix / "conda-meta/history"
        snapshot[name] = {
            "python": str(python),
            "python_version": subprocess.check_output(
                [str(python), "--version"], text=True, stderr=subprocess.STDOUT
            ).strip(),
            "conda_history_sha256": sha256_file(history),
            "normalized_pip_freeze_sha256": normalized_pip_freeze_sha256(python),
            "editable_entries_excluded": True,
        }
    return snapshot


def no_final_outputs() -> bool:
    protected = (
        ARTIFACTS / "final_raw",
        ARTIFACTS / "final_gpu_execution.json",
        ARTIFACTS / "rollout_completeness.json",
        ARTIFACTS / "paired_statistics.json",
        ARTIFACTS / "contact_process_metrics.json",
        ARTIFACTS / "claim_freeze.json",
        ARTIFACTS / "final_decision.json",
        ARTIFACTS / "final_launch.json",
        ARTIFACTS / "final_job_status.json",
        ARTIFACTS / "final_heartbeat.json",
        ROOT / ".local/logs/simulation/s4_3_pi2n/final",
        ROOT / ".local/logs/simulation/s4_3_pi2n/final_orchestrator.log",
        ROOT / ".local/cache/simulation/s4_3_pi2n/final",
        ROOT / ".local/tmp/s43n_final",
    )
    return not any(path.exists() for path in protected)


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit("refusing to overwrite PI2N pre-FINAL freeze")
    required_files = (*FROZEN_INPUTS.values(), *(ROOT / row for row in SOURCE_FILES))
    if any(not path.is_file() for path in required_files) or any(
        not path.is_dir() for path in CHECKPOINT_PATHS.values()
    ):
        raise SystemExit("PI2N_PRE_FINAL_REQUIRED_INPUT_MISSING")
    if not no_final_outputs():
        raise SystemExit("PI2N FINAL performance or launch state already exists")
    if subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    ).strip():
        raise SystemExit("tracked worktree must be clean before PI2N FINAL freeze")
    if subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip() != "develop/sim-benchmark":
        raise SystemExit("PI2N FINAL freeze requires develop/sim-benchmark")

    evaluation = read_json(EVALUATION_PROTOCOL)
    runtime = read_json(RUNTIME_V2)
    final_manifest = read_json(FINAL_MANIFEST)
    development = read_json(DEVELOPMENT_RESULTS)
    development_execution = read_json(DEVELOPMENT_EXECUTION)
    development_manifest = read_json(DEVELOPMENT_MANIFEST)
    pre_dev_freeze = read_json(PRE_DEV_FREEZE)
    exposure_ledger = read_json(EXPOSURE_LEDGER)
    selection = read_json(SELECTION)
    eligibility = read_json(ELIGIBILITY)
    refinement = read_json(REFINEMENT)
    protected = read_json(PROTECTED)
    checkpoint_manifests = {
        model: read_json(path) for model, path in CHECKPOINT_MANIFESTS.items()
    }
    expected_hashes = {
        **EXPECTED_PROTECTED,
        "B_VA27": checkpoint_manifests["B_VA27"].get("checkpoint_tree_sha256"),
        "B_VAC_V": checkpoint_manifests["B_VAC_V"].get("checkpoint_tree_sha256"),
    }
    live_hashes = {
        model: checkpoint_tree_sha256(path)
        for model, path in CHECKPOINT_PATHS.items()
    }
    initial = protected["live_migrated_policy_checkpoints"]
    final_protocol = evaluation["cohorts"]["PI2N_FINAL"]
    gates = {
        "evaluation_protocol_frozen": evaluation.get("status")
        == "FROZEN_BEFORE_NEW_POLICY_TRAINING",
        "runtime_v2_PASS": runtime.get("status") == "PASS"
        and runtime.get("scientific_rollouts_authorized") is True
        and runtime.get("official_async_evaluator_retained") is True,
        "development_complete_before_final": development.get("status") == "PASS"
        and development.get("total_canonical_outcomes") == 90
        and development.get("final_performance_accessed") is False
        and development_execution.get("status") == "PASS"
        and development_manifest.get("status") == "PASS"
        and development_manifest.get("policy_performance_seen") is False
        and pre_dev_freeze.get("status") == "PASS",
        "reset_exposure_audit_unchanged": exposure_ledger.get("status") == "PASS"
        and all(
            count == 0
            for count in exposure_ledger.get("overlap_counts", {}).values()
        ),
        "vac_star_exact": selection.get("status") == "PASS"
        and selection.get("selected_vac_star") == "B_VAC_V"
        and selection.get("development_result_sha256")
        == sha256_file(DEVELOPMENT_RESULTS)
        and selection.get("final_performance_accessed") is False,
        "X_remains_not_authorized": refinement.get("route")
        == "NOT_RUN_NOT_JUSTIFIED"
        and refinement.get("X_policy_authorized") is False
        and eligibility["candidates"]["X"]["eligible"] is False,
        "final_manifest_exact_unseen": final_manifest.get("status") == "PASS"
        and final_manifest.get("cohort") == "PI2N_FINAL"
        and final_manifest.get("episodes") == 200
        and final_manifest.get("all_reset_identities_unique") is True
        and final_manifest.get("policy_performance_seen") is False,
        "formal_protocol_exact": final_protocol.get("models")
        == ["B0", "B_VA27", "B1", "B_HVA", "B2", "VAC_STAR"]
        and final_protocol.get("seed_blocks") == [12, 13, 14, 15]
        and final_protocol.get("resets_per_block") == 50
        and evaluation["formal_statistics"].get("paired_bootstrap_samples")
        == 100_000
        and evaluation["formal_statistics"].get("holm_family_size") == 6,
        "candidate_completion_manifests_PASS": all(
            checkpoint_manifests[model].get("status") == "PASS"
            and checkpoint_manifests[model].get("frozen") is True
            and checkpoint_manifests[model].get("optimizer_steps") == 30_000
            for model in ("B_VA27", "B_VAC_V")
        ),
        "historical_checkpoint_manifests_PASS": all(
            checkpoint_manifests[model].get("status") == "PASS"
            and checkpoint_manifests[model].get("frozen") is True
            and checkpoint_manifests[model].get("optimizer_steps") == 30_000
            and checkpoint_manifests[model].get("checkpoint_tree_sha256")
            == EXPECTED_PROTECTED[model]
            for model in ("B1", "B_HVA", "B2")
        ),
        "protected_hashes_match_starting_audit": all(
            initial[model].get("tree_sha256") == EXPECTED_PROTECTED[model]
            and initial[model].get("matches_frozen") is True
            for model in EXPECTED_PROTECTED
        ),
        "all_live_checkpoint_trees_exact": live_hashes == expected_hashes,
        "selected_checkpoint_hash_exact": selection.get(
            "selected_checkpoint_tree_sha256"
        )
        == expected_hashes["B_VAC_V"],
        "no_final_outputs_before_freeze": no_final_outputs(),
    }
    if not all(gates.values()):
        failed = [name for name, value in gates.items() if not value]
        raise SystemExit("PI2N_PRE_FINAL_FREEZE_GATE_FAIL: " + ",".join(failed))

    payload = {
        "schema": "tactile3d-unit.s4-3-pi2n-pre-final-freeze.v1",
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "formal_models": list(MODELS),
        "formal_display_models": [
            "B0",
            "B_VA27",
            "B1",
            "B_HVA",
            "B2",
            "VAC_STAR",
        ],
        "vac_star": "B_VAC_V",
        "episodes_per_model": 200,
        "total_canonical_outcomes": 1200,
        "seed_blocks": [12, 13, 14, 15],
        "episodes_per_block": 50,
        "maximum_parallel_heavy_workers": 4,
        "official_async_evaluator": True,
        "replan_ratio": 0.8,
        "native_failure_or_timeout_in_denominator": True,
        "no_best_of_retry_splicing": True,
        "retry": runtime["retry_contract"],
        "required_server_environment": runtime["required_server_environment"],
        "environment_snapshot": environment_snapshot(),
        "checkpoint_paths": {
            model: symbolic(path) for model, path in CHECKPOINT_PATHS.items()
        },
        "checkpoint_tree_sha256": live_hashes,
        "frozen_input_files": {
            name: {"path": symbolic(path), "sha256": sha256_file(path)}
            for name, path in FROZEN_INPUTS.items()
        },
        "sources_sha256": {
            symbolic(ROOT / relative): sha256_file(ROOT / relative)
            for relative in SOURCE_FILES
        },
        "final_reset_sequence_sha256": final_manifest[
            "ordered_reset_sequence_sha256"
        ],
        "formal_comparisons": evaluation["formal_comparisons"],
        "formal_statistics": evaluation["formal_statistics"],
        "final_performance_seen_before_freeze": False,
        "gates": {name: "PASS" for name in gates},
    }
    atomic_json(OUTPUT, payload)
    print(
        json.dumps(
            {
                "status": "PASS",
                "formal_models": payload["formal_models"],
                "outcomes": payload["total_canonical_outcomes"],
                "final_performance_seen": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
