#!/usr/bin/env python3
"""Finalize S4.3-PI2M from frozen artifacts without launching new work."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2m"
START_HEAD = "df7fe033b5c982f07ff0e08468e4e1016d5bf225"
DEXJOCO_HEAD = "8d23b0fab23b17a58c4b55f3942e17013aaf8267"
EXPECTED_CHECKPOINTS = {
    "B0": "1e7a6ace5d69a988a8b258e1c56a24d88b077580a05b27be3f510df9ac3864f3",
    "B1": "04b59cbc3491bf4e88dc75558a08a94e5588e2fbe398d413e1766a87c7ab6f4f",
    "B2": "86c4908533ca24da06ae66f943e3605b5ccbae455441290ca8fb35514359e949",
    "BVA": "04b609d7cc89e5fffdfab8219bf34da362117a15d0c7e3d5cd4ee20a9ee4770d",
    "B_HVA": "82469f1d48b09f1c6152019512dba82f0f899f07bb0aab76bb58c0e0ba1e3a8b",
}
CHECKPOINTS = {
    "B0": ROOT / ".local/experiments/simulation/s4_3_pi0/training/pinch_tongs/s43_pi0_official_seed42/29999",
    "B1": ROOT / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/s43_pi1b_contact_tokens_seed42/29999",
    "B2": ROOT / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/s43_pi1c_contact_tokens_physical_aux_seed42/29999",
    "BVA": ROOT / ".local/experiments/simulation/s4_3_pi2u/bva/pinch_tongs/s43_pi2u_bva_seed42/29999",
    "B_HVA": ROOT / ".local/experiments/simulation/s4_3_pi2m/bhva/pinch_tongs/s43_pi2m_bhva_seed42/29999",
}
EXPECTED_S42 = {
    "s4_2": "7da07e8f5babe705d1fd643fc54dcce6db25e3636edf043aae2f7fec940a35ba",
    "s4_2_formal": "4404d9bd732999a6bec98bfcd9ea072d6215c642c60f66bfa0aa6671319035e9",
    "s4_2dr": "48bdf8a02773a898c25725d3afa46d083371e5113dee71a1f8ed34e596fa5e75",
    "s4_2ds": "528c2ecd55e529cba4604c5ecfd57466b56f9fcdeff900f86714ee9e44624d7e",
    "s4_2r": "90d801dfd541bc3d8065fb8e3a9d88d443377b51a653188290ea4fbd6a37b3eb",
}
EXPECTED_SIDECARS = {
    "B2_contact": "833db9ddb4d37534bf38a7ed0b214fee2f000bb4e489d3fa9507b4e7bca5bd8e",
    "B_HVA_corrected_VA": "7d51a23672273ec3ea46f947080c0c3ea66db55332522b199caab00ad4fc2126",
    "historical_BVA_0p32s": "c596b2f56880a969148f7cf06268ecfa9ad23bac01014be6c73ad20afd0d0612",
}
SIDECARS = {
    "B2_contact": ROOT / ".local/datasets/simulation/s4_3_pi1/pinch_tongs_official_tactile/sidecar.npz",
    "B_HVA_corrected_VA": ROOT / ".local/datasets/simulation/s4_3_pi2m/pinch_tongs_va_t27/sidecar.npz",
    "historical_BVA_0p32s": ROOT / ".local/datasets/simulation/s4_3_pi2u/pinch_tongs_va/sidecar.npz",
}
EXPECTED_RAW = {
    "B1": "8e6ac39c60d6a84e0df953f6c4076ebd55e0680b312b486a0814699e346680cf",
    "B_HVA": "078d8abb6fd933a9b4a42053212ec808098c6111de2e309e06599bc6bf69a18e",
    "B2": "4953b6fdb54d003823c30760c2cf97be09f541b0815215b367a5a0cb056a5a13",
}
RAW = {
    "B1": ARTIFACTS / "b1_raw_rollouts.json",
    "B_HVA": ARTIFACTS / "b_hva_raw_rollouts.json",
    "B2": ARTIFACTS / "b2_raw_rollouts.json",
}


def command(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(32 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(path: Path) -> str:
    rows = []
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        rows.append((file.relative_to(path).as_posix(), sha256_file(file), file.stat().st_size))
    digest = hashlib.sha256()
    for relative, sha256, size in rows:
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(sha256.encode())
        digest.update(b"\0")
        digest.update(str(size).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def symbolic(path: Path) -> str:
    return "$REPO_ROOT/" + path.resolve().relative_to(ROOT).as_posix()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def archive_existing(path: Path) -> None:
    if not path.exists():
        return
    for index in range(1, 100):
        archived = path.with_name(f"{path.stem}_event_{index:03d}{path.suffix}")
        if not archived.exists():
            shutil.move(path, archived)
            return
    raise RuntimeError(f"too many archived versions for {path}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    head = command("git", "rev-parse", "HEAD")
    branch = command("git", "branch", "--show-current")
    require(branch == "develop/sim-benchmark", f"unexpected branch: {branch}")
    require(command("git", "status", "--porcelain") == "", "tracked worktree must be clean")
    require(command("git", "rev-parse", "HEAD", cwd=ROOT / "third_party/dexjoco") == DEXJOCO_HEAD, "DexJoCo revision changed")

    completeness = read_json(ARTIFACTS / "preanalysis_completeness_audit.json")
    statistics = read_json(ARTIFACTS / "paired_statistics.json")
    claim = read_json(ARTIFACTS / "claim_freeze.json")
    independent = read_json(ARTIFACTS / "statistics_independent_audit.json")
    plots = read_json(ARTIFACTS / "plot_manifest.json")
    training = read_json(ARTIFACTS / "training_completion.json")
    target = read_json(ARTIFACTS / "corrected_target_audit.json")
    pi2b = read_json(ROOT / "configs/simulation/s4_3_pi2b_recommendation.json")
    for name, payload in {
        "completeness": completeness,
        "statistics": statistics,
        "independent statistics": independent,
        "plots": plots,
        "training": training,
        "target": target,
    }.items():
        require(payload.get("status") == "PASS", f"{name} is not PASS")
    require(claim["ENGINEERING_STATUS"] == "COMPLETE_VALID", "claim engineering status")
    require(claim["PRIMARY_TARGET_EFFECT"] == "VA_TARGET_ADVANTAGE_CONFIRMED", "primary claim drift")
    require(pi2b["status"] == "DRAFT_NOT_EXECUTED" and not pi2b["training_authorized"], "PI2B draft status")
    require(not (ROOT / ".local/experiments/simulation/s4_3_pi2b").exists(), "PI2B experiment directory exists")

    paper = ROOT / ".local/paper/PAPER_CORE.md"
    paper_text = paper.read_text()
    for required in (
        "PI2M_status: COMPLETE_VALID_FIXED_SEED",
        "B2−B_HVA (primary)",
        "VA_TARGET_ADVANTAGE_CONFIRMED",
        "+16 rows ≈ 0.32 s",
        "DRAFT_NOT_EXECUTED",
    ):
        require(required in paper_text, f"PAPER_CORE missing {required}")

    checkpoint_hashes = {name: tree_hash(path) for name, path in CHECKPOINTS.items()}
    s42_hashes = {
        name: tree_hash(ROOT / ".local/experiments/simulation" / name)
        for name in EXPECTED_S42
    }
    sidecar_hashes = {name: sha256_file(path) for name, path in SIDECARS.items()}
    raw_hashes = {name: sha256_file(path) for name, path in RAW.items()}
    e_t_path = ROOT / ".local/experiments/simulation/s4_2r/contact_state/accepted.pt"
    e_t_hash = sha256_file(e_t_path)
    require(checkpoint_hashes == EXPECTED_CHECKPOINTS, "protected checkpoint hash drift")
    require(s42_hashes == EXPECTED_S42, "protected S4.2 tree hash drift")
    require(sidecar_hashes == EXPECTED_SIDECARS, "target/contact sidecar hash drift")
    require(raw_hashes == EXPECTED_RAW, "canonical raw rollout hash drift")
    require(e_t_hash == "3d4519195e7a0d9f5af63399840a48121d5f614ea587c80b9461fc696a372a19", "E_T hash drift")

    now = datetime.now(timezone.utc).isoformat()
    protected = {
        "schema": "tactile3d-unit.s4-3-pi2m-protected-integrity-after.v1",
        "status": "PASS",
        "audited_at": now,
        "repo": {"branch": branch, "head": head, "dexjoco_head": DEXJOCO_HEAD, "tracked_worktree_clean": True},
        "checkpoint_tree_sha256": checkpoint_hashes,
        "checkpoint_paths": {name: symbolic(path) for name, path in CHECKPOINTS.items()},
        "E_T": {"path": symbolic(e_t_path), "sha256": e_t_hash},
        "S4_2_tree_sha256": s42_hashes,
        "sidecar_sha256": sidecar_hashes,
        "raw_rollout_sha256": raw_hashes,
        "historical_BVA": {
            "target_offset_rows": 16,
            "physical_horizon_seconds": 0.32,
            "modified_by_PI2M": False,
            "matched_horizon_control_for_B2": False,
        },
        "corrected_B_HVA_target": {
            "target_offset_control_ticks": 27,
            "physical_horizon_seconds": 0.54,
            "valid_identity_equal_B2": True,
        },
        "PI2B_started": False,
        "push_performed": False,
    }

    regression = {
        "schema": "tactile3d-unit.s4-3-pi2m-regression-tests.v2",
        "status": "PASS",
        "applicable_suites": [
            {"environment": "unit", "tests": "PI2M + PI2U + PI1", "result": "34 passed", "exit_code": 0},
            {"environment": "openpi CPU", "tests": "PI2M mode-contract", "result": "18 passed", "exit_code": 0},
            {"environment": "tactile-unit-dexjoco CPU", "tests": "rollout transport/runtime", "result": "7 passed", "exit_code": 0},
            {"environment": "tactile-unit-dexjoco EGL GPU1 after idle scan/lock", "tests": "real pinch_tongs reset/step/RGB/contact", "result": "1 passed; 116 deprecation warnings", "exit_code": 0},
            {"environment": "unit CPU", "tests": "simulation contact models + S4.2R contact dynamics", "result": "12 passed", "exit_code": 0},
        ],
        "environment_partition_event": {
            "status": "RECORDED_NOT_ALGORITHM_FAILURE",
            "detail": "A mixed DexJoCo command first collected two torch suites in an environment without torch (exit2); the suites were rerun in their supported unit environment and passed. OSMesa was unavailable, so the one real-render runtime test was rerun under EGL only after a two-snapshot GPU0-3 idle audit and GPU1 advisory lock.",
        },
        "frozen_runtime_evidence": {
            "mode_parity_sha256": sha256_file(ARTIFACTS / "mode_parity.json"),
            "production_smoke_sha256": sha256_file(ARTIFACTS / "production_smoke.json"),
            "pretrain_gate_sha256": sha256_file(ARTIFACTS / "pretrain_gate_launch.json"),
        },
        "historical_artifacts_written_by_regression": False,
    }

    pi2b_local = ARTIFACTS / "pi2b_recommendation_draft.json"
    atomic_json(pi2b_local, pi2b)
    archive_existing(ARTIFACTS / "protected_integrity_after.json")
    atomic_json(ARTIFACTS / "protected_integrity_after.json", protected)
    archive_existing(ARTIFACTS / "regression_tests.json")
    atomic_json(ARTIFACTS / "regression_tests.json", regression)

    comparisons = statistics["paired_comparisons"]
    final = {
        "schema": "tactile3d-unit.s4-3-pi2m-final-decision.v2",
        "status": "COMPLETE_VALID",
        "completed_at": now,
        "repo": {"starting_head": START_HEAD, "ending_head": head, "branch": branch, "tracked_worktree_clean": True, "NO_PUSH": True},
        "ENGINEERING_STATUS": claim["ENGINEERING_STATUS"],
        "PRIMARY_TARGET_EFFECT": claim["PRIMARY_TARGET_EFFECT"],
        "VA_AUX_VS_NO_AUX": claim["VA_AUX_VS_NO_AUX"],
        "CONTACT_AUX_REPLICATION": claim["CONTACT_AUX_REPLICATION"],
        "CLAIM_LEVEL": claim["CLAIM_LEVEL"],
        "model_statistics": statistics["model_statistics"],
        "paired_comparisons": comparisons,
        "target_gate": {
            "status": "PASS",
            "B_HVA_target": "clean frozen VA-only teacher, episode-local raw/control t-to-t+27 = 0.54 s",
            "valid_rows_equal_B2": True,
            "historical_BVA_preserved": "+16 rows = 0.32 s; context only",
        },
        "B_HVA_training": {
            "seed": 42,
            "steps": 30000,
            "global_batch_size": 32,
            "lambda_phys": training["lambda_phys"],
            "training_wall_seconds": training["training_wall_seconds"],
            "gpu_hours": training["gpu_hours"],
            "checkpoint_tree_sha256": training["checkpoint_tree_sha256"],
            "params_tree_sha256": training["params_tree_sha256"],
        },
        "evaluation": {"seed": 7, "triple_aligned_resets": 200, "canonical_rollouts": 600},
        "recovery_limitation": completeness["recovery_provenance"],
        "allowed_wording": [
            "Under matched online inputs, auxiliary-head architecture, and a 0.54-s target horizon, the clean VA-only target outperformed the Contact/VAC target in the tested fixed-training-seed evaluation.",
            "The VA-target auxiliary improved over the no-auxiliary Contact-State policy in this fixed-seed cohort; the Contact/VAC auxiliary replication versus the same baseline was inconclusive.",
        ],
        "prohibited_wording": [
            "tactile or Contact-State is useless",
            "VA targets universally outperform Contact/VAC targets",
            "cross-modal alignment is unnecessary",
            "the effect is stable across training seeds, tasks, or real hardware",
            "an inconclusive B2-B1 comparison proves equivalence",
        ],
        "artifact_sha256": {
            "paired_statistics": sha256_file(ARTIFACTS / "paired_statistics.json"),
            "claim_freeze": sha256_file(ARTIFACTS / "claim_freeze.json"),
            "independent_statistics_audit": sha256_file(ARTIFACTS / "statistics_independent_audit.json"),
            "preanalysis_completeness": sha256_file(ARTIFACTS / "preanalysis_completeness_audit.json"),
            "protected_integrity_after": sha256_file(ARTIFACTS / "protected_integrity_after.json"),
            "regression_tests": sha256_file(ARTIFACTS / "regression_tests.json"),
            "pi2b_recommendation_draft": sha256_file(pi2b_local),
            "PAPER_CORE": sha256_file(paper),
        },
        "PAPER_CORE_updated": True,
        "PI2B": {"status": "DRAFT_NOT_EXECUTED", "minimum_new_runs": 6, "full_matrix_new_runs": 11, "started": False},
        "push_performed": False,
        "new_branch_or_worktree": False,
    }
    archive_existing(ARTIFACTS / "final_decision.json")
    atomic_json(ARTIFACTS / "final_decision.json", final)

    commits = command("git", "log", "--format=- `%h` %s", f"{START_HEAD}..{head}")
    acceptance = f"""# S4.3-PI2M Human Acceptance

Status: **COMPLETE_VALID — FIXED_SEED_ONLY**

## Identity and scope

- Git: `develop/sim-benchmark`, start `{START_HEAD}`, end `{head}`; tracked worktree clean at finalization.
- DexJoCo: `{DEXJOCO_HEAD}`. No push, PR, merge, tag, release, branch, or worktree was created.
- Only B_HVA seed42 was newly trained: 30,000 steps, global batch32, lambda `{training['lambda_phys']}`, wall/GPU time `{training['training_duration']}` / `{training['gpu_hours']:.4f}` GPU-hours.
- B_HVA checkpoint tree: `{training['checkpoint_tree_sha256']}`; params tree: `{training['params_tree_sha256']}`.
- PI2B is `DRAFT_NOT_EXECUTED`; zero PI2B runs were started.

## Mandatory horizon gate

- PASS: the frozen clean VA teacher was itself trained at raw/control `t→t+27 = 0.54 s`.
- B_HVA uses a PI2M-only corrected `+27` cache with exactly B2's valid-row identity, per-episode tail-27 masking, no cross-episode transition, no interpolation, no Contact/tactile target input, no B3 projector, and finite targets.
- Historical BVA remains byte-identical and explicitly historical: `+16 rows = 0.32 s`. It is not a matched-horizon B2 control.

## Formal seed7 result (200 aligned resets each)

| Model | Success | Wilson 95% CI |
|---|---:|---:|
| B1 | 34/200 (17.0%) | [12.43, 22.82]% |
| B_HVA | 85/200 (42.5%) | [35.85, 49.43]% |
| B2 | 46/200 (23.0%) | [17.71, 29.31]% |

| Contrast | Delta | Paired 95% CI | Raw p | Holm p | Frozen decision |
|---|---:|---:|---:|---:|---|
| B2−B_HVA | −19.5 pp | [−29, −10] pp | 0.000130822 | 0.000261645 | VA_TARGET_ADVANTAGE_CONFIRMED |
| B_HVA−B1 | +25.5 pp | [17, 34] pp | 7.24537e−8 | 2.17361e−7 | positive confirmed |
| B2−B1 | +6.0 pp | [−2, 14] pp | 0.168643 | 0.168643 | INCONCLUSIVE |

The interrupted B2 attempt was never spliced into the result. One complete clean 200-reset B2 replay is canonical. Among the interrupted attempt's first 31 completed resets, 18 outcomes agreed and 13 differed on replay despite identical reset IDs. The asynchronous evaluator is therefore not outcome-deterministic across retries; this is a disclosed limitation, not a selection gate.

## Claim freeze

- `ENGINEERING_STATUS=COMPLETE_VALID`
- `PRIMARY_TARGET_EFFECT=VA_TARGET_ADVANTAGE_CONFIRMED`
- `VA_AUX_VS_NO_AUX=positive_confirmed`
- `CONTACT_AUX_REPLICATION=inconclusive`
- `CLAIM_LEVEL=FIXED_SEED_ONLY`

Allowed wording and prohibited extrapolations are machine-bound in `final_decision.json`. In particular, this result does not show that tactile input is useless, that Contact targets are universally worse, that alignment is unnecessary, or that the result generalizes across seeds/tasks/hardware.

## Acceptance paths

- `paired_statistics.json`, `claim_freeze.json`, `statistics_independent_audit.json`
- `protected_integrity_after.json`, `regression_tests.json`, `final_decision.json`
- `pi2b_recommendation_draft.json`, `plots/`, and `.local/paper/PAPER_CORE.md`

## Task commits

{commits}

No further action is authorized by this acceptance file.
"""
    archive_existing(ARTIFACTS / "HUMAN_ACCEPTANCE.md")
    atomic_text(ARTIFACTS / "HUMAN_ACCEPTANCE.md", acceptance)
    print(json.dumps({"status": "PASS", "head": head, "protected": protected["status"], "final": final["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
