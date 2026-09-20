#!/usr/bin/env python3
"""Freeze every PI2N development-evaluation input before performance access."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2n"
OUTPUT = ARTIFACTS / "pre_dev_freeze.json"
EVALUATION_PROTOCOL = ROOT / "configs/simulation/s4_3_pi2n_evaluation_protocol.json"
CANDIDATE_PROTOCOL = ROOT / "configs/simulation/s4_3_pi2n_candidate_protocol.json"
RUNTIME_V2 = ARTIFACTS / "runtime_protocol_v2.json"
DEVELOPMENT_MANIFEST = ARTIFACTS / "development_manifest.json"
FINAL_MANIFEST = ARTIFACTS / "final_reset_manifest.json"
EXPOSURE_LEDGER = ARTIFACTS / "reset_exposure_ledger.json"
CANDIDATE_ELIGIBILITY = ARTIFACTS / "candidate_eligibility.json"
REFINEMENT_DECISION = ARTIFACTS / "refinement_decision.json"

MODELS = ("B_HVA", "B2", "B_VAC_V")
NEW_VAC_CANDIDATES = ("B_VAC_V",)
SEED_BLOCKS = (9, 10, 11)
EPISODES_PER_BLOCK = 10
EXECUTION_ORDER = {
    9: ("B_HVA", "B2", "B_VAC_V"),
    10: ("B2", "B_VAC_V", "B_HVA"),
    11: ("B_VAC_V", "B_HVA", "B2"),
}
CHECKPOINT_MANIFESTS = {
    "B_HVA": ROOT / ".local/artifacts/simulation/s4_3_pi2m/bhva_checkpoint_manifest.json",
    "B2": ROOT / ".local/artifacts/simulation/s4_3_pi1/pi1c_checkpoint_manifest.json",
    "B_VAC_V": ARTIFACTS / "b_vac_v_checkpoint_manifest.json",
}
CHECKPOINT_PATHS = {
    "B_HVA": ROOT
    / ".local/experiments/simulation/s4_3_pi2m/bhva/pinch_tongs/"
    "s43_pi2m_bhva_seed42/29999",
    "B2": ROOT
    / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/"
    "s43_pi1c_contact_tokens_physical_aux_seed42/29999",
    "B_VAC_V": Path(
        "/mnt/ugreen_nas/storage/UniT_storage/experiments/simulation/s4_3_pi2n/"
        "runs/B_VAC_V/pinch_tongs/s43_pi2n_b_vac_v_seed42/29999"
    ),
}
SOURCE_FILES = (
    "scripts/simulation/run_s4_3_pi2n_development.py",
    "scripts/simulation/launch_s4_3_pi2n_development.py",
    "scripts/simulation/freeze_s4_3_pi2n_development.py",
    "scripts/simulation/serve_s4_3_pi2n_policy.py",
    "scripts/simulation/run_s4_3_pi2u_eval.py",
    "scripts/simulation/evaluate_s4_3_pi1d_augmented.py",
    "scripts/simulation/serve_s4_3_pi1_contact_state.py",
    "gr00t/simulation/pi05_tactile_unit.py",
    "gr00t/simulation/s4_3_pi1.py",
    "configs/simulation/s4_3_pi2n_evaluation_protocol.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def checkpoint_tree_sha256(path: Path) -> str:
    rows = [
        (
            file.relative_to(path).as_posix(),
            sha256_file(file),
            file.stat().st_size,
        )
        for file in sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    ]
    digest = hashlib.sha256()
    for relative, value, size in rows:
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(value.encode())
        digest.update(b"\0")
        digest.update(str(size).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def load_pass(path: Path, accepted: tuple[str, ...] = ("PASS",)) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("status") not in accepted:
        raise SystemExit(f"required artifact is not accepted: {path}")
    return payload


def no_development_performance_outputs() -> bool:
    protected = (
        ARTIFACTS / "development_gpu_execution.json",
        ARTIFACTS / "development_results.json",
        ARTIFACTS / "vac_star_selection.json",
        ARTIFACTS / "development_raw",
        ROOT / ".local/cache/simulation/s4_3_pi2n/development",
        ROOT / ".local/logs/simulation/s4_3_pi2n/development",
        ROOT / ".local/tmp/s43n_dev",
    )
    return not any(path.exists() for path in protected)


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit("refusing to overwrite PI2N pre-DEV freeze")
    required = (
        EVALUATION_PROTOCOL,
        CANDIDATE_PROTOCOL,
        RUNTIME_V2,
        DEVELOPMENT_MANIFEST,
        FINAL_MANIFEST,
        EXPOSURE_LEDGER,
        CANDIDATE_ELIGIBILITY,
        REFINEMENT_DECISION,
        *CHECKPOINT_MANIFESTS.values(),
        *(ROOT / relative for relative in SOURCE_FILES),
    )
    if any(not path.is_file() for path in required) or any(
        not path.is_dir() for path in CHECKPOINT_PATHS.values()
    ):
        raise SystemExit("PI2N_PRE_DEV_REQUIRED_INPUT_MISSING")
    if not no_development_performance_outputs():
        raise SystemExit("PI2N development performance already exists")

    evaluation = load_pass(
        EVALUATION_PROTOCOL, ("FROZEN_BEFORE_NEW_POLICY_TRAINING",)
    )
    candidate_protocol = load_pass(
        CANDIDATE_PROTOCOL, ("FROZEN_BEFORE_NEW_POLICY_TRAINING",)
    )
    runtime = load_pass(RUNTIME_V2)
    development = load_pass(DEVELOPMENT_MANIFEST)
    final = load_pass(FINAL_MANIFEST)
    exposure = load_pass(EXPOSURE_LEDGER)
    eligibility = load_pass(CANDIDATE_ELIGIBILITY, ("FROZEN_AFTER_ONE_SHOT_ROUTE",))
    refinement = load_pass(REFINEMENT_DECISION, ("COMPLETE_VALID",))
    checkpoint_payloads = {
        model: load_pass(path) for model, path in CHECKPOINT_MANIFESTS.items()
    }
    live_checkpoint_hashes = {
        model: checkpoint_tree_sha256(path)
        for model, path in CHECKPOINT_PATHS.items()
    }

    dev_protocol = evaluation["cohorts"]["PI2N_DEV"]
    gates = {
        "runtime_v2_authorizes_scientific_rollouts": runtime.get(
            "scientific_rollouts_authorized"
        )
        is True,
        "official_async_evaluator_retained": runtime.get(
            "official_async_evaluator_retained"
        )
        is True,
        "development_exact_30_reset_manifest": development.get("cohort")
        == "PI2N_DEV"
        and development.get("episodes") == 30
        and development.get("all_reset_identities_unique") is True,
        "development_seed_blocks_exact": dev_protocol.get("seed_blocks")
        == list(SEED_BLOCKS)
        and dev_protocol.get("resets_per_block") == EPISODES_PER_BLOCK,
        "development_performance_unseen": development.get("policy_performance_seen")
        is False,
        "final_performance_unseen": final.get("policy_performance_seen") is False,
        "reset_exposure_disjoint": all(
            count == 0 for count in exposure.get("overlap_counts", {}).values()
        ),
        "only_B_VAC_V_is_new_vac_candidate": NEW_VAC_CANDIDATES
        == ("B_VAC_V",)
        and eligibility["candidates"]["B_VAC_V"]["eligible"] is True,
        "X_not_authorized_or_replaced": eligibility["candidates"]["X"][
            "eligible"
        ]
        is False
        and eligibility["candidates"]["X"]["policy_run_authorized"] is False
        and refinement.get("X_policy_authorized") is False
        and refinement.get("route") == "NOT_RUN_NOT_JUSTIFIED",
        "all_dev_checkpoint_manifests_frozen": all(
            payload.get("frozen") is True
            and payload.get("optimizer_steps") == 30_000
            and isinstance(payload.get("checkpoint_tree_sha256"), str)
            and len(payload["checkpoint_tree_sha256"]) == 64
            for payload in checkpoint_payloads.values()
        ),
        "all_live_checkpoint_trees_match_manifests": all(
            live_checkpoint_hashes[model]
            == checkpoint_payloads[model]["checkpoint_tree_sha256"]
            for model in MODELS
        ),
        "candidate_protocol_final_only": candidate_protocol["common_training"].get(
            "checkpoint_selection"
        )
        == "final only",
        "no_development_outputs_before_freeze": no_development_performance_outputs(),
    }
    if not all(gates.values()):
        failed = [name for name, value in gates.items() if not value]
        raise SystemExit("PI2N_PRE_DEV_FREEZE_GATE_FAIL: " + ",".join(failed))

    hashes = {
        "evaluation_protocol": sha256_file(EVALUATION_PROTOCOL),
        "candidate_protocol": sha256_file(CANDIDATE_PROTOCOL),
        "runtime_protocol_v2": sha256_file(RUNTIME_V2),
        "development_manifest": sha256_file(DEVELOPMENT_MANIFEST),
        "final_reset_manifest": sha256_file(FINAL_MANIFEST),
        "reset_exposure_ledger": sha256_file(EXPOSURE_LEDGER),
        "candidate_eligibility": sha256_file(CANDIDATE_ELIGIBILITY),
        "refinement_decision": sha256_file(REFINEMENT_DECISION),
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2n-pre-development-freeze.v1",
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "models": list(MODELS),
        "new_vac_candidates": list(NEW_VAC_CANDIDATES),
        "X_policy": "NOT_RUN_NOT_JUSTIFIED",
        "seed_blocks": list(SEED_BLOCKS),
        "episodes_per_block": EPISODES_PER_BLOCK,
        "episodes_per_model": len(SEED_BLOCKS) * EPISODES_PER_BLOCK,
        "total_canonical_outcomes": len(MODELS)
        * len(SEED_BLOCKS)
        * EPISODES_PER_BLOCK,
        "execution_contract": {
            "wave_order": [
                {"seed": seed, "models": list(EXECUTION_ORDER[seed])}
                for seed in SEED_BLOCKS
            ],
            "fresh_policy_server_per_model_seed_block": True,
            "fresh_contact_service_per_model_seed_block": True,
            "maximum_parallel_heavy_workers": 3,
            "gpu_assignment": "select checked-idle locked GPUs at launch; same balanced wave structure for every seed block",
            "official_async_evaluator": True,
            "replan_ratio": 0.8,
            "native_failure_or_timeout_in_denominator": True,
            "no_best_of_retry_splicing": True,
            "retry": runtime["retry_contract"],
        },
        "required_server_environment": runtime["required_server_environment"],
        "selection": {
            "eligible_vac_candidates": list(NEW_VAC_CANDIDATES),
            "rule": dev_protocol["selection"],
            "single_valid_candidate_contract": "B_VAC_V becomes VAC_STAR after all DEV raw cohorts pass integrity; anchors cannot become VAC_STAR",
            "performance_confirmation_role": False,
        },
        "checkpoint_manifests": {
            model: {
                "path": "$REPO_ROOT/" + path.relative_to(ROOT).as_posix(),
                "manifest_sha256": sha256_file(path),
                "checkpoint": checkpoint_payloads[model]["checkpoint"],
                "checkpoint_tree_sha256": checkpoint_payloads[model][
                    "checkpoint_tree_sha256"
                ],
                "live_checkpoint_path": str(CHECKPOINT_PATHS[model]),
                "live_checkpoint_tree_sha256": live_checkpoint_hashes[model],
            }
            for model, path in CHECKPOINT_MANIFESTS.items()
        },
        "inputs_sha256": hashes,
        "sources_sha256": {
            "$REPO_ROOT/" + relative: sha256_file(ROOT / relative)
            for relative in SOURCE_FILES
        },
        "development_reset_sequence_sha256": development[
            "ordered_reset_sequence_sha256"
        ],
        "development_performance_seen_before_freeze": False,
        "final_performance_seen_before_freeze": False,
        "candidate_selection_performed_before_freeze": False,
        "gates": {name: "PASS" for name in gates},
    }
    atomic_json(OUTPUT, payload)
    print(
        json.dumps(
            {
                "status": "PASS",
                "models": payload["models"],
                "total_canonical_outcomes": payload["total_canonical_outcomes"],
                "development_performance_seen": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
