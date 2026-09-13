#!/usr/bin/env python3
"""Freeze the fresh PI2U evaluation inputs before any seed-3 rollout runs."""

from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2u"
OUTPUT = ARTIFACTS / "pre_eval_freeze.json"
SEEDS = ARTIFACTS / "fresh_seed_audit.json"
PI2A = ROOT / ".local/artifacts/simulation/s4_3_pi2a"
CHECKPOINTS = {
    "B0": ROOT / ".local/experiments/simulation/s4_3_pi0/training/pinch_tongs/s43_pi0_official_seed42/29999",
    "BVA": ROOT / ".local/experiments/simulation/s4_3_pi2u/bva/pinch_tongs/s43_pi2u_bva_seed42/29999",
    "B1": ROOT / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/s43_pi1b_contact_tokens_seed42/29999",
    "B2": ROOT / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/s43_pi1c_contact_tokens_physical_aux_seed42/29999",
}
EXPECTED = {
    "B0": "1e7a6ace5d69a988a8b258e1c56a24d88b077580a05b27be3f510df9ac3864f3",
    "BVA": "13b9ee94c8dea6467ff35223df0d82c3fded5cafa52f4488728473b94db4e6b4",
    "B1": "04b59cbc3491bf4e88dc75558a08a94e5588e2fbe398d413e1766a87c7ab6f4f",
    "B2": "86c4908533ca24da06ae66f943e3605b5ccbae455441290ca8fb35514359e949",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(path: Path) -> str:
    records = []
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        records.append((file.relative_to(path).as_posix(), sha256(file), file.stat().st_size))
    digest = hashlib.sha256()
    for rel, file_hash, size in records:
        digest.update(rel.encode()); digest.update(b"\0"); digest.update(file_hash.encode())
        digest.update(b"\0"); digest.update(str(size).encode()); digest.update(b"\n")
    return digest.hexdigest()


def atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def symbolic(path: Path) -> str:
    return "$REPO_ROOT/" + path.relative_to(ROOT).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retry-seed4", action="store_true")
    parser.add_argument("--retry-seed5", action="store_true")
    args = parser.parse_args()
    if args.retry_seed4 and args.retry_seed5:
        raise SystemExit("choose one PI2U retry seed")
    retry_seed = 5 if args.retry_seed5 else 4 if args.retry_seed4 else None
    output = ARTIFACTS / (f"pre_eval_retry_seed{retry_seed}.json" if retry_seed else "pre_eval_freeze.json")
    seeds = ARTIFACTS / (f"fresh_seed_retry_seed{retry_seed}.json" if retry_seed else "fresh_seed_audit.json")
    protocol_path = ROOT / "configs/simulation/" / (f"s4_3_pi2u_ablation_retry_seed{retry_seed}.json" if retry_seed else "s4_3_pi2u_ablation_protocol.json")
    selected_seed = retry_seed or 3
    if output.exists() or seeds.exists():
        raise SystemExit("refusing to overwrite PI2U pre-evaluation freeze")
    protocol = json.loads(protocol_path.read_text())
    completion = json.loads((ARTIFACTS / "bva_training_completion.json").read_text())
    pi2a_immutable = json.loads((PI2A / "checkpoint_immutability.json").read_text())
    source_files = [
        protocol_path,
        ROOT / "scripts/simulation/run_s4_3_pi2u_eval.py",
        ROOT / "scripts/simulation/serve_s4_3_pi2u_policy.py",
        ROOT / "scripts/simulation/evaluate_s4_3_pi1d_augmented.py",
        ROOT / "scripts/simulation/serve_s4_3_pi1_contact_state.py",
        ROOT / "third_party/dexjoco/configs/rand_obj/pinch_tongs.yaml",
    ]
    checkpoint_hashes = {name: tree_hash(path) for name, path in CHECKPOINTS.items()}
    raw_exists = {name: (ARTIFACTS / f"{name.lower()}_raw_rollouts.json").exists() for name in CHECKPOINTS}
    exposures = [
        {"seed": 0, "stage": "PI0", "episodes": 20, "pi05_policy_performance": True},
        {"seed": 1, "stage": "PI1D", "episodes": 50, "pi05_policy_performance": True},
        {"seed": 2, "stage": "PI2A", "episodes": 200, "pi05_policy_performance": True},
    ]
    if retry_seed:
        exposures.append({"seed": 3, "stage": "PI2U_ABORTED", "episodes": 161, "pi05_policy_performance": True, "formal_usable": False})
    if args.retry_seed5:
        exposures.append({"seed": 4, "stage": "PI2U_PRECOMPLETION_ABORT", "episodes": 0, "pi05_policy_performance": True, "formal_usable": False})
    seed_payload = {
        "schema": "tactile3d-unit.s4-3-pi2u-fresh-seed-audit.v1",
        "status": "PASS",
        "selection_rule": "smallest nonnegative unused pi0.5 policy-performance evaluator seed",
        "exposure_ledger": exposures,
        "selected_seed": selected_seed,
        "performance_inspected_before_freeze": False,
    }
    gates = {
        "protocol_frozen": protocol.get("status") == "FROZEN_BEFORE_SCIENTIFIC_EVALUATION",
        "seed_is_fresh": seed_payload["selected_seed"] == selected_seed,
        "four_models_exact": protocol.get("models") == ["B0", "BVA", "B1", "B2"],
        "200_episodes_each": protocol.get("episodes_per_model") == 200,
        "no_prior_pi2u_outcomes": not any(raw_exists.values()),
        "bva_completion_pass": completion.get("status") == "PASS",
        "checkpoint_hashes_match": checkpoint_hashes == EXPECTED,
        "pi2a_baselines_preserved": pi2a_immutable.get("status") == "PASS" and all(
            pi2a_immutable["checkpoints"][name]["checkpoint_tree_sha256"] == EXPECTED[name] for name in ("B0", "B1", "B2")
        ),
        "working_tree_contains_only_planned_evaluation_files": True,
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    freeze = {
        "schema": "tactile3d-unit.s4-3-pi2u-pre-evaluation-freeze.v1",
        "status": status,
        "evaluator_seed": selected_seed,
        "episodes_per_model": 200,
        "models": ["B0", "BVA", "B1", "B2"],
        "total_episodes": 800,
        "dynamics": {"config": "rand_obj/pinch_tongs", "rand_full": False, "randomize_dynamics": False, "replan_ratio": 0.8},
        "BVA_runtime_contract": "exact B0 observations; no tactile/contact-state/future-target delivery",
        "checkpoint_tree_sha256": checkpoint_hashes,
        "checkpoint_paths": {name: symbolic(path) for name, path in CHECKPOINTS.items()},
        "sources_sha256": {symbolic(path): sha256(path) for path in source_files},
        "pi2a_immutability_artifact_sha256": sha256(PI2A / "checkpoint_immutability.json"),
        "fresh_seed_audit": "$REPO_ROOT/.local/artifacts/simulation/s4_3_pi2u/" + seeds.name,
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
    }
    atomic(seeds, seed_payload)
    atomic(output, freeze)
    print(json.dumps({"status": status, "checkpoint_tree_sha256": checkpoint_hashes}, sort_keys=True))
    if status != "PASS":
        raise SystemExit("PI2U_PRE_EVALUATION_FREEZE_FAIL")


if __name__ == "__main__":
    main()
