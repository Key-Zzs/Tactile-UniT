#!/usr/bin/env python3
"""Final evidence audit for the completed S4.3 PI2U BVA ablation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2u"
EXPECTED = {
    "B0": "1e7a6ace5d69a988a8b258e1c56a24d88b077580a05b27be3f510df9ac3864f3",
    "BVA": "13b9ee94c8dea6467ff35223df0d82c3fded5cafa52f4488728473b94db4e6b4",
    "B1": "04b59cbc3491bf4e88dc75558a08a94e5588e2fbe398d413e1766a87c7ab6f4f",
    "B2": "86c4908533ca24da06ae66f943e3605b5ccbae455441290ca8fb35514359e949",
}


def load(name: str) -> dict[str, Any]:
    return json.loads((ARTIFACTS / name).read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic(name: str, value: dict[str, Any]) -> None:
    target = ARTIFACTS / name
    if target.exists():
        raise SystemExit(f"refusing to overwrite {target}")
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)


def main() -> None:
    if (ARTIFACTS / "final_decision.json").exists():
        raise SystemExit("final PI2U audit was already completed")
    required = [
        "bva_training_completion.json", "bva_checkpoint_manifest.json", "fresh_seed_retry_seed5.json", "pre_eval_retry_seed5.json",
        "b0_eval.json", "bva_eval.json", "b1_eval.json", "b2_eval.json", "paired_ablation_statistics.json", "mechanism_interpretation.json",
    ]
    if any(not (ARTIFACTS / name).is_file() for name in required):
        raise SystemExit("PI2U final audit called before all required evidence exists")
    completion, freeze, statistics, mechanism = (load(name) for name in ("bva_training_completion.json", "pre_eval_retry_seed5.json", "paired_ablation_statistics.json", "mechanism_interpretation.json"))
    source_hashes = freeze["sources_sha256"]
    source_gates = {}
    for symbolic, expected in source_hashes.items():
        current = sha256(ROOT / symbolic.removeprefix("$REPO_ROOT/"))
        source_gates[symbolic] = current == expected
    raw = {model: load(f"{model.lower()}_raw_rollouts.json") for model in EXPECTED}
    runtime_gates = {}
    for model, payload in raw.items():
        rows = payload.get("episode_results", [])
        runtime_gates[f"{model}_complete"] = payload.get("status") == "PASS" and len(rows) == 200
        if model in ("B0", "BVA"):
            runtime_gates[f"{model}_no_contact_delivery"] = all(
                not event.get("contact_state_sent", False) and not event.get("training_only_fields_sent", [])
                for row in rows for event in row.get("action_chunks", [])
            )
    test = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/simulation/test_s4_3_pi0.py", "tests/simulation/test_s4_3_pi1.py", "tests/simulation/test_s4_3_pi2u.py"],
        cwd=ROOT, text=True, capture_output=True,
    )
    gates = {
        "bva_training_completion": completion.get("status") == "PASS",
        "pre_eval_freeze": freeze.get("status") == "PASS",
        "analysis": statistics.get("status") == "PASS" and mechanism.get("status") == "PASS",
        "checkpoint_identities": freeze.get("checkpoint_tree_sha256") == EXPECTED,
        "frozen_eval_sources_unchanged": all(source_gates.values()),
        "all_runtime_integrity_gates": all(runtime_gates.values()),
        "project_regression_tests": test.returncode == 0,
        "pi2a_immutability_reference": (ROOT / ".local/artifacts/simulation/s4_3_pi2a/checkpoint_immutability.json").is_file(),
    }
    outcome = statistics["BVA_outcome"]
    decision = {
        "schema": "tactile3d-unit.s4-3-pi2u-final-decision.v1", "status": "PASS" if all(gates.values()) else "FAIL",
        "decision": outcome, "pi2b_readiness": statistics["PI2B_readiness"],
        "BVA_vs_B0": statistics["contrasts"]["BVA-B0"],
        "B2_vs_BVA": statistics["contrasts"]["B2-BVA"],
        "gates": {key: "PASS" if value else "FAIL" for key, value in gates.items()},
    }
    audit = {
        "schema": "tactile3d-unit.s4-3-pi2u-final-repository-audit.v1", "status": decision["status"],
        "required_evidence": required, "source_gates": {key: "PASS" if value else "FAIL" for key, value in source_gates.items()},
        "runtime_gates": {key: "PASS" if value else "FAIL" for key, value in runtime_gates.items()},
        "regression_stdout": test.stdout, "regression_stderr": test.stderr, "gates": decision["gates"],
    }
    atomic("final_repository_audit.json", audit)
    atomic("final_decision.json", decision)
    acceptance = ARTIFACTS / "HUMAN_ACCEPTANCE.md"
    acceptance.write_text(
        "# S4.3 PI2U BVA acceptance\n\n"
        f"- Final decision: `{outcome}`\n"
        f"- PI2B readiness: `{statistics['PI2B_readiness']}`\n"
        "- Evidence: `bva_training_completion.json`, `pre_eval_freeze.json`, `paired_ablation_statistics.json`, `final_repository_audit.json`.\n"
    )
    print(json.dumps({"status": decision["status"], "decision": outcome}, sort_keys=True))
    if decision["status"] != "PASS":
        raise SystemExit("PI2U_FINAL_AUDIT_FAIL")


if __name__ == "__main__":
    main()
