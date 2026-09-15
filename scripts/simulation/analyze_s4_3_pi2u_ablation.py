#!/usr/bin/env python3
"""Analyze the preregistered fresh B0/BVA/B1/B2 PI2U evaluation."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2u"
MODELS = ("B0", "BVA", "B1", "B2")
CONTRASTS = (("B0", "BVA"), ("BVA", "B2"), ("B1", "B2"), ("B0", "B1"))
N = 200
RESAMPLES = 100_000
EVALUATOR_SEED = 6
CANONICAL_FREEZE_ALIASES = {
    "fresh_seed_audit.json": "fresh_seed_retry_seed6.json",
    "pre_eval_freeze.json": "pre_eval_retry_seed6.json",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(name: str, value: dict[str, Any]) -> None:
    out = ARTIFACTS / name
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)


def materialize_canonical_freeze_aliases() -> None:
    """Preserve the exact preregistered seed6 payloads under required artifact names."""
    for canonical_name, remediation_name in CANONICAL_FREEZE_ALIASES.items():
        canonical = ARTIFACTS / canonical_name
        remediation = ARTIFACTS / remediation_name
        if canonical.exists():
            raise SystemExit(f"refusing to overwrite {canonical}")
        if not remediation.is_file():
            raise SystemExit(f"missing preregistered remediation freeze: {remediation}")
        temporary = canonical.with_suffix(canonical.suffix + ".tmp")
        temporary.write_bytes(remediation.read_bytes())
        temporary.replace(canonical)


def wilson(k: int, n: int) -> list[float]:
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [center - radius, center + radius]


def mcnemar(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if not discordant:
        return 1.0
    return min(1.0, 2 * sum(math.comb(discordant, i) for i in range(min(first_only, second_only) + 1)) / 2**discordant)


def contrast(first: str, second: str, outcomes: dict[str, np.ndarray], seed: int) -> dict[str, Any]:
    left, right = outcomes[first], outcomes[second]
    diff = right.astype(np.int8) - left.astype(np.int8)
    rng = np.random.default_rng(seed)
    boot = np.empty(RESAMPLES)
    for start in range(0, RESAMPLES, 5000):
        stop = min(RESAMPLES, start + 5000)
        boot[start:stop] = diff[rng.integers(0, N, size=(stop - start, N))].mean(axis=1)
    table = {
        "both_success": int(np.sum(left & right)),
        "first_only": int(np.sum(left & ~right)),
        "second_only": int(np.sum(~left & right)),
        "both_fail": int(np.sum(~left & ~right)),
    }
    ci = [float(x) for x in np.quantile(boot, [0.025, 0.975])]
    delta = float(diff.mean())
    p = mcnemar(table["first_only"], table["second_only"])
    confirmed = delta > 0 and ci[0] > 0 and p < 0.05
    material = delta >= .10 and confirmed
    hurt = delta < 0 and ci[1] < 0 and p < .05
    return {
        "effect": f"{second}-{first}", "first": first, "second": second, "episodes": N,
        "success_difference": delta, "delta_percentage_points": 100 * delta,
        "paired_bootstrap_95ci": ci, "paired_bootstrap_95ci_percentage_points": [100*x for x in ci],
        "bootstrap_resamples": RESAMPLES, "bootstrap_seed": seed,
        "paired_outcome_table": table, "exact_mcnemar_two_sided_p": p,
        "statistically_confirmed_improvement": confirmed, "material_improvement": material,
        "statistically_confirmed_hurt": hurt,
        "classification": ("MATERIAL_GAIN" if material else "STATISTICAL_GAIN" if confirmed else "HURT" if hurt else "POSITIVE_TREND" if delta > 0 else "NO_GAIN"),
    }


def holm(rows: dict[str, dict[str, Any]]) -> dict[str, float]:
    ordered = sorted(rows, key=lambda name: rows[name]["exact_mcnemar_two_sided_p"])
    adjusted, running, count = {}, 0.0, len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (count-rank)*rows[name]["exact_mcnemar_two_sided_p"]))
        adjusted[name] = running
    return adjusted


def main() -> None:
    if (ARTIFACTS / "paired_ablation_statistics.json").exists():
        raise SystemExit("refusing to overwrite completed PI2U analysis")
    artifacts = {model: load(ARTIFACTS / f"{model.lower()}_raw_rollouts.json") for model in MODELS}
    outcomes: dict[str, np.ndarray] = {}
    summaries = {}
    for model, data in artifacts.items():
        rows = data.get("episode_results", [])
        if data.get("status") != "PASS" or data.get("episodes") != N or len(rows) != N:
            raise SystemExit(f"incomplete raw outcomes for {model}")
        reset_ids = [row.get("episode_index", index) for index, row in enumerate(rows)]
        if reset_ids != list(range(N)):
            raise SystemExit(f"noncanonical reset ordering for {model}")
        values = np.array([bool(row["success"]) for row in rows], dtype=bool)
        outcomes[model] = values
        terminations = Counter(row.get("termination", "UNKNOWN") for row in rows)
        summaries[model] = {
            "schema": "tactile3d-unit.s4-3-pi2u-model-evaluation.v1", "status": "PASS", "model": model,
            "episodes": N, "successes": int(values.sum()), "success_rate": float(values.mean()),
            "wilson_95ci": wilson(int(values.sum()), N), "termination_breakdown": dict(sorted(terminations.items())),
            "raw_artifact": f"$REPO_ROOT/.local/artifacts/simulation/s4_3_pi2u/{model.lower()}_raw_rollouts.json",
            "raw_artifact_sha256": sha256(ARTIFACTS / f"{model.lower()}_raw_rollouts.json"),
            "ordered_reset_identities_sha256": hashlib.sha256(
                "\n".join(str(row["reset_identity"]) for row in rows).encode()
            ).hexdigest(),
        }
    if len({value["ordered_reset_identities_sha256"] for value in summaries.values()}) != 1:
        raise SystemExit("models did not receive byte-identical ordered fresh resets")
    materialize_canonical_freeze_aliases()
    for model, value in summaries.items():
        atomic(f"{model.lower()}_eval.json", value)
    rows = {f"{second}-{first}": contrast(first, second, outcomes, 4306 + i) for i, (first, second) in enumerate(CONTRASTS)}
    adjusted = holm(rows)
    for name, value in rows.items():
        value["holm_adjusted_p"] = adjusted[name]
    bva = rows["BVA-B0"]
    b2_bva = rows["B2-BVA"]
    outcome = ("BVA_MATERIAL_GAIN" if bva["material_improvement"] else "BVA_STATISTICAL_GAIN" if bva["statistically_confirmed_improvement"] else "BVA_HURT" if bva["statistically_confirmed_hurt"] else "BVA_POSITIVE_TREND" if bva["success_difference"] > 0 else "BVA_NO_GAIN")
    bva_competitive = (
        summaries["BVA"]["success_rate"] >= summaries["B2"]["success_rate"]
        and not b2_bva["statistically_confirmed_improvement"]
    )
    readiness = (
        "PI2B_READY_B0_BVA_B1_B2"
        if bva["statistically_confirmed_improvement"] or bva_competitive
        else "PI2B_READY_B0_B1_B2"
        if b2_bva["statistically_confirmed_improvement"]
        else "PI2B_MECHANISM_DIAGNOSIS_FIRST"
    )
    b2_b1 = rows["B2-B1"]
    b1_b0 = rows["B1-B0"]
    if bva["statistically_confirmed_improvement"] and b2_bva["statistically_confirmed_improvement"]:
        pattern = "CASE_B_VA_HELPS_AND_FULL_TACTILE_UNIT_ADDS_VALUE"
    elif not bva["statistically_confirmed_improvement"] and b2_bva["statistically_confirmed_improvement"]:
        pattern = "CASE_A_VA_DOES_NOT_EXPLAIN_FULL_TACTILE_UNIT"
    elif b2_bva["statistically_confirmed_hurt"]:
        pattern = "CASE_D_VA_EXCEEDS_FULL_TACTILE_UNIT"
    elif bva["statistically_confirmed_improvement"] and not b2_bva["statistically_confirmed_improvement"]:
        pattern = "CASE_C_COMPATIBLE_VA_MAY_EXPLAIN_MOST_INCREMENT"
    else:
        pattern = "INDETERMINATE_PATTERN"
    mechanism = {
        "schema": "tactile3d-unit.s4-3-pi2u-mechanism-interpretation.v1", "status": "PASS",
        "evaluator_seed": EVALUATOR_SEED,
        "BVA_outcome": outcome, "VA_explains_incremental_value": bva["statistically_confirmed_improvement"],
        "B2_exceeds_BVA": b2_bva["statistically_confirmed_improvement"], "pi2b_readiness": readiness,
        "BVA_competitive_for_multiseed": bva_competitive,
        "pi2b_readiness_rationale": "BVA is retained when its point estimate is at least B2 and B2 does not statistically exceed it, as required by the preregistered competitive-BVA guidance.",
        "pattern": pattern,
        "questions": {
            "BVA_minus_B0_VA_only_supervision": bva["classification"],
            "B2_minus_BVA_full_Tactile_UniT_vs_VA_only": b2_bva["classification"],
            "B2_minus_B1_VAC_auxiliary_increment": b2_b1["classification"],
            "B1_minus_B0_tactile_conditioning_increment": b1_b0["classification"],
        },
        "interpretation_boundary": "B2-BVA is full Tactile-UniT versus VA-only supervision, not an isolated Contact-target effect.",
        "runtime_fact": "BVA changes training supervision only; its inference payload is exactly B0 and contains no tactile/contact/future target.",
    }
    statistics = {
        "schema": "tactile3d-unit.s4-3-pi2u-paired-ablation-statistics.v1", "status": "PASS",
        "evaluator_seed": EVALUATOR_SEED,
        "episodes_per_model": N, "summaries": summaries, "contrasts": rows,
        "holm_family": list(rows), "BVA_outcome": outcome, "PI2B_readiness": readiness,
        "raw_outcomes_sha256": {model: summaries[model]["raw_artifact_sha256"] for model in MODELS},
    }
    atomic("paired_ablation_statistics.json", statistics)
    atomic("mechanism_interpretation.json", mechanism)
    print(json.dumps({"status": "PASS", "BVA_outcome": outcome, "PI2B_readiness": readiness}, sort_keys=True))


if __name__ == "__main__":
    main()
