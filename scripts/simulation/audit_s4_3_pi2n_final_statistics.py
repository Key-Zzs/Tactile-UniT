#!/usr/bin/env python3
"""Independently recompute the frozen six-model PI2N FINAL statistics."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2n"
FROZEN = ARTIFACTS / "paired_statistics.json"
COMPLETENESS = ARTIFACTS / "rollout_completeness.json"
RESET_MANIFEST = ARTIFACTS / "final_reset_manifest.json"
RAW = ARTIFACTS / "final_raw"
OUTPUT = ARTIFACTS / "statistics_independent_audit.json"
MODELS = ("B0", "B_VA27", "B1", "B_HVA", "B2", "B_VAC_V")
SEEDS = (12, 13, 14, 15)
COMPARISONS = (
    ("VAC_STAR-B_HVA", "B_VAC_V", "B_HVA"),
    ("VAC_STAR-B2", "B_VAC_V", "B2"),
    ("VAC_STAR-B0", "B_VAC_V", "B0"),
    ("B_HVA-B_VA27", "B_HVA", "B_VA27"),
    ("B_VA27-B0", "B_VA27", "B0"),
    ("B1-B0", "B1", "B0"),
)
BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEED = 4317


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_path(seed: int, model: str) -> Path:
    return RAW / f"seed_{seed}" / f"{model.lower()}_raw_rollouts.json"


def load_outcomes(
    completeness: dict[str, Any], reset_manifest: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    expected = reset_manifest["ordered_reset_identities"]
    values: dict[str, list[bool]] = {model: [] for model in MODELS}
    identities: dict[str, list[str]] = {model: [] for model in MODELS}
    raw_hashes: dict[str, str] = {}
    for seed in SEEDS:
        for model in MODELS:
            path = raw_path(seed, model)
            key = f"seed_{seed}/{model}"
            observed_hash = sha256_file(path)
            if observed_hash != completeness["raw_artifacts"][key]["sha256"]:
                raise SystemExit(f"PI2N independent audit raw drift: {key}")
            payload = read_json(path)
            values[model].extend(bool(row["success"]) for row in payload["episode_results"])
            identities[model].extend(
                row["reset_identity"] for row in payload["episode_results"]
            )
            raw_hashes[key] = observed_hash
    if not all(
        len(values[model]) == 200 and identities[model] == expected
        for model in MODELS
    ):
        raise SystemExit("PI2N independent audit reset alignment failed")
    return {
        model: np.asarray(rows, dtype=np.bool_) for model, rows in values.items()
    }, raw_hashes


def paired_bootstrap(differences: np.ndarray) -> list[float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    chunks = []
    for _ in range(50):
        indices = rng.integers(
            0, len(differences), size=(2_000, len(differences))
        )
        chunks.append(differences[indices].mean(axis=1))
    return np.percentile(np.concatenate(chunks), [2.5, 97.5]).tolist()


def holm(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, name in enumerate(ordered):
        running = max(running, (len(ordered) - rank) * raw[name])
        adjusted[name] = min(1.0, running)
    return adjusted


def close(left: Any, right: Any, atol: float = 1e-12) -> bool:
    return bool(
        np.allclose(np.asarray(left), np.asarray(right), atol=atol, rtol=0)
    )


def recompute(outcomes: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    models: dict[str, Any] = {}
    for model, values in outcomes.items():
        successes = int(values.sum())
        interval = binomtest(successes, len(values)).proportion_ci(
            confidence_level=0.95, method="wilson"
        )
        models[model] = {
            "successes": successes,
            "n": len(values),
            "success_rate": successes / len(values),
            "wilson_95ci": [float(interval.low), float(interval.high)],
        }
    pairs: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    for name, left_name, right_name in COMPARISONS:
        left, right = outcomes[left_name], outcomes[right_name]
        left_only = int(np.count_nonzero(left & ~right))
        right_only = int(np.count_nonzero(~left & right))
        discordant = left_only + right_only
        p_value = (
            float(
                binomtest(
                    min(left_only, right_only),
                    discordant,
                    p=0.5,
                    alternative="two-sided",
                ).pvalue
            )
            if discordant
            else 1.0
        )
        differences = left.astype(np.float64) - right.astype(np.float64)
        raw_p[name] = p_value
        pairs[name] = {
            "table": {
                "both_success": int(np.count_nonzero(left & right)),
                "left_only_success": left_only,
                "right_only_success": right_only,
                "both_failure": int(np.count_nonzero(~left & ~right)),
            },
            "risk_difference": float(differences.mean()),
            "paired_bootstrap_95ci": paired_bootstrap(differences),
            "scipy_binomtest_exact_two_sided_p": p_value,
        }
    adjusted = holm(raw_p)
    for name, row in pairs.items():
        row["holm_adjusted_p"] = adjusted[name]
    return models, pairs


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite {OUTPUT}")
    required = (FROZEN, COMPLETENESS, RESET_MANIFEST)
    if any(not path.is_file() for path in required):
        raise SystemExit("PI2N independent statistics prerequisite missing")
    frozen = read_json(FROZEN)
    completeness = read_json(COMPLETENESS)
    reset_manifest = read_json(RESET_MANIFEST)
    if (
        frozen.get("status") != "PASS"
        or completeness.get("status") != "PASS"
        or completeness.get("total_canonical_outcomes") != 1200
    ):
        raise SystemExit("PI2N independent statistics integrity gate failed")
    outcomes, raw_hashes = load_outcomes(completeness, reset_manifest)
    models, pairs = recompute(outcomes)
    model_gates: dict[str, bool] = {}
    for model, row in models.items():
        expected = frozen["model_statistics"][model]
        model_gates[f"{model}_successes"] = row["successes"] == expected["successes"]
        model_gates[f"{model}_n"] = row["n"] == expected["n"] == 200
        model_gates[f"{model}_rate"] = close(
            row["success_rate"], expected["success_rate"]
        )
        model_gates[f"{model}_wilson"] = close(
            row["wilson_95ci"], expected["wilson_95ci"]
        )
    pair_gates: dict[str, bool] = {}
    for name, row in pairs.items():
        expected = frozen["paired_comparisons"][name]
        pair_gates[f"{name}_table"] = row["table"] == expected["table"]
        pair_gates[f"{name}_delta"] = close(
            row["risk_difference"], expected["risk_difference"]
        )
        pair_gates[f"{name}_bootstrap"] = close(
            row["paired_bootstrap_95ci"], expected["paired_bootstrap_95ci"]
        )
        pair_gates[f"{name}_raw_p"] = close(
            row["scipy_binomtest_exact_two_sided_p"],
            expected["mcnemar_exact_two_sided_raw_p"],
        )
        pair_gates[f"{name}_holm"] = close(
            row["holm_adjusted_p"], expected["holm_adjusted_p"]
        )
    interaction = (
        outcomes["B_HVA"].astype(np.float64)
        - outcomes["B_VA27"].astype(np.float64)
        - outcomes["B1"].astype(np.float64)
        + outcomes["B0"].astype(np.float64)
    )
    frozen_interaction = frozen["secondary_difference_in_differences"]
    interaction_row = {
        "estimate": float(interaction.mean()),
        "paired_bootstrap_95ci": paired_bootstrap(interaction),
    }
    interaction_gates = {
        "estimate": close(
            interaction_row["estimate"], frozen_interaction["estimate"]
        ),
        "bootstrap": close(
            interaction_row["paired_bootstrap_95ci"],
            frozen_interaction["paired_bootstrap_95ci"],
        ),
        "causal_synergy_not_claimed": frozen_interaction.get(
            "causal_synergy_claimed"
        )
        is False,
    }
    gates = {
        "frozen_statistics_PASS": frozen.get("status") == "PASS",
        "fixed_seed_only": frozen.get("claim_level") == "FIXED_SEED_ONLY",
        "six_model_statistics_match": all(model_gates.values()),
        "six_paired_comparisons_match": len(pairs) == 6
        and all(pair_gates.values()),
        "secondary_interaction_matches": all(interaction_gates.values()),
        "inconclusive_not_equivalence": frozen.get(
            "inconclusive_interpreted_as_equivalence"
        )
        is False,
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2n-statistics-independent-audit.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "independent_implementation": {
            "marginal_interval": "scipy.stats.binomtest.proportion_ci(method=wilson)",
            "paired_test": "scipy.stats.binomtest on discordant pairs",
            "bootstrap": "independent NumPy 50x2000 tuple-resampling implementation",
            "holm": "independent monotone step-down implementation",
        },
        "model_statistics": models,
        "paired_comparisons": pairs,
        "secondary_difference_in_differences": interaction_row,
        "model_gates": {
            name: "PASS" if value else "FAIL" for name, value in model_gates.items()
        },
        "pair_gates": {
            name: "PASS" if value else "FAIL" for name, value in pair_gates.items()
        },
        "interaction_gates": {
            name: "PASS" if value else "FAIL"
            for name, value in interaction_gates.items()
        },
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "input_sha256": {
            **{f"raw_{name}": value for name, value in raw_hashes.items()},
            "paired_statistics": sha256_file(FROZEN),
            "rollout_completeness": sha256_file(COMPLETENESS),
            "final_reset_manifest": sha256_file(RESET_MANIFEST),
        },
        "scope": "one checkpoint per model, training seed42, 200 matched FINAL resets",
        "PI2B_started": False,
        "real_robot_started": False,
    }
    atomic_json(OUTPUT, payload)
    print(json.dumps({"status": payload["status"], "gates": payload["gates"]}, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit("PI2N_STATISTICS_INDEPENDENT_AUDIT_FAIL")


if __name__ == "__main__":
    main()
