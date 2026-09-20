#!/usr/bin/env python3
"""Independently recompute and audit the frozen PI2M paired statistics."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2m"
FROZEN = ARTIFACTS / "paired_statistics.json"
COMPLETENESS = ARTIFACTS / "preanalysis_completeness_audit.json"
OUTPUT = ARTIFACTS / "statistics_independent_audit.json"
MODELS = ("B1", "B_HVA", "B2")
COMPARISONS = (
    ("B2-B_HVA", "B2", "B_HVA"),
    ("B_HVA-B1", "B_HVA", "B1"),
    ("B2-B1", "B2", "B1"),
)


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_bootstrap(left: np.ndarray, right: np.ndarray) -> list[float]:
    differences = left.astype(np.float64) - right.astype(np.float64)
    rng = np.random.default_rng(4317)
    values = []
    for _ in range(50):
        indices = rng.integers(0, len(differences), size=(2_000, len(differences)))
        values.append(differences[indices].mean(axis=1))
    return np.percentile(np.concatenate(values), [2.5, 97.5]).tolist()


def holm(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, name in enumerate(ordered):
        running = max(running, (len(ordered) - rank) * raw[name])
        adjusted[name] = min(1.0, running)
    return adjusted


def close(left: Any, right: Any, atol: float = 1e-12) -> bool:
    return bool(np.allclose(np.asarray(left), np.asarray(right), atol=atol, rtol=0))


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite {OUTPUT}")
    if read_json(COMPLETENESS).get("status") != "PASS":
        raise SystemExit("preanalysis completeness audit is not PASS")
    frozen = read_json(FROZEN)
    raw_paths = {model: ARTIFACTS / f"{model.lower()}_raw_rollouts.json" for model in MODELS}
    raw = {model: read_json(path) for model, path in raw_paths.items()}
    outcomes = {
        model: np.asarray(
            [bool(row["success"]) for row in raw[model]["episode_results"]],
            dtype=np.bool_,
        )
        for model in MODELS
    }
    independent_models: dict[str, Any] = {}
    model_gates: dict[str, bool] = {}
    for model, values in outcomes.items():
        successes = int(values.sum())
        interval = binomtest(successes, len(values)).proportion_ci(
            confidence_level=0.95, method="wilson"
        )
        independent_models[model] = {
            "successes": successes,
            "n": len(values),
            "success_rate": successes / len(values),
            "wilson_95ci": [float(interval.low), float(interval.high)],
        }
        expected = frozen["model_statistics"][model]
        model_gates[f"{model}_successes"] = successes == expected["successes"]
        model_gates[f"{model}_rate"] = close(
            independent_models[model]["success_rate"], expected["success_rate"]
        )
        model_gates[f"{model}_wilson"] = close(
            independent_models[model]["wilson_95ci"], expected["wilson_95ci"]
        )

    independent_pairs: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    for name, left_name, right_name in COMPARISONS:
        left, right = outcomes[left_name], outcomes[right_name]
        left_only = int(np.count_nonzero(left & ~right))
        right_only = int(np.count_nonzero(~left & right))
        discordant = left_only + right_only
        p_value = (
            float(
                binomtest(
                    min(left_only, right_only), discordant, p=0.5, alternative="two-sided"
                ).pvalue
            )
            if discordant
            else 1.0
        )
        raw_p[name] = p_value
        independent_pairs[name] = {
            "table": {
                "both_success": int(np.count_nonzero(left & right)),
                "left_only_success": left_only,
                "right_only_success": right_only,
                "both_failure": int(np.count_nonzero(~left & ~right)),
            },
            "risk_difference": float(left.mean() - right.mean()),
            "paired_bootstrap_95ci": paired_bootstrap(left, right),
            "scipy_binomtest_exact_two_sided_p": p_value,
        }
    adjusted = holm(raw_p)
    pair_gates: dict[str, bool] = {}
    for name, row in independent_pairs.items():
        row["holm_adjusted_p"] = adjusted[name]
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
    gates = {
        "frozen_statistics_PASS": frozen.get("status") == "PASS",
        "claim_level_fixed_seed_only": frozen.get("claim_level") == "FIXED_SEED_ONLY",
        "historical_BVA_excluded_and_0p32s": frozen.get("historical_BVA_excluded") is True
        and frozen.get("historical_BVA_actual_target_horizon_seconds") == 0.32,
        "all_model_statistics_match": all(model_gates.values()),
        "all_paired_statistics_match": all(pair_gates.values()),
        "primary_negative_confirmed": frozen["paired_comparisons"]["B2-B_HVA"][
            "classification"
        ]
        == "NEGATIVE_CONFIRMED",
        "VA_aux_positive_confirmed": frozen["paired_comparisons"]["B_HVA-B1"][
            "classification"
        ]
        == "POSITIVE_CONFIRMED",
        "contact_aux_replication_inconclusive": frozen["paired_comparisons"]["B2-B1"][
            "classification"
        ]
        == "INCONCLUSIVE",
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2m-statistics-independent-audit.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "independent_implementation": {
            "marginal_interval": "scipy.stats.binomtest.proportion_ci(method=wilson)",
            "paired_test": "scipy.stats.binomtest on discordant pairs",
            "bootstrap": "independent NumPy tuple-resampling implementation",
            "holm": "independent monotone step-down implementation",
        },
        "model_statistics": independent_models,
        "paired_comparisons": independent_pairs,
        "model_gates": {name: "PASS" if value else "FAIL" for name, value in model_gates.items()},
        "pair_gates": {name: "PASS" if value else "FAIL" for name, value in pair_gates.items()},
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "input_sha256": {
            **{f"raw_{model}": sha256_file(path) for model, path in raw_paths.items()},
            "paired_statistics": sha256_file(FROZEN),
            "preanalysis_completeness": sha256_file(COMPLETENESS),
        },
        "scope": "one checkpoint per model, training seed42, evaluator seed7, 200 matched resets",
        "recovery_limit": "B2 is one complete clean retry; 13/31 interrupted-prefix outcomes differed despite identical reset identities, so fixed seed does not imply bit/outcome-deterministic asynchronous execution",
        "PI2B_started": False,
    }
    atomic_json(OUTPUT, payload)
    print(json.dumps({"status": payload["status"], "gates": payload["gates"]}, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit("PI2M_STATISTICS_INDEPENDENT_AUDIT_FAIL")


if __name__ == "__main__":
    main()
