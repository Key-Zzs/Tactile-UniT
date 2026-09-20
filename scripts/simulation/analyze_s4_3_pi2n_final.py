#!/usr/bin/env python3
"""Analyze only a complete frozen 1,200-outcome PI2N FINAL cohort."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2n"
PROTOCOL = ROOT / "configs/simulation/s4_3_pi2n_evaluation_protocol.json"
PRE_FREEZE = ARTIFACTS / "pre_final_freeze.json"
COMPLETENESS = ARTIFACTS / "rollout_completeness.json"
RESET_MANIFEST = ARTIFACTS / "final_reset_manifest.json"
RAW = ARTIFACTS / "final_raw"
STATISTICS = ARTIFACTS / "paired_statistics.json"
PROCESS_METRICS = ARTIFACTS / "contact_process_metrics.json"
CLAIM_FREEZE = ARTIFACTS / "claim_freeze.json"
FINAL_DECISION = ARTIFACTS / "final_decision.json"
MODELS = ("B0", "B_VA27", "B1", "B_HVA", "B2", "B_VAC_V")
SEED_BLOCKS = (12, 13, 14, 15)
COMPARISONS = (
    ("VAC_STAR-B_HVA", "B_VAC_V", "B_HVA"),
    ("VAC_STAR-B2", "B_VAC_V", "B2"),
    ("VAC_STAR-B0", "B_VAC_V", "B0"),
    ("B_HVA-B_VA27", "B_HVA", "B_VA27"),
    ("B_VA27-B0", "B_VA27", "B0"),
    ("B1-B0", "B1", "B0"),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson(successes: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(
        p * (1 - p) / n + z * z / (4 * n * n)
    ) / denominator
    return center - radius, center + radius


def exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    probability = sum(
        math.comb(discordant, value) for value in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * probability)


def paired_bootstrap(
    differences: np.ndarray, *, resamples: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=np.float64)
    chunk = 2_000
    for start in range(0, resamples, chunk):
        stop = min(resamples, start + chunk)
        indices = rng.integers(
            0, len(differences), size=(stop - start, len(differences))
        )
        values[start:stop] = differences[indices].mean(axis=1)
    lower, upper = np.percentile(values, [2.5, 97.5])
    return float(lower), float(upper)


def holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, (total - rank) * raw[name])
        adjusted[name] = min(1.0, running)
    return adjusted


def classification(delta: float, lower: float, upper: float, adjusted: float) -> str:
    if delta > 0 and lower > 0 and adjusted < 0.05:
        return "POSITIVE_CONFIRMED"
    if delta < 0 and upper < 0 and adjusted < 0.05:
        return "NEGATIVE_CONFIRMED"
    return "INCONCLUSIVE"


def raw_path(seed: int, model: str) -> Path:
    return RAW / f"seed_{seed}" / f"{model.lower()}_raw_rollouts.json"


def load_canonical_rows(
    completeness: dict[str, Any], reset_manifest: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    expected = reset_manifest["ordered_reset_identities"]
    rows_by_model: dict[str, list[dict[str, Any]]] = {model: [] for model in MODELS}
    for seed in SEED_BLOCKS:
        for model in MODELS:
            path = raw_path(seed, model)
            key = f"seed_{seed}/{model}"
            if not path.is_file():
                raise SystemExit(f"PI2N FINAL raw artifact missing: {key}")
            if sha256_file(path) != completeness["raw_artifacts"][key]["sha256"]:
                raise SystemExit(f"PI2N FINAL raw artifact drifted: {key}")
            payload = json.loads(path.read_text())
            rows_by_model[model].extend(payload["episode_results"])
    identities = {
        model: [row["reset_identity"] for row in rows]
        for model, rows in rows_by_model.items()
    }
    if not all(
        len(rows_by_model[model]) == 200 and identities[model] == expected
        for model in MODELS
    ):
        raise SystemExit("PI2N FINAL canonical row identity audit failed")
    return rows_by_model


def numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"available": False, "n": 0, "mean": None, "median": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "available": True,
        "n": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def process_metrics(rows_by_model: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for model, rows in rows_by_model.items():
        fields: dict[str, list[float]] = {
            "final_pinch_count": [],
            "max_pinch_count": [],
            "max_normal_force_proxy": [],
            "mean_tactile_l2_proxy": [],
            "active_contact_samples": [],
            "matched_contact_count_sum": [],
            "executed_action_mean_l2": [],
            "termination_steps": [],
        }
        for row in rows:
            progress = row.get("task_progress", {})
            tactile = row.get("tactile_diagnostics", {})
            action = row.get("executed_action_stats", {})
            mappings = {
                "final_pinch_count": progress.get("final_pinch_count"),
                "max_pinch_count": progress.get("max_pinch_count"),
                "max_normal_force_proxy": tactile.get("max_normal_force"),
                "mean_tactile_l2_proxy": tactile.get("mean_l2"),
                "active_contact_samples": tactile.get("active_samples"),
                "matched_contact_count_sum": tactile.get(
                    "matched_contact_count_sum"
                ),
                "executed_action_mean_l2": action.get("mean_l2"),
                "termination_steps": row.get("steps"),
            }
            for name, value in mappings.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    fields[name].append(float(value))
        metrics[model] = {
            name: numeric_summary(values) for name, values in fields.items()
        }
    return {
        "schema": "tactile3d-unit.s4-3-pi2n-contact-process-metrics.v1",
        "status": "PASS",
        "created_at": now(),
        "models": metrics,
        "physics_force_units": "DexJoCo telemetry units as emitted by the frozen evaluator; not a real tactile sensor",
        "force_integral": "NA_NOT_EMITTED_BY_FROZEN_RUNTIME",
        "tangential_force": "NA_NOT_EMITTED_BY_FROZEN_RUNTIME",
        "grasp_stage_time": "NA_NOT_EMITTED_BY_FROZEN_RUNTIME",
        "slip_or_torque": "NA_NOT_EMITTED_BY_FROZEN_RUNTIME",
        "native_success_unchanged": True,
        "privileged_metrics_sent_to_policy": False,
    }


def axis_from_pair(value: str, *, positive: str, negative: str) -> str:
    if value == "POSITIVE_CONFIRMED":
        return positive
    if value == "NEGATIVE_CONFIRMED":
        return negative
    return "INCONCLUSIVE"


def analyze(
    rows_by_model: dict[str, list[dict[str, Any]]],
    protocol: dict[str, Any],
    pre_freeze: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    statistical = protocol["formal_statistics"]
    resamples = int(statistical["paired_bootstrap_samples"])
    seed = int(statistical["paired_bootstrap_seed"])
    outcomes = {
        model: np.asarray(
            [bool(row["success"]) for row in rows], dtype=np.bool_
        )
        for model, rows in rows_by_model.items()
    }
    model_statistics: dict[str, Any] = {}
    for model, rows in rows_by_model.items():
        successes = int(outcomes[model].sum())
        interval = wilson(successes, 200)
        model_statistics[model] = {
            "display_model": "VAC_STAR" if model == "B_VAC_V" else model,
            "successes": successes,
            "n": 200,
            "success_rate": successes / 200,
            "success_percent": 100 * successes / 200,
            "wilson_95ci": list(interval),
            "wilson_95ci_percent": [100 * interval[0], 100 * interval[1]],
            "termination_counts": dict(
                Counter(row["termination"] for row in rows)
            ),
            "mean_steps": float(np.mean([row["steps"] for row in rows])),
        }

    paired: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    for name, left_name, right_name in COMPARISONS:
        left, right = outcomes[left_name], outcomes[right_name]
        both_success = int(np.count_nonzero(left & right))
        left_only = int(np.count_nonzero(left & ~right))
        right_only = int(np.count_nonzero(~left & right))
        both_failure = int(np.count_nonzero(~left & ~right))
        differences = left.astype(np.float64) - right.astype(np.float64)
        delta = float(differences.mean())
        interval = paired_bootstrap(
            differences, resamples=resamples, seed=seed
        )
        p_value = exact_mcnemar(left_only, right_only)
        raw_p[name] = p_value
        paired[name] = {
            "left": "VAC_STAR" if left_name == "B_VAC_V" else left_name,
            "right": right_name,
            "left_checkpoint_model": left_name,
            "right_checkpoint_model": right_name,
            "table": {
                "both_success": both_success,
                "left_only_success": left_only,
                "right_only_success": right_only,
                "both_failure": both_failure,
            },
            "risk_difference": delta,
            "risk_difference_percentage_points": 100 * delta,
            "paired_bootstrap_95ci": list(interval),
            "paired_bootstrap_95ci_percentage_points": [
                100 * interval[0],
                100 * interval[1],
            ],
            "bootstrap_resamples": resamples,
            "bootstrap_seed": seed,
            "mcnemar_exact_two_sided_raw_p": p_value,
        }
    adjusted = holm_adjust(raw_p)
    for name, row in paired.items():
        row["holm_adjusted_p"] = adjusted[name]
        lower, upper = row["paired_bootstrap_95ci"]
        row["classification"] = classification(
            row["risk_difference"], lower, upper, adjusted[name]
        )
        row["large_effect_point_estimate"] = (
            abs(row["risk_difference_percentage_points"])
            >= float(statistical["material_threshold_pp"])
        )
        row["material_confirmed"] = bool(
            row["large_effect_point_estimate"]
            and row["classification"] != "INCONCLUSIVE"
        )

    interaction_vector = (
        outcomes["B_HVA"].astype(np.float64)
        - outcomes["B_VA27"].astype(np.float64)
        - outcomes["B1"].astype(np.float64)
        + outcomes["B0"].astype(np.float64)
    )
    interaction_interval = paired_bootstrap(
        interaction_vector, resamples=resamples, seed=seed
    )
    interaction = {
        "definition": "[p(B_HVA)-p(B_VA27)]-[p(B1)-p(B0)]",
        "estimate": float(interaction_vector.mean()),
        "estimate_percentage_points": 100 * float(interaction_vector.mean()),
        "paired_bootstrap_95ci": list(interaction_interval),
        "paired_bootstrap_95ci_percentage_points": [
            100 * interaction_interval[0],
            100 * interaction_interval[1],
        ],
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "causal_synergy_claimed": False,
    }
    statistics = {
        "schema": "tactile3d-unit.s4-3-pi2n-paired-statistics.v1",
        "status": "PASS",
        "created_at": now(),
        "claim_level": "FIXED_SEED_ONLY",
        "vac_star": pre_freeze["vac_star"],
        "model_statistics": model_statistics,
        "paired_comparisons": paired,
        "secondary_difference_in_differences": interaction,
        "multiplicity_family": [name for name, _, _ in COMPARISONS],
        "holm_family_size": len(COMPARISONS),
        "protocol_sha256": sha256_file(PROTOCOL),
        "pre_final_freeze_sha256": sha256_file(PRE_FREEZE),
        "rollout_completeness_sha256": sha256_file(COMPLETENESS),
        "inconclusive_interpreted_as_equivalence": False,
    }
    process = process_metrics(rows_by_model)
    primary = paired["VAC_STAR-B_HVA"]["classification"]
    refinement = paired["VAC_STAR-B2"]["classification"]
    h_increment = paired["B_HVA-B_VA27"]["classification"]
    full_method = paired["VAC_STAR-B0"]["classification"]
    claim = {
        "schema": "tactile3d-unit.s4-3-pi2n-claim-freeze.v1",
        "ENGINEERING_STATUS": "COMPLETE_VALID",
        "VAC_READOUT_DIAGNOSIS": "INCONCLUSIVE",
        "VAC_VS_STRONG_VA": axis_from_pair(
            primary,
            positive="VAC_ADVANTAGE_CONFIRMED",
            negative="STRONG_VA_ADVANTAGE_CONFIRMED",
        ),
        "VAC_REFINEMENT_VS_B2": axis_from_pair(
            refinement,
            positive="IMPROVEMENT_CONFIRMED",
            negative="HURT_CONFIRMED",
        ),
        "H_INCREMENT_GIVEN_VA": axis_from_pair(
            h_increment,
            positive="POSITIVE_CONFIRMED",
            negative="NEGATIVE_CONFIRMED",
        ),
        "FULL_METHOD_VS_B0": axis_from_pair(
            full_method,
            positive="IMPROVEMENT_CONFIRMED",
            negative="HURT_CONFIRMED",
        ),
        "CLAIM_LEVEL": "FIXED_SEED_ONLY",
        "vac_star": pre_freeze["vac_star"],
        "teacher_comparison_causal_isolation_claimed": False,
        "cross_training_seed_stability_claimed": False,
        "cross_task_or_real_robot_claimed": False,
        "inconclusive_interpreted_as_equivalence": False,
    }
    decision = {
        "schema": "tactile3d-unit.s4-3-pi2n-final-decision.v1",
        "status": "COMPLETE_VALID",
        "created_at": now(),
        "decision_axes": claim,
        "statistics_sha256": None,
        "contact_process_metrics_sha256": None,
        "PI2B_started": False,
        "real_robot_started": False,
        "push_performed": False,
    }
    return statistics, process, claim, decision


def main() -> None:
    outputs = (STATISTICS, PROCESS_METRICS, CLAIM_FREEZE, FINAL_DECISION)
    if any(path.exists() for path in outputs):
        raise SystemExit("refusing to overwrite PI2N FINAL analysis")
    required = (PROTOCOL, PRE_FREEZE, COMPLETENESS, RESET_MANIFEST)
    if any(not path.is_file() for path in required):
        raise SystemExit("PI2N FINAL analysis prerequisite missing")
    protocol = json.loads(PROTOCOL.read_text())
    pre_freeze = json.loads(PRE_FREEZE.read_text())
    completeness = json.loads(COMPLETENESS.read_text())
    reset_manifest = json.loads(RESET_MANIFEST.read_text())
    if (
        pre_freeze.get("status") != "PASS"
        or completeness.get("status") != "PASS"
        or completeness.get("total_canonical_outcomes") != 1200
        or reset_manifest.get("status") != "PASS"
    ):
        raise SystemExit("PI2N FINAL analysis integrity gate failed")
    if protocol["formal_comparisons"] != [name for name, _, _ in COMPARISONS]:
        raise SystemExit("PI2N FINAL comparison family drifted")
    rows_by_model = load_canonical_rows(completeness, reset_manifest)
    statistics, process, claim, decision = analyze(
        rows_by_model, protocol, pre_freeze
    )
    atomic_json(STATISTICS, statistics)
    atomic_json(PROCESS_METRICS, process)
    claim["statistics_sha256"] = sha256_file(STATISTICS)
    claim["contact_process_metrics_sha256"] = sha256_file(PROCESS_METRICS)
    atomic_json(CLAIM_FREEZE, claim)
    decision["statistics_sha256"] = sha256_file(STATISTICS)
    decision["contact_process_metrics_sha256"] = sha256_file(PROCESS_METRICS)
    decision["claim_freeze_sha256"] = sha256_file(CLAIM_FREEZE)
    atomic_json(FINAL_DECISION, decision)
    print(json.dumps(claim, sort_keys=True))


if __name__ == "__main__":
    main()
