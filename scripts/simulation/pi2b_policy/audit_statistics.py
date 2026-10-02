#!/usr/bin/env python3
"""Independently audit Track-A statistics with the Python standard library."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2b_policy"
COMPLETENESS = ARTIFACTS / "rollout_completeness.json"
PER_SEED = ARTIFACTS / "per_seed_statistics.json"
CROSSED = ARTIFACTS / "crossed_seed_reset_analysis.json"
NEW_ONLY = ARTIFACTS / "new_seeds_only_sensitivity.json"
DECISION = ARTIFACTS / "final_decision.json"
OUTPUT = ARTIFACTS / "statistics_independent_audit.json"
MODELS = ("B0", "B_VA27", "B1", "B_HVA", "B2")
SEEDS = (42, 43, 44)
FAMILIES = {
    "primary": (("B_HVA-B_VA27", "B_HVA", "B_VA27"), ("B2-B_HVA", "B2", "B_HVA")),
    "key_secondary": (
        ("B2-B1", "B2", "B1"),
        ("B_HVA-B0", "B_HVA", "B0"),
        ("B2-B0", "B2", "B0"),
    ),
    "auxiliary_secondary": (("B_VA27-B0", "B_VA27", "B0"), ("B1-B0", "B1", "B0")),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(first: float, second: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(first), float(second), rel_tol=tolerance, abs_tol=tolerance)


def wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return (0.0 if successes == 0 else center - half, 1.0 if successes == total else center + half)


def paired(first: list[bool], second: list[bool]) -> tuple[float, dict[str, int], float]:
    table = {
        "both_fail": sum(not a and not b for a, b in zip(first, second, strict=True)),
        "first_only": sum(a and not b for a, b in zip(first, second, strict=True)),
        "second_only": sum(not a and b for a, b in zip(first, second, strict=True)),
        "both_success": sum(a and b for a, b in zip(first, second, strict=True)),
    }
    discordant = table["first_only"] + table["second_only"]
    if discordant:
        tail = sum(math.comb(discordant, index) for index in range(min(table["first_only"], table["second_only"]) + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    else:
        p_value = 1.0
    difference = statistics.fmean(int(a) - int(b) for a, b in zip(first, second, strict=True))
    return difference, table, p_value


def holm(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def sign_flip(values: list[float]) -> float:
    observed = abs(statistics.fmean(values))
    samples = [
        abs(statistics.fmean(sign * value for sign, value in zip(signs, values, strict=True)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return sum(value + 1e-15 >= observed for value in samples) / len(samples)


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise SystemExit("independent statistics audit failure: " + message)
    checks.append(message)


def main() -> None:
    completeness = read_json(COMPLETENESS)
    per_seed = read_json(PER_SEED)
    crossed = read_json(CROSSED)
    new_only = read_json(NEW_ONLY)
    decision = read_json(DECISION)
    checks: list[str] = []
    outcomes: dict[tuple[str, int], dict[str, bool]] = {
        (model, seed): {} for model in MODELS for seed in SEEDS
    }
    terminations: dict[tuple[str, int], Counter[str]] = {
        key: Counter() for key in outcomes
    }
    steps: dict[tuple[str, int], list[int]] = {key: [] for key in outcomes}
    raw_hashes: dict[str, str] = {}
    for name, record in completeness["raw_artifacts"].items():
        path = Path(record["path"])
        raw_hashes[name] = sha256_file(path)
        require(raw_hashes[name] == record["sha256"], f"raw hash {name}", checks)
        payload = read_json(path)
        key = (payload["model"], int(payload["training_seed"]))
        for row in payload["episode_results"]:
            identity = row["reset_identity"]
            require(identity not in outcomes[key], f"unique tuple {key}/{identity}", checks)
            outcomes[key][identity] = bool(row["success"])
            terminations[key][row["termination"]] += 1
            steps[key].append(int(row["steps"]))
    require(len(raw_hashes) == 60, "60 raw blocks", checks)
    require(sum(len(rows) for rows in outcomes.values()) == 3000, "3000 canonical rows", checks)
    identity_order = list(next(iter(outcomes.values())).keys())
    for key, values in outcomes.items():
        require(len(values) == 200 and list(values.keys()) == identity_order, f"shared ordered resets {key}", checks)

    vectors = {key: list(values.values()) for key, values in outcomes.items()}
    for model in MODELS:
        for seed in SEEDS:
            key = (model, seed)
            report = per_seed["checkpoint_statistics"][model][str(seed)]
            successes = sum(vectors[key])
            interval = wilson(successes, 200)
            require(report["successes"] == successes, f"success count {key}", checks)
            require(report["failures"] == 200 - successes, f"failure count {key}", checks)
            require(report["termination_counts"] == dict(sorted(terminations[key].items())), f"termination counts {key}", checks)
            require(close(report["steps"]["mean"], statistics.fmean(steps[key])), f"mean steps {key}", checks)
            require(all(close(a, b) for a, b in zip(report["wilson_95ci"], interval, strict=True)), f"Wilson interval {key}", checks)

    contrast_seed_deltas: dict[str, list[float]] = {row[0]: [] for family in FAMILIES.values() for row in family}
    for seed in SEEDS:
        for family, members in FAMILIES.items():
            raw_p_values = []
            local = []
            for name, first_name, second_name in members:
                difference, table, p_value = paired(vectors[(first_name, seed)], vectors[(second_name, seed)])
                report = per_seed["paired_comparisons"][str(seed)][name]
                require(close(report["risk_difference"], difference), f"paired difference seed{seed}/{name}", checks)
                require(report["discordant_table"] == table, f"discordant table seed{seed}/{name}", checks)
                require(close(report["mcnemar_exact_two_sided_raw_p"], p_value), f"McNemar seed{seed}/{name}", checks)
                require(report["paired_tuple_bootstrap_95ci"][0] <= difference <= report["paired_tuple_bootstrap_95ci"][1], f"paired bootstrap contains estimate seed{seed}/{name}", checks)
                raw_p_values.append(p_value)
                local.append((name, report))
                contrast_seed_deltas[name].append(difference)
            adjusted = holm(raw_p_values)
            for (_, report), expected in zip(local, adjusted, strict=True):
                require(close(report["holm_adjusted_p_within_seed_family"], expected), f"Holm seed{seed}/{family}", checks)

    for name, deltas in contrast_seed_deltas.items():
        report = crossed["contrasts"][name]
        expected_mean = statistics.fmean(deltas)
        expected_sd = statistics.stdev(deltas)
        require(close(report["training_seed_summary"]["mean"], expected_mean), f"crossed mean {name}", checks)
        require(close(report["training_seed_summary"]["sample_sd"], expected_sd), f"crossed SD {name}", checks)
        require(all(close(a, b) for a, b in zip(report["training_seed_summary"]["range"], (min(deltas), max(deltas)), strict=True)), f"crossed range {name}", checks)
        require(close(report["training_seed_summary"]["exact_two_sided_sign_flip_p"], sign_flip(deltas)), f"sign flip {name}", checks)
        require(close(report["conditional_shared_reset_bootstrap"]["estimate"], expected_mean), f"conditional estimate {name}", checks)
        require(close(report["two_way_seed_by_reset_sensitivity"]["estimate"], expected_mean), f"two-way estimate {name}", checks)
        new_deltas = deltas[1:]
        new_report = new_only["contrasts"][name]
        require(close(new_report["training_seed_summary"]["mean"], statistics.fmean(new_deltas)), f"new-only mean {name}", checks)
        require(close(new_report["training_seed_summary"]["exact_two_sided_sign_flip_p"], sign_flip(new_deltas)), f"new-only sign flip {name}", checks)

    interaction = [
        contrast_seed_deltas["B_HVA-B_VA27"][index] - contrast_seed_deltas["B1-B0"][index]
        for index in range(3)
    ]
    interaction_report = crossed["interaction_diagnostic"]
    require(close(interaction_report["training_seed_summary"]["mean"], statistics.fmean(interaction)), "interaction mean", checks)
    require(close(interaction_report["training_seed_summary"]["exact_two_sided_sign_flip_p"], sign_flip(interaction)), "interaction sign flip", checks)
    require(decision["ENGINEERING_STATUS"] == "COMPLETE_VALID", "engineering complete-valid", checks)
    require(decision["TRAINING_SEED_SCOPE"] == "THREE_SEEDS_LIMITED", "limited seed scope", checks)
    require(decision["EQUIVALENCE_CLAIMED"] is False, "no equivalence claim", checks)
    require(decision["TRACK_B_NEW_TEACHER_READ"] is False, "Track-B isolated", checks)

    payload = {
        "schema": "tactile3d-unit.s4-3-pi2b-policy-statistics-independent-audit.v1",
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation": "Python standard library only; does not import Track-A analysis/statistics modules",
        "checks_performed": len(checks),
        "gates": {
            "raw_hashes_and_cardinality": "PASS",
            "checkpoint_counts_wilson_terminations_steps": "PASS",
            "paired_differences_discordance_exact_mcnemar": "PASS",
            "within_seed_family_holm": "PASS",
            "crossed_seed_mean_sd_range_sign_flip": "PASS",
            "new_seeds_only_mean_sign_flip": "PASS",
            "bootstrap_estimates_and_containment": "PASS",
            "interaction_arithmetic": "PASS",
            "decision_scope_and_isolation": "PASS",
        },
        "total_successes": sum(sum(vector) for vector in vectors.values()),
        "total_failures": 3000 - sum(sum(vector) for vector in vectors.values()),
        "input_sha256": {
            "rollout_completeness.json": sha256_file(COMPLETENESS),
            "per_seed_statistics.json": sha256_file(PER_SEED),
            "crossed_seed_reset_analysis.json": sha256_file(CROSSED),
            "new_seeds_only_sensitivity.json": sha256_file(NEW_ONLY),
            "final_decision.json": sha256_file(DECISION),
        },
        "raw_artifact_sha256": raw_hashes,
    }
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUTPUT)
    print(json.dumps({"status": "PASS", "checks": len(checks), "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
