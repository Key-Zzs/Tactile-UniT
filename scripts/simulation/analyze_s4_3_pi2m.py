#!/usr/bin/env python3
"""Analyze the frozen 200-reset B1/B_HVA/B2 matched-input comparison."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2m"
PROTOCOL = ROOT / "configs/simulation/s4_3_pi2m_statistical_protocol.json"
MODELS = ("B1", "B_HVA", "B2")
COMPARISONS = (
    ("B2-B_HVA", "B2", "B_HVA"),
    ("B_HVA-B1", "B_HVA", "B1"),
    ("B2-B1", "B2", "B1"),
)


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
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - radius, center + radius


def exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def paired_bootstrap(
    left: np.ndarray, right: np.ndarray, *, resamples: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    differences = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    values = np.empty(resamples, dtype=np.float64)
    chunk = 2_000
    for start in range(0, resamples, chunk):
        stop = min(resamples, start + chunk)
        indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
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


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    if protocol.get("status") != "FROZEN_BEFORE_BHVA_TRAINING":
        raise SystemExit("PI2M statistical protocol is not frozen")
    raw_paths = {model: ARTIFACTS / f"{model.lower()}_raw_rollouts.json" for model in MODELS}
    if any(not path.is_file() for path in raw_paths.values()):
        raise SystemExit("PI2M requires all three canonical raw rollout artifacts")
    raw = {model: json.loads(path.read_text()) for model, path in raw_paths.items()}
    completeness_gates = {
        f"{model}_PASS_200": payload.get("status") == "PASS"
        and payload.get("episodes") == 200
        and len(payload.get("episode_results", [])) == 200
        for model, payload in raw.items()
    }
    identities = {
        model: [row["reset_identity"] for row in payload["episode_results"]]
        for model, payload in raw.items()
    }
    completeness_gates.update(
        {
            "exactly_600_canonical_outcomes": sum(
                len(payload["episode_results"]) for payload in raw.values()
            )
            == 600,
            "200_triple_aligned_reset_identities": all(
                identities[model] == identities["B1"] for model in MODELS[1:]
            )
            and len(set(identities["B1"])) == 200,
            "no_duplicate_episode_indices": all(
                [row["episode_index"] for row in payload["episode_results"]]
                == list(range(200))
                for payload in raw.values()
            ),
            "seed7_all_models": all(payload.get("evaluator_seed") == 7 for payload in raw.values()),
            "matched_online_contact_delivery": all(
                payload.get("contact_state_sent_to_policy") is True for payload in raw.values()
            ),
            "training_targets_never_sent": all(
                payload["gates"].get("training_only_targets_never_sent") == "PASS"
                for payload in raw.values()
            ),
        }
    )
    completeness = {
        "schema": "tactile3d-unit.s4-3-pi2m-rollout-completeness.v1",
        "status": "PASS" if all(completeness_gates.values()) else "FAIL",
        "counts": {model: len(raw[model]["episode_results"]) for model in MODELS},
        "total": sum(len(raw[model]["episode_results"]) for model in MODELS),
        "reset_sequence_sha256": hashlib.sha256(
            "\n".join(identities["B1"]).encode()
        ).hexdigest(),
        "raw_sha256": {model: sha256_file(path) for model, path in raw_paths.items()},
        "gates": {
            name: "PASS" if value else "FAIL" for name, value in completeness_gates.items()
        },
    }
    atomic_json(ARTIFACTS / "rollout_completeness.json", completeness)
    if completeness["status"] != "PASS":
        raise SystemExit("PI2M_ROLLOUT_COMPLETENESS_FAIL")

    outcomes = {
        model: np.asarray(
            [bool(row["success"]) for row in raw[model]["episode_results"]], dtype=np.bool_
        )
        for model in MODELS
    }
    model_stats = {}
    for model in MODELS:
        episodes = raw[model]["episode_results"]
        successes = int(outcomes[model].sum())
        interval = wilson(successes, 200)
        model_stats[model] = {
            "successes": successes,
            "n": 200,
            "success_rate": successes / 200,
            "success_percent": 100 * successes / 200,
            "wilson_95ci": list(interval),
            "wilson_95ci_percent": [100 * interval[0], 100 * interval[1]],
            "termination_counts": dict(Counter(row["termination"] for row in episodes)),
            "mean_steps": float(np.mean([row["steps"] for row in episodes])),
            "contact_telemetry_available": all(
                "contact_state_diagnostics" in row and "tactile_diagnostics" in row
                for row in episodes
            ),
        }

    paired = {}
    raw_p = {}
    for name, left_name, right_name in COMPARISONS:
        left, right = outcomes[left_name], outcomes[right_name]
        both_success = int(np.count_nonzero(left & right))
        left_only = int(np.count_nonzero(left & ~right))
        right_only = int(np.count_nonzero(~left & right))
        both_failure = int(np.count_nonzero(~left & ~right))
        delta = float(left.mean() - right.mean())
        interval = paired_bootstrap(
            left,
            right,
            resamples=int(protocol["bootstrap_resamples"]),
            seed=int(protocol["bootstrap_seed"]),
        )
        p_value = exact_mcnemar(left_only, right_only)
        raw_p[name] = p_value
        paired[name] = {
            "left": left_name,
            "right": right_name,
            "table": {
                "both_success": both_success,
                "left_only_success": left_only,
                "right_only_success": right_only,
                "both_failure": both_failure,
            },
            "risk_difference": delta,
            "risk_difference_percentage_points": 100 * delta,
            "paired_bootstrap_95ci": list(interval),
            "paired_bootstrap_95ci_percentage_points": [100 * interval[0], 100 * interval[1]],
            "bootstrap_resamples": int(protocol["bootstrap_resamples"]),
            "bootstrap_seed": int(protocol["bootstrap_seed"]),
            "mcnemar_exact_two_sided_raw_p": p_value,
        }
    adjusted = holm_adjust(raw_p)
    for name, row in paired.items():
        row["holm_adjusted_p"] = adjusted[name]
        lower, upper = row["paired_bootstrap_95ci"]
        delta = row["risk_difference"]
        if delta > 0 and lower > 0 and adjusted[name] < 0.05:
            classification = "POSITIVE_CONFIRMED"
        elif delta < 0 and upper < 0 and adjusted[name] < 0.05:
            classification = "NEGATIVE_CONFIRMED"
        else:
            classification = "INCONCLUSIVE"
        row["classification"] = classification
        row["large_effect_point_estimate"] = abs(row["risk_difference_percentage_points"]) >= 10
        row["material_confirmed"] = row["large_effect_point_estimate"] and classification != "INCONCLUSIVE"

    payload = {
        "schema": "tactile3d-unit.s4-3-pi2m-paired-statistics.v1",
        "status": "PASS",
        "claim_level": "FIXED_SEED_ONLY",
        "model_statistics": model_stats,
        "paired_comparisons": paired,
        "protocol_sha256": sha256_file(PROTOCOL),
        "multiplicity_family": list(name for name, _, _ in COMPARISONS),
        "historical_BVA_excluded": True,
        "historical_BVA_actual_target_horizon_seconds": 0.32,
    }
    atomic_json(ARTIFACTS / "paired_statistics.json", payload)

    primary = paired["B2-B_HVA"]["classification"]
    if primary == "POSITIVE_CONFIRMED":
        primary_effect = "CONTACT_TARGET_ADVANTAGE_CONFIRMED"
    elif primary == "NEGATIVE_CONFIRMED":
        primary_effect = "VA_TARGET_ADVANTAGE_CONFIRMED"
    else:
        primary_effect = "INCONCLUSIVE"
    claim = {
        "schema": "tactile3d-unit.s4-3-pi2m-claim-freeze.v1",
        "ENGINEERING_STATUS": "COMPLETE_VALID",
        "PRIMARY_TARGET_EFFECT": primary_effect,
        "VA_AUX_VS_NO_AUX": paired["B_HVA-B1"]["classification"].lower(),
        "CONTACT_AUX_REPLICATION": paired["B2-B1"]["classification"].lower(),
        "CLAIM_LEVEL": "FIXED_SEED_ONLY",
        "cross_modal_alignment_necessity_proven": False,
        "inconclusive_interpreted_as_equivalence": False,
        "statistics_sha256": sha256_file(ARTIFACTS / "paired_statistics.json"),
    }
    atomic_json(ARTIFACTS / "claim_freeze.json", claim)
    print(json.dumps(claim, sort_keys=True))


if __name__ == "__main__":
    main()
