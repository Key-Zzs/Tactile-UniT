#!/usr/bin/env python3
"""Validate paired PI1D outputs and compute all preregistered statistics."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi1"
MODELS = ("R0", "B0", "B1", "B2")
PRIMARY = (("B0", "B1"), ("B1", "B2"), ("B0", "B2"))
BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEED = 4301


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(name: str, payload: Any) -> None:
    path = ARTIFACTS / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def wilson(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def exact_mcnemar(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(first_only, second_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_contrast(first: str, second: str, outcomes: dict[str, np.ndarray]) -> dict[str, Any]:
    first_values, second_values = outcomes[first], outcomes[second]
    differences = second_values.astype(np.float64) - first_values.astype(np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(differences), size=(BOOTSTRAP_RESAMPLES, len(differences)))
    distribution = differences[indices].mean(axis=1)
    interval = np.quantile(distribution, [0.025, 0.975], method="linear").tolist()
    delta = float(differences.mean())
    both_success = int(np.sum(first_values & second_values))
    first_only = int(np.sum(first_values & ~second_values))
    second_only = int(np.sum(~first_values & second_values))
    both_fail = int(np.sum(~first_values & ~second_values))
    if delta >= 0.10 and interval[0] > 0:
        material = "MATERIAL_IMPROVEMENT"
    elif delta <= -0.10 and interval[1] < 0:
        material = "MATERIAL_HURT"
    else:
        material = "NO_MATERIAL_DIFFERENCE"
    trend = "POSITIVE_TREND" if delta > 0 else "NEGATIVE_TREND" if delta < 0 else "NO_POINT_DIFFERENCE"
    return {
        "first": first,
        "second": second,
        "effect": f"{second}-{first}",
        "success_difference": delta,
        "percentage_point_difference": 100 * delta,
        "paired_bootstrap_95ci": interval,
        "paired_bootstrap_95ci_percentage_points": [100 * value for value in interval],
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "paired_outcome_table": {
            "both_success": both_success,
            "first_only": first_only,
            "second_only": second_only,
            "both_fail": both_fail,
        },
        "exact_mcnemar_two_sided_p": exact_mcnemar(first_only, second_only),
        "material_classification": material,
        "descriptive_trend": trend,
    }


def decide(contrasts: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    b1_b0 = contrasts["B1-B0"]["material_classification"]
    b2_b1 = contrasts["B2-B1"]["material_classification"]
    b2_b0 = contrasts["B2-B0"]["material_classification"]
    values = {b1_b0, b2_b1, b2_b0}
    if "MATERIAL_HURT" in values and "MATERIAL_IMPROVEMENT" in values:
        return "S4_3_PI1D_MIXED_OR_INCONCLUSIVE", [
            "Primary paired contrasts contain both material improvement and material hurt.",
            "The frozen 50-episode evidence does not support one monotonic method interpretation.",
        ]
    if "MATERIAL_HURT" in values:
        return "S4_3_PI1D_TACTILE_UNIT_HURTS_PI05", [
            "At least one primary train-matched tactile contrast meets the frozen material-hurt rule.",
            "The conclusion follows paired rollouts rather than training loss.",
        ]
    if b2_b0 == "MATERIAL_IMPROVEMENT" and b2_b1 == "MATERIAL_IMPROVEMENT":
        return "S4_3_PI1D_FULL_TACTILE_UNIT_IMPROVES_PI05", [
            "B2 materially improves over the train-matched B0 baseline.",
            "B2 also materially improves over B1, isolating added physical-auxiliary benefit.",
        ]
    if b1_b0 == "MATERIAL_IMPROVEMENT" and b2_b1 != "MATERIAL_IMPROVEMENT":
        return "S4_3_PI1D_CONTACT_STATE_TOKENS_IMPROVE_PI05", [
            "B1 materially improves over train-matched B0.",
            "B2 does not materially improve over B1.",
        ]
    if b2_b1 == "MATERIAL_IMPROVEMENT" and b1_b0 != "MATERIAL_IMPROVEMENT":
        return "S4_3_PI1D_PHYSICAL_AUXILIARY_IMPROVES_PI05", [
            "B2 materially improves over B1.",
            "B1 does not materially improve over train-matched B0.",
        ]
    if values == {"NO_MATERIAL_DIFFERENCE"}:
        return "S4_3_PI1D_NO_MATERIAL_GAIN", [
            "All evaluation pipelines passed, but no primary contrast meets the frozen material-gain rule.",
            "No primary contrast meets the material-hurt rule.",
        ]
    return "S4_3_PI1D_MIXED_OR_INCONCLUSIVE", [
        "The primary contrasts do not map uniquely to a preregistered improvement category.",
        "The 50 paired episodes are retained without post-hoc model selection.",
    ]


def main() -> None:
    freeze_path = ARTIFACTS / "pre_pi1d_freeze.json"
    freeze = json.loads(freeze_path.read_text())
    if freeze["status"] != "PASS" or freeze["evaluation_performance_seen"] is not False:
        raise SystemExit("PI1D pre-evaluation freeze is invalid")
    artifacts = {
        model: json.loads((ARTIFACTS / f"pi1d_{model.lower()}_eval.json").read_text())
        for model in MODELS
    }
    reset_sequences = {
        model: [row["reset_identity"] for row in artifact["episode_results"]]
        for model, artifact in artifacts.items()
    }
    outcomes = {
        model: np.asarray([row["success"] for row in artifact["episode_results"]], dtype=bool)
        for model, artifact in artifacts.items()
    }
    gates = {
        "all_model_runtime_artifacts_pass": all(artifact["status"] == "PASS" for artifact in artifacts.values()),
        "evaluation_order_R0_B0_B1_B2": [
            model for model, _ in sorted(artifacts.items(), key=lambda item: (ARTIFACTS / f"pi1d_{item[0].lower()}_eval.json").stat().st_mtime_ns)
        ]
        == list(MODELS),
        "exactly_50_each": all(len(values) == 50 for values in outcomes.values()),
        "same_ordered_reset_identities": all(reset_sequences[model] == reset_sequences["R0"] for model in MODELS[1:]),
        "no_reset_replacement": all(
            [row["episode_index"] for row in artifact["episode_results"]] == list(range(50))
            for artifact in artifacts.values()
        ),
        "fresh_seed1": all(artifact["evaluator_seed"] == 1 for artifact in artifacts.values()),
        "same_official_semantics": all(
            artifact["physics_action_success_reset_camera_prompt_modified"] is False
            for artifact in artifacts.values()
        ),
        "B0_R0_tactile_ignored": all(
            artifacts[model]["contact_state_sent_to_policy"] is False for model in ("R0", "B0")
        ),
        "B1_B2_current_contact_consumed": all(
            artifacts[model]["contact_state_sent_to_policy"] is True for model in ("B1", "B2")
        ),
        "no_training_only_target_at_runtime": all(
            artifact["gates"]["training_only_targets_never_sent"] == "PASS"
            for artifact in artifacts.values()
        ),
        "no_failed_episode_deletion": all(
            artifact["gates"]["no_episode_deleted"] == "PASS" for artifact in artifacts.values()
        ),
        "no_model_selection_remains": freeze["no_model_selection_remains"] is True,
    }
    if not all(gates.values()):
        failed = [name for name, value in gates.items() if not value]
        raise SystemExit("S4_3_PI1D_RUNTIME_FAIL: " + ",".join(failed))

    model_results = {}
    for model in MODELS:
        successes = int(outcomes[model].sum())
        model_results[model] = {
            "successes": successes,
            "episodes": 50,
            "success_rate": successes / 50,
            "wilson_95ci": wilson(successes, 50),
            "reset_sequence_sha256": hashlib.sha256("\n".join(reset_sequences[model]).encode()).hexdigest(),
            "runtime_artifact": f"$REPO_ROOT/.local/artifacts/simulation/s4_3_pi1/pi1d_{model.lower()}_eval.json",
            "runtime_artifact_sha256": sha256_file(ARTIFACTS / f"pi1d_{model.lower()}_eval.json"),
        }
    contrasts = {}
    for first, second in (("B0", "B1"), ("B1", "B2"), ("B0", "B2"), ("B0", "R0")):
        row = paired_contrast(first, second, outcomes)
        contrasts[row["effect"]] = row

    decision, reasons = decide(contrasts)
    point_gains = [contrasts["B1-B0"]["success_difference"], contrasts["B2-B0"]["success_difference"]]
    reference_spread = abs(contrasts["R0-B0"]["success_difference"])
    if max(point_gains) >= 0.10 or any(value > 0 for value in point_gains) or reference_spread >= 0.10:
        multiseed = "MULTI_SEED_CONFIRMATION_RECOMMENDED"
        multiseed_reason = "A positive tactile trend/material point gain or >=10pp R0-B0 reproduction spread warrants seed-level confirmation."
    elif max(point_gains) <= 0 and all(
        contrasts[name]["descriptive_trend"] != "POSITIVE_TREND"
        for name in ("B1-B0", "B2-B1", "B2-B0")
    ):
        multiseed = "NEGATIVE_RESULT_STABLE_ENOUGH_FOR_DIAGNOSIS"
        multiseed_reason = "No primary contrast has a positive paired point trend in the frozen 50 episodes."
    else:
        multiseed = "NOT_YET_JUSTIFIED"
        multiseed_reason = "The frozen paired results do not yet justify the cost of three training seeds."

    payload = {
        "schema": "tactile3d-unit.s4-3-pi1d-paired-statistics.v1",
        "status": "PASS",
        "task": "pinch_tongs",
        "regime": "rand_obj",
        "episodes": 50,
        "fresh_evaluator_seed": 1,
        "evaluation_order": list(MODELS),
        "same_ordered_resets": True,
        "reset_sequence": reset_sequences["R0"],
        "model_results": model_results,
        "primary_contrasts": {name: contrasts[name] for name in ("B1-B0", "B2-B1", "B2-B0")},
        "reference_contrast": contrasts["R0-B0"],
        "reference_interpretation": "R0-B0 is contextual training-seed/reproduction spread, not a primary method effect.",
        "material_rule": {
            "improvement": "delta >= +10 percentage points AND paired CI lower > 0",
            "hurt": "delta <= -10 percentage points AND paired CI upper < 0",
            "otherwise": "NO_MATERIAL_DIFFERENCE",
        },
        "small_effect_precision_warning": "n=50; do not claim publication-level small-effect precision",
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
    }
    atomic_json("pi1d_paired_statistics.json", payload)
    final = {
        "schema": "tactile3d-unit.s4-3-pi1-final-decision.v1",
        "status": "PASS",
        "decision": decision,
        "reasons": reasons,
        "multi_seed_readiness": multiseed,
        "multi_seed_reason": multiseed_reason,
        "additional_training_launched": False,
        "paired_statistics": "$REPO_ROOT/.local/artifacts/simulation/s4_3_pi1/pi1d_paired_statistics.json",
        "paired_statistics_sha256": sha256_file(ARTIFACTS / "pi1d_paired_statistics.json"),
    }
    atomic_json("final_decision.json", final)
    acceptance = f"""# S4.3-PI1 Human Acceptance\n\n- PI1A alignment: PASS\n- PI1 mode implementation: PASS\n- PI1B 30k checkpoint freeze: PASS\n- PI1C 30k checkpoint freeze: PASS\n- PI1D fresh paired seed-1 reset contract: PASS\n- PI1D runtime/statistics: PASS\n- Final decision: `{decision}`\n- Multi-seed readiness: `{multiseed}`\n- Additional seeds launched: NO\n- Push performed: NO\n"""
    (ARTIFACTS / "HUMAN_ACCEPTANCE.md").write_text(acceptance)
    print(json.dumps({"status": "PASS", "results": model_results, "decision": decision, "multi_seed": multiseed}, sort_keys=True))


if __name__ == "__main__":
    main()
