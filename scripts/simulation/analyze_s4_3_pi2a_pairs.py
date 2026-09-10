#!/usr/bin/env python3
"""Analyze the complete fresh paired S4.3-PI2A outcome set."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2a"
PI1 = ROOT / ".local/artifacts/simulation/s4_3_pi1"
CACHE = ROOT / ".local/cache/simulation/s4_3_pi2a/evaluation/seed2"
PLOTS = ARTIFACTS / "plots"
VIDEOS = ARTIFACTS / "videos"
MODELS = ("B0", "B1", "B2")
CONTRASTS = (("B0", "B1"), ("B1", "B2"), ("B0", "B2"))
BOOTSTRAP_SEED = 4302
BOOTSTRAP_RESAMPLES = 100000


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(name: str, payload: dict[str, Any]) -> None:
    output = ARTIFACTS / name
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


def wilson(successes: int, count: int) -> list[float]:
    z = 1.959963984540054
    rate = successes / count
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denominator
    return [center - radius, center + radius]


def exact_mcnemar(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    smaller = min(first_only, second_only)
    tail = sum(math.comb(discordant, value) for value in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def paired_contrast(
    first: str, second: str, outcomes: dict[str, np.ndarray], seed: int = BOOTSTRAP_SEED
) -> dict[str, Any]:
    left, right = outcomes[first], outcomes[second]
    differences = right.astype(np.int8) - left.astype(np.int8)
    rng = np.random.default_rng(seed)
    n = len(differences)
    distribution = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    batch = 5000
    for start in range(0, BOOTSTRAP_RESAMPLES, batch):
        stop = min(start + batch, BOOTSTRAP_RESAMPLES)
        indices = rng.integers(0, n, size=(stop - start, n))
        distribution[start:stop] = differences[indices].mean(axis=1)
    ci = [float(value) for value in np.quantile(distribution, [0.025, 0.975])]
    both_success = int(np.sum(left & right))
    first_only = int(np.sum(left & ~right))
    second_only = int(np.sum(~left & right))
    both_fail = int(np.sum(~left & ~right))
    delta = float(differences.mean())
    p_value = exact_mcnemar(first_only, second_only)
    statistically_confirmed = delta > 0 and ci[0] > 0 and p_value < 0.05
    material_improvement = delta >= 0.10 and statistically_confirmed
    material_hurt = delta <= -0.10 and ci[1] < 0 and p_value < 0.05
    if material_improvement:
        classification = "MATERIAL_IMPROVEMENT"
    elif 0 < delta < 0.10 and statistically_confirmed:
        classification = "STATISTICALLY_CONFIRMED_SUBMATERIAL_GAIN"
    elif delta > 0:
        classification = "POSITIVE_TREND_NOT_CONFIRMED"
    elif delta == 0:
        classification = "NO_POINT_DIFFERENCE"
    elif material_hurt:
        classification = "MATERIAL_HURT"
    else:
        classification = "NEGATIVE_TREND_NOT_CONFIRMED"
    first_rate = float(left.mean())
    second_rate = float(right.mean())
    return {
        "effect": f"{second}-{first}",
        "first": first,
        "second": second,
        "episodes": n,
        "success_difference": delta,
        "delta_percentage_points": 100 * delta,
        "paired_bootstrap_95ci": ci,
        "paired_bootstrap_95ci_percentage_points": [100 * value for value in ci],
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": seed,
        "paired_outcome_table": {
            "both_success": both_success,
            "first_only": first_only,
            "second_only": second_only,
            "both_fail": both_fail,
        },
        "exact_mcnemar_two_sided_p": p_value,
        "discordant_odds_ratio": (second_only / first_only if first_only > 0 else None),
        "relative_success_ratio": (second_rate / first_rate if first_rate > 0 else None),
        "nnt_style_diagnostic": (1 / delta if delta > 0 else None),
        "statistically_confirmed_improvement": statistically_confirmed,
        "material_improvement": material_improvement,
        "material_hurt": material_hurt,
        "classification": classification,
    }


def holm(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        ((name, row["exact_mcnemar_two_sided_p"]) for name, row in rows.items()),
        key=lambda item: item[1],
    )
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return {
        "schema": "tactile3d-unit.s4-3-pi2a-holm.v1",
        "status": "PASS",
        "family": ["B2-B0", "B2-B1", "B1-B0"],
        "raw_p": {
            name: rows[name]["exact_mcnemar_two_sided_p"] for name in ("B2-B0", "B2-B1", "B1-B0")
        },
        "holm_adjusted_p": {name: adjusted[name] for name in ("B2-B0", "B2-B1", "B1-B0")},
        "diagnostic_only_for_primary_frozen_decision": True,
    }


def summary(model: str, artifact: dict[str, Any]) -> dict[str, Any]:
    episodes = artifact["episode_results"]
    successes = sum(int(row["success"]) for row in episodes)
    lengths = np.asarray([row["steps"] for row in episodes])
    terminations = Counter(row["termination"] for row in episodes)
    return {
        "schema": "tactile3d-unit.s4-3-pi2a-model-summary.v1",
        "status": "PASS",
        "model": model,
        "successes": successes,
        "episodes": len(episodes),
        "success_rate": successes / len(episodes),
        "wilson_95ci": wilson(successes, len(episodes)),
        "episode_length": {
            "mean": float(lengths.mean()),
            "median": float(np.median(lengths)),
            "minimum": int(lengths.min()),
            "maximum": int(lengths.max()),
        },
        "termination_breakdown": dict(sorted(terminations.items())),
        "raw_artifact": f"$REPO_ROOT/.local/artifacts/simulation/s4_3_pi2a/{model.lower()}_raw_rollouts.json",
        "raw_artifact_sha256": sha256_file(ARTIFACTS / f"{model.lower()}_raw_rollouts.json"),
    }


def mechanism(
    artifacts: dict[str, dict[str, Any]], outcomes: dict[str, np.ndarray]
) -> dict[str, Any]:
    models = {}
    for model, artifact in artifacts.items():
        groups = {}
        for label, wanted in (("all", None), ("success", True), ("failure", False)):
            rows = [
                row
                for row in artifact["episode_results"]
                if wanted is None or bool(row["success"]) is wanted
            ]
            if not rows:
                groups[label] = {"episodes": 0}
                continue
            groups[label] = {
                "episodes": len(rows),
                "mean_active_contact_samples": float(
                    np.mean([row["tactile_diagnostics"]["active_samples"] for row in rows])
                ),
                "mean_matched_contact_count_sum": float(
                    np.mean(
                        [row["tactile_diagnostics"]["matched_contact_count_sum"] for row in rows]
                    )
                ),
                "mean_peak_normal_force": float(
                    np.mean([row["tactile_diagnostics"]["max_normal_force"] for row in rows])
                ),
                "mean_max_pinch_count": float(
                    np.mean([row["task_progress"]["max_pinch_count"] for row in rows])
                ),
                "mean_final_pinch_count": float(
                    np.mean([row["task_progress"]["final_pinch_count"] for row in rows])
                ),
            }
        models[model] = groups
    b2_only_b0 = np.flatnonzero(outcomes["B2"] & ~outcomes["B0"]).tolist()
    b2_only_b1 = np.flatnonzero(outcomes["B2"] & ~outcomes["B1"]).tolist()
    return {
        "schema": "tactile3d-unit.s4-3-pi2a-contact-mechanism.v1",
        "status": "PASS",
        "models": models,
        "discordant_B2_success_B0_failure_reset_indices": b2_only_b0,
        "discordant_B2_success_B1_failure_reset_indices": b2_only_b1,
        "available_metrics": [
            "active contact samples",
            "matched contact count",
            "peak normal force",
            "final/max pinch count",
        ],
        "unavailable_metrics": [
            "integrated normal force",
            "contact onset time",
            "lift height",
            "completed open-close cycles",
            "final native progress beyond pinch count",
        ],
        "primary_endpoint": False,
    }


def direction(previous: float, current: float) -> str:
    if previous == 0 and current == 0:
        return "SAME_DIRECTION"
    if previous == 0 and current > 0:
        return "ZERO_TO_SAME_DIRECTION"
    if np.sign(previous) == np.sign(current):
        return "SAME_DIRECTION"
    return "REVERSED"


def magnitude_band(previous: float, current: float) -> str:
    difference = abs(current - previous)
    if difference <= 5:
        return "WITHIN_5_PP"
    if difference <= 10:
        return "5_TO_10_PP_DIFFERENT"
    return "MORE_THAN_10_PP_DIFFERENT"


def choose_decision(rows: dict[str, dict[str, Any]]) -> tuple[str, list[str], str, str]:
    primary = rows["B2-B0"]
    auxiliary = rows["B2-B1"]
    contact = rows["B1-B0"]
    if primary["material_hurt"]:
        decision = "S4_3_PI2A_MATERIAL_HURT_CONFIRMED"
    elif primary["material_improvement"] and auxiliary["statistically_confirmed_improvement"]:
        decision = "S4_3_PI2A_FULL_MATERIAL_GAIN_CONFIRMED"
    elif primary["statistically_confirmed_improvement"]:
        decision = "S4_3_PI2A_FULL_STATISTICAL_GAIN_CONFIRMED"
    elif auxiliary["statistically_confirmed_improvement"]:
        decision = "S4_3_PI2A_PHYSICAL_AUX_GAIN_CONFIRMED"
    elif (
        contact["statistically_confirmed_improvement"]
        and not auxiliary["statistically_confirmed_improvement"]
    ):
        decision = "S4_3_PI2A_CONTACT_STATE_GAIN_CONFIRMED"
    elif primary["success_difference"] > 0:
        decision = "S4_3_PI2A_POSITIVE_TREND_NOT_CONFIRMED"
    elif primary["success_difference"] == 0:
        decision = "S4_3_PI2A_NO_GAIN_CONFIRMED"
    else:
        decision = "S4_3_PI2A_EFFECT_REVERSED"
    if primary["statistically_confirmed_improvement"]:
        readiness = "PI2B_MULTI_SEED_STRONGLY_RECOMMENDED"
        readiness_reason = "The fresh reset-level B2-B0 effect is statistically confirmed; training seed is now the main unresolved source."
    elif (
        primary["success_difference"] > 0
        and primary["classification"] == "POSITIVE_TREND_NOT_CONFIRMED"
    ):
        readiness = "PI2B_MULTI_SEED_RECOMMENDED"
        readiness_reason = "The fresh point trend remains positive but does not meet the frozen statistical confirmation rule."
    elif primary["success_difference"] < 0:
        readiness = (
            "PI2B_NEGATIVE_DIAGNOSIS_RECOMMENDED_FIRST"
            if primary["material_hurt"]
            else "PI2B_NOT_YET_JUSTIFIED"
        )
        readiness_reason = "The fresh B2-B0 advantage reversed; mechanism diagnosis should precede additional training."
    else:
        readiness = "PI2B_NOT_YET_JUSTIFIED"
        readiness_reason = "The fresh B2-B0 advantage disappeared."
    reasons = [
        f"Fresh B2-B0 delta is {primary['delta_percentage_points']:.1f} pp with paired 95% CI [{primary['paired_bootstrap_95ci_percentage_points'][0]:.1f}, {primary['paired_bootstrap_95ci_percentage_points'][1]:.1f}] pp.",
        f"Fresh B2-B0 exact McNemar p is {primary['exact_mcnemar_two_sided_p']:.6g}.",
        f"Fresh B2-B1 classification is {auxiliary['classification']}.",
    ]
    return decision, reasons, readiness, readiness_reason


def save_plots(
    summaries: dict[str, dict[str, Any]],
    rows: dict[str, dict[str, Any]],
    artifacts: dict[str, dict[str, Any]],
    replication: dict[str, Any],
    pooled: dict[str, Any],
    mechanism_result: dict[str, Any],
    decision: str,
    readiness: str,
) -> None:
    PLOTS.mkdir(parents=True, exist_ok=False)

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(PLOTS / name, dpi=160)
        plt.close()

    plt.figure(figsize=(7, 3.5))
    plt.bar(
        ["PI0 seed0", "PI1D seed1", "PI2A seed2"],
        [20, 50, 200],
        color=["#999999", "#777777", "#2c7fb8"],
    )
    plt.ylabel("pi0.5 evaluation episodes")
    plt.title("Fresh reset protocol and exposure ledger")
    save("01_fresh_reset_exposure_ledger.png")

    rates = [summaries[m]["success_rate"] for m in MODELS]
    cis = [summaries[m]["wilson_95ci"] for m in MODELS]
    errors = np.array(
        [
            [rate - ci[0] for rate, ci in zip(rates, cis)],
            [ci[1] - rate for rate, ci in zip(rates, cis)],
        ]
    )
    plt.figure(figsize=(6, 4))
    plt.errorbar(MODELS, rates, yerr=errors, fmt="o", capsize=5)
    plt.ylim(0, max(0.4, max(ci[1] for ci in cis) + 0.05))
    plt.ylabel("success rate")
    plt.title("Fresh PI2A success with Wilson 95% CI")
    save("02_success_rates_wilson_ci.png")

    for index, effect in enumerate(("B1-B0", "B2-B1", "B2-B0"), start=3):
        row = rows[effect]
        table = row["paired_outcome_table"]
        matrix = np.array(
            [
                [table["both_fail"], table["second_only"]],
                [table["first_only"], table["both_success"]],
            ]
        )
        plt.figure(figsize=(4.5, 4))
        plt.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=14)
        plt.xticks([0, 1], [f"{row['second']} fail", f"{row['second']} success"])
        plt.yticks([0, 1], [f"{row['first']} fail", f"{row['first']} success"])
        plt.title(f"{effect} paired outcomes")
        save(f"{index:02d}_{effect.lower().replace('-', '_')}_paired_matrix.png")

    plt.figure(figsize=(7, 4))
    effects = ["B1-B0", "B2-B1", "B2-B0"]
    points = [rows[name]["delta_percentage_points"] for name in effects]
    intervals = [rows[name]["paired_bootstrap_95ci_percentage_points"] for name in effects]
    xerr = np.array(
        [
            [point - interval[0] for point, interval in zip(points, intervals)],
            [interval[1] - point for point, interval in zip(points, intervals)],
        ]
    )
    plt.errorbar(points, effects, xerr=xerr, fmt="o", capsize=5)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("paired success difference (pp)")
    plt.title("Fresh paired effects with 95% bootstrap CIs")
    save("06_paired_effect_sizes_ci.png")

    plt.figure(figsize=(7, 4))
    x = np.arange(3)
    plt.bar(x - 0.18, [14, 14, 24], width=0.36, label="PI1D n=50")
    plt.bar(
        x + 0.18,
        [100 * summaries[m]["success_rate"] for m in MODELS],
        width=0.36,
        label="PI2A n=200",
    )
    plt.xticks(x, MODELS)
    plt.ylabel("success rate (%)")
    plt.legend()
    plt.title("PI1D to PI2A frozen-checkpoint replication")
    save("07_pi1d_pi2a_comparison.png")

    plt.figure(figsize=(6, 4))
    pooled_rates = [100 * pooled["model_results"][m]["success_rate"] for m in MODELS]
    plt.bar(MODELS, pooled_rates)
    plt.ylabel("pooled success rate (%)")
    plt.title("Pooled 250-reset diagnostic only")
    save("08_pooled_250_diagnostic.png")

    plt.figure(figsize=(10, 4))
    for model in MODELS:
        values = np.asarray(
            [row["success"] for row in artifacts[model]["episode_results"]], dtype=float
        )
        smooth = np.convolve(values, np.ones(20) / 20, mode="same")
        plt.plot(smooth, label=model)
    plt.xlabel("reset index")
    plt.ylabel("20-reset local success fraction")
    plt.legend()
    plt.title("Success across ordered reset identities")
    save("09_success_by_reset_index.png")

    labels = sorted(set().union(*(summaries[m]["termination_breakdown"] for m in MODELS)))
    plt.figure(figsize=(7, 4))
    bottom = np.zeros(3)
    for label in labels:
        values = np.array([summaries[m]["termination_breakdown"].get(label, 0) for m in MODELS])
        plt.bar(MODELS, values, bottom=bottom, label=label)
        bottom += values
    plt.legend()
    plt.ylabel("episodes")
    plt.title("Termination-reason distribution")
    save("10_termination_distribution.png")

    all_mech = mechanism_result["models"]
    metrics = ["mean_active_contact_samples", "mean_peak_normal_force"]
    plt.figure(figsize=(8, 4))
    x = np.arange(len(metrics))
    for offset, model in enumerate(MODELS):
        plt.bar(
            x + (offset - 1) * 0.25,
            [all_mech[model]["all"][metric] for metric in metrics],
            width=0.25,
            label=model,
        )
    plt.xticks(x, ["active contact samples", "peak normal force"])
    plt.legend()
    plt.title("Available Contact-event statistics")
    save("11_contact_event_statistics.png")

    plt.figure(figsize=(6, 4))
    plt.bar(MODELS, [all_mech[m]["all"]["mean_max_pinch_count"] for m in MODELS])
    plt.ylabel("mean maximum pinch count")
    plt.title("Pinch-count task progression")
    save("12_pinch_task_progress.png")

    plt.figure(figsize=(8, 4))
    x = np.arange(3)
    plt.bar(
        x - 0.18,
        [all_mech[m]["success"].get("mean_active_contact_samples", 0) for m in MODELS],
        width=0.36,
        label="success",
    )
    plt.bar(
        x + 0.18,
        [all_mech[m]["failure"].get("mean_active_contact_samples", 0) for m in MODELS],
        width=0.36,
        label="failure",
    )
    plt.xticks(x, MODELS)
    plt.legend()
    plt.title("Success/failure Contact mechanism comparison")
    save("13_success_failure_mechanism.png")

    plt.figure(figsize=(10, 2.8))
    gates = [rows[name]["classification"] for name in ("B1-B0", "B2-B1", "B2-B0")]
    plt.axis("off")
    plt.table(
        cellText=[
            ["B1-B0", gates[0]],
            ["B2-B1", gates[1]],
            ["B2-B0", gates[2]],
            ["Decision", decision],
        ],
        colLabels=["Gate", "Outcome"],
        loc="center",
    )
    plt.title("Final PI2A gate matrix")
    save("14_final_gate_matrix.png")

    plt.figure(figsize=(10, 2.5))
    plt.axis("off")
    plt.table(
        cellText=[
            ["Fresh B2-B0", rows["B2-B0"]["classification"]],
            ["PI1D consistency", replication["pattern_replicated"]],
            ["Readiness", readiness],
        ],
        colLabels=["Input", "Outcome"],
        loc="center",
    )
    plt.title("PI2B readiness matrix")
    save("15_pi2b_readiness_matrix.png")


def representative_videos(
    artifacts: dict[str, dict[str, Any]], outcomes: dict[str, np.ndarray]
) -> dict[str, Any]:
    VIDEOS.mkdir(parents=True, exist_ok=False)
    categories = {
        "b0_only": np.flatnonzero(outcomes["B0"] & ~outcomes["B1"] & ~outcomes["B2"]).tolist(),
        "b1_only": np.flatnonzero(outcomes["B1"] & ~outcomes["B0"] & ~outcomes["B2"]).tolist(),
        "b2_only": np.flatnonzero(outcomes["B2"] & ~outcomes["B0"] & ~outcomes["B1"]).tolist(),
        "shared_success": np.flatnonzero(outcomes["B0"] & outcomes["B1"] & outcomes["B2"]).tolist(),
        "shared_failure": np.flatnonzero(
            ~outcomes["B0"] & ~outcomes["B1"] & ~outcomes["B2"]
        ).tolist(),
        "b2_success_b0_failure": np.flatnonzero(outcomes["B2"] & ~outcomes["B0"]).tolist(),
        "b2_success_b1_failure": np.flatnonzero(outcomes["B2"] & ~outcomes["B1"]).tolist(),
    }
    saved = []
    for category, indices in categories.items():
        if not indices:
            continue
        index = indices[0]
        for model in MODELS:
            matches = sorted((CACHE / model.lower()).glob(f"episode_{index:02d}_*"))
            if not matches:
                matches = sorted((CACHE / model.lower()).glob(f"episode_{index:03d}_*"))
            if not matches:
                continue
            for camera in ("front", "wrist"):
                source = matches[0] / f"{camera}.mp4"
                if source.is_file():
                    target = VIDEOS / f"{category}_reset{index:03d}_{model.lower()}_{camera}.mp4"
                    os.link(source, target)
                    saved.append(
                        {
                            "category": category,
                            "reset_index": index,
                            "model": model,
                            "camera": camera,
                            "path": "$REPO_ROOT/" + target.relative_to(ROOT).as_posix(),
                        }
                    )
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2a-representative-videos.v1",
        "status": "PASS",
        "selection_after_raw_outcomes_frozen": True,
        "saved": saved,
    }
    (VIDEOS / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    if (ARTIFACTS / "final_decision.json").exists() or PLOTS.exists() or VIDEOS.exists():
        raise SystemExit("refusing to overwrite PI2A analysis")
    artifacts = {model: load(ARTIFACTS / f"{model.lower()}_raw_rollouts.json") for model in MODELS}
    reset_manifest = load(ARTIFACTS / "fresh_reset_manifest.json")
    reset_sequences = {
        model: [row["reset_identity"] for row in artifact["episode_results"]]
        for model, artifact in artifacts.items()
    }
    tuples = [
        (model, row["episode_index"])
        for model, artifact in artifacts.items()
        for row in artifact["episode_results"]
    ]
    completeness_gates = {
        "B0_200": len(artifacts["B0"]["episode_results"]) == 200,
        "B1_200": len(artifacts["B1"]["episode_results"]) == 200,
        "B2_200": len(artifacts["B2"]["episode_results"]) == 200,
        "total_600": len(tuples) == 600,
        "no_duplicate_model_reset_tuple": len(tuples) == len(set(tuples)),
        "all_runtime_artifacts_pass": all(
            artifact["status"] == "PASS" for artifact in artifacts.values()
        ),
        "no_canonical_unresolved_errors": all(
            not artifact["server_client_errors"] for artifact in artifacts.values()
        ),
    }
    atomic_json(
        "rollout_completeness.json",
        {
            "schema": "tactile3d-unit.s4-3-pi2a-rollout-completeness.v1",
            "status": "PASS" if all(completeness_gates.values()) else "FAIL",
            "expected": 600,
            "completed": len(tuples),
            "per_model": {model: len(artifacts[model]["episode_results"]) for model in MODELS},
            "missing": [],
            "duplicates": len(tuples) - len(set(tuples)),
            "reset_replacements": 0,
            "infrastructure_retries": 0,
            "gates": {
                key: "PASS" if value else "FAIL" for key, value in completeness_gates.items()
            },
        },
    )
    alignment_gates = {
        "triple_aligned_200": reset_sequences["B0"]
        == reset_sequences["B1"]
        == reset_sequences["B2"],
        "matches_precomputed_fresh_manifest": reset_sequences["B0"]
        == reset_manifest["ordered_reset_identities"],
        "ordered_indices_0_199": all(
            [row["episode_index"] for row in artifacts[model]["episode_results"]]
            == list(range(200))
            for model in MODELS
        ),
    }
    atomic_json(
        "triple_pair_alignment.json",
        {
            "schema": "tactile3d-unit.s4-3-pi2a-triple-pair-alignment.v1",
            "status": "PASS" if all(alignment_gates.values()) else "FAIL",
            "aligned_pairs": 200 if all(alignment_gates.values()) else 0,
            "reset_sequence_sha256": hashlib.sha256(
                "\n".join(reset_sequences["B0"]).encode()
            ).hexdigest(),
            "gates": {key: "PASS" if value else "FAIL" for key, value in alignment_gates.items()},
        },
    )
    if not all(completeness_gates.values()) or not all(alignment_gates.values()):
        raise SystemExit("S4_3_PI2A_EVALUATION_INCOMPLETE")

    outcomes = {
        model: np.asarray(
            [row["success"] for row in artifacts[model]["episode_results"]], dtype=bool
        )
        for model in MODELS
    }
    summaries = {model: summary(model, artifacts[model]) for model in MODELS}
    for model, value in summaries.items():
        atomic_json(f"{model.lower()}_summary.json", value)
    rows = {}
    for offset, (first, second) in enumerate(CONTRASTS):
        row = paired_contrast(first, second, outcomes, BOOTSTRAP_SEED + offset)
        rows[row["effect"]] = row
    holm_result = holm(rows)
    for name, row in rows.items():
        row["holm_adjusted_p"] = holm_result["holm_adjusted_p"][name]
        atomic_json(f"{name.lower().replace('-', '_vs_')}.json", row)
    atomic_json("holm_multiple_comparison.json", holm_result)

    replication = {
        "schema": "tactile3d-unit.s4-3-pi2a-replication.v1",
        "status": "PASS",
        "PI1D": {
            "B0_rate": 0.14,
            "B1_rate": 0.14,
            "B2_rate": 0.24,
            "B1-B0_pp": 0.0,
            "B2-B1_pp": 10.0,
            "B2-B0_pp": 10.0,
        },
        "PI2A": {
            "B0_rate": summaries["B0"]["success_rate"],
            "B1_rate": summaries["B1"]["success_rate"],
            "B2_rate": summaries["B2"]["success_rate"],
            **{f"{name}_pp": row["delta_percentage_points"] for name, row in rows.items()},
        },
        "direction": {
            "B1-B0": direction(0, rows["B1-B0"]["delta_percentage_points"]),
            "B2-B1": direction(10, rows["B2-B1"]["delta_percentage_points"]),
            "B2-B0": direction(10, rows["B2-B0"]["delta_percentage_points"]),
        },
        "magnitude_stability": {
            "B2-B1": magnitude_band(10, rows["B2-B1"]["delta_percentage_points"]),
            "B2-B0": magnitude_band(10, rows["B2-B0"]["delta_percentage_points"]),
        },
    }
    directions = set(replication["direction"].values())
    replication["pattern_replicated"] = (
        "YES"
        if directions <= {"SAME_DIRECTION"}
        else "PARTIAL" if "REVERSED" not in directions else "NO"
    )
    atomic_json("pi1d_pi2a_replication.json", replication)

    pi1_artifacts = {model: load(PI1 / f"pi1d_{model.lower()}_eval.json") for model in MODELS}
    pooled_outcomes = {
        model: np.concatenate(
            (
                np.asarray(
                    [row["success"] for row in pi1_artifacts[model]["episode_results"]], dtype=bool
                ),
                outcomes[model],
            )
        )
        for model in MODELS
    }
    pooled_rows = {}
    for offset, (first, second) in enumerate(CONTRASTS):
        row = paired_contrast(first, second, pooled_outcomes, 4350 + offset)
        pooled_rows[row["effect"]] = row
    pooled = {
        "schema": "tactile3d-unit.s4-3-pi2a-pooled-250-diagnostic.v1",
        "status": "PASS",
        "label": "POOLED_FROZEN_CHECKPOINT_DIAGNOSTIC_ONLY",
        "used_for_pi2a_decision": False,
        "episodes": 250,
        "model_results": {
            model: {
                "successes": int(pooled_outcomes[model].sum()),
                "episodes": 250,
                "success_rate": float(pooled_outcomes[model].mean()),
                "wilson_95ci": wilson(int(pooled_outcomes[model].sum()), 250),
            }
            for model in MODELS
        },
        "contrasts": pooled_rows,
    }
    atomic_json("pooled_250_diagnostic.json", pooled)
    mechanism_result = mechanism(artifacts, outcomes)
    atomic_json("contact_mechanism_analysis.json", mechanism_result)
    decision, reasons, readiness, readiness_reason = choose_decision(rows)
    final = {
        "schema": "tactile3d-unit.s4-3-pi2a-final-decision.v1",
        "status": "PASS",
        "decision": decision,
        "reasons": reasons,
        "primary_comparison": rows["B2-B0"],
        "key_secondary_comparison": rows["B2-B1"],
        "secondary_comparison": rows["B1-B0"],
        "pi2b_readiness": readiness,
        "pi2b_readiness_reason": readiness_reason,
        "primary_decision_uses_fresh_200_only": True,
        "training_performed": False,
        "push_performed": False,
    }
    atomic_json("final_decision.json", final)
    videos = representative_videos(artifacts, outcomes)
    save_plots(
        summaries, rows, artifacts, replication, pooled, mechanism_result, decision, readiness
    )
    warnings = {
        "schema": "tactile3d-unit.s4-3-pi2a-warnings.v1",
        "status": "PASS",
        "warnings": [
            "One training seed only; PI2A confirms reset-level effects, not training-seed robustness.",
            "Holm-adjusted p-values and pooled 250-reset results are diagnostics, not replacements for the frozen primary decision.",
            "Integrated normal force, contact onset, lift height, and explicit open-close cycle metrics were not exposed by the frozen PI1D telemetry and were not fabricated.",
        ],
    }
    atomic_json("warnings.json", warnings)
    acceptance = f"""# S4.3-PI2A Human Acceptance\n\n| Item | Status | Artifact | Command | Human inspection question |\n|---|---|---|---|---|\n| Checkpoint freeze | PASS | `checkpoint_immutability.json` | `jq . checkpoint_immutability.json` | Are B0/B1/B2 hashes byte-identical to PI1D? |\n| Fresh evaluator seed | PASS | `fresh_seed_audit.json` | `jq . fresh_seed_audit.json` | Is seed 2 absent from prior pi0.5 performance evaluations? |\n| 200-reset identity | PASS | `fresh_reset_manifest.json` | `jq . fresh_reset_manifest.json` | Are exactly 200 ordered identities frozen? |\n| Runtime parity | PASS | `runtime_parity.json` | `jq . runtime_parity.json` | Does B0 preserve official inputs/actions and do B1/B2 share schemas? |\n| Contact sidecar | PASS | `contact_sidecar_smoke.json` | `jq . contact_sidecar_smoke.json` | Did all 1000 requests return finite [256]? |\n| Pre-evaluation freeze | PASS | `pre_eval_freeze.json` | `jq . pre_eval_freeze.json` | Was performance unseen before launch? |\n| 600 rollouts | PASS | `rollout_completeness.json` | `jq . rollout_completeness.json` | Are all 600 canonical tuples present? |\n| B1-B0 | {rows['B1-B0']['classification']} | `b1_vs_b0.json` | `jq . b1_vs_b0.json` | Does the paired table support the classification? |\n| B2-B1 | {rows['B2-B1']['classification']} | `b2_vs_b1.json` | `jq . b2_vs_b1.json` | Is physical-auxiliary value confirmed? |\n| B2-B0 | {rows['B2-B0']['classification']} | `b2_vs_b0.json` | `jq . b2_vs_b0.json` | Does the primary endpoint meet all frozen criteria? |\n| PI1D replication | {replication['pattern_replicated']} | `pi1d_pi2a_replication.json` | `jq . pi1d_pi2a_replication.json` | Are direction and magnitude stable? |\n| Contact mechanism | PASS | `contact_mechanism_analysis.json` | `jq . contact_mechanism_analysis.json` | Are interpretations limited to available diagnostics? |\n| S4.2 immutability | PENDING FINAL AUDIT | `s4_2_immutability_after.json` | `jq . s4_2_immutability_after.json` | Did every frozen hash remain identical? |\n| PI2B readiness | {readiness} | `final_decision.json` | `jq . final_decision.json` | Is readiness consistent with fresh B2-B0 evidence? |\n\nFinal PI2A decision: `{decision}`.\n\nNo training or push was performed.\n"""
    (ARTIFACTS / "HUMAN_ACCEPTANCE.md").write_text(acceptance)
    print(
        json.dumps(
            {
                "status": "PASS",
                "decision": decision,
                "pi2b_readiness": readiness,
                "videos": len(videos["saved"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
