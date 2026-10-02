#!/usr/bin/env python3
"""Analyze the complete Track-A FINAL cohort under the frozen protocol."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics as py_statistics
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.pi2b_policy.analysis import (  # noqa: E402
    direction_label,
    percent,
    percent_interval,
    summarize_seed_deltas,
)
from gr00t.simulation.pi2b_policy.statistics import (  # noqa: E402
    conditional_reset_bootstrap,
    discordant_table,
    exact_mcnemar,
    holm_adjust,
    paired_risk_difference,
    two_way_bootstrap,
    wilson_interval,
)


ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2b_policy"
PREFINAL = ARTIFACTS / "pre_final_freeze.json"
COMPLETENESS = ARTIFACTS / "rollout_completeness.json"
RESET_MANIFEST = ARTIFACTS / "reset_manifest.json"
RUNTIME_AMENDMENT = ARTIFACTS / "runtime_amendment_cross_device_publish.json"
STATISTICS_CONFIG = ROOT / "configs/simulation/pi2b_policy/statistics.json"
MODELS = ("B0", "B_VA27", "B1", "B_HVA", "B2")
SEEDS = (42, 43, 44)
NEW_SEEDS = (43, 44)
FAMILIES = {
    "primary": (("B_HVA-B_VA27", "B_HVA", "B_VA27"), ("B2-B_HVA", "B2", "B_HVA")),
    "key_secondary": (
        ("B2-B1", "B2", "B1"),
        ("B_HVA-B0", "B_HVA", "B0"),
        ("B2-B0", "B2", "B0"),
    ),
    "auxiliary_secondary": (
        ("B_VA27-B0", "B_VA27", "B0"),
        ("B1-B0", "B1", "B0"),
    ),
}
CONTRASTS = tuple(item for family in FAMILIES.values() for item in family)
OUTPUTS = {
    "per_seed": ARTIFACTS / "per_seed_statistics.json",
    "crossed": ARTIFACTS / "crossed_seed_reset_analysis.json",
    "new_only": ARTIFACTS / "new_seeds_only_sensitivity.json",
    "audit": ARTIFACTS / "analysis_input_audit.json",
    "decision": ARTIFACTS / "final_decision.json",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def load_and_validate() -> tuple[dict[str, np.ndarray], dict[tuple[str, int], list[dict[str, Any]]], dict[str, Any]]:
    if git("branch", "--show-current") != "develop/pi2b-policy":
        raise SystemExit("analysis requires develop/pi2b-policy")
    prefinal = read_json(PREFINAL)
    completeness = read_json(COMPLETENESS)
    reset_manifest = read_json(RESET_MANIFEST)
    amendment = read_json(RUNTIME_AMENDMENT)
    config = read_json(STATISTICS_CONFIG)
    if prefinal.get("status") != "PASS" or completeness.get("status") != "PASS":
        raise SystemExit("formal completeness inputs are not PASS")
    if completeness.get("total_canonical_outcomes") != 3000:
        raise SystemExit("formal cohort is not exactly 3000 outcomes")
    if tuple(prefinal.get("formal_models", ())) != MODELS or tuple(prefinal.get("training_seeds", ())) != SEEDS:
        raise SystemExit("pre-final model/seed identity drift")
    if sha256_file(RESET_MANIFEST) != prefinal["reset_manifest_sha256"]:
        raise SystemExit("reset manifest drift")
    amended_source = "scripts/simulation/pi2b_policy/evaluate.py"
    amendment_gates = {
        "status": amendment.get("status") == "PASS",
        "scope": amendment.get("scope") == "ARTIFACT_PUBLICATION_ONLY",
        "prefinal": amendment.get("pre_final_freeze_sha256") == sha256_file(PREFINAL),
        "old_hash": amendment.get("original_frozen_evaluate_sha256")
        == prefinal["sources_sha256"][amended_source],
        "new_hash": amendment.get("amended_evaluate_sha256")
        == sha256_file(ROOT / amended_source),
        "performance_blind": amendment.get("performance_seen") is False,
        "policy_semantics": amendment.get("policy_or_evaluator_semantics_changed") is False,
        "reset_semantics": amendment.get("reset_or_sampling_semantics_changed") is False,
        "success_rules": amendment.get("success_or_timeout_rules_changed") is False,
        "track_b_isolated": amendment.get("track_b_new_teacher_read") is False,
    }
    if not all(amendment_gates.values()):
        failed = [name for name, passed in amendment_gates.items() if not passed]
        raise SystemExit("runtime amendment gate failure: " + ",".join(failed))
    for relative, expected in prefinal["sources_sha256"].items():
        path = ROOT / relative
        if relative == amended_source:
            expected = amendment["amended_evaluate_sha256"]
        if not path.is_file() or sha256_file(path) != expected:
            raise SystemExit(f"frozen source drift: {relative}")
    for absolute, expected in prefinal["external_sources_sha256"].items():
        path = Path(absolute)
        if not path.is_file() or sha256_file(path) != expected:
            raise SystemExit(f"frozen external source drift: {absolute}")
    if config.get("status") != "FROZEN_BEFORE_NEW_TRAINING":
        raise SystemExit("statistics protocol is not frozen")
    if int(config["bootstrap_repetitions"]) != int(prefinal["statistics"]["bootstrap_repetitions"]):
        raise SystemExit("bootstrap repetition drift")
    if int(config["bootstrap_seed"]) != int(prefinal["statistics"]["bootstrap_seed"]):
        raise SystemExit("bootstrap seed drift")
    if reset_manifest.get("status") != "PASS" or reset_manifest.get("episodes") != 200:
        raise SystemExit("reset manifest is not a complete 200-reset PASS cohort")
    ordered_ids = reset_manifest["ordered_reset_identities"]
    if len(ordered_ids) != 200 or len(set(ordered_ids)) != 200:
        raise SystemExit("reset identity cardinality failure")
    expected_by_block = {
        (int(row["seed"]), int(row["block_index"])): row["reset_identity"]
        for row in reset_manifest["reset_specs"]
    }
    if len(expected_by_block) != 200:
        raise SystemExit("reset spec mapping failure")
    checkpoint_hash = {
        (row["model"], int(row["training_seed"])): row["tree_sha256"]
        for row in prefinal["checkpoints"]
    }
    if len(checkpoint_hash) != 15:
        raise SystemExit("checkpoint mapping failure")

    rows: dict[tuple[str, int], list[dict[str, Any]]] = {
        (model, seed): [] for model in MODELS for seed in SEEDS
    }
    tuples: set[tuple[str, int, str]] = set()
    raw_hashes: dict[str, str] = {}
    for key, raw_record in sorted(completeness["raw_artifacts"].items()):
        path = Path(raw_record["path"])
        payload_bytes = path.read_bytes()
        observed_hash = sha256_bytes(payload_bytes)
        if observed_hash != raw_record["sha256"]:
            raise SystemExit(f"raw artifact drift: {key}")
        raw_hashes[key] = observed_hash
        payload = json.loads(payload_bytes)
        model = payload["model"]
        seed = int(payload["training_seed"])
        reset_seed = int(payload["reset_seed"])
        expected_checkpoint = checkpoint_hash.get((model, seed))
        if (model, seed) not in rows or expected_checkpoint is None:
            raise SystemExit(f"unexpected model/seed in {key}")
        if payload.get("status") != "PASS" or payload.get("episodes") != 50:
            raise SystemExit(f"block status/count failure: {key}")
        if payload.get("checkpoint_tree_sha256") != expected_checkpoint:
            raise SystemExit(f"block checkpoint mismatch: {key}")
        if payload.get("server_client_errors"):
            raise SystemExit(f"block has server/client errors: {key}")
        episode_rows = payload.get("episode_results", [])
        if len(episode_rows) != 50:
            raise SystemExit(f"episode count mismatch: {key}")
        observed_successes = 0
        for episode in episode_rows:
            index = int(episode["episode_index"])
            identity = episode["reset_identity"]
            expected_identity = expected_by_block.get((reset_seed, index))
            if expected_identity != identity:
                raise SystemExit(f"reset identity mismatch: {key}:{index}")
            if episode.get("model") != model or int(episode.get("training_seed")) != seed:
                raise SystemExit(f"row model/seed mismatch: {key}:{index}")
            if int(episode.get("reset_seed")) != reset_seed:
                raise SystemExit(f"row reset-seed mismatch: {key}:{index}")
            if episode.get("checkpoint_tree_sha256") != expected_checkpoint:
                raise SystemExit(f"row checkpoint mismatch: {key}:{index}")
            if episode.get("server_client_errors"):
                raise SystemExit(f"row has server/client errors: {key}:{index}")
            if not isinstance(episode.get("success"), bool):
                raise SystemExit(f"non-boolean outcome: {key}:{index}")
            if not isinstance(episode.get("steps"), int) or episode["steps"] <= 0:
                raise SystemExit(f"invalid step count: {key}:{index}")
            canonical = (model, seed, identity)
            if canonical in tuples:
                raise SystemExit(f"duplicate canonical tuple: {canonical}")
            tuples.add(canonical)
            rows[(model, seed)].append(
                {
                    "global_index": reset_manifest["ordered_reset_identities"].index(identity),
                    "reset_identity": identity,
                    "success": episode["success"],
                    "termination": episode["termination"],
                    "steps": episode["steps"],
                }
            )
            observed_successes += int(episode["success"])
        if observed_successes != int(payload["successes"]):
            raise SystemExit(f"block success total mismatch: {key}")
    if len(raw_hashes) != 60 or len(tuples) != 3000:
        raise SystemExit("raw artifact or canonical tuple cardinality failure")

    outcomes: dict[str, np.ndarray] = {}
    for model in MODELS:
        matrix = np.empty((len(SEEDS), 200), dtype=np.bool_)
        for seed_index, seed in enumerate(SEEDS):
            ordered = sorted(rows[(model, seed)], key=lambda row: row["global_index"])
            if len(ordered) != 200 or [row["reset_identity"] for row in ordered] != ordered_ids:
                raise SystemExit(f"shared reset sequence failure: {model}/seed{seed}")
            rows[(model, seed)] = ordered
            matrix[seed_index] = [row["success"] for row in ordered]
        outcomes[model] = matrix
    evidence = {
        "prefinal_sha256": sha256_file(PREFINAL),
        "completeness_sha256": sha256_file(COMPLETENESS),
        "reset_manifest_sha256": sha256_file(RESET_MANIFEST),
        "statistics_config_sha256": sha256_file(STATISTICS_CONFIG),
        "runtime_amendment_sha256": sha256_file(RUNTIME_AMENDMENT),
        "runtime_amendment_gates": amendment_gates,
        "raw_artifact_sha256": raw_hashes,
        "ordered_reset_sequence_sha256": prefinal["ordered_reset_sequence_sha256"],
        "git_head_at_analysis": git("rev-parse", "HEAD"),
    }
    return outcomes, rows, evidence


def bootstrap_pair(first: np.ndarray, second: np.ndarray, *, repetitions: int, seed: int) -> dict[str, Any]:
    result = conditional_reset_bootstrap(first, second, repetitions=repetitions, seed=seed)
    result["interval"] = list(result["interval"])
    return result


def two_way_pair(first: np.ndarray, second: np.ndarray, *, repetitions: int, seed: int) -> dict[str, Any]:
    result = two_way_bootstrap(first, second, repetitions=repetitions, seed=seed)
    result["interval"] = list(result["interval"])
    return result


def checkpoint_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(int(row["success"]) for row in rows)
    total = len(rows)
    terminations = Counter(row["termination"] for row in rows)
    steps = [int(row["steps"]) for row in rows]
    interval = wilson_interval(successes, total)
    timeouts = sum(value for key, value in terminations.items() if "timeout" in key.lower())
    return {
        "successes": successes,
        "episodes": total,
        "failures": total - successes,
        "timeouts": timeouts,
        "success_rate": successes / total,
        "success_percent": 100.0 * successes / total,
        "wilson_95ci": list(interval),
        "wilson_95ci_percent": percent_interval(interval),
        "termination_counts": dict(sorted(terminations.items())),
        "steps": {
            "mean": float(py_statistics.fmean(steps)),
            "median": float(py_statistics.median(steps)),
            "min": min(steps),
            "max": max(steps),
            "total": sum(steps),
        },
    }


def analyze() -> dict[str, Any]:
    outcomes, rows, evidence = load_and_validate()
    config = read_json(STATISTICS_CONFIG)
    repetitions = int(config["bootstrap_repetitions"])
    bootstrap_seed = int(config["bootstrap_seed"])
    checkpoint_statistics = {
        model: {str(seed): checkpoint_summary(rows[(model, seed)]) for seed in SEEDS}
        for model in MODELS
    }
    per_seed_pairs: dict[str, dict[str, dict[str, Any]]] = {str(seed): {} for seed in SEEDS}
    for seed_index, seed in enumerate(SEEDS):
        for family, members in FAMILIES.items():
            family_rows = []
            for name, first_name, second_name in members:
                first = outcomes[first_name][seed_index]
                second = outcomes[second_name][seed_index]
                bootstrap = bootstrap_pair(first[None, :], second[None, :], repetitions=repetitions, seed=bootstrap_seed)
                table = discordant_table(first, second)
                family_rows.append(
                    {
                        "name": name,
                        "family": family,
                        "first": first_name,
                        "second": second_name,
                        "risk_difference": paired_risk_difference(first, second),
                        "risk_difference_percentage_points": percent(paired_risk_difference(first, second)),
                        "paired_tuple_bootstrap_95ci": bootstrap["interval"],
                        "paired_tuple_bootstrap_95ci_percentage_points": percent_interval(bootstrap["interval"]),
                        "discordant_table": table,
                        "mcnemar_exact_two_sided_raw_p": exact_mcnemar(first, second),
                        "bootstrap_repetitions": repetitions,
                        "bootstrap_seed": bootstrap_seed,
                        "bootstrap_unit": "paired reset tuple within this training seed",
                    }
                )
            adjusted = holm_adjust(row["mcnemar_exact_two_sided_raw_p"] for row in family_rows)
            for row, adjusted_p in zip(family_rows, adjusted, strict=True):
                row["holm_adjusted_p_within_seed_family"] = adjusted_p
                per_seed_pairs[str(seed)][row.pop("name")] = row

    crossed: dict[str, Any] = {}
    new_only: dict[str, Any] = {}
    for name, first_name, second_name in CONTRASTS:
        first = outcomes[first_name]
        second = outcomes[second_name]
        deltas = [paired_risk_difference(first[index], second[index]) for index in range(len(SEEDS))]
        conditional = bootstrap_pair(first, second, repetitions=repetitions, seed=bootstrap_seed)
        two_way = two_way_pair(first, second, repetitions=repetitions, seed=bootstrap_seed)
        summary = summarize_seed_deltas(deltas)
        crossed[name] = {
            "first": first_name,
            "second": second_name,
            "per_seed_risk_difference": {str(seed): value for seed, value in zip(SEEDS, deltas, strict=True)},
            "per_seed_risk_difference_percentage_points": {str(seed): percent(value) for seed, value in zip(SEEDS, deltas, strict=True)},
            "training_seed_summary": summary,
            "training_seed_summary_percentage_points": {
                "mean": percent(summary["mean"]),
                "sample_sd": percent(summary["sample_sd"]),
                "range": percent_interval(summary["range"]),
            },
            "conditional_shared_reset_bootstrap": conditional,
            "conditional_shared_reset_bootstrap_percentage_points": {
                "estimate": percent(conditional["estimate"]),
                "interval": percent_interval(conditional["interval"]),
            },
            "two_way_seed_by_reset_sensitivity": two_way,
            "two_way_seed_by_reset_sensitivity_percentage_points": {
                "estimate": percent(two_way["estimate"]),
                "interval": percent_interval(two_way["interval"]),
            },
            "material_10pp_reference_met_by_absolute_mean": abs(summary["mean"]) >= 0.10,
            "material_reference_is_not_an_engineering_gate": True,
        }
        new_first = first[1:, :]
        new_second = second[1:, :]
        new_deltas = [paired_risk_difference(new_first[index], new_second[index]) for index in range(len(NEW_SEEDS))]
        new_conditional = bootstrap_pair(new_first, new_second, repetitions=repetitions, seed=bootstrap_seed)
        new_two_way = two_way_pair(new_first, new_second, repetitions=repetitions, seed=bootstrap_seed)
        new_summary = summarize_seed_deltas(new_deltas)
        new_only[name] = {
            "per_seed_risk_difference": {str(seed): value for seed, value in zip(NEW_SEEDS, new_deltas, strict=True)},
            "per_seed_risk_difference_percentage_points": {str(seed): percent(value) for seed, value in zip(NEW_SEEDS, new_deltas, strict=True)},
            "training_seed_summary": new_summary,
            "training_seed_summary_percentage_points": {
                "mean": percent(new_summary["mean"]),
                "sample_sd": percent(new_summary["sample_sd"]),
                "range": percent_interval(new_summary["range"]),
            },
            "conditional_shared_reset_bootstrap": new_conditional,
            "conditional_shared_reset_bootstrap_percentage_points": {
                "estimate": percent(new_conditional["estimate"]),
                "interval": percent_interval(new_conditional["interval"]),
            },
            "two_way_seed_by_reset_sensitivity": new_two_way,
            "two_way_seed_by_reset_sensitivity_percentage_points": {
                "estimate": percent(new_two_way["estimate"]),
                "interval": percent_interval(new_two_way["interval"]),
            },
        }

    interaction = outcomes["B_HVA"].astype(np.float64) - outcomes["B_VA27"].astype(np.float64) - outcomes["B1"].astype(np.float64) + outcomes["B0"].astype(np.float64)
    zeros = np.zeros_like(interaction)
    interaction_deltas = [float(interaction[index].mean()) for index in range(len(SEEDS))]
    interaction_conditional = bootstrap_pair(interaction, zeros, repetitions=repetitions, seed=bootstrap_seed)
    interaction_two_way = two_way_pair(interaction, zeros, repetitions=repetitions, seed=bootstrap_seed)
    interaction_summary = summarize_seed_deltas(interaction_deltas)
    interaction_record = {
        "definition": "(B_HVA-B_VA27)-(B1-B0)",
        "role": "predeclared interaction diagnostic; not pure causal synergy",
        "per_seed": {str(seed): value for seed, value in zip(SEEDS, interaction_deltas, strict=True)},
        "per_seed_percentage_points": {str(seed): percent(value) for seed, value in zip(SEEDS, interaction_deltas, strict=True)},
        "training_seed_summary": interaction_summary,
        "training_seed_summary_percentage_points": {
            "mean": percent(interaction_summary["mean"]),
            "sample_sd": percent(interaction_summary["sample_sd"]),
            "range": percent_interval(interaction_summary["range"]),
        },
        "conditional_shared_reset_bootstrap": interaction_conditional,
        "conditional_shared_reset_bootstrap_percentage_points": {
            "estimate": percent(interaction_conditional["estimate"]),
            "interval": percent_interval(interaction_conditional["interval"]),
        },
        "two_way_seed_by_reset_sensitivity": interaction_two_way,
        "two_way_seed_by_reset_sensitivity_percentage_points": {
            "estimate": percent(interaction_two_way["estimate"]),
            "interval": percent_interval(interaction_two_way["interval"]),
        },
    }
    new_interaction = interaction[1:, :]
    new_interaction_deltas = [float(new_interaction[index].mean()) for index in range(len(NEW_SEEDS))]
    new_interaction_conditional = bootstrap_pair(new_interaction, np.zeros_like(new_interaction), repetitions=repetitions, seed=bootstrap_seed)
    new_interaction_two_way = two_way_pair(new_interaction, np.zeros_like(new_interaction), repetitions=repetitions, seed=bootstrap_seed)
    new_interaction_summary = summarize_seed_deltas(new_interaction_deltas)
    new_interaction_record = {
        "definition": "(B_HVA-B_VA27)-(B1-B0)",
        "per_seed": {str(seed): value for seed, value in zip(NEW_SEEDS, new_interaction_deltas, strict=True)},
        "per_seed_percentage_points": {str(seed): percent(value) for seed, value in zip(NEW_SEEDS, new_interaction_deltas, strict=True)},
        "training_seed_summary": new_interaction_summary,
        "conditional_shared_reset_bootstrap": new_interaction_conditional,
        "two_way_seed_by_reset_sensitivity": new_interaction_two_way,
    }

    hva = crossed["B_HVA-B_VA27"]
    vac_vs_va = crossed["B2-B_HVA"]
    vac_vs_h = crossed["B2-B1"]
    h_label = direction_label(
        hva["per_seed_risk_difference"].values(),
        hva["conditional_shared_reset_bootstrap"]["interval"],
        positive="CONSISTENT_POSITIVE",
        negative="NEGATIVE_DIRECTION",
        mixed="MIXED",
        inconclusive="NO_CLEAR_DIFFERENCE",
    )
    vac_va_label = direction_label(
        vac_vs_va["per_seed_risk_difference"].values(),
        vac_vs_va["conditional_shared_reset_bootstrap"]["interval"],
        positive="VAC_C_FAVORED",
        negative="VA_FAVORED",
        mixed="MIXED",
        inconclusive="INCONCLUSIVE",
    )
    vac_h_label = direction_label(
        vac_vs_h["per_seed_risk_difference"].values(),
        vac_vs_h["conditional_shared_reset_bootstrap"]["interval"],
        positive="VAC_C_AUX_FAVORED",
        negative="H_ONLY_FAVORED",
        mixed="MIXED",
        inconclusive="INCONCLUSIVE",
    )
    decision = {
        "schema": "tactile3d-unit.s4-3-pi2b-policy-final-decision.v1",
        "status": "COMPLETE_VALID",
        "created_at_utc": now(),
        "ENGINEERING_STATUS": "COMPLETE_VALID",
        "H_GIVEN_VA27": h_label,
        "VAC_C_VS_VA_TARGET": vac_va_label,
        "VAC_C_VS_H_ONLY": vac_h_label,
        "FULL_RECIPES_VS_B0": {
            name: {
                "mean_effect_percentage_points": crossed[name]["training_seed_summary_percentage_points"]["mean"],
                "conditional_95ci_percentage_points": crossed[name]["conditional_shared_reset_bootstrap_percentage_points"]["interval"],
            }
            for name in ("B_HVA-B0", "B2-B0")
        },
        "TRAINING_SEED_SCOPE": "THREE_SEEDS_LIMITED",
        "SEED42_DISCLOSURE": "previously selected and development-exposed",
        "SEEDS43_44_DISCLOSURE": "preregistered independent training additions",
        "LABEL_RULE": "direction label requires all three seed point differences to have the same strict sign and the conditional shared-reset 95% interval to exclude zero in that direction; opposite signs yield MIXED; otherwise inconclusive/no-clear-difference",
        "MATERIAL_EFFECT_REFERENCE_PP": 10,
        "MATERIAL_REFERENCE_IS_ENGINEERING_GATE": False,
        "EQUIVALENCE_CLAIMED": False,
        "POPULATION_LEVEL_TRAINING_SEED_SIGNIFICANCE_CLAIMED": False,
        "TRACK_B_NEW_TEACHER_READ": False,
        "AUTO_EXPANSION_AUTHORIZED": False,
    }
    per_seed_payload = {
        "schema": "tactile3d-unit.s4-3-pi2b-policy-per-seed-statistics.v1",
        "status": "PASS",
        "created_at_utc": now(),
        "checkpoint_statistics": checkpoint_statistics,
        "paired_comparisons": per_seed_pairs,
        "families": {name: [row[0] for row in members] for name, members in FAMILIES.items()},
        "multiplicity": "Holm within each declared family separately within each training seed",
        "mcnemar": "exact two-sided within each training seed; seed-reset rows are never pooled",
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": bootstrap_seed,
        "evidence": evidence,
    }
    crossed_payload = {
        "schema": "tactile3d-unit.s4-3-pi2b-policy-crossed-seed-reset-analysis.v1",
        "status": "PASS",
        "created_at_utc": now(),
        "training_seeds": list(SEEDS),
        "contrasts": crossed,
        "interaction_diagnostic": interaction_record,
        "estimand": "mean of three within-training-seed paired reset risk differences",
        "conditional_interval_scope": "conditional on these three trained checkpoints",
        "two_way_warning": "three training seeds make outer-level uncertainty estimation weak; intervals do not guarantee future-run performance",
        "seed_level_sign_flip_warning": "with three training seeds the minimum attainable two-sided exact sign-flip p-value is 0.25",
        "equivalence_claimed": False,
        "evidence": evidence,
    }
    new_only_payload = {
        "schema": "tactile3d-unit.s4-3-pi2b-policy-new-seeds-only-sensitivity.v1",
        "status": "PASS",
        "created_at_utc": now(),
        "training_seeds": list(NEW_SEEDS),
        "contrasts": new_only,
        "interaction_diagnostic": new_interaction_record,
        "scope": "transparent sensitivity restricted to preregistered independent additions seeds43/44",
        "warning": "two training seeds are insufficient for population-level certainty; minimum exact two-sided sign-flip p-value is 0.5",
        "evidence": evidence,
    }
    audit_payload = {
        "schema": "tactile3d-unit.s4-3-pi2b-policy-statistics-audit.v1",
        "status": "PASS",
        "created_at_utc": now(),
        "gates": {
            "60_raw_artifact_hashes_match": len(evidence["raw_artifact_sha256"]) == 60,
            "3000_unique_canonical_tuples": True,
            "all_15_share_ordered_200_resets": True,
            "checkpoint_hashes_match_prefinal": True,
            "frozen_statistics_sources_match": True,
            "no_unresolved_server_client_errors": True,
            "success_totals_match_raw_blocks": True,
            "all_families_have_within_seed_holm": True,
            "seed_reset_rows_not_pooled_for_mcnemar": True,
            "track_b_new_teacher_read": False,
        },
        "total_successes": int(sum(outcomes[model].sum() for model in MODELS)),
        "total_failures": int(3000 - sum(outcomes[model].sum() for model in MODELS)),
        "evidence": evidence,
    }
    return {
        "per_seed": per_seed_payload,
        "crossed": crossed_payload,
        "new_only": new_only_payload,
        "audit": audit_payload,
        "decision": decision,
    }


def main() -> None:
    existing = [str(path) for path in OUTPUTS.values() if path.exists()]
    if existing:
        raise SystemExit("refusing to overwrite analysis artifacts: " + ", ".join(existing))
    results = analyze()
    for key, path in OUTPUTS.items():
        atomic_json(path, results[key])
    print(json.dumps({"status": "PASS", "outputs": {key: str(path) for key, path in OUTPUTS.items()}}, indent=2))


if __name__ == "__main__":
    main()
