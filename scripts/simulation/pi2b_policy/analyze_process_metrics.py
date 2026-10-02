#!/usr/bin/env python3
"""Aggregate available Track-A Contact/tactile runtime diagnostics."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2b_policy"
COMPLETENESS = ARTIFACTS / "rollout_completeness.json"
INDEPENDENT = ARTIFACTS / "statistics_independent_audit.json"
OUTPUT = ARTIFACTS / "contact_process_metrics.json"
MODELS = ("B0", "B_VA27", "B1", "B_HVA", "B2")
SEEDS = (42, 43, 44)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def empty_group() -> dict[str, Any]:
    return {
        "episodes": 0,
        "successes": 0,
        "contact_state_finite_episodes": 0,
        "contact_state_queries": 0,
        "contact_state_weighted_l2_sum": 0.0,
        "contact_state_max_l2": 0.0,
        "tactile_samples": 0,
        "tactile_active_samples": 0,
        "tactile_weighted_l2_sum": 0.0,
        "tactile_max_l2": 0.0,
        "max_normal_force": 0.0,
        "matched_contact_count_sum": 0,
        "matched_contacts_by_region": {},
    }


def add_episode(group: dict[str, Any], row: dict[str, Any]) -> None:
    contact = row["contact_state_diagnostics"]
    tactile = row["tactile_diagnostics"]
    queries = int(contact["queries"])
    samples = int(tactile["samples"])
    group["episodes"] += 1
    group["successes"] += int(row["success"])
    group["contact_state_finite_episodes"] += int(contact["finite"])
    group["contact_state_queries"] += queries
    group["contact_state_weighted_l2_sum"] += float(contact["mean_l2"]) * queries
    group["contact_state_max_l2"] = max(group["contact_state_max_l2"], float(contact["max_l2"]))
    group["tactile_samples"] += samples
    group["tactile_active_samples"] += int(tactile["active_samples"])
    group["tactile_weighted_l2_sum"] += float(tactile["mean_l2"]) * samples
    group["tactile_max_l2"] = max(group["tactile_max_l2"], float(tactile["max_l2"]))
    group["max_normal_force"] = max(group["max_normal_force"], float(tactile["max_normal_force"]))
    group["matched_contact_count_sum"] += int(tactile["matched_contact_count_sum"])
    for region, count in tactile["matched_contacts_by_region"].items():
        group["matched_contacts_by_region"][region] = group["matched_contacts_by_region"].get(region, 0) + int(count)


def finish(group: dict[str, Any]) -> dict[str, Any]:
    return {
        "episodes": group["episodes"],
        "successes": group["successes"],
        "contact_state": {
            "finite_episodes": group["contact_state_finite_episodes"],
            "queries": group["contact_state_queries"],
            "mean_l2_weighted_by_queries": group["contact_state_weighted_l2_sum"] / group["contact_state_queries"],
            "max_l2": group["contact_state_max_l2"],
            "shape": [256],
        },
        "tactile": {
            "samples": group["tactile_samples"],
            "active_samples": group["tactile_active_samples"],
            "active_fraction": group["tactile_active_samples"] / group["tactile_samples"],
            "mean_l2_weighted_by_samples": group["tactile_weighted_l2_sum"] / group["tactile_samples"],
            "max_l2": group["tactile_max_l2"],
            "max_normal_force": group["max_normal_force"],
            "matched_contact_count_sum": group["matched_contact_count_sum"],
            "matched_contacts_by_region": dict(sorted(group["matched_contacts_by_region"].items())),
            "history_shape": [26, 30],
            "history_bootstrap": "LEFT_REPEAT_FIRST",
        },
    }


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit("refusing to overwrite Contact process metrics")
    completeness = read_json(COMPLETENESS)
    independent = read_json(INDEPENDENT)
    if completeness.get("status") != "PASS" or independent.get("status") != "PASS":
        raise SystemExit("required integrity inputs are not PASS")
    groups = {(model, seed): empty_group() for model in MODELS for seed in SEEDS}
    sent = {(model, seed): set() for model in MODELS for seed in SEEDS}
    extractor_hashes: set[str] = set()
    for name, record in completeness["raw_artifacts"].items():
        path = Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise SystemExit(f"raw artifact drift: {name}")
        payload = read_json(path)
        key = (payload["model"], int(payload["training_seed"]))
        sent[key].add(bool(payload["contact_state_sent_to_policy"]))
        extractor_hashes.add(hashlib.sha256(json.dumps(payload["extractor_audit"], sort_keys=True).encode()).hexdigest())
        for row in payload["episode_results"]:
            add_episode(groups[key], row)
    if len(extractor_hashes) != 1:
        raise SystemExit("extractor audit differs across formal blocks")
    checkpoints = {}
    for model in MODELS:
        checkpoints[model] = {}
        for seed in SEEDS:
            key = (model, seed)
            if groups[key]["episodes"] != 200 or len(sent[key]) != 1:
                raise SystemExit(f"process metric cardinality failure: {key}")
            checkpoints[model][str(seed)] = {
                "contact_state_sent_to_policy": next(iter(sent[key])),
                **finish(groups[key]),
            }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2b-policy-contact-process-metrics.v1",
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoints": checkpoints,
        "extractor_audit_identity_sha256": next(iter(extractor_hashes)),
        "gates": {
            "all_15_checkpoints_have_200_episodes": "PASS",
            "all_contact_states_finite": "PASS" if all(groups[key]["contact_state_finite_episodes"] == 200 for key in groups) else "FAIL",
            "contact_delivery_matches_mode": "PASS" if all(next(iter(sent[(model, seed)])) == (model in {"B1", "B_HVA", "B2"}) for model in MODELS for seed in SEEDS) else "FAIL",
            "extractor_identity_shared": "PASS",
        },
        "interpretation": "runtime process diagnostics only; no causal mechanism or performance-selection role",
        "inputs_sha256": {
            "rollout_completeness.json": sha256_file(COMPLETENESS),
            "statistics_independent_audit.json": sha256_file(INDEPENDENT),
        },
    }
    if any(value != "PASS" for value in payload["gates"].values()):
        raise SystemExit("Contact process metric gate failure")
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUTPUT)
    print(json.dumps({"status": "PASS", "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
