#!/usr/bin/env python3
"""Bind PI2S-R H parity to frozen fixed-observation action diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.pi2sr_provenance import sha256_file  # noqa: E402

ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_pi2sr"
CONTRACT = ROOT / "configs/simulation/pi2sr/runtime_contract_v2.json"
MODELS = ("B1_43", "B_HVA_42", "B_HVA_43", "B2_43")
POSITIONS = tuple(range(8))
CONDITIONS = ("correct", "same_episode_lag5", "zero")


def experiment_root() -> Path:
    value = os.environ.get("UNIT_EXPERIMENT_ROOT")
    return Path(value) if value else ROOT / ".local/experiments"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_identity = {
        (row["checkpoint_id"], int(row["fixed_position"]), row["condition"]): row
        for row in records
        if row["checkpoint_id"] in MODELS
        and int(row["fixed_position"]) in POSITIONS
        and row["condition"] in CONDITIONS
    }
    expected = {
        (model, position, condition)
        for model in MODELS
        for position in POSITIONS
        for condition in CONDITIONS
    }
    missing = sorted(expected - set(by_identity))
    if missing:
        raise RuntimeError(f"frozen fixed-observation records are incomplete: {missing[:5]}")

    canonical_vs_alternate = []
    perturbed: dict[str, list[float]] = {condition: [] for condition in CONDITIONS[1:]}
    for model in MODELS:
        for position in POSITIONS:
            correct = by_identity[(model, position, "correct")]
            action = np.asarray(correct["action"], dtype=np.float32)
            if action.shape != (30, 22) or not np.isfinite(action).all():
                raise RuntimeError("frozen correct-H action has invalid shape or values")
            canonical_vs_alternate.append(
                {
                    "model": model,
                    "fixed_position": position,
                    "global_row": int(correct["global_row"]),
                    "action_sha256": correct["action_sha256"],
                    "max_absolute_difference": 0.0,
                    "mean_absolute_difference": 0.0,
                    "reason": "canonical and alternate H are byte-identical under matched PI2S-R compute",
                }
            )
            for condition in CONDITIONS[1:]:
                row = by_identity[(model, position, condition)]
                perturbed[condition].append(float(row["delta"]["chunk_l2"]["all"]))
    return {
        "canonical_vs_alternate": canonical_vs_alternate,
        "historical_perturbations": {
            condition: {
                "samples": len(values),
                "mean_chunk_l2": float(np.mean(values)),
                "median_chunk_l2": float(np.median(values)),
                "max_chunk_l2": float(np.max(values)),
                "nonzero_count": int(np.count_nonzero(values)),
            }
            for condition, values in perturbed.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    expected_models = [
        name.replace("seed", "") for name in contract["action_sensitivity"]["models"]
    ]
    if expected_models != list(MODELS):
        raise RuntimeError("action-sensitivity implementation/model freeze mismatch")
    compute = json.loads((args.artifact_root / "compute_parity.json").read_text(encoding="utf-8"))
    if not compute["cross_path"]["byte_equal"]:
        raise RuntimeError(
            "alternate H is not byte-identical; fresh policy inference would be required"
        )
    historical = (
        experiment_root() / "simulation/s4_3_pi2s/artifacts/fixed_observation_interventions.json"
    )
    source = json.loads(historical.read_text(encoding="utf-8"))
    if source["status"] != "PASS" or source["gates"]["fixed_observation_actions_exact"] != "PASS":
        raise RuntimeError("historical fixed-observation action artifact is not qualified")
    result = summarize(source["records"])
    payload = {
        "schema": "tactile3d-unit.pi2sr-fixed-observation-action-sensitivity.v1",
        "status": "PASS",
        "models": list(MODELS),
        "fixed_observation_positions": list(POSITIONS),
        "samples_per_condition": len(MODELS) * len(POSITIONS),
        "canonical_runtime": "DIRECT_IN_PROCESS",
        "alternate_runtime": "UNIX_SERVICE_CPU_MATCHED_BATCH_ONE",
        "canonical_vs_alternate_h": compute["cross_path"],
        "canonical_vs_alternate_action": {
            "max_absolute_difference": 0.0,
            "mean_absolute_difference": 0.0,
            "byte_equal": True,
            "derivation": "Identical H bytes select the same already-frozen correct-H action record at fixed observation/noise; no policy rerun or checkpoint selection.",
        },
        **result,
        "historical_source": {
            "logical_path": "$UNIT_EXPERIMENT_ROOT/simulation/s4_3_pi2s/artifacts/fixed_observation_interventions.json",
            "sha256": sha256_file(historical),
            "status": source["status"],
            "shared_noise": source["gates"]["shared_noise_across_h_conditions"],
        },
        "interpretation": "Matched PI2S-R direct/service H creates no action difference on the frozen substitution check; historically defined larger H perturbations create nonzero fixed-observation action differences. Neither statement identifies a closed-loop failure cause.",
        "new_policy_inference": False,
        "training_or_optimizer_updates": 0,
        "checkpoint_selection": False,
        "causal_rollout_claim": False,
    }
    atomic_json(args.artifact_root / "fixed_observation_action_sensitivity.json", payload)
    print(
        json.dumps(
            {
                "canonical_alternate_action_max": 0.0,
                "historical_perturbed_nonzero": {
                    name: row["nonzero_count"]
                    for name, row in payload["historical_perturbations"].items()
                },
                "status": payload["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
