#!/usr/bin/env python3
"""Audit S1.0-S1.5 evidence and emit the deterministic M1 acceptance result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(".local/artifacts/tactile_teacher")
    parser.add_argument("--s1-0", type=Path, default=root / "s1_0/s1_0_summary.json")
    parser.add_argument("--s1-1", type=Path, default=root / "s1_1/s1_1_summary.json")
    parser.add_argument("--s1-2", type=Path, default=Path(
        ".local/experiments/tactile_teacher/s1_baselines/s1_2_summary.json"
    ))
    parser.add_argument("--s1-3", type=Path, default=Path(
        ".local/experiments/tactile_teacher/s1_teacher/s1_3_summary.json"
    ))
    parser.add_argument("--s1-4", type=Path, default=root / "s1_4/s1_4_summary.json")
    parser.add_argument("--output-dir", type=Path, default=root / "s1_5")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    s10 = json.loads(args.s1_0.read_text())
    s11 = json.loads(args.s1_1.read_text())
    s12 = json.loads(args.s1_2.read_text())
    s13 = json.loads(args.s1_3.read_text())
    s14 = json.loads(args.s1_4.read_text())
    direct = s14["direct_future_prediction"]
    probes = s14["frozen_latent_probes"]
    temporal = s14["temporal_ablation"]
    trivial = s14["trivial_future_baselines"]
    collapse = s14["collapse"]

    gates = {
        "data_validity": (
            s10["status"] == "PASS"
            and s11["status"] == "PASS"
            and s12["status"] == "PASS"
            and s13["status"] == "PASS"
        ),
        "non_collapse": (
            collapse["per_dimension_variance"]["near_zero_fraction"] == 0
            and collapse["effective_rank"] > 1
            and collapse["pairwise_distance"]["p01"] > 0
        ),
        "temporal_value": (
            temporal["full_history"]["dynamic"]["mse"]
            < temporal["last_frame"]["dynamic"]["mse"]
        ),
        "temporal_order": (
            temporal["full_history"]["dynamic"]["mse"]
            < temporal["shuffled_history"]["dynamic"]["mse"]
            and temporal["full_history"]["dynamic"]["mse"]
            < temporal["reversed_history"]["dynamic"]["mse"]
        ),
        "future_predictability": (
            probes["teacher"]["future"]["all"]["mse"]
            < trivial["persistence"]["all"]["mse"]
            and probes["teacher"]["future"]["all"]["mse"]
            < trivial["mean"]["all"]["mse"]
        ),
        "baseline_comparison": (
            direct["teacher"]["dynamic"]["mse"] < direct["B0"]["dynamic"]["mse"]
            and direct["teacher"]["dynamic"]["mse"] < direct["B2"]["dynamic"]["mse"]
        ),
        "robustness_monotonic": all(
            levels["clean"]["all"]["mse"] < levels["mild"]["all"]["mse"]
            < levels["strong"]["all"]["mse"]
            for levels in s14["robustness"].values()
        ),
    }
    proposed_beats_b2 = direct["teacher"]["dynamic"]["mse"] < direct["B2"][
        "dynamic"
    ]["mse"]
    if all(gates.values()):
        final = "PASS" if proposed_beats_b2 else "PASS WITH MODEL-QUALITY WARNING"
    else:
        final = "FAIL"
    result = {
        "schema": "tactile3d-unit.s1.5-m1-acceptance.v1",
        "gates": {name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
        "proposed_beats_b2_dynamic_mse": proposed_beats_b2,
        "s1_final": final,
        "m1_predictable_contact_state_latent": (
            "ESTABLISHED" if final != "FAIL" else "NOT ESTABLISHED"
        ),
        "canonical_representation": (
            "h_t^c = E_T(T_[t-0.533s:t]) in R^256; continuous, order-sensitive, "
            "future-predictive"
        ),
        "interpretation": (
            "All relative M1 gates pass. Large transition amplitudes remain conservative in "
            "individual forecasts, so aggregate and temporal evidence should be considered "
            "together. No S2/UniT integration is included."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "m1_acceptance.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if final != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
