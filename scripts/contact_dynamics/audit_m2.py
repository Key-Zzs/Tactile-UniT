#!/usr/bin/env python3
"""Apply the explicit S2 gates and emit the deterministic M2 decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transition-manifest",
        type=Path,
        default=Path(".local/artifacts/contact_dynamics/s2_1/transition_manifest.json"),
    )
    parser.add_argument(
        "--training-summary",
        type=Path,
        default=Path(".local/experiments/contact_dynamics/s2_models/s2_training_summary.json"),
    )
    parser.add_argument(
        "--evaluation-summary",
        type=Path,
        default=Path(".local/artifacts/contact_dynamics/s2_4/s2_evaluation_summary.json"),
    )
    parser.add_argument(
        "--visualization-summary",
        type=Path,
        default=Path(
            ".local/artifacts/contact_dynamics/s2_4/plots/visualization_summary.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".local/artifacts/contact_dynamics/s2_5"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    transition = json.loads(args.transition_manifest.read_text())
    training = json.loads(args.training_summary.read_text())
    evaluation = json.loads(args.evaluation_summary.read_text())
    visualization = json.loads(args.visualization_summary.read_text())
    contract = transition["canonical_contract"]
    ablation = evaluation["ablations"]
    full_dynamic = ablation["full"]["dynamic"]["future_mse"]
    probes = evaluation["probes"]["transition_code"]
    semantics = {
        "contact_transition_macro_f1": probes["contact_transition"]["macro_f1"],
        "contact_transition_majority_macro_f1": probes["contact_transition"]["majority"][
            "macro_f1"
        ],
        "force_trend_macro_f1": probes["force_trend"]["macro_f1"],
        "force_trend_majority_macro_f1": probes["force_trend"]["majority"]["macro_f1"],
        "per_finger_change_only_accuracy": probes["per_finger_change"][
            "change_only_accuracy"
        ],
    }
    gates = {
        "A_teacher_freeze": (
            transition["teacher"]["parameters_changed"] is False
            and transition["teacher"]["requires_grad"] is False
            and transition["teacher"]["eval_deterministic_exact"] is True
        ),
        "B_transition_data_validity": (
            all(value == 0 for value in transition["episode_leakage_counts"].values())
            and contract["horizon_frames"] == 16
            and contract["overlap_samples"] == 0
            and transition["status"] == "PASS"
        ),
        "C_future_reconstruction": (
            evaluation["gates"]["future_reconstruction_finite"]
            and evaluation["baselines"]["proposed"]["all"]["future_mse"] >= 0
        ),
        "D_transition_code_necessity": (
            full_dynamic < ablation["zero"]["dynamic"]["future_mse"]
            and full_dynamic < ablation["shuffled_code"]["dynamic"]["future_mse"]
        ),
        "E_temporal_direction_or_pairing": (
            full_dynamic < ablation["reversed_transition"]["dynamic"]["future_mse"]
            or full_dynamic < ablation["shuffled_future"]["dynamic"]["future_mse"]
        ),
        "F_dynamic_semantics": (
            semantics["contact_transition_macro_f1"]
            > semantics["contact_transition_majority_macro_f1"]
            or semantics["force_trend_macro_f1"]
            > semantics["force_trend_majority_macro_f1"]
            or semantics["per_finger_change_only_accuracy"] > 0
        ),
        "G_non_collapse": evaluation["gates"]["non_collapse"],
        "H_interface_compatibility": evaluation["gates"]["interface_8x32"],
    }
    all_pass = all(gates.values())
    proposed_better = evaluation["proposed_vs_delta"]["proposed_better"]
    if all_pass:
        final = "PASS" if proposed_better else "PASS WITH MODEL-QUALITY WARNING"
    else:
        final = "FAIL"
    result = {
        "schema": "tactile3d-unit.s2.5-m2-acceptance.v1",
        "gates": {name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
        "teacher_checkpoint_sha256": transition["teacher"]["checkpoint_sha256"],
        "canonical_contract": contract,
        "pair_counts": {
            split: value["pairs"] for split, value in transition["splits"].items()
        },
        "selected_lambda_delta": training["selected_lambda_delta"],
        "test_metrics": evaluation["baselines"]["proposed"],
        "transition_code_controls": ablation,
        "dynamic_semantics": semantics,
        "non_collapse": {
            "effective_rank": evaluation["collapse"]["flattened_8x32"]["effective_rank"],
            "near_zero_fraction": evaluation["collapse"]["flattened_8x32"][
                "per_dimension_variance"
            ]["near_zero_fraction"],
            "query_collapsed_fraction": evaluation["collapse"]["query_diversity"][
                "collapsed_sample_fraction"
            ],
        },
        "proposed_vs_delta": evaluation["proposed_vs_delta"],
        "visualizations": visualization["files"],
        "s2_final": final,
        "m2_predictive_contact_dynamics_representation": (
            "ESTABLISHED" if final != "FAIL" else "NOT ESTABLISHED"
        ),
        "canonical_definition": (
            "z_c = E_c(h_t^c, h_{t+16}^c) in R^(8x32); "
            "h_hat_{t+16}^c = D_c(z_c, h_t^c); "
            "anchor delta 16/30 = 0.533333 s; non-overlapping Teacher windows"
        ),
        "s3_started": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "m2_acceptance.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if final != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
