"""Frozen S4.2-R Contact-State no-collapse gate evaluation."""

from __future__ import annotations

from typing import Any, Mapping


def evaluate_no_collapse_contract(
    protocol: Mapping[str, Any],
    latent_metrics: Mapping[str, Any],
    pairwise: Mapping[str, float],
    original_gates: Mapping[str, bool],
    perturbation_controls: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate gates A-G exactly as frozen in the S4.2-R JSON contract."""

    contract = protocol["new_structural_collapse_contract"]
    variance = contract["gate_a_variance"]
    dominant = contract["gate_b_dominant_component"]
    rank_floor = contract["gate_c_effective_rank_floor"]
    relative = contract["gate_d_source_relative_diversity"]
    distance = contract["gate_e_pairwise_diversity"]
    semantic = contract["gate_f_semantic_functionality"]
    perturbation = contract["gate_g_perturbation_sensitivity"]
    relative_ratio = float(latent_metrics["effective_rank"]) / min(
        float(relative["d_src"]), 16.0
    )
    required_original = semantic["require_all_original_validation_gates"]
    expected_controls = perturbation["correct_history_must_beat"]
    details = {
        "A_variance": {
            "pass": float(latent_metrics["near_zero_variance_fraction"])
            <= float(variance["near_zero_variance_fraction_max"]),
            "value": float(latent_metrics["near_zero_variance_fraction"]),
            "maximum": float(variance["near_zero_variance_fraction_max"]),
            "threshold_definition": variance["near_zero_definition"],
        },
        "B_dominant_component": {
            "pass": float(latent_metrics["top_pc_explained_variance"])
            < float(dominant["top_1_pca_explained_variance_max_exclusive"]),
            "value": float(latent_metrics["top_pc_explained_variance"]),
            "maximum_exclusive": float(
                dominant["top_1_pca_explained_variance_max_exclusive"]
            ),
        },
        "C_effective_rank_floor": {
            "pass": float(latent_metrics["effective_rank"])
            >= float(rank_floor["effective_rank_min"]),
            "value": float(latent_metrics["effective_rank"]),
            "minimum": float(rank_floor["effective_rank_min"]),
        },
        "D_source_relative_diversity": {
            "pass": relative_ratio >= float(relative["ratio_min"]),
            "value": relative_ratio,
            "minimum": float(relative["ratio_min"]),
            "d_src": float(relative["d_src"]),
        },
        "E_pairwise_diversity": {
            "pass": (
                float(pairwise["same_sample_duplicate_distance_mean"])
                <= float(distance["same_sample_duplicate_distance_max"])
                and float(pairwise["different_sample_distance_mean"])
                > float(pairwise["same_sample_duplicate_distance_mean"])
                and float(pairwise["different_sample_distance_p05"]) > 0.0
            ),
            "same_sample_duplicate_distance_mean": float(
                pairwise["same_sample_duplicate_distance_mean"]
            ),
            "same_sample_duplicate_distance_max": float(
                distance["same_sample_duplicate_distance_max"]
            ),
            "different_sample_distance_mean": float(
                pairwise["different_sample_distance_mean"]
            ),
            "different_sample_distance_p05": float(
                pairwise["different_sample_distance_p05"]
            ),
        },
        "F_semantic_functionality": {
            "pass": all(bool(original_gates.get(name, False)) for name in required_original),
            "required_original_gates": {
                name: bool(original_gates.get(name, False)) for name in required_original
            },
        },
        "G_perturbation_sensitivity": {
            "pass": all(
                name in perturbation_controls
                and float(perturbation_controls[name]["control_mse"])
                > float(perturbation_controls[name]["correct_mse"])
                and float(perturbation_controls[name]["improvement_ci95"][0])
                > float(perturbation["bootstrap_error_difference_ci_lower_strictly_above"])
                for name in expected_controls
            ),
            "controls": {
                name: dict(perturbation_controls[name])
                for name in expected_controls
                if name in perturbation_controls
            },
        },
    }
    return {
        "gates": details,
        "overall_pass": all(row["pass"] for row in details.values()),
    }
