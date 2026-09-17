#!/usr/bin/env python3
"""Consolidate PI2W evidence and render the required static plots."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / ".local/artifacts/simulation/s4_3_pi2w"
PLOTS = OUT / "plots"
PI2U = ROOT / ".local/artifacts/simulation/s4_3_pi2u"
PI2A = ROOT / ".local/artifacts/simulation/s4_3_pi2a"
CACHE = ROOT / ".local/cache/simulation/s4_3_pi2w/dev_diagnostic_arrays.npz"


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def write(name: str, payload: Any) -> None:
    path = OUT / name
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def paired_precision(delta: float, discordance: float, episodes: int) -> dict[str, float]:
    standard_error = math.sqrt(max(discordance - delta * delta, 0.0) / episodes)
    z_alt = abs(delta) / standard_error if standard_error else math.inf
    power = normal_cdf(-1.96 - z_alt) + 1.0 - normal_cdf(1.96 - z_alt)
    return {
        "episodes": episodes,
        "delta": delta,
        "discordance": discordance,
        "standard_error": standard_error,
        "approximate_95ci_half_width": 1.96 * standard_error,
        "approximate_two_sided_power_alpha_0p05": power,
    }


def wilson_width(p: float, n: int) -> float:
    z = 1.96
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 2 * half


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(PLOTS / name, dpi=180, bbox_inches="tight")
    plt.close()


def render_plots(branch: dict[str, Any], paired: dict[str, Any], static: dict[str, Any],
                 high: dict[str, Any], rq: dict[str, Any], pi2u: dict[str, Any],
                 power: dict[str, Any]) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    colors = ["#64748b", "#2563eb", "#f59e0b", "#7c3aed"]
    paths = list(branch["paths"])
    labels = ["D0 static", "D1 vision", "D2 action", "D3 fusion"]

    means = [branch["paths"][p]["mean"] for p in paths]
    cis = [branch["paths"][p]["bootstrap_95ci"] for p in paths]
    errors = np.array([[m - ci[0], ci[1] - m] for m, ci in zip(means, cis)]).T
    plt.figure(figsize=(7, 4.2))
    plt.bar(labels, means, color=colors, yerr=errors, capsize=4)
    plt.ylabel("Future-DINO cosine loss")
    plt.title("DexJoCo DEV whole-frame reconstruction")
    savefig("01_dev_branch_whole_frame_loss.png")

    names = list(paired["comparisons"])
    pmeans = [paired["comparisons"][n]["mean"] for n in names]
    pcis = [paired["comparisons"][n]["bootstrap_95ci"] for n in names]
    perr = np.array([[m - ci[0], ci[1] - m] for m, ci in zip(pmeans, pcis)]).T
    plt.figure(figsize=(7, 4.2))
    plt.errorbar(range(3), pmeans, yerr=perr, fmt="o", capsize=5, color="#dc2626")
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(range(3), ["Vision−D0", "Action−D0", "Fusion−D0"])
    plt.ylabel("Paired loss difference")
    plt.title("All learned routes are worse on whole-frame loss")
    savefig("02_paired_loss_differences_vs_static.png")

    groups = list(branch["paths"][paths[0]]["per_source_group_mean"])
    x = np.arange(len(groups))
    plt.figure(figsize=(11, 4.8))
    width = 0.19
    for i, path in enumerate(paths):
        vals = [branch["paths"][path]["per_source_group_mean"][g] for g in groups]
        plt.bar(x + (i - 1.5) * width, vals, width, label=labels[i], color=colors[i])
    plt.xticks(x, [g.replace("-script-", "\ns") for g in groups], fontsize=8)
    plt.ylabel("Mean loss")
    plt.title("Per-source-group branch loss")
    plt.legend(ncol=4, fontsize=8)
    savefig("03_per_source_group_branch_loss.png")

    with np.load(CACHE, allow_pickle=False) as arrays:
        similarity = arrays["current_future_similarity"]
        motion = arrays["patch_motion"].reshape(-1)
    plt.figure(figsize=(7, 4.2))
    plt.hist(similarity, bins=50, color="#0f766e", alpha=0.85)
    plt.axvline(similarity.mean(), color="black", linestyle="--", label=f"mean={similarity.mean():.3f}")
    plt.xlabel("Current→future DINO cosine similarity")
    plt.ylabel("Samples")
    plt.title("Persistence strength on DexJoCo DEV")
    plt.legend()
    savefig("04_current_future_dino_similarity.png")

    plt.figure(figsize=(7, 4.2))
    plt.hist(motion, bins=np.linspace(0, 1.01, 101), color="#ea580c", alpha=0.85)
    plt.yscale("log")
    plt.xlabel("Patch motion magnitude (1−cosine)")
    plt.ylabel("Patches (log scale)")
    plt.title("Patch feature-change distribution")
    savefig("05_patch_motion_magnitude.png")

    strata = ["all_patches", "top_10_percent", "top_25_percent", "bottom_50_percent"]
    x = np.arange(len(strata))
    plt.figure(figsize=(9, 4.8))
    for i, path in enumerate(paths):
        vals = [high["results"][s][path]["mean"] for s in strata]
        plt.bar(x + (i - 1.5) * width, vals, width, label=labels[i], color=colors[i])
    plt.xticks(x, ["All", "Top 10%", "Top 25%", "Bottom 50%"])
    plt.ylabel("Cosine loss")
    plt.title("Static dominance reverses on high-change patches")
    plt.legend(ncol=4, fontsize=8)
    savefig("06_patch_stratified_branch_results.png")

    rnames = ["D1_vision_only", "D2_action_only", "D3_fusion"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(range(3), [rq["routes"][n]["pre_post_cosine"]["mean"] for n in rnames], color=colors[1:])
    axes[0].set_xticks(range(3), ["Vision", "Action", "Fusion"])
    axes[0].set_ylim(0.95, 1.0)
    axes[0].set_ylabel("Pre/post-RQ cosine")
    axes[1].bar(range(3), [rq["routes"][n]["quantization_mse"]["mean"] for n in rnames], color=colors[1:])
    axes[1].set_xticks(range(3), ["Vision", "Action", "Fusion"])
    axes[1].set_ylabel("Quantization MSE")
    fig.suptitle("RQ distortion diagnostics")
    savefig("07_pre_post_rq_distortion.png")

    models = ["B0", "BVA", "B1", "B2"]
    rates = [pi2u["summaries"][m]["success_rate"] for m in models]
    ci = [pi2u["summaries"][m]["wilson_95ci"] for m in models]
    err = np.array([[r - c[0], c[1] - r] for r, c in zip(rates, ci)]).T
    plt.figure(figsize=(7, 4.2))
    plt.bar(models, rates, yerr=err, capsize=4, color=["#64748b", "#0f766e", "#2563eb", "#7c3aed"])
    plt.ylabel("Success rate")
    plt.ylim(0, 0.36)
    plt.title("Formal seed-42 policy evidence (evaluator seed 6)")
    savefig("08_policy_evidence_b0_bva_b1_b2.png")

    matrix = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    plt.figure(figsize=(6, 4.4))
    plt.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.yticks(range(4), models)
    plt.xticks(range(2), ["Online tactile", "Physical auxiliary"])
    for i in range(4):
        for j in range(2):
            plt.text(j, i, "Yes" if matrix[i, j] else "No", ha="center", va="center")
    plt.title("Method information / auxiliary ablation matrix")
    savefig("09_method_information_auxiliary_matrix.png")

    plt.figure(figsize=(9, 4.6))
    plt.axis("off")
    boxes = [
        (0.03, 0.63, "BVA formal\n24.0%"),
        (0.37, 0.63, "B2 formal\n21.0%"),
        (0.70, 0.63, "BVA > B2\npoint estimate"),
        (0.37, 0.15, "PI2B_MECHANISM_\nDIAGNOSIS_FIRST"),
    ]
    for x0, y0, text in boxes:
        plt.text(x0, y0, text, transform=plt.gca().transAxes, ha="left", va="center",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#334155"))
    plt.annotate("", xy=(0.36, 0.66), xytext=(0.20, 0.66), xycoords="axes fraction", arrowprops=dict(arrowstyle="->"))
    plt.annotate("", xy=(0.68, 0.66), xytext=(0.53, 0.66), xycoords="axes fraction", arrowprops=dict(arrowstyle="->"))
    plt.annotate("", xy=(0.52, 0.34), xytext=(0.80, 0.60), xycoords="axes fraction", arrowprops=dict(arrowstyle="->"))
    plt.title("PI2B readiness decision")
    savefig("10_pi2b_minimal_model_set_decision.png")

    plt.figure(figsize=(8, 4.5))
    for contrast, rows in power["scenarios"].items():
        episodes = [row["episodes"] for row in rows]
        widths = [200 * row["approximate_95ci_half_width"] for row in rows]
        plt.plot(episodes, widths, marker="o", label=contrast)
    plt.xlabel("Paired episodes / checkpoint")
    plt.ylabel("Approx. full 95% CI width (percentage points)")
    plt.title("Prospective rollout precision from observed discordance")
    plt.legend(fontsize=8)
    savefig("11_pi2b_prospective_precision.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    branch = load(OUT / "dev_branch_diagnostic.json")
    paired = load(OUT / "dev_branch_paired_statistics.json")
    static = load(OUT / "static_dominance_diagnostic.json")
    high = load(OUT / "high_change_patch_diagnostic.json")
    rq = load(OUT / "rq_localization.json")
    native = load(OUT / "native_unit_gate_sanity.json")
    bva_stats = load(PI2U / "paired_ablation_statistics.json")
    bva_manifest = load(PI2U / "bva_checkpoint_manifest.json")
    bva_completion = load(PI2U / "bva_training_completion.json")
    pi2a_decision = load(PI2A / "final_decision.json")

    diagnosis = {
        "schema": "tactile3d-unit.s4-3-pi2w-pi2v-failure-diagnosis.v1",
        "status": "PASS",
        "historical_decision_preserved": "S4_3_PI2V_UNIT_ADAPTER_REPRESENTATION_FAIL",
        "primary_diagnosis": "PI2V_FAIL_VISION_DOMAIN_TRANSFER",
        "secondary_factors": [
            "WHOLE_FRAME_DINO_STATIC_REGION_DOMINANCE",
            "VISUAL_ENCODER_FUSION_DECODER_COMPONENT_NOT_SEPARATELY_IDENTIFIED",
        ],
        "evidence": {
            "dexjoco_D0": branch["paths"]["D0_no_motion"]["mean"],
            "dexjoco_D1": branch["paths"]["D1_vision_only"]["mean"],
            "dexjoco_D1_minus_D0": paired["comparisons"]["D1_vision_only_minus_D0"],
            "dexjoco_fraction_D1_beating_D0": branch["paths"]["D1_vision_only"]["fraction_samples_beating_D0"],
            "native_D0": native["paths"]["D0_no_motion"]["mean"],
            "native_D1": native["paths"]["D1_vision_only"]["mean"],
            "native_D3": native["paths"]["D3_fusion"]["mean"],
            "native_fraction_D1_beating_D0": native["paths"]["D1_vision_only"]["fraction_samples_beating_D0"],
            "dexjoco_current_future_similarity": static["current_to_future_cosine_similarity"]["mean"],
            "top10_change_concentration": static["change_concentration"]["top_10_percent_fraction_of_cosine_change"]["mean"],
            "top10_D1_improvement": high["results"]["top_10_percent"]["D1_vision_only"]["motion_explanation_improvement_D0_minus_model"]["mean"],
        },
        "rq_classification": "RQ_HEALTHY",
        "rq_reason": "All routes retain mean pre/post-RQ cosine near 0.98, cross-modal relation preservation is 0.972 Pearson, and both RVQ levels use many active codes; these data do not support RQ as the dominant failure source.",
        "decoder_localization": "INCONCLUSIVE_WITHIN_VISUAL_PATH",
        "decoder_reason": "The official decoder accepts only bridge-projected quantized tokens, so no untrained pre-RQ decode was invented. Native GR1 success shows the released end-to-end visual path works in-domain, but DexJoCo cannot isolate encoder/M-Former/fusion/decoder contributions without intervention.",
        "does_not_overturn_historical_gate": True,
    }
    write("pi2v_failure_diagnosis.json", diagnosis)

    claims = {
        "schema": "tactile3d-unit.s4-3-pi2w-original-unit-claim-boundary.v1",
        "status": "PASS",
        "official_code_used": True,
        "official_implementation_used": True,
        "released_tokenizer_weights_used": True,
        "untouched_official_unit_baseline": False,
        "official_original_unit_policy_evaluated_on_dexjoco": False,
        "can_claim_original_unit_fundamentally_fails": False,
        "can_claim_rq_is_worse_than_continuous": False,
        "can_claim_adapter_only_transfer_failed_preregistered_gate": True,
        "allowed_wording": "Using the official UniT implementation and released initialization, an adapter-only DexJoCo transfer failed our preregistered representation validation and was therefore not used as a downstream policy baseline.",
        "forbidden_wording": [
            "Original UniT performs worse than Tactile-UniT on DexJoCo.",
            "Our experiments prove discrete RQ representations are inferior.",
        ],
    }
    write("original_unit_claim_boundary.json", claims)

    bva_source = {
        "schema": "tactile3d-unit.s4-3-pi2w-bva-source-of-truth.v1",
        "status": "BVA_FORMAL_COMPLETE",
        "training_seed": 42,
        "optimizer_steps": 30000,
        "global_batch_size": bva_completion["global_batch_size"],
        "checkpoint": bva_manifest["checkpoint"],
        "checkpoint_tree_sha256": bva_manifest["checkpoint_tree_sha256"],
        "training_status": bva_completion["status"],
        "formal_evaluator_seed": bva_stats["evaluator_seed"],
        "episodes_per_model": bva_stats["episodes_per_model"],
        "formal_evaluation_status": bva_stats["status"],
        "retrained_in_pi2w": False,
    }
    write("bva_source_of_truth.json", bva_source)
    write("bva_formal_results.json", {
        "schema": "tactile3d-unit.s4-3-pi2w-bva-formal-results.v1",
        "status": "PASS",
        "source": "$REPO_ROOT/.local/artifacts/simulation/s4_3_pi2u/paired_ablation_statistics.json",
        "evaluator_seed": bva_stats["evaluator_seed"],
        "episodes_per_model": bva_stats["episodes_per_model"],
        "summaries": bva_stats["summaries"],
        "contrasts": bva_stats["contrasts"],
        "holm_family": bva_stats["holm_family"],
        "mechanism_classification": bva_stats["BVA_outcome"],
    })

    methods = {
        "B0": {"online_tactile": False, "physical_supervision": "NONE", "representation_target": "NONE", "training_seed": 42},
        "BVA": {"online_tactile": False, "physical_supervision": "VA_ONLY_CONTINUOUS", "representation_target": "continuous visual-action physical auxiliary", "training_seed": 42},
        "B1": {"online_tactile": True, "physical_supervision": "NONE", "representation_target": "online Contact-State tokens only", "training_seed": 42},
        "B2": {"online_tactile": True, "physical_supervision": "VAC", "representation_target": "Contact-State tokens plus continuous VAC physical auxiliary", "training_seed": 42},
    }
    evidence_rows = []
    for model, details in methods.items():
        if model in bva_stats["summaries"]:
            evidence_rows.append({"protocol": "PI2U_FORMAL_SEED6", "model": model, **details, **bva_stats["summaries"][model]})
        if model != "BVA":
            summary = load(PI2A / f"{model.lower()}_summary.json")
            evidence_rows.append({"protocol": "PI2A_FRESH200_SEED2", "model": model, **details, **summary})
    mechanism = {
        "schema": "tactile3d-unit.s4-3-pi2w-mechanism-evidence-table.v1",
        "status": "PASS",
        "rows": evidence_rows,
        "paired_effects": {
            "PI2A": {
                "B1-B0": pi2a_decision["secondary_comparison"],
                "B2-B1": pi2a_decision["key_secondary_comparison"],
                "B2-B0": pi2a_decision["primary_comparison"],
            },
            "PI2U": bva_stats["contrasts"],
        },
        "pooling": "NOT_PERFORMED; PI2A seed2 and PI2U seed6 remain separate protocols.",
    }
    write("mechanism_evidence_table.json", mechanism)

    questions = {
        "schema": "tactile3d-unit.s4-3-pi2w-core-question-answers.v1",
        "status": "PASS",
        "Q1_online_contact_state_alone_improves_pi05": {"answer": "NO", "evidence": "B1-B0 is -3.5 pp on PI2A and -2.5 pp on PI2U; neither is confirmed."},
        "Q2_vac_aux_improves_over_contact_state": {"answer": "YES", "evidence": "PI2A B2-B1 is +10.5 pp, paired 95% CI [3.0,18.0] pp, exact p=0.0111, Holm-adjusted p=0.0334; PI2U remains +5.0 pp trend."},
        "Q3_va_only_supervision_improves_pi05": {"answer": "TREND", "evidence": "PI2U BVA-B0 is +5.5 pp, paired 95% CI [-2.0,13.0] pp, exact p=0.207; not confirmed."},
        "Q4_b2_outperforms_bva": {"answer": "NO", "evidence": "PI2U B2-BVA is -3.0 pp, paired 95% CI [-11.5,5.5] pp."},
        "Q5_generic_vs_contact_aware_auxiliary_distinguished": {"answer": "INCONCLUSIVE", "evidence": "BVA is competitive and B2-BVA is not an isolated Contact-target contrast because online information also differs."},
        "Q6_bunit_necessary_for_core_claim": {"answer": "NO", "evidence": "The supported core mechanism claim is B2 versus B1/B0, not superiority to Original UniT."},
    }
    write("core_question_answers.json", questions)

    bunit = {
        "schema": "tactile3d-unit.s4-3-pi2w-bunit-disposition.v1",
        "status": "PASS",
        "decision": "BUNIT_APPENDIX_FAILED_TRANSFER_ONLY",
        "reason": "No policy-level BUniT exists, PI2V failed only the adapter representation gate, and the paper's core claim does not require superiority over Original UniT.",
        "main_paper": "Omit BUniT from the main policy ablation and make no Original-UniT superiority claim.",
        "appendix": "Report the preregistered adapter-only transfer failure, native sanity, diagnosis, and claim boundary.",
        "retry": "Future work only under a new preregistration; no retuning or checkpoint shopping of PI2V.",
    }
    write("bunit_disposition.json", bunit)

    scenarios = {
        "B2-B1_PI2A": (0.105, 63 / 200),
        "B2-B1_PI2U": (0.05, 58 / 200),
        "B2-B0_PI2A": (0.07, 74 / 200),
        "BVA-B0_PI2U": (0.055, 63 / 200),
        "B2-BVA_PI2U": (-0.03, 74 / 200),
    }
    power = {
        "schema": "tactile3d-unit.s4-3-pi2w-pi2b-power-precision-analysis.v1",
        "status": "PASS",
        "method": "Normal approximation for paired binary difference using Var(delta_hat)=(discordance-delta^2)/N; diagnostic planning only.",
        "scenarios": {
            name: [paired_precision(delta, q, n) for n in (100, 200)]
            for name, (delta, q) in scenarios.items()
        },
        "marginal_rate_wilson_full_width_at_p0p20": {str(n): wilson_width(0.20, n) for n in (100, 200)},
        "recommendation": 200,
        "reason": "At observed discordance, 200 gives roughly 14.8-16.8 pp full paired-difference CI width versus 20.7-23.8 pp at 100; PI2A-like +10.5 pp has about 75% rather than 46% approximate within-seed power.",
        "hierarchy_warning": "These calculations condition on one trained checkpoint. With three training seeds, training-seed variation is the higher-level uncertainty and must not be erased by pooling rollouts.",
    }
    write("pi2b_power_precision_analysis.json", power)

    model_set = {
        "schema": "tactile3d-unit.s4-3-pi2w-pi2b-model-set-recommendation.v1",
        "status": "DRAFT_NOT_EXECUTED",
        "decision": "PI2B_MECHANISM_DIAGNOSIS_FIRST",
        "trigger": "BVA point estimate (24.0%) exceeds B2 (21.0%) in the formal common seed6 evaluation and generic auxiliary benefit remains a serious explanation.",
        "immediate_new_training_runs": 0,
        "conditional_model_set_after_claim_resolution": ["B0", "BVA", "B1", "B2"],
        "conditional_new_training_seeds": [43, 44],
        "training_seed_rule": "next two consecutive integers after the sole canonical training seed 42; frozen without observing outcomes",
        "conditional_new_30k_runs": 8,
        "why_not_B0_B1_B2_only": "Dropping competitive BVA would leave the generic physical-auxiliary alternative unresolved.",
        "required_precondition": "Before authorizing eight runs, decide whether the manuscript needs a generic-vs-contact-aware mechanism claim. If yes, preregister a clean isolating control/design; if no, narrow the claim and retain all four existing mechanisms in multi-seed confirmation.",
    }
    write("pi2b_model_set_recommendation.json", model_set)

    recommendation = {
        "schema": "tactile3d-unit.s4-3-pi2w-pi2b-recommendation.v1",
        "status": "DRAFT_NOT_EXECUTED",
        "training_authorized": False,
        "readiness": model_set["decision"],
        "existing_training_seed": 42,
        "proposed_additional_training_seeds": [43, 44],
        "conditional_models_per_seed": ["B0", "BVA", "B1", "B2"],
        "conditional_new_runs": 8,
        "initialization": "Each model independently initializes from the exact frozen official pi0.5 base, matching its seed42 recipe; no warm-start from another ablation.",
        "optimizer_steps": 30000,
        "global_batch_size": 32,
        "dataset": "same frozen official 100-episode pinch_tongs dataset and existing sidecars/targets",
        "recipe": "same official pi0.5 optimization recipe and final-checkpoint-only rule",
        "fresh_evaluator_seed": 7,
        "evaluator_seed_reason": "Seeds 0-6 have all been exposed, including incomplete/invalid PI2U attempts 3-5 and formal seed6; 7 is the smallest unexposed seed.",
        "episodes_per_checkpoint": 200,
        "common_reset_rule": "One frozen seed7 reset sequence shared by every checkpoint; no outcome-dependent replacement.",
        "primary_comparison": "B2-B1",
        "key_secondary": "B2-B0",
        "mechanism_secondary": "B2-BVA",
        "statistical_treatment": "Report paired effects separately per training seed, then use a training-seed hierarchical bootstrap or hierarchical binary model; do not pool 3xN rollouts as independent.",
        "compute_estimate": {
            "observed_seconds_per_step": {"B0_derived": 2.63, "B2": 2.7, "BVA": 3.9},
            "conditional_sequential_wall_hours_for_8_runs_approx": 198,
            "conditional_gpu_hours_approx_at_two_gpus_per_run": 396,
            "note": "Planning estimate only; no process launched.",
        },
    }
    write("pi2b_recommendation.json", recommendation)

    render_plots(branch, paired, static, high, rq, bva_stats, power)
    print(json.dumps({
        "status": "PASS",
        "PI2V": diagnosis["primary_diagnosis"],
        "BUniT": bunit["decision"],
        "PI2B": model_set["decision"],
    }))


if __name__ == "__main__":
    main()
