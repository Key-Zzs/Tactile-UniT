# S4.3-PI2B Multi-Seed Confirmation — Preregistration Draft

Status: **DRAFT — NOT EXECUTED — TRAINING NOT AUTHORIZED**

PI2B is not ready to launch. The formal seed42/evaluator-seed6 result places BVA at 24.0% and B2 at 21.0%; because BVA exceeds B2 by point estimate, the frozen PI2W decision is `PI2B_MECHANISM_DIAGNOSIS_FIRST`. Before training, the paper must decide whether it needs to distinguish generic physical-auxiliary benefit from Contact-aware VAC benefit. If it does, a clean isolating control/design must be preregistered. If it does not, the claim must be narrowed and the four-model confirmation below may be separately authorized.

## Conditional model set

The minimum fair set after that claim decision is B0, BVA, B1, and B2. Omitting BVA is not scientifically acceptable because BVA-B0 is a +5.5 pp positive trend and BVA is competitive with B2.

The existing training seed is 42. The two proposed additional seeds are 43 and 44, selected by the frozen rule “next two consecutive integers after the sole canonical seed42,” without observing outcomes. This implies eight new 30k runs if the conditional four-model protocol is later authorized.

## Frozen training contract

- Initialize every model/seed independently from the exact frozen official pi0.5 base; do not warm-start from another ablation.
- Use the same frozen official 100-episode `pinch_tongs` dataset and the existing frozen sidecars/targets.
- Use the same official pi0.5 optimization recipe, 30,000 optimizer steps, and global batch size 32.
- Select only the final checkpoint. No rollout-based early stopping, intermediate checkpoint selection, or checkpoint shopping.
- Preserve B0, BVA, B1, and B2 information contracts exactly as already audited.

## Evaluation contract

Evaluator seeds 0–6 are exposed, including incomplete or invalid PI2U attempts 3–5 and the formal seed6 evaluation. Seed7 is therefore the smallest unexposed evaluator seed. Freeze one seed7 reset sequence and reuse it for every checkpoint.

Evaluate 200 paired episodes per checkpoint. With observed paired discordance of 0.29–0.37, 200 episodes yields an approximate full paired-difference 95% interval width of 14.8–16.8 pp, versus 20.7–23.8 pp at 100. For the PI2A-like +10.5 pp B2-B1 effect, approximate within-checkpoint power rises from 47.8% at 100 to 76.8% at 200. These are planning diagnostics, not guarantees across training seeds.

The primary contrast is B2-B1. B2-B0 is the key secondary contrast. B2-BVA is the mechanism secondary if the four-model protocol is authorized.

## Statistical hierarchy

Training seed is the higher-level source of randomness. Report the paired effect and uncertainty separately for each training seed, then use a hierarchical bootstrap with training seeds as the outer resampling unit or a hierarchical binary model with training-seed effects. Do not pool `3 × N` rollout outcomes as if all rollouts were independent.

## Compute planning

Observed prior timing is approximately 2.63 s/step for B0, 2.7 s/step for B2, and 3.9 s/step for BVA. Eight conditional runs are roughly 198 sequential wall-hours or 396 GPU-hours at two GPUs per run. This is an estimate only; PI2W launched no training process.
