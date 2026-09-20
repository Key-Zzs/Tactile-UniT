# S4.3-PI2B Multi-Seed Confirmation — Preregistration Draft

Status: **DRAFT — NOT EXECUTED — TRAINING NOT AUTHORIZED**

S4.3-PI2M produced a valid fixed-training-seed result in the direction opposite to the original Contact-target superiority hypothesis. At training seed42 and evaluator seed7, B_HVA reached 85/200 (42.5%) and B2 reached 46/200 (23.0%). The paired B2−B_HVA difference was −19.5 percentage points, with a frozen paired-bootstrap 95% CI of [−29.0, −10.0] points and Holm-adjusted p=0.000261645. This is `VA_TARGET_ADVANTAGE_CONFIRMED` only in the tested fixed-seed setting.

This draft records two scientifically coherent next-step choices. It does not authorize either choice, and it starts zero runs.

## Historical BVA boundary

The existing model named BVA is a historical VA auxiliary result whose target cache uses `+16 rows`, physically about `0.32 s`. Its checkpoint, cache, masks, training result, and evaluation result remain unchanged. It is not a matched-horizon VA-vs-VAC control for B2, whose contract is episode-local raw/control `t→t+27`, about `0.54 s`.

Any new no-contact VA model using the corrected `+27` target must receive a distinct identity such as `B_VA27`. It must never overwrite, relabel, or masquerade as historical BVA.

## Option A — Minimum matched-target confirmation

Confirm the already matched B1/B_HVA/B2 experiment across training seeds:

- retain the frozen seed42 checkpoints;
- independently train B1, B_HVA, and B2 at preregistered seeds43 and 44;
- run count: `3 models × 2 new seeds = 6` new 30k runs;
- primary: B2−B_HVA;
- key secondary: B_HVA−B1;
- replication secondary: B2−B1.

This option is the minimum design for testing whether the PI2M target ordering persists across training randomness. It does not estimate the no-contact VA condition at the corrected horizon.

## Option B — Full online-input × target matrix

If the paper requires the full input/target factorial context, use B0, a newly named B_VA27, B1, B_HVA, and B2 over seeds42/43/44:

- train B_VA27 at seeds42, 43, and 44: 3 new runs;
- train B0, B1, B_HVA, and B2 at seeds43 and 44: 8 new runs;
- total: 11 new 30k runs;
- preserve the existing seed42 B0/B1/B_HVA/B2 checkpoints unchanged;
- retain historical BVA only as explicitly labeled `+16 / 0.32 s` context.

B_VA27 must use the same episode-local raw/control `t→t+27` transition and valid-row identity as B2/B_HVA, mask every episode's final 27 rows, avoid cross-episode transitions and future-RGB interpolation, and use only the clean frozen VA teacher.

## Shared training contract if later authorized

- Initialize every new model/seed independently from the exact frozen official π0.5 base; do not warm-start from any trained ablation.
- Use the frozen 100-episode `pinch_tongs` dataset, 30,000 optimizer steps, global batch size 32, and the matching official π0.5 optimization recipe.
- Use final checkpoints only. No rollout-based early stopping, intermediate checkpoint selection, or checkpoint shopping.
- Choose seeds43/44 before observing their outcomes and preserve every result.

## Evaluation and recovery contract if later authorized

Evaluator seeds0–7 are exposed. Seed8 is only the current smallest candidate; its actual reset/configuration novelty must be re-audited and frozen before use. Use 200 shared reset identities per checkpoint in whichever option is authorized.

PI2M exposed an evaluator limitation that must be preregistered rather than hidden: for the interrupted B2 attempt's first 31 completed resets, a clean replay with identical reset IDs agreed on 18 outcomes and differed on 13. The formal PI2M result used one complete clean 200-reset B2 replay as canonical, with no per-reset splicing or best-of-retry selection. Therefore a future cohort must freeze an exact retry policy and disclose that the asynchronous evaluator is not outcome-deterministic across retries.

If a synchronous deterministic evaluator is adopted instead, every comparison group in that new cohort must be rerun under it after separate authorization. Results from different runtime semantics must not be silently combined.

## Statistical hierarchy

The reset is the paired unit within a checkpoint; training seed is the outer source of randomness. Report each seed's paired effect separately, then use a hierarchical bootstrap with training seed as the outer resampling unit or a preregistered hierarchical binary model. Never pool all rollout rows across seeds as independent observations. Three training seeds still give only a limited estimate of training variance.

## Explicit stop

No PI2B training, target generation, smoke evaluation, formal rollout, or GPU reservation is authorized by this document. A later authorization must choose Option A or B, refresh all protected hashes, re-audit GPU availability, and freeze a new protocol before any run starts.
