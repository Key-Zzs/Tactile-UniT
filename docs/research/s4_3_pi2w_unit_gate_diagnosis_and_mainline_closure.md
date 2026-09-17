# S4.3-PI2W Original-UniT Gate Diagnosis & Mainline Closure

## Scope and historical closure

PI2W was read-only diagnosis, evidence consolidation, and next-stage design. It did not train or resume any tokenizer, adapter, pi0.5 policy, raw-tactile baseline, or world model. It did not retune a threshold, select an intermediate checkpoint, retrain BVA, generate BUniT targets, start PI2B, or mutate learned weights.

The historical decision remains unchanged: `S4_3_PI2V_UNIT_ADAPTER_REPRESENTATION_FAIL`. PI2V failed its preregistered future-vision reconstruction gate at the single preregistered step80000 checkpoint. This does not establish that Original UniT fundamentally fails on DexJoCo, and it does not establish that RQ is inferior to a continuous representation.

## Official visual-reconstruction contract

At official UniT commit `0d762e32180bddd765694ef3846a3a5053f9d37f`, current and future 224×224 ImageNet-normalized RGB frames pass through the same DINOv2-large backbone. The wrapper selects hidden-state layer `-2`, removes CLS, and returns 256 ordered patch features of dimension 1024. The Vision branch uses current features as condition and future features as target to produce eight 1024-dimensional query features. Vision and Action queries enter the `pv`/`pa` routed Fusion module, are projected to 32 dimensions, quantized by the released two-stage 128-code RVQ, projected back to 1024 dimensions, and decoded with the current DINO patch tensor as condition.

For batch element (i) and patch (j), the official DINO-mode loss is:

\[
L = \frac{1}{B}\sum_i \left[1 - \frac{1}{256}\sum_j
\cos\left(\hat f_{i,j}^{future}, f_{i,j}^{future}\right)\right].
\]

Cosine similarity is over the last, 1024-dimensional feature axis; patches and then samples are averaged. Current and future features have identical preprocessing, layer, CLS exclusion, patch ordering, and no asymmetric target projection.

## No-motion gate audit

The historical validator implements

\[
L_{static}=\frac{1}{B}\sum_i\left[1-\frac{1}{256}\sum_j
\cos(f_{i,j}^{current},f_{i,j}^{future})\right].
\]

The static predictor and future target are in exactly the same DINO layer, patch order, normalization, and temporal pair as the learned reconstruction target, with the same cosine and reductions and no one-sided projection. The implementation classification is `NO_MOTION_GATE_IMPLEMENTATION_VALID`.

This is nevertheless a custom preregistered persistence diagnostic, not an official UniT acceptance criterion.

## DexJoCo read-only DEV diagnostic

The exact frozen 4,860-row, nine-source-group DEV split and step80000 adapter were evaluated with a common FP32 cosine calculation for D0–D3. No path-specific target or metric was used.

| Path | Mean loss | 95% bootstrap CI | Mean difference vs D0 | Fraction beating D0 |
|---|---:|---:|---:|---:|
| D0 no-motion | 0.08118 | [0.07990, 0.08245] | reference | — |
| D1 Vision-only | 0.09884 | [0.09767, 0.10002] | +0.01766 | 0.0% |
| D2 Action-only | 0.09797 | [0.09680, 0.09914] | +0.01679 | 0.0% |
| D3 Fusion | 0.09795 | [0.09679, 0.09912] | +0.01677 | 0.0% |

Vision-only fails before the Action adapter can be the sole explanation. Action-only and Fusion also fail, but their nearly identical whole-frame loss does not localize the primary problem to Action/Fusion.

## RQ and decoder localization

Mean pre/post-RQ cosine is approximately 0.981 for Vision-only, 0.983 for Action-only, and 0.980 for Fusion. Pre/post Vision–Action relation has Pearson correlation 0.972. The routes use 31–44 active first-level codes and 71–85 active second-level codes, with normalized entropy well above the preregistered collapse floor. The diagnosis is `RQ_HEALTHY`, not `RQ_DOMINANT_FAILURE_CANDIDATE`.

No pre-RQ tensor was fed through the frozen decoder because the released decoder contract accepts bridge-projected post-RQ tokens. The data therefore cannot isolate visual encoder/M-Former/Fusion from decoder transfer without an intervention. Decoder-only causality remains inconclusive within the broader visual-domain path.

## Static-dominance diagnosis

Mean current→future DINO cosine similarity is 0.91882. The median patch motion magnitude is only 0.00868. The top 10% changing patches contain 66.24% of total cosine change, and the top 25% contain 90.67%.

On the top 10% changing patches, D1 improves over persistence by 0.03068 mean loss, D2 by 0.01294, and D3 by 0.02481; D1 and D3 beat D0 on 76.1% and 74.8% of samples. On the bottom 50%, persistence wins on every sample. The whole-frame metric is therefore static-region dominated even though it is implemented correctly. Persistence is a strong approximation in this feature space; this does not mean the future equals the current.

This diagnostic does not replace or overturn the frozen PI2V gate.

## Native released-UniT sanity

A bounded read-only sanity test used 512 samples from the existing local held-out GR1 representation subset, the released fulldata tokenizer, and the same D0–D3 metric. D0 loss is 0.31680, D1 loss is 0.13666, and D3 loss is 0.16173. D1 and D3 beat persistence on 99.8% and 99.6% of samples. The custom gate is therefore capable of recognizing released UniT in-domain; the DexJoCo Vision-only failure is strong evidence of visual-domain transfer limitation rather than a universally unfaithful gate.

The primary diagnosis is `PI2V_FAIL_VISION_DOMAIN_TRANSFER`. Whole-frame static dominance is a secondary factor. The exact frozen visual component responsible is not separately identified.

## Original UniT claim boundary

Official code, the released tokenizer weights, and the official reconstruction implementation were used. The evaluated system was not an untouched official UniT baseline: it added a DexJoCo embodiment adapter. No official Original-UniT policy was trained or evaluated on DexJoCo.

Allowed wording:

> Using the official UniT implementation and released initialization, an adapter-only DexJoCo transfer failed our preregistered representation validation and was therefore not used as a downstream policy baseline.

Forbidden without a fair policy-level comparison: “Original UniT performs worse than Tactile-UniT on DexJoCo.” Also forbidden: “Our experiments prove discrete RQ representations are inferior.”

## BVA source-of-truth and mechanism evidence

BVA is `BVA_FORMAL_COMPLETE`: seed42, 30,000 steps, global batch 32, formal evaluator seed6, 200 episodes/model. PI2W did not retrain or reevaluate it.

| Method | Online tactile | Physical supervision | PI2U seed6 success | Wilson 95% CI |
|---|---|---|---:|---:|
| B0 | No | None | 18.5% | [13.7%, 24.5%] |
| BVA | No | VA-only continuous | 24.0% | [18.6%, 30.4%] |
| B1 | Yes | None | 16.0% | [11.6%, 21.7%] |
| B2 | Yes | VAC | 21.0% | [15.9%, 27.2%] |

BVA-B0 is +5.5 pp, paired 95% CI [-2.0, 13.0] pp, exact McNemar p=0.207 and Holm-adjusted p=0.829: a positive trend, not confirmation. B2-BVA is -3.0 pp, paired 95% CI [-11.5, 5.5] pp. B2 does not outperform BVA.

PI2A fresh-200 remains separate and is not pooled: B0 18.5%, B1 15.0%, B2 25.5%; B2-B1 is +10.5 pp with paired 95% CI [3.0, 18.0] pp and Holm-adjusted p=0.0334, while B2-B0 is +7.0 pp with CI [-1.5, 15.5] pp.

The mechanism answers are:

- Q1 Contact-State alone: **NO**.
- Q2 VAC auxiliary beyond Contact-State: **YES**.
- Q3 VA-only physical supervision: **TREND**.
- Q4 B2 versus BVA: **NO**.
- Q5 generic auxiliary versus Contact-aware benefit: **INCONCLUSIVE**.
- Q6 BUniT required for the core claim: **NO**.

## BUniT disposition and PI2B

BUniT disposition is `BUNIT_APPENDIX_FAILED_TRANSFER_ONLY`. It is omitted from the main policy ablation; the adapter-only representation failure and its limitations belong in the appendix. Any retry is future work under a new preregistration.

PI2B readiness is `PI2B_MECHANISM_DIAGNOSIS_FIRST`. Because BVA’s formal point estimate exceeds B2, spending six or eight new runs before resolving the intended mechanism claim would violate the frozen decision rule. Immediate new training runs: zero.

If the claim is resolved and a four-model confirmation is separately authorized, retain B0/BVA/B1/B2, add deterministic training seeds 43 and 44, run eight new final-checkpoint-only 30k jobs, and evaluate 200 paired seed7 resets per checkpoint. Treat training seed hierarchically; do not pool all rollouts as independent. The full non-executing draft is in `configs/simulation/s4_3_pi2b_recommendation.json` and `docs/research/s4_3_pi2b_preregistration_draft.md`.

## Final decisions

- PI2V diagnosis: `PI2V_FAIL_VISION_DOMAIN_TRANSFER`
- BUniT disposition: `BUNIT_APPENDIX_FAILED_TRANSFER_ONLY`
- PI2B readiness: `PI2B_MECHANISM_DIAGNOSIS_FIRST`

PI2V remains historically failed and closed. PI2W is complete after final integrity checks. PI2B is not started. M4 is not established.
