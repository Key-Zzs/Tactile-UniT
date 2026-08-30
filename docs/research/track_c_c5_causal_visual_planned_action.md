# Full Track C — C5 Causal Visual Substitution and Planned-Action Interface

C5 freezes every accepted C1--C4 representation and predictor, introduces a
typed policy-planned Action interface, and evaluates whether causal Vision can
replace the unavailable Contact context `H`. The locked result is
`C5_CAUSAL_SYSTEM_READY_A_ONLY_FALLBACK`: the full `F_AH` path is retained, but
missing-`H` runtime traffic must use the frozen C4 Action-only fallback. The C5
causal visual fallback remains diagnostic and is not runtime-approved.

## Frozen contracts and integrity

`PlannedActionChunk` contains exactly 16 steps, covering `a_t` through
`a_t+15`, with shapes `[B,16,58]` before and `[B,16,128]` after the frozen A-R
transform. Its embodiment ID is 31. The raw ordering is left arm 7, left hand
22, right arm 7, and right hand 22. Normalization reuses the accepted
train-only statistics; state-relative features and first differences are
recomputed by frozen A-R; the representation is continuous pre-RQ and does not
use RQ. There is no `a_t+16`.

Sources are mandatory and typed as `POLICY_GENERATED`,
`DEMONSTRATION_TEACHER`, or `ORACLE_EVAL`. Only `POLICY_GENERATED` is legal at
runtime. Demonstration and oracle plans are restricted to explicit offline
evaluation, and source tags do not change numeric encoding: the maximum
cross-source difference was exactly 0. The accepted A-R reproduction maximum
was 0.0000107363 against the already accepted tolerance 0.0005.

No identity-locked policy-plan artifact was found in the accepted tracked
configuration or research record. Consequently, real policy-generated plans
were unavailable and their domain was not validated. This is recorded as
`POLICY_PLAN_DOMAIN_WARNING`; no policy was trained or fabricated.

The before/after identity audit passed for C1, C2, C2-R, C3-DP, the accepted
full predictor, C4 offline VA and emergency A predictors, C4 uncertainty, A-R,
S1, S2, the shared space, and the Original UniT tokenizer/DINO files. The
locked evaluation also reproduced the shared-state digest exactly.

## Causal visual boundary and cache

The legal visual supports were current frame `I_t` and causal history
`I_t-7:t`; future frames were forbidden. Each frame was processed by the
frozen Original UniT DINO patch features, frozen M-Former input projection,
frozen `vq_down_resampler`, and fixed 2x4 spatial pooling, producing `[8,32]`
tokens. Train-only normalization was fit on 4,130,576 tokens from 516,322
deduplicated training frames. The cache contains:

| Split | Rows | Unique frames | Status |
| --- | ---: | ---: | --- |
| train | 65,536 | 516,322 | complete before selection |
| validation | 8,192 | 65,236 | complete before selection |
| test | 17,504 | 138,430 | built only after the five pretest hashes froze |

All array hashes, shapes, and finite-value checks passed. Current indices were
exactly `t`; histories were exactly `t-7:t`, remained within the episode, and
never exceeded `t`. Packed-video timestamp alignment and repeated frozen
feature extraction passed. Neither validation nor test contributed to visual
normalization.

## Bounded validation selection

Exactly six preregistered trials were run, and selection used validation only.
The test split remained unloaded. `R_contact` and `R_force` below are semantic
retention ratios.

| Trial | Visual support | Family | Distill | Physics/cov | Epoch | Utility | R_contact | R_force | MSE | Rank | All gates | Selected |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| T0 | current | direct | no | no | 12 | 0.377825 | 0.512204 | 0.682455 | 0.061647 | 5.884907 | no | no |
| T1 | history-8 | direct | no | no | 12 | 0.385600 | 0.528140 | 0.709574 | 0.062673 | 5.882293 | no | no |
| T2 | current | direct | yes | yes | 12 | 0.383633 | 0.510272 | 0.713707 | 0.061908 | 5.757941 | no | no |
| T3 | history-8 | direct | yes | yes | 12 | 0.383976 | 0.528319 | 0.688580 | 0.061504 | 5.911995 | no | no |
| T4 | current | modular | yes | no | 4 | 0.384313 | 0.506554 | 0.706884 | 0.060628 | 5.772166 | yes | yes |
| T5 | history-8 | modular | yes | no | 10 | 0.392670 | 0.527262 | 0.709469 | 0.059667 | 5.781500 | no | no |

T4 was the only candidate to pass every validation gate, so the frozen
simplicity preferences did not override it. It uses a bounded current-frame
visual encoder plus a modular Contact fallback, with 17,888 trainable
parameters and a frozen 17,600-parameter offline-VA predictor component. Its
checkpoint SHA-256 is
`5cf8c0a001eba174e47b18712f6624c961a383c33338a8b0a0ce35254b4f3e9c`.
The selection artifact SHA-256 is
`f16fea2a10a91a671d794d22205a08ccbd7b24664bf186e1cd02d593bf71c1ed`.

The best current candidate was T4 and the best history candidate was T5. T4
minus T5 shared MSE was 0.000961 with 95% bootstrap CI
[0.000783, 0.001156]. The best direct trial was T1 (utility 0.385600), while
the best modular trial was T5 (0.392670); T5 nevertheless failed the shared,
physics, and visual-context gates.

## Locked post-hoc engineering evaluation

The single completed locked evaluation used 17,504 rows on isolated physical
GPU1 with no fallback. The pretest contract hashes were validated before any
test array was opened. The first launch attempt was rejected by the GPU
isolation preflight before test access because a required device-order
environment variable was absent; the completed evaluation followed with the
same frozen code, data, and checkpoints. Repeated causal prediction was bitwise
exact.

The selected causal fallback achieved Contact macro-F1 0.401769,
`R_contact=0.515171`, force macro-F1 0.465360, `R_force=0.685720`, future-change
macro-F1 0.492911, and shared MSE 0.061658. It passed Contact, force,
future-change, visual-gain-over-A, teacher-side dynamic physics, exact Action
temporal use, and noncollapse gates. It failed:

- shared prediction, because the wrong-time-past control improvement CI
  [-0.0000519, 0.0000834] included zero; and
- visual-context use for the same strongest invalid control.

The causal path improved shared MSE over A-only from 0.067945 to 0.061658; the
95% MSE-gain CI was [0.006078, 0.006517]. Its Contact macro-F1 gain CI over
A-only was [0.014828, 0.024937], while force gain was inconclusive at
[-0.000122, 0.012145]. The offline future-Vision oracle upper bound achieved
Contact retention 0.507362, force retention 0.696988, and shared MSE 0.062637;
it remains excluded from the runtime router.

The causal physics MSE was 0.011703 overall and 0.037267 on dynamic rows. The
dynamic improvement CI over the strongest non-oracle control was
[0.0000145, 0.0000510], with teacher-side `H` isolated from the fallback input.
Exact raw planned-Action perturbations increased dynamic shared error by
0.002044 for reversal (CI [0.001729, 0.002362]), 0.001827 for shuffle (CI
[0.001487, 0.002177]), and 0.050554 for a different plan (CI
[0.048829, 0.052245]). The Action temporal gate passed.

The frozen full path passed non-regression: Contact retention 0.900316, force
retention 0.794947, shared MSE 0.046534, dynamic physics MSE 0.036072, effective
rank 8.307557, checkpoint identity, `H`-use, and Action temporal-use gates all
passed.

## Planned-Action domain diagnostic

The oracle demonstration surrogate had shared MSE 0.060628 and mean calibrated
uncertainty 0.058805. Mild raw-58 noise increased representation RMS by
0.038066, MSE by 0.000404, and uncertainty by 0.000440. Strong noise increased
representation RMS by 0.113067, MSE by 0.004122, and uncertainty by 0.003654.
A different-episode plan increased representation RMS by 0.321521, MSE by
0.044631, and uncertainty by 0.011432. Oracle, mild, and strong uncertainty
were monotonic. Temporal smoothing and one-step lag were benign diagnostics,
not policy-domain calibration. Actual policy-domain validation remains not
available.

## Runtime router

Availability routing is an exhaustive deterministic truth table, not a neural
missingness classifier:

| Action | H | causal Vision | Mode |
| --- | --- | --- | --- |
| available | available | either | `FULL_AH` |
| available | missing | available | `FALLBACK_CAUSAL_VA` capability |
| available | missing | missing | `FALLBACK_A` |
| missing | either | either | `ABSTAIN_NO_ACTION` |

The router audit passed, rejects demonstration Action at runtime, excludes
offline oracle VA from runtime, and abstains without Action. Because the
locked causal visual gates did not all pass, the approved missing-`H` mode in
the final decision is `FALLBACK_A`, not `FALLBACK_CAUSAL_VA`.

## Availability-aware uncertainty

The frozen `C5ContactUncertaintyEstimator` has 19,161 trainable parameters and
mode-specific inputs for `FULL_AH`, `FALLBACK_CAUSAL_VA`, and `FALLBACK_A`.
Every mean predictor remained frozen. Epoch 16 was selected on validation;
one common calibration scale (0.967985), one validation-defined high-error
threshold (0.080109), and a constant-variance baseline (0.057877) were frozen.
The checkpoint SHA-256 is
`5ea1ff866fbc60cacb5735dd6655b7e8343be11f46b3d06ad53b5900fa6f812f`.

| Mode | Spearman (95% CI) | AUROC | AUPRC | NLL | Constant NLL | Top-20% removal |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| full | 0.856725 ([0.852515, 0.861060]) | 0.904739 | 0.765492 | -1.269933 | -1.022714 | 0.370947 |
| causal | 0.785650 ([0.779231, 0.792070]) | 0.836321 | 0.632003 | -1.036125 | -0.892049 | 0.231545 |
| A-only | 0.733530 ([0.725268, 0.741092]) | 0.800168 | 0.592540 | -0.928244 | -0.837741 | 0.180663 |

All informative uncertainty gates passed for both fallback modes. Mean
uncertainty was 0.043113 for full, 0.058761 for causal, and 0.071912 for
A-only. Causal-minus-full uncertainty had 95% CI [0.015262, 0.016055]. Dynamic
mean uncertainty exceeded static mean uncertainty in every mode. For causal,
the dynamic/static values were 0.087996/0.049067; free-to-contact and
contact-to-free values were 0.088689 and 0.079762.

## Leakage, geometry, tests, and decision

The causal audit found no future Vision, future tactile, true `u_v`, true
`u_c`/`z_c`, demonstration Action, private residual, or pair-ID runtime input.
Nested oracle guards passed. Frozen identities and state digests passed before
and after evaluation.

Effective ranks were 25.503495 for oracle `u_c`, 8.307557 for full, 5.763700
for the offline fallback, 5.772700 for the causal fallback, and 5.273499 for
A-only. The causal representation had minimum per-dimension variance 0.001956,
zero near-zero dimensions, query collapsed-pair fraction 0, mean query cosine
distance 0.320676, and CKA with oracle 0.526381. It did not collapse, but the
material oracle rank gap retains `RANK_WARNING`.

The complete regression suite finished with 403 passed, 0 failed, and 0
skipped. This covers C5, C4, C3-MS-CC-R, C3-MS-CC, C3-R0, C3-DP, C2-R, C2,
C1, integration, A-R, C0, S1/S2, and Original UniT guards. Twenty-one
decision-focused plots and `HUMAN_ACCEPTANCE.md` were generated under the
ignored C5 artifact directory.

Final decision: `C5_CAUSAL_SYSTEM_READY_A_ONLY_FALLBACK` with
`POLICY_PLAN_DOMAIN_WARNING`, `CAUSAL_VISUAL_SUBSTITUTION_WARNING`, and
`RANK_WARNING`. C6 readiness is `READY_WITH_WARNING`, but C6 was not started.
M3 is not established. This result does not claim policy deployment readiness
or publication-level confirmation.

STOP AFTER C5. Do not start C6.
