# Track C C3-MS-CC-R: exact Action temporal evidence and minimal-source closure

## Scope and protocol

C3-MS-CC-R is a post-C3-MS-CC closure stage. It does not begin C4, C5, C6/M3,
change the shared space, fine-tune a native encoder, or predict the Contact-private
residual. Model and epoch selection use train and validation only. The final 17,504-row
evaluation is explicitly a locked post-hoc closure re-evaluation, not a first-look test.

The accepted A-R checkpoint is not self-contained: its T-Rex overlay and R1-P transition
adapter reconstruct the frozen shared Action path from the canonical Original UniT
tokenizer. The canonical base, both shards, index, A-R checkpoint, train-only transition
statistics, T-Rex embodiment ID 31, and released Action rows 0–29 were all identity-checked.

## Exact Action construction

For each pair, the implementation reconstructs the raw `[16,58]` Action sequence before
creating correct, time-reversed, temporally shuffled, and deterministic same-split
different-episode sources. Each source is then normalized and padded afresh, so
raw-state-relative and first-difference features are recomputed by the accepted R1-P
transform. The full frozen A-R path produces continuous `z_a [8,32]`; frozen `P_a` then
produces exact `u_a [8,32]`. No RQ and no shared-token reversal surrogate is used.

On validation dynamic windows, exact raw reversal changed `u_a` by MSE 0.003300 and
shuffle changed it by 0.004452. A-R decoder error increased from 0.066748 for correct
Action to 0.068927 under reversal, 0.076018 under shuffle, and 0.377582 for a different
episode. All paired difference confidence intervals had lower bounds above zero.

## Reducer audit and bounded remediation

The original reducer implementation is valid. Frozen T0 A+H originally failed the
physics, Action-surrogate, and exact-A-R gates, so the reducer chose T1 V+A+H. With exact
evidence, frozen T0 passed both raw-reversed and raw-shuffled Action gates but retained
one genuine validation physics gap. This legitimately triggered one bounded A+H-only
R1 remediation initialized from T0. It retained the accepted architecture and added no
Vision input. Its fixed objective used exact temporal ranking and the existing shared
Contact physics terms. Epoch 12 was selected by validation utility with every validation
hard gate passing.

## Locked closure result

The validation selection artifact was written with `test_loaded: false` and hashed before
the locked benchmark was loaded. The resulting A+H predictor achieved:

- Contact macro-F1 0.567907 and semantic retention 0.900316;
- force macro-F1 0.511595 and semantic retention 0.794947;
- future-change macro-F1 0.591074;
- free-to-contact F1 0.258877 and contact-to-free F1 0.229861;
- shared-target MSE 0.046534;
- exact reversed dynamic MSE increase 0.001156, 95% CI [0.000925, 0.001406];
- exact shuffled dynamic MSE increase 0.000891, 95% CI [0.000655, 0.001141];
- all-window physics MSE 0.011384 versus strongest control 0.011811;
- dynamic physics improvement CI [0.001538, 0.002016];
- effective rank 8.307557 and CKA with oracle Contact 0.726869.

All semantic, H-context, exact Action temporal-use, shared-latent/retrieval, all-window
physics, dynamic physics, boundary-validity, non-collapse, and frozen-identity hard gates
pass. Effective rank remains strongly contracted relative to oracle Contact and is retained
as a warning.

## Decision

Final decision: `C3MSCCR_READY_AH_MINIMAL_WITH_RANK_WARNING`.

Classification: `A_PLUS_H_CANONICAL_MINIMAL_SOURCE`.

At the canonical 16-frame / 0.533333-second T-Rex Contact-transition horizon, given the
accepted Action transition representation and current Contact context, the Vision
transition teacher does not provide a statistically established semantic benefit sufficient
to justify inclusion in the canonical Contact predictor. Vision remains an optional
short-horizon context and ablation route; this result does not claim that Vision is generally
unnecessary for robotics.

The canonical representation-level contract is `u_hat_c = F(u_a, h_t^c)` targeting shared
`u_c`. Actual Contact retains `r_c^priv`; it is unchanged and is neither predicted nor used
as a source or target. This is not an online policy or deployment-readiness claim. C4 may be
recommended with a rank warning, but C4, C5, C6/M3 remain not started and M3 is not
established.
