# S4.2-R Contact-State Rank Remediation Protocol

## Separate follow-up and preserved failure

This protocol is a separately preregistered follow-up to the original S4.2
Contact-State experiment. The original result remains a failure:

- decision: `S4_2_CONTACT_STATE_FAIL`;
- selected checkpoint: `B3-N1-R0.25`;
- effective rank: `11.255626`;
- preregistered minimum: `16.0`.

Nothing in S4.2-R changes that threshold, decision, commit, or evidence. In
particular, S4.2-R must not describe the original experiment as passing. Its
question is narrower: did the original absolute rank gate identify structural
collapse, or did it penalize a low-dimensional simulated tactile source?

## Evidence available before this freeze

R1 and R2 ran before this protocol freeze and without training. They used the
formal train and validation splits only. R1 found a TRAIN-only source reference
`d_src = 3.121517446651421`, with canonical source-term SHA-256
`62a33f61000606431bd3b34e6cb7dd3b6a903d3851ad56c0e152f6b6a47ddb7b`.
R2 resolved every thumb body/geom, found no raw object–thumb contact in exact
three-task deterministic replays, retained four meaningfully active regions,
and classified the zero thumb channel as data coverage with warning
`THUMB_ZERO_ACTIVITY`.

The frozen teacher checkpoint SHA-256 is
`3d4519195e7a0d9f5af63399840a48121d5f614ea587c80b9461fc696a372a19`.
The formal dataset manifest canonical SHA-256 is
`652181d45bc317a1b17debe5e9c58529ae7d096bb75441cd17d0165bf2d701d4`.

## New structural-collapse contract

All gates are evaluated on validation. Source-relative quantities use the
already frozen TRAIN-only source reference.

1. Variance: near-zero variance fraction must be at most `1%`. A dimension is
   near-zero when its variance is strictly below `1e-4` times the median
   positive variance.
2. Dominant component: top-1 PCA explained variance must be strictly below
   `50%`.
3. Effective-rank floor: effective rank must be at least `8.0`. This is a
   conservative minimum diverse-factor count, not a capacity-utilization target
   and not a retroactive change to the original threshold.
4. Source-relative diversity: `effective_rank(h_val) / min(d_src, 16)` must be
   at least `0.75`.
5. Pairwise diversity: exact duplicate distance must be at most `1e-8`, mean
   different-sample distance must be strictly larger, and its fifth percentile
   must be strictly positive.
6. Semantic functionality: every original predictive, dynamic, temporal,
   contact, force, trend, finite, and deterministic-reload validation gate must
   remain passed.
7. Perturbation sensitivity: on dynamic-q70 validation, the correct history
   must beat shuffled, reversed, and repeated-last-frame histories. Each paired
   error-difference bootstrap lower 95% confidence bound must be positive, using
   5,000 resamples and the frozen seed base in the JSON contract.

Rank below `16`, an unused thumb region, and low static/free-window rank are
warnings rather than individual hard failures. Rank below `16` emits
`LOW_EFFECTIVE_RANK_WARNING`.

## Existing checkpoint first

R4 performs no training. It evaluates the exact existing checkpoint on
validation under the new contract. If every hard gate passes, the classification
is `LOW_INTRINSIC_DIMENSION_NOT_COLLAPSE` and the state becomes
`S4_2R_CONTACT_STATE_ACCEPTED_EXISTING`. No replacement model is trained.

## Conditional bounded remediation

R5 is entered only if R4 fails. It permits at most three new trials, all using
the existing B3 architecture and frozen N1 normalization:

- `R0`: original objective with new seed `5242`;
- `R1`: weak VICReg-style preservation with `lambda_variance = 1e-3` and
  `lambda_covariance = 1e-4`;
- `R2`: weak temporal InfoNCE with weight `1e-3` and temperature `0.1`.

No architecture sweep or validation lambda sweep is allowed. A single
deterministic rescaling is allowed only if a TRAIN-only magnitude analysis is
written as a protocol addendum before the first new trial. The model parameter
count may increase by at most 5%.

Every new structural and original functional gate must pass. Selection priority
is existing checkpoint, `R0`, `R1`, then `R2`. Among passing models within 3% of
the best validation future MSE, rank health, parameter count, and intervention
size break ties; rank alone cannot select a model.

## Test and downstream boundary

The locked test remains closed for model performance. Training and selection
use train/validation only. S4.2-3 is permitted only after R6 accepts a checkpoint.
All later stages retain the hard dependency order. This goal stops after S4.2-5
bridge validation and does not begin S4.2-6, S4.2-7, the locked final test,
S4.3, or policy training.
