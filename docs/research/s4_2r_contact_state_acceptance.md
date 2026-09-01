# S4.2-R Contact-State Acceptance

## Decision

The separately preregistered S4.2-R validation protocol accepts the existing
`B3-N1-R0.25` checkpoint without retraining:

`S4_2R_CONTACT_STATE_ACCEPTED_EXISTING`

The scientific classification is `LOW_INTRINSIC_DIMENSION_NOT_COLLAPSE`. This
does not change the original S4.2 result, which remains
`S4_2_CONTACT_STATE_FAIL` because effective rank `11.255626` was below the
original preregistered threshold `16.0`.

## Validation evidence

The exact checkpoint retains SHA-256
`3d4519195e7a0d9f5af63399840a48121d5f614ea587c80b9461fc696a372a19`.
R4 used validation only and performed no training. Its measured future MSE is
`0.155457109`; effective rank is `11.257566`; top-PC explained variance is
`0.359593`; near-zero variance fraction is `0`; and source-relative rank ratio is
`3.606440` against the frozen TRAIN-only `d_src = 3.121517`.

All preregistered gates pass:

- A, variance: `0.0 <= 0.01`;
- B, dominant component: `0.359593 < 0.5`;
- C, effective-rank floor: `11.257566 >= 8.0`;
- D, source-relative diversity: `3.606440 >= 0.75`;
- E, pairwise diversity: duplicate distance is zero and the different-sample
  fifth percentile is positive;
- F, all original predictive/semantic/temporal gates remain passed;
- G, shuffled, reversed, and repeated-last-frame controls are all worse on
  dynamic validation with positive paired-bootstrap lower confidence bounds.

The checkpoint is promoted byte-for-byte beneath the S4.2-R local experiment
root. No R5 trial ran. Warnings `LOW_EFFECTIVE_RANK_WARNING` and
`THUMB_ZERO_ACTIVITY` remain explicit.

The locked test was not loaded and did not participate in selection. S4.2-3 is
now allowed under the hard dependency order.
