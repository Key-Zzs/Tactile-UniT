# S4.2-3 Rank-Audited Simulated Contact-Dynamics Result

## Decision

S4.2-3 stops with `S4_2_3_CONTACT_DYNAMICS_FAIL`.

The stage ran only after S4.2-R accepted the existing Contact-State checkpoint.
It used exact current histories ending at `t` and future histories ending at
`t+27`: current samples `[t-25, ..., t]`, future samples
`[t+2, ..., t+27]`, zero raw overlap, and an anchor delta of `0.54 s`.
Train/validation caches contain 22,680/4,860 aligned pairs; no test values were
loaded.

## Candidate and baselines

The frozen candidates were C0 persistence, C1 current-only, C2 delta MLP, and
C3 proposed dynamics encoder/decoder with delta weights `0.5` and `1.0`. All
models satisfy the 2M parameter budget. Validation selected `C3-lambda1.0`
(1,251,520 parameters), checkpoint SHA-256
`f67b8b0d6f519944adafecbb0a93bb4387213fa1a4f33f7f770a55938de05602`.

On 1,492 dynamic validation pairs:

- C0: `1.935077` MSE;
- C1: `0.392784` MSE;
- C2: `0.0181563` MSE;
- selected C3: `0.0171561` MSE.

C3 therefore improves over the strongest baseline C2 by `5.509%`, below the
frozen hard minimum `10%`. The paired absolute-error improvement CI is positive
(`[0.0008774, 0.0011298]`), so the small improvement is real but insufficient.

## Passed evidence

Every other gate passes. Full-code dynamic MSE beats zero (`0.849114`),
different-episode (`0.403402`), reversed (`2.147885`), and mismatched-future
(`0.403903`) controls, each with a positive bootstrap lower bound. Reload is
deterministic, output shape is `[8,32]`, values are finite, effective rank is
`31.0402`, near-zero variance and collapsed-query fractions are zero, and no
sample collapses across queries.

The transition code remains semantically strong: contact-transition macro-F1 is
`0.9784`, force-trend macro-F1 is `0.9504`, and the four active regions have
strong contact-change probes. The thumb change probe is correctly unusable
because its train label is a single class.

## Hard stop

The one failed gate is sufficient to fail the stage. S4.2-4 Action/Vision/paired
VAC and S4.2-5 Bridge are not run. The locked test remains closed. No dynamics
threshold change, extra candidate, or unregistered remediation is performed.

A minimal future remedy is a separately preregistered Contact-Dynamics study
that strengthens the proposed transition inductive bias or objective against the
matched C2 delta-autoencoder ceiling while retaining this recorded failure.
