# S4.3-PI2A Frozen-Checkpoint Statistical Confirmation

Status: frozen before scientific evaluation.

PI2A is a no-training, no-retuning, fresh-reset paired closed-loop
confirmation of the frozen PI1D checkpoints. It does not attempt to improve or
select a policy. The historical PI1D decision remains
`S4_3_PI1D_NO_MATERIAL_GAIN`.

## Frozen question

The primary question is whether B2, the complete Tactile-UniT variant,
outperforms the train-matched B0 local official pi0.5 baseline on a new set of
200 `pinch_tongs` / `rand_obj` reset conditions. B2-B1 is the key secondary
comparison of shared-physical training supervision beyond Contact-State
conditioning, and B1-B0 is secondary.

The evaluated models are byte-identical to PI1D:

- B0: local official pi0.5, 30k, seed 42.
- B1: B0 plus the frozen eight-token Contact-State adapter, 30k, seed 42.
- B2: the B1 online observation path plus the frozen physical auxiliary used
  only during training, 30k, seed 42.

B2 receives no physical target, future Contact, future vision, demonstration
contact, or expert action during evaluation. B1 and B2 receive exactly the
same inference information.

## Fresh paired protocol

Seed 0 was exposed by PI0 and seed 1 by PI1D. Seed 2 is the smallest
nonnegative evaluator seed not previously used for pi0.5 policy-performance
evaluation. Other historical ACT/PD/S4.2 seed namespaces are not pi0.5 policy
performance exposure.

Each model receives the same ordered 200 reset conditions. There is no reset
replacement, filtering, dynamics randomization, `rand_full`, or model-specific
reset. Evaluation uses the frozen official prompt, 0.8 replan ratio, official
success definition, and the PI1D augmented runtime. B0 computes tactile only
for parity and logging and does not send Contact-State to the policy.

All 600 canonical outcomes must exist before effect classification. Scientific
timeouts remain benchmark failures. An infrastructure failure may be retried
only for the identical model/reset/code/checkpoint/config and must retain its
retry record.

## Frozen statistics

Per-model reporting includes successes out of 200, rate, Wilson 95% interval,
episode lengths, and termination reasons. Each paired contrast reports the
success-rate difference, a 100,000-resample paired percentile bootstrap 95%
interval with seed 4302, discordant table, exact two-sided McNemar p-value,
discordant odds ratio where defined, and diagnostic risk ratio. Holm-adjusted
p-values across B2-B0, B2-B1, and B1-B0 are robustness diagnostics.

An improvement is statistically confirmed only when its point delta is
positive, the paired interval lower bound is positive, and exact McNemar
two-sided p is below 0.05. It is material only when the delta is also at least
10 percentage points. A positive 0--10 point effect meeting all statistical
conditions is a statistically confirmed submaterial gain. The fresh 200-reset
set alone controls the PI2A decision; the independent pooled 250-reset result
is diagnostic only.

## Freeze rules

After the preregistration commit, scientific code, checkpoints, thresholds,
seed, prompt, reset generator, tactile contract, normalization, evaluator
semantics, and inference schemas are immutable for PI2A. Any required semantic
change after the first scientific rollout is
`S4_3_PI2A_RUNTIME_FREEZE_FAIL`, not a patch-and-continue opportunity.

PI2A ends after its final decision and PI2B-readiness classification. It does
not start new training seeds, raw-tactile work, a world model, real-robot work,
or any push/release action.
