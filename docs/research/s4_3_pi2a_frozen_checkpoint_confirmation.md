# S4.3-PI2A Frozen-Checkpoint Statistical Confirmation

Status: complete.

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

## Integrity and completion

All frozen checkpoint, runtime-parity, fresh-seed, reset-identity, environment,
and S4.2 immutability gates passed. The evaluator completed all 600 canonical
rollouts: 200 per model and 200/200 triple-aligned reset pairs. There were no
missing or duplicate tuples, infrastructure retries, reset replacements,
unresolved policy-server errors, or Contact-State sidecar errors. No training,
retuning, checkpoint switch, package mutation, or push occurred.

The frozen checkpoint tree hashes were:

- B0: `1e7a6ace5d69a988a8b258e1c56a24d88b077580a05b27be3f510df9ac3864f3`.
- B1: `04b59cbc3491bf4e88dc75558a08a94e5588e2fbe398d413e1766a87c7ab6f4f`.
- B2: `86c4908533ca24da06ae66f943e3605b5ccbae455441290ca8fb35514359e949`.

## Fresh 200-reset results

| Model | Successes | Rate | Wilson 95% CI | Mean / median steps |
|---|---:|---:|---:|---:|
| B0 | 37/200 | 18.5% | [13.73%, 24.46%] | 941.325 / 1000 |
| B1 | 30/200 | 15.0% | [10.71%, 20.61%] | 939.645 / 1000 |
| B2 | 51/200 | 25.5% | [19.96%, 31.96%] | 914.710 / 1000 |

Every non-success terminated at the frozen 1000-step limit. The paired results
were:

| Contrast | Delta | Paired bootstrap 95% CI | Exact McNemar p | Holm p | Classification |
|---|---:|---:|---:|---:|---|
| B1-B0 | -3.5 pp | [-11.0, 4.0] pp | 0.442626 | 0.442626 | `NEGATIVE_TREND_NOT_CONFIRMED` |
| B2-B1 | +10.5 pp | [3.0, 18.0] pp | 0.011141 | 0.033424 | `MATERIAL_IMPROVEMENT` |
| B2-B0 (primary) | +7.0 pp | [-1.5, 15.5] pp | 0.130178 | 0.260355 | `POSITIVE_TREND_NOT_CONFIRMED` |

The B2-B1 discordant counts were 42 B2-only successes versus 21 B1-only
successes (discordant odds ratio 2.0; relative success ratio 1.70). Thus the
pre-registered key-secondary comparison confirms a material benefit from the
training-only shared-physical auxiliary beyond Contact-State conditioning.
The primary B2-B0 comparison remains positive but does not meet the frozen
statistical-confirmation rule because its interval includes zero and its exact
p-value exceeds 0.05.

## Replication and diagnostics

The core B2 pattern is stable: B2-B1 remained near +10 pp (+10.5 pp), and
B2-B0 remained positive and within 5 pp of the PI1D estimate (+7.0 pp versus
+10 pp). The complete three-contrast pattern is not replicated because B1-B0
reversed from zero to -3.5 pp. This descriptive result does not change the
pre-registered decision.

The pooled, diagnostic-only 250-reset rates were B0 17.6%, B1 14.8%, and B2
25.2%. Pooled B2-B0 was +7.6 pp and remained unconfirmed; pooled B2-B1 was
+10.4 pp and confirmed. These pooled results were not used to determine PI2A.

Available contact telemetry showed mean peak normal force of 24.49, 25.44,
and 25.79 for B0, B1, and B2, respectively, and mean maximum pinch counts of
1.705, 1.605, and 1.825. Successful episodes reached pinch count 3 in every
model. Integrated normal force, contact onset, lift height, and explicit cycle
counts were unavailable in the frozen telemetry and were not inferred.

## Decision

Final PI2A decision:
`S4_3_PI2A_PHYSICAL_AUX_GAIN_CONFIRMED`.

The primary B2-B0 endpoint is not confirmed, but B2-B1 is a statistically
confirmed material improvement under the frozen rule. Accordingly, PI2B is
`PI2B_MULTI_SEED_RECOMMENDED`: a future, separately preregistered study may
evaluate two additional training seeds across B0/B1/B2 (six 30k runs). PI2A
does not start those runs.

Post-evaluation regression testing passed with 399 tests passed, one skipped,
and zero failures. PI2A stops here.
