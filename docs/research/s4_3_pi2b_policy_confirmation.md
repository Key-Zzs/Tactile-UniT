# S4.3-PI2B Policy Multi-Seed Confirmation

Status: **COMPLETE_VALID — THREE_SEEDS_LIMITED**

This Track A study completed the frozen five-recipe, three-training-seed
confirmation on `pinch_tongs`. It added exactly ten independently initialized
30,000-step runs (seeds 43 and 44), reused the five audited seed42 checkpoints,
and evaluated all fifteen checkpoints on the same ordered 200-reset cohort.
All 3,000 canonical outcomes passed the completeness and runtime-contract gates.

The scientific result is mixed, not a confirmation of a universal positive
effect. `B_HVA-B_VA27`, `B2-B_HVA`, and `B2-B1` all change direction across
training seeds. The engineering completion status is independent of that
outcome.

## Frozen scope and identities

- Git base: `8f39aed123adf0a8b7241e75472e678355444e5e`
- Branch: `develop/pi2b-policy`
- Task/regime: `pinch_tongs` / official frozen `rand_obj`
- Training: seeds 43 and 44, five recipes each, 30,000 optimizer steps,
  global batch 32, one GPU per run; no teacher training
- Evaluation: seeds 42/43/44 × five recipes × 200 shared resets = 3,000
- Reset blocks: 16/17/18/19, 50 resets each
- Ordered reset-sequence SHA256:
  `983160e58a9718aea825999f09a78b449bdbb3151ecc9e2168b6da7585085713`
- Track B new teacher read or substituted: **no**
- Main `PAPER_CORE.md`, the other worktree, push, PR, merge, tag: **not modified/performed**

The five exact recipes are:

| ID | Online H | Training-only auxiliary target | Exact lambda | Added recipe parameters |
|---|---|---|---:|---:|
| B0 | no | none | 0 | 0 |
| B_VA27 | no | clean `u_v^VA27`, +27 ticks / 0.54 s | 0.03077957631925596 | 658,176 |
| B1 | yes | none | 0 | 199,680 |
| B_HVA | yes | clean `u_v^VA27`, +27 ticks / 0.54 s | 0.026468189597253295 | 857,856 |
| B2 | yes | frozen `u_c^VAC27`, +27 ticks / 0.54 s | 0.020927851827513076 | 857,856 |

The auxiliary targets remain training-only. All H-enabled modes use the same
frozen E_T `[26,30] -> [256]` causal history with `LEFT_REPEAT_FIRST`; B0 and
B_VA27 compute Contact diagnostics but do not send H to the policy. The data
contract retained 40,065 base rows, 37,365 valid +27 targets and 2,700
episode-tail masks. Historical BVA (`+16` ticks / 0.32 s) is not in this cohort.

## Training completion

Every new run cold-restored at train state 30,000 with finite arrays. The wall
time below is launch-manifest creation to terminal status update; because each
run used one GPU it is also the measured GPU-hour estimate. Total was 396.62
GPU-hours.

| Model | Seed | GPU | Hours | Checkpoint tree SHA256 |
|---|---:|---:|---:|---|
| B0 | 43 | 2 | 39.49 | `33888318e7dcb56ef8262784589ae9caf29dbd3b7568e722258e89bb13ec4a87` |
| B_VA27 | 43 | 3 | 39.57 | `347f0f40a660ee8a1627374fcaeeab2edc1fecd4ff3840ae293c48fdfbfd53c8` |
| B1 | 43 | 1 | 39.81 | `cb15ab1d2b9f3dd738359bd1b65d81cfc562f03b3bc752dd15cdfef0f7e7fdae` |
| B_HVA | 43 | 2 | 39.82 | `528ba1888d094dc903f32818d6eea5d57c847a2c73d90057fb1ce679f8814d2a` |
| B2 | 43 | 3 | 39.90 | `b3a143e3262f47c61d0dfa14dacda62ad6ba1f7574239d488d886db9f56a50a8` |
| B0 | 44 | 1 | 39.78 | `2355c820e8e0a003ee744426f4b375e40b492d808ab4239186b06d63552591f3` |
| B_VA27 | 44 | 2 | 39.81 | `86adfaf983c4fcd6a39f2fb4413c246f94baa161f7e8cfe24754a1209baecb98` |
| B1 | 44 | 3 | 39.93 | `63b9038261ea2e5d3559c7ffaa0d59c117dcbe3914f16890ad23703b075a1bbc` |
| B_HVA | 44 | 1 | 39.21 | `b342e4c0a4303f338906ef0030e3d86730730ed443698d0aa3392ecaaf3f594c` |
| B2 | 44 | 2 | 39.30 | `9f46aae6eed9f1523168fc83fe1236f2b0413e5fc38b488c12b84eed3d4ce203` |

The five seed42 tree hashes were live-verified and reused unchanged:

| B0 | B_VA27 | B1 | B_HVA | B2 |
|---|---|---|---|---|
| `1e7a6ace…3864f3` | `14e6ea9b…c52cc` | `04b59cbc…ab6f4f` | `82469f1d…e3a8b` | `86c49085…9e949` |

## Formal evaluation integrity

The official DexJoCo/OpenPI asynchronous runtime, action chunks, replan ratio
0.8, sampling contract, success/timeout rules and deterministic environment
settings were held fixed. The Prompt1 exclusive evaluation barrier covered the
whole wave, and workers used the shared GPU locks. Completeness gates verify:

- 60/60 model × training-seed × reset-block workers passed;
- 3,000/3,000 unique canonical tuples exist;
- all fifteen checkpoints share the exact ordered 200 reset identities;
- native failures are retained, with no replacement resets;
- no unresolved server/client error or training-only runtime input exists.

The first three completed blocks exposed a filesystem-only publication bug:
POSIX rename from `/tmp` to NAS returned `EXDEV` after evaluator output was
complete. The versioned amendment changed only publication to copy + hash/size
verify + fsync + same-filesystem atomic rename. It was performance-blind, kept
the first complete attempt canonical, and did not change policy, reset,
sampling, success, or timeout semantics.

## Checkpoint results

Wilson intervals are per checkpoint over its 200 resets. All 2,197 failures
ended natively at `max_steps`; there were zero timeouts and zero unresolved
runtime errors.

| Model | Seed | Success | Rate | Wilson 95% CI | Fail | Timeout | Mean steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 42 | 47/200 | 23.5% | [18.16, 29.84]% | 153 | 0 | 920.720 |
| B0 | 43 | 82/200 | 41.0% | [34.42, 47.92]% | 118 | 0 | 846.440 |
| B0 | 44 | 70/200 | 35.0% | [28.73, 41.84]% | 130 | 0 | 865.800 |
| B_VA27 | 42 | 56/200 | 28.0% | [22.24, 34.59]% | 144 | 0 | 921.205 |
| B_VA27 | 43 | 97/200 | 48.5% | [41.67, 55.39]% | 103 | 0 | 811.425 |
| B_VA27 | 44 | 86/200 | 43.0% | [36.33, 49.93]% | 114 | 0 | 835.255 |
| B1 | 42 | 22/200 | 11.0% | [7.38, 16.09]% | 178 | 0 | 957.010 |
| B1 | 43 | 41/200 | 20.5% | [15.49, 26.63]% | 159 | 0 | 922.960 |
| B1 | 44 | 42/200 | 21.0% | [15.93, 27.16]% | 158 | 0 | 934.105 |
| B_HVA | 42 | 82/200 | 41.0% | [34.42, 47.92]% | 118 | 0 | 862.965 |
| B_HVA | 43 | 1/200 | 0.5% | [0.09, 2.78]% | 199 | 0 | 999.510 |
| B_HVA | 44 | 50/200 | 25.0% | [19.51, 31.43]% | 150 | 0 | 908.915 |
| B2 | 42 | 45/200 | 22.5% | [17.26, 28.77]% | 155 | 0 | 937.475 |
| B2 | 43 | 14/200 | 7.0% | [4.22, 11.41]% | 186 | 0 | 981.665 |
| B2 | 44 | 68/200 | 34.0% | [27.79, 40.81]% | 132 | 0 | 882.345 |

Seed42 was previously selected/development-exposed. Seeds43/44 are the
preregistered independent training additions. Their absolute rates should not
be conflated with the older PI2N cohort, which used different formal resets.

## Paired effects within each training seed

Each cell is `risk difference pp; exact two-sided McNemar raw p / Holm p`.
Holm correction is applied separately within each declared family and each
training seed. The 600 seed-reset rows are never pooled into one McNemar test.

| Family / contrast | Seed42 | Seed43 | Seed44 |
|---|---:|---:|---:|
| primary: B_HVA−B_VA27 | +13.0; .008781 / .008781 | −48.0; 2.52e−29 / 5.05e−29 | −18.0; .000534 / .001069 |
| primary: B2−B_HVA | −18.5; .000132 / .000264 | +6.5; .000977 / .000977 | +9.0; .053544 / .053544 |
| key: B2−B1 | +11.5; .001403 / .002805 | −13.5; .000142 / .000142 | +13.0; .003836 / .011509 |
| key: B_HVA−B0 | +17.5; .000224 / .000671 | −40.5; 8.27e−25 / 2.48e−24 | −10.0; .030786 / .061573 |
| key: B2−B0 | −1.0; .899076 / .899076 | −34.0; 5.02e−15 / 1.00e−14 | −1.0; .920411 / .920411 |
| auxiliary: B_VA27−B0 | +4.5; .355699 / .355699 | +7.5; .137368 / .137368 | +8.0; .121371 / .121371 |
| auxiliary: B1−B0 | −12.5; .000802 / .001605 | −20.5; 1.27e−5 / 2.53e−5 | −14.0; .003747 / .007493 |

The authoritative JSON also records every discordant table and every 100,000
replicate paired tuple-bootstrap interval.

## Crossed training-seed × reset analysis

The estimand is the arithmetic mean of the three within-seed paired risk
differences. `Conditional CI` resamples the shared reset index while
conditioning on the fifteen trained checkpoints. `Two-way CI` additionally
resamples the three shared training-seed indices and is only a weak
finite-sample sensitivity analysis.

| Contrast | Seed42 / 43 / 44 (pp) | Mean ± seed SD (pp) | Conditional 95% CI | Two-way 95% CI | Exact seed sign-flip p |
|---|---|---:|---:|---:|---:|
| B_HVA−B_VA27 | +13.0 / −48.0 / −18.0 | −17.67 ± 30.50 | [−23.00, −12.33] | [−46.50, +11.00] | .50 |
| B2−B_HVA | −18.5 / +6.5 / +9.0 | −1.00 ± 15.21 | [−5.17, +3.33] | [−17.00, +11.83] | 1.00 |
| B2−B1 | +11.5 / −13.5 / +13.0 | +3.67 ± 14.89 | [−0.67, +8.17] | [−12.00, +16.67] | 1.00 |
| B_HVA−B0 | +17.5 / −40.5 / −10.0 | −11.00 ± 29.01 | [−15.83, −6.33] | [−39.00, +15.50] | .75 |
| B2−B0 | −1.0 / −34.0 / −1.0 | −12.00 ± 19.05 | [−17.00, −7.00] | [−32.00, +4.00] | .25 |
| B_VA27−B0 | +4.5 / +7.5 / +8.0 | +6.67 ± 1.89 | [+1.33, +12.00] | [−0.33, +13.83] | .25 |
| B1−B0 | −12.5 / −20.5 / −14.0 | −15.67 ± 4.25 | [−20.33, −11.00] | [−23.33, −8.50] | .25 |

The predeclared interaction diagnostic
`(B_HVA−B_VA27)−(B1−B0)` is +25.5/−27.5/−4.0 pp, mean −2.0 pp,
conditional CI [−9.0,+5.0] pp and two-way CI [−26.83,+24.33] pp. It is not a
pure causal synergy estimate.

The seed43/44-only sensitivity gives:

| Contrast | Mean (pp) | Conditional 95% CI | Two-way 95% CI |
|---|---:|---:|---:|
| B_HVA−B_VA27 | −33.00 | [−38.75, −27.25] | [−52.50, −11.50] |
| B2−B_HVA | +7.75 | [+2.75, +12.75] | [+2.50, +14.75] |
| B2−B1 | −0.25 | [−5.75, +5.25] | [−18.00, +18.50] |
| B_HVA−B0 | −25.25 | [−30.75, −19.75] | [−45.00, −4.50] |
| B2−B0 | −17.50 | [−23.75, −11.25] | [−39.00, +5.50] |
| B_VA27−B0 | +7.75 | [+0.75, +14.50] | [−0.50, +16.00] |
| B1−B0 | −17.25 | [−23.50, −11.00] | [−26.50, −8.00] |

With three training seeds the minimum exact two-sided seed-level sign-flip p
is 0.25; with two it is 0.5. Reset-level precision cannot be used to claim
population-level training-seed significance.

## Frozen decisions and interpretation

- `ENGINEERING_STATUS = COMPLETE_VALID`
- `H_GIVEN_VA27 = MIXED`
- `VAC_C_VS_VA_TARGET = MIXED`
- `VAC_C_VS_H_ONLY = MIXED`
- `TRAINING_SEED_SCOPE = THREE_SEEDS_LIMITED`

`B_HVA-B_VA27` reverses from +13 pp at exposed seed42 to −48/−18 pp at the
two new seeds. `B2-B_HVA` reverses from −18.5 pp to +6.5/+9 pp. These are real
canonical outcomes, not engineering failures, and no checkpoint or seed is
discarded. The result does not support a stable positive H increment under
VA27 or a stable ordering of Contact/VAC versus clean VA target recipes.

The consistent point directions for B_VA27−B0 (+4.5/+7.5/+8 pp) and B1−B0
(−12.5/−20.5/−14 pp) are descriptive under only three seeds. In particular,
the 10 pp material-effect reference is not an engineering gate, conditional
reset intervals do not represent unseen training runs, and absence of a
difference is not equivalence. No human–robot transfer, real-hardware,
Original-UniT superiority, or cross-task claim follows.

Contact/tactile process fields are available for every episode: all 21 hand
collision geometries were mapped, histories were finite `[26,30]`, Contact
states were finite `[256]`, and delivery matched each mode. These fields audit
the runtime mechanism; they do not isolate why seed43 B_HVA collapsed or
establish a causal representation mechanism.

## Integrity and authoritative artifacts

The independent standard-library audit does not import the Track-A statistics
module. It passed 3,319 checks covering raw hashes/cardinality, Wilson counts,
terminations/steps, paired differences, discordance, exact McNemar, per-family
Holm, crossed-seed arithmetic, exact sign flips, new-seed sensitivity,
interaction arithmetic and decision scope.

Key artifacts under `.local/artifacts/simulation/s4_3_pi2b_policy/`:

- `training_completion.json`
- `pre_final_freeze.json`
- `runtime_amendment_cross_device_publish.json`
- `rollout_completeness.json`
- `final_gpu_execution.json`
- `per_seed_statistics.json`
- `crossed_seed_reset_analysis.json`
- `new_seeds_only_sensitivity.json`
- `statistics_independent_audit.json`
- `final_decision.json`
- `plots_manifest.json` and `plots/`

The tracked result freeze is
`configs/simulation/pi2b_policy/final_decision.json`. The paper delta and
integration handoff are emitted separately; the main PAPER_CORE is left
untouched. No additional seed, task, teacher, real-robot run, push, PR or merge
is authorized by this completion.
