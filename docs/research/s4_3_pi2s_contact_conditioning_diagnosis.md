# S4.3-PI2S contact-conditioning diagnosis

Status: `COMPLETE_VALID_DIAGNOSIS_INCONCLUSIVE`

PI2S is a bounded post-hoc development diagnosis of the PI2B seed instability. It does not
replace the 3,000 Track A formal outcomes. The frozen focus is B0_43, B_VA27_43, B1_43,
B_HVA_42, B_HVA_43 and B2_43.

## Completed evidence

- All 15 final checkpoints, six focus checkpoints and four B_HVA 10k/20k intermediates are
  bound to prior full-tree identities. Seed42 and seeds43/44 are not source-identical cohorts.
- The raw 50 Hz history, 26-frame `LEFT_REPEAT_FIRST` construction, reset isolation and
  `t+27 = 0.54 s` target/mask contract pass.
- Cached H and direct E_T are bitwise equal on 64 frozen histories. The live Unix-service path
  is finite but differs from direct E_T by up to `2.7418136596679688e-6`, exceeding the frozen
  numeric tolerance. This is an engineering mismatch, not a demonstrated rollout cause.
- Prefix and loaded-weight contracts pass. Five H conditions produce 704 frozen action records;
  H-enabled checkpoints show non-zero action deltas under lagged, other-episode, mean and zero H.
- Seventy-two read-only fixed-time/noise gradient records pass with no optimizer or checkpoint
  writes. Main/aux cosine signs vary by model, batch and flow time, so the evidence does not
  identify persistent gradient conflict as a unique cause.
- Historical episode aggregates preserve all 3,000 Track A outcomes. H norms for B_HVA
  seeds42/43/44 remain finite and similar, which does not support a simple norm collapse label.

## Bounded closed-loop diagnosis

The base development wave completed 72/72 tuples with 20 successes and 52 failures:

| Checkpoint | Success / 12 |
|---|---:|
| B0_43 | 4 |
| B_VA27_43 | 8 |
| B1_43 | 3 |
| B_HVA_42 | 4 |
| B_HVA_43 | 0 |
| B2_43 | 1 |

All 12 B_HVA_43 episodes reached object contact and native lift; none reached the native trigger
or success. The defensible localization is therefore after contact/lift and before completion of
the native pinch-cycle trigger. Exact APPROACH and STABLE_GRASP first failure is unavailable
because those stages had no preregistered native predicate.

The optional wave completed 48/48 tuples with nine successes and 39 failures. Relative to the
same-identity correct-H subsets, lag5/mean effects are direction-mixed: B1 mean improves the
point count, B_HVA42 declines, and B_HVA43 has only one mean-H success. These are eight-pair
development cells in a runtime with known cross-attempt outcome non-determinism; they establish
sensitivity, not a causal repair. The earlier v1 stopped lag5 attempts and the v2 hash-domain
audit failures remain preserved beside the v2.1 recovery.

The 72 + 48 rollouts exactly exhaust the authorized maximum of 120. No further search is
authorized. There were zero training steps, optimizer updates, checkpoint writes or real-robot
runs.

## Decision boundary

The evidence supports a real H-conditioned action pathway, a small live-service numeric mismatch,
source-cohort nonidentity and coarse failure localization. It does not support a unique root cause,
a universal method ranking, a replacement of Track A teachers with Track B teachers, or a real
hardware claim.

The minimum next engineering action is to version and qualify a deterministic H transport/runtime
contract under matched device, dtype, thread and kernel settings. Any later scientific run requires
separate authorization, prospective stepwise failure telemetry, one immutable source/config, and
persisted initialization/PRNG traces. Existing checkpoints and scores must remain unchanged.

Canonical machine-readable evidence lives at
`$EXPERIMENT_ROOT/simulation/s4_3_pi2s/artifacts/diagnosis_and_evidence_strength.json`.
