# S3.3-R Action Transition Remediation

This document records Route A-R only. It does not start Track C, Vision–Action–Contact
integration, Contact tokenization, shared-RQ training, VLA policy training, 3D, or real-robot
work.

## Starting point and isolation

The mandatory startup audit was performed before any edit. The worktree was clean on
`develop/action-transition-remediation` at `6156bb2`, and both that accepted Track A baseline and
`develop/tactile-action-bootstrap` were ancestors. The remote was fetched with prune and tags;
there was no unknown divergence. The independent Contact-tokenizer worktree was inspected but not
modified.

Route A-R adds a T-Rex-specific transition module, remediation scripts/config/tests, and this
research record. It does not edit the shared Original-UniT Action implementation, the Contact
path, or either README. The R1-P candidate wraps the accepted shared Action path without changing
shared tensors. The R1-N candidate is entirely T-Rex-native. Consequently there is no shared-core
integration conflict introduced by this route.

## Frozen data and source semantics

The accepted data contract is unchanged:

- frozen episode split: 4,370 train / 547 validation / 547 untouched test episodes;
- cached windows: 65,536 train / 8,192 validation / 536,499 untouched test;
- raw current state `[58]` and planned target-action chunk `[16,58]`;
- action support exactly `a_t:t+15`, with no `a_t+16`;
- ordering: left arm 7, left hand 22, right arm 7, right hand 22;
- canonical state/action: `[128]` and `[16,128]`, with the first 58 channels valid;
- accepted state/action mean and standard deviation fitted on frozen train episodes only;
- split leakage: zero for train/validation, train/test, and validation/test.

The dataset source describes `observation.state` as joint positions and `action` as target joint
positions in the identical 58-dimensional anatomical order. Dimension names pair each current
arm/hand joint with its corresponding target joint. State-relative subtraction is therefore
semantically legal when performed in raw joint space. It is deliberately not performed between
independently normalized state and action vectors.

## A-R0: raw negative strength

A-R0 performs no training. The raw pass covers every cached window. Reversal and shuffling stay
within the same episode and the same valid 16-frame support. The shuffle uses one frozen seed and
permutation, there is no padding, and neither whole-chunk negative has an unchanged-frame bug.

The table reports normalized sequence MSE between each negative and the correct action chunk.

| Split | Subset | Reversed | Shuffled | Different episode |
|---|---:|---:|---:|---:|
| train | all | 0.011212 | 0.009321 | 1.994105 |
| train | dynamic | 0.064923 | 0.054257 | 2.414487 |
| validation | all | 0.011026 | 0.009168 | 1.968903 |
| validation | dynamic | 0.062709 | 0.052422 | 2.523144 |
| untouched test | all | 0.010728 | 0.008925 | 2.044495 |
| untouched test | dynamic | 0.059060 | 0.049431 | 2.411847 |

The all-window negatives are close because the train-derived split labels about 89% of windows as
static. They are nevertheless non-trivial on the frozen dynamic subset, consistently across all
three episode partitions. First-difference MSE makes the shuffle especially visible: on untouched
test it is 0.006642 overall, versus 0.001144 for reversal. The raw evidence therefore rules out
“the data contains no usable order signal” as a sufficient diagnosis. Model-side encoder and
decoder audits remain necessary before assigning the A-R0 primary diagnosis.

Authoritative raw evidence is stored at
`.local/artifacts/tactile_unit/s3_3_r/a_r0_raw_negative_strength.json`.

## Transition-centered representation

Features are constructed in raw joint space and then normalized with frozen-train-only moments:

1. accepted normalized absolute target action;
2. state-relative target `a_(t+tau) - s_t`;
3. first difference `a_(t+tau) - a_(t+tau-1)` for `tau=1..15`, with a documented zero sentinel at
   the first temporal position.

The absolute action is retained, so derivative features cannot destroy the ability to reconstruct
the planned absolute target. Relative and velocity moments are fitted over the complete accepted
train cache; validation and test never contribute statistics.

### R1-P: transition preprocessor plus shared Action path

R1-P is the minimal shared-path candidate. A unified 174-to-128 transition adapter is initialized
as an exact pass-through for the 58 absolute normalized action channels, with zero-initialized
temporal residuals. It feeds the accepted selected T-Rex row-isolated A2 Action path. Trainable
values are limited to the transition adapter, T-Rex row 31, the existing T-Rex-only A2 residual,
and T-Rex-specific decoder rows. Rows 0–29 and every shared Action tensor remain frozen. Original
UniT RQ is never part of training.

### R1-N: grouped T-Rex-native path

R1-N is only executed if R1-P fails the frozen validation gates. It contains four anatomical
temporal branches (left arm, left hand, right arm, right hand), explicit side/type embeddings,
compact dilated residual temporal blocks, two small temporal Transformer layers, and eight learned
queries. It emits continuous `z_a [B,8,32]`. Its token-gated decoder conditions on the current
state but bounds the initial state-path contribution, retaining the legal condition without
removing the need for action tokens. The native model has approximately 0.78 million parameters.

R1-S remains prohibited unless both R1-P and R1-N are insufficient. If it ever becomes necessary,
it requires a separate GR1 rehearsal protocol and final T4 Action-side non-regression; the T4
held-out 960 may not be used for training or selection.

## Objectives and frozen validation selection

Both candidates use absolute reconstruction, raw-state-relative reconstruction, first-difference
reconstruction, temporal ranking against reversed/shuffled/different-episode chunks, zero-token
necessity ranking, train-derived dynamic weighting, and variance/query-diversity anti-collapse
terms. Current state remains present in every decoder path.

Candidate checkpoints are ranked on validation only. A gate-passing checkpoint must satisfy:

- normalized reconstruction MSE no greater than 1.0;
- dynamic reversed/correct and shuffled/correct ratios at least 1.05;
- different-episode/correct ratio at least 1.05;
- zero/full and mean/full ratios at least 1.10;
- effective rank at least 8 and collapsed-query fraction at most 0.05;
- finite deterministic `[B,8,32]` output.

Only after R1-P/R1-N selection is frozen may the untouched test evaluator run. The test gate adds
paired bootstrap evidence: the 95% lower bounds for dynamic reversed and shuffled ratios must be
greater than 1.0. All-window results remain reported even when the dynamic gate is primary.

## Required held-out evidence

The evaluator is implemented to cover normalized and raw-unit MSE/MAE for all/dynamic windows and
all four anatomical groups; correct, reversed, shuffled, different-episode, zero, mean, and
state-only controls; magnitude/trend/active-side/arm-versus-hand/primitive probes; variance,
effective rank, norms, query diversity, collapse, and pairwise distance; exact repeat and cold
reload; validation-only feature ablations; and read-only Original-UniT frozen-RQ distortion,
cosine, code use, perplexity, reconstruction retention, probe retention, and temporal-control
retention.

Original-UniT preservation is structural for R1-N because its checkpoint contains no shared or
old-row value. For R1-P the deployable overlay remains T-Rex-owned and the source digest proof for
rows 0–29 is unchanged. GR1 continues to select untouched row 24 and unchanged shared tensors.

## GPU isolation and current execution state

Every model-side scientific process uses the Route A-R advisory-lock wrapper. It audits physical
GPUs 2 and 3, tries physical GPU 3 before GPU 2, rechecks compute occupancy after acquiring the
lock, exposes exactly one device, and uses logical `cuda:0`. Physical GPUs 0 and 1 are forbidden.
No process is killed or preempted.

At the latest execution audit, both permitted GPUs had conflicting external compute workloads, so
the scientific state was `GPU_RESOURCE_BUSY`. The complete raw A-R0 pass, train-only transition
statistics, implementation, and CPU tests proceeded. Model-side A-R0, R1-P/R1-N training,
untouched-test evaluation, plots, HUMAN_ACCEPTANCE, and the final decision remain pending compliant
GPU evidence. No provisional number in this document is used to claim readiness.

## Current verification

The complete `tests/tactile_unit` suite passes 72 tests. It covers the new feature semantics,
anatomical ordering, temporal sensitivity, controls, `[B,8,32]` interface, gradient routing,
checkpoint reload, deterministic behavior, frozen data interval/leakage, old-row and GR1
preservation, and privacy rules. No `.local` file is tracked.

## Final decision

Pending model-side A-R0, validation selection, and untouched-test evaluation. The route must end in
exactly one of the objective's allowed `ACTION_TRANSITION_*` or `STRUCTURAL_FAIL` decisions; no
readiness decision is made from the raw audit alone.
