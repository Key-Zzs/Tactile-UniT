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
decoder audits were therefore used to assign the A-R0 diagnosis.

Authoritative raw evidence is stored at
`.local/artifacts/tactile_unit/s3_3_r/a_r0_raw_negative_strength.json`.

### Model-side A-R0 diagnosis

The accepted A2 encoder was audited on fixed-seed random samples without replacement, avoiding
cache-stride aliasing. On untouched test, reversed and shuffled chunks remained almost identical
to correct in the continuous latent: flattened cosine was 0.999562 and 0.999594, with mean
distances 0.206506 and 0.200707. A different-episode chunk was clearly separated (cosine 0.740623,
distance 7.615767). The baseline decoder was not bypassing its tokens: all-window normalized MSE
was 0.069189 for full tokens, 1.142953 for zero/state-only, and 0.503642 for the mean token.
However, its dynamic reversed/correct and shuffled/correct ratios were only 1.1016 and 1.0975,
consistent with a representation that preserves action content much more strongly than order.

The primary diagnosis is `MIXED`, with factors `ABSOLUTE_ACTION_ORDER_INVARIANCE` and
`FROZEN_TRUNK_DOMAIN_MISMATCH`. `WEAK_TEMPORAL_NEGATIVES` and `DECODER_STATE_BYPASS` are not
sufficient diagnoses. The complete evidence is stored in
`.local/artifacts/tactile_unit/s3_3_r/a_r0_diagnosis.json`.

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

R1-P passed the frozen validation gate from step 500 onward and was selected at step 800. Its
validation normalized MSE was 0.071214; dynamic reversed/correct and shuffled/correct were 1.0821
and 1.3368; zero/full and mean/full were 8.5161 and 4.7708; effective rank was 11.3828 with no
collapsed query. Training took 112.33 seconds. Because the smaller-change shared-path candidate
passed, R1-N and R1-S were not executed.

The untouched-test evaluator covered all 536,499 reconstruction windows. All-window normalized
MSE/MAE were 0.069665/0.179546 and raw-unit MSE/MAE were 0.004646/0.042043. Dynamic normalized
MSE/MAE were 0.123770/0.241333 and raw-unit MSE/MAE were 0.007721/0.055351. All-window normalized
MSE by group was 0.065000 left arm, 0.072587 left hand, 0.066514 right arm, and 0.069230 right
hand; dynamic values were 0.111258, 0.134145, 0.107512, and 0.122548 respectively.

Temporal controls used a fixed 16,384-window test sample. Correct all/dynamic MSE was
0.069656/0.124062. Reversed was 0.072342/0.140084 and shuffled was 0.082987/0.189652. The primary
dynamic ratios were 1.1292 for reversal (paired-bootstrap 95% CI 1.1173–1.1412) and 1.5287 for
shuffle (1.4875–1.5701). Different-episode, zero/state-only, and mean all-window ratios were
10.7686, 16.3226, and 5.8655. Every preregistered gate passed.

Frozen probes retained action semantics: active-side balanced accuracy 0.7925, arm-versus-hand
0.6528, primitive balanced accuracy 0.5174, magnitude R2 0.2487, and trend R2 0.3031. Effective
rank was 11.3073; mean query cosine distance was 0.8220; collapsed-query fraction was zero; repeat
and cold reload were exact; outputs were finite `[B,8,32]`.

The validation-only feature ablation favored the full absolute + relative + velocity input
(normalized MSE 0.070789) over absolute only (0.072469), absolute + relative (0.070904), or
absolute + velocity (0.071643).

Original-UniT preservation is structural for R1-N because its checkpoint contains no shared or
old-row value. For R1-P the deployable overlay remains T-Rex-owned and the source digest proof for
rows 0–29 is unchanged. GR1 continues to select untouched row 24 and unchanged shared tensors.

Rows 0–29 were bit-identical before and after training, with matching digest
`e92ced68df2247c19dd99f5be8b165922fbcc06ffa0c27597999ebb4b54d803c`. Shared tensors remained
frozen, so GR1 is preserved and R1-S-only T4 non-regression was not required.

The Original-UniT RQ was evaluated read-only. Mean pre/post cosine was 0.9326 and relative
distortion was 0.3049, but reconstruction MSE increased from 0.069817 continuous to 0.146826
quantized (2.1030x). Stage 0/1 used 104/120 codes with perplexities 34.22/32.97. Quantized dynamic
reversed/shuffled ratios were 1.0502/1.2805, while magnitude, trend, and primitive probe retention
dropped materially. This is an RQ compatibility warning, not a failure of the continuous path;
the RQ was not modified.

## GPU isolation and current execution state

Every model-side scientific process uses the Route A-R advisory-lock wrapper. It audits physical
GPUs 2 and 3 and the explicitly authorized GPU 1, tries physical GPU 3 before GPU 2 and GPU 1,
rechecks compute occupancy after acquiring the lock, exposes exactly one device, and uses logical
`cuda:0`. The user explicitly authorized physical GPU 1 on 2026-08-26 after repeated GPU 2/3
contention. Physical GPU 0 remains forbidden. No process is killed or preempted.

Physical GPUs 2 and 3 had conflicting external workloads throughout the scientific run. The
wrapper therefore used the explicitly authorized physical GPU 1, exposed as the sole logical
`cuda:0`; `torch.cuda.device_count()` was one and the isolation audit passed. GPU 0 was never used.
Model-side A-R0, training, and untouched-test evaluation all record the same compliant device
provenance.

## Current verification

The complete `tests/tactile_unit` suite passes 74 tests. It covers the new feature semantics,
anatomical ordering, temporal sensitivity, controls, `[B,8,32]` interface, gradient routing,
checkpoint reload, deterministic behavior, frozen data interval/leakage, old-row and GR1
preservation, and privacy rules. `git diff --check` and the explicit private-path scan pass. No
`.local` file is tracked.

## Final decision and Track C contract

The final decision is `ACTION_TRANSITION_READY_WITH_RQ_WARNING`. The selected continuous shared
path passes reconstruction, temporal, token-necessity, non-collapse, determinism, and preservation
gates. Only the unchanged frozen RQ is incompatible enough to warrant a warning.

The frozen downstream contract is:

- encoder type: shared R1-P with a T-Rex-only transition adapter;
- input: current normalized state `[B,128]` plus planned/target action chunk `[B,16,128]`, using
  absolute, raw-state-relative, and first-difference transition features;
- output: continuous pre-RQ `z_a [B,8,32]`;
- normalization: `.local/artifacts/tactile_unit/s3_3_r/transition_feature_stats.json`;
- checkpoint: `.local/experiments/tactile_unit/s3_3_r/selected.pt`;
- checkpoint SHA256: `a28b73a1148712bdf3b040305ca6942e552a3f2369950fe1a1562dafa9febfa0`;
- decoder: `decode(z_a, current_state_features, embodiment_id=31)` produces planned target action
  `[B,16,128]`;
- online semantics: the action chunk is a planned/target transition representation, not a current
  observation;
- frozen-RQ status: warning; Track C should consume the continuous pre-RQ representation.

Route A-R is complete. Track C was not started.
