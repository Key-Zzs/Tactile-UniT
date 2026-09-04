# S4.3-RR ACT Rollout Runtime Remediation & Frozen Benchmark Resume Result

## 1. Git

Branch:
develop/sim-benchmark

Starting HEAD:
862c93681d68fa86b628e62b219b702680a114ab

Final HEAD:
SELF — the local commit containing this report

Commits:

- eeece4f5333375ab4e4f11b45a4533699315c6b0 — fix(sim): harden ACT rollout IPC transport
- 86da21bc03f12f52a22073cc003ec275438f6b35 — test(sim): refreeze ACT closed-loop rollout harness
- final local commit — eval(sim): close frozen ACT policy benchmark

Push:
NOT PERFORMED

PR:
NOT CREATED

Working tree:
Clean after the final local commit.

## 2. Historical S4.3-2 Failure Preservation

Historical decision:
S4_3_2_ENVIRONMENT_FAIL

Historical cause:
AF_UNIX_PATH_TOO_LONG

Old endpoint bytes:
118

Old platform payload limit:
107

Scientific rollouts previously started:
0

Policy performance previously seen:
NO

Historical result modified:
NO

## 3. Frozen Scientific State

ACT checkpoints:
36

Checkpoint hashes unchanged:
PASS — all 36 are byte-identical; checkpoint-set SHA256 is `152ce90961e0c943d21a554b4b564ba80b4a6520e78073f6edba1624512b78c0`.

Policy variants:
P0/P1/P2/P3

Training seeds:
0, 1, 2

Evaluation resets:
90 frozen `POLICY_EVAL_V1` reset specifications; 30 per task.

Success predicates:
Unchanged native pinch-height/pinch-count, nail-depth, and mouse-display contracts; contract SHA256 `7268e8d3dce846621157dcf982e15b38321d0cf8952024f03fdfcd77c729d2f9`.

Timeouts:
pinch_tongs 823, hammer_nail 635, click_mouse 1000 post-warm-up control steps.

Stride:
5

Scientific contract mutation:
NO

## 4. AF_UNIX Remediation

Root cause:
IPC_TRANSPORT_PATH_LENGTH_BUG, independent of checkpoints, task physics, DexJoCo state, Action adaptation, success criteria, and S4.2 representations.

Old construction:
`$REPOSITORY/.local/tmp/simulation/s4_3_restart/rollout_sockets/<task>_<variant>_seed<seed>.sock`

New construction:
Short private runtime namespace plus a digest, process identity, and nonce.

Runtime root:
Prefer `$XDG_RUNTIME_DIR/tu3d`; otherwise use `$SYSTEM_TEMP/tu3d-<uid>`.

Endpoint format:
`$RUNTIME_TMP/tu3d_<short_hash>_<pid>_<nonce>.sock`

Maximum encoded endpoint length:
54 bytes in the 36-identity transport audit; 57 bytes in production smoke and scientific workers.

Hard endpoint ceiling:
80 bytes

Server/client centralized builder:
PASS

Stale cleanup:
Private namespace, uid, registered-PID liveness, socket type, and inode ownership are all checked before reclamation; normal and failed startup clean up their own endpoint.

Scientific runtime semantics changed:
NO

## 5. Transport Regression

Historical long-path regression:
PASS

36 job endpoint identities:
36

Unique:
36

All <=80 bytes:
YES; maximum 54 bytes in the synthetic identity audit.

Four-worker bind/connect:
PASS, with no cross-talk.

Cleanup:
PASS for active, stale, arbitrary-file, failed-connect, failed-startup, and normal-exit cases.

RPC payload parity:
PASS

Gate:
PASS

## 6. Production-Path Smoke

Exact production worker:
YES

Smoke identities:
6 disjoint `RR_SMOKE` runs, each with 100 post-warm-up control steps.

Tasks:
pinch_tongs, hammer_nail, click_mouse

Variants:
P0 and P3 for every task.

Server bind:
PASS

Client connect:
PASS

EGL:
PASS

Checkpoint:
PASS for all six frozen checkpoint loads.

Action [27,22]:
PASS

Adapter:
PASS, central 22D-to-23D Action adapter.

Stride:
PASS, 5.

Causality:
PASS; no future read, and P3 Contact-State/auxiliary runtime paths were exercised.

Socket cleanup:
PASS; no endpoint or process leak.

Gate:
PASS

## 7. Rollout Harness V2 Freeze

Artifact:
`.local/artifacts/simulation/s4_3_rr/pre_rollout_freeze_v2.json`

SHA:
`77752956370a4ca63398dc585c01c5dccf5ffe2803a6cf5187280b189bf8096c`

36 checkpoint hashes:
PASS; frozen in the artifact.

90 reset manifest hash:
Evaluation config SHA256 `e04144eef2f5b6914bafb43611ff69821ad1a894e89424af40a7dc0d35063a08`; reset-identity SHA256 `07056988216afee716aca87f73cd599d5bd15957ba585fac599d74575ec22c6d`.

Statistics hash:
`7fd284bbf8c85ad6c3f78210cb7dff79baa9e3e6b158f9d67e08e2bcd223ab40`

Only transport code changed:
YES

Scientific rollout performance seen before freeze:
NO

Gate:
PASS

## 8. GPU Execution

Eligible physical GPUs:
0,1,2,3 when genuinely idle.

Workers:
36 repository-owned scientific jobs, one worker at a time on the only eligible device at each launch.

Per-GPU rollout counts:
GPU1: 1080 rollouts across 36 jobs; GPU0/GPU2/GPU3: 0.

Busy conflicts:
0; unrelated allocations were excluded.

Lock violations:
0

Oversubscription:
NO

## 9. Closed-Loop Completeness

Expected:
1080

Canonical completed:
1080 across 36/36 jobs and 1080 unique canonical identities.

Infrastructure failures:
Two controller-session exits: one at a completed-job boundary after 600 rollouts and one after 720 rollouts. The latter interrupted a newly launched worker before any reset metadata existed and was relaunched with identical code, checkpoint, configuration, and ordered reset stream.

Exact retries:
0 canonical scientific retries; 1 exact pre-reset infrastructure relaunch. The controller restart record was reconstructed after queue completion because the original attempt-1 log path was reused.

Scientific failures/timeouts retained:
YES — all 1062 timeouts were retained; there were 18 successes.

Reset replacements:
0

Duplicate canonical identities:
0

Missing identities:
0

Gate:
PASS

## 10. Success by Task

| Task | P0 | P1 | P2 | P3 |
|---|---:|---:|---:|---:|
| pinch_tongs | 0/90 (0.00%) | 8/90 (8.89%) | 9/90 (10.00%) | 0/90 (0.00%) |
| hammer_nail | 0/90 (0.00%) | 1/90 (1.11%) | 0/90 (0.00%) | 0/90 (0.00%) |
| click_mouse | 0/90 (0.00%) | 0/90 (0.00%) | 0/90 (0.00%) | 0/90 (0.00%) |

Per-training-seed results, in seed0/seed1/seed2 order:

- pinch_tongs — P0 0/0/0%; P1 16.67/6.67/3.33%; P2 3.33/10.00/16.67%; P3 0/0/0%.
- hammer_nail — P0 0/0/0%; P1 0/3.33/0%; P2 0/0/0%; P3 0/0/0%.
- click_mouse — every variant 0/0/0%.

## 11. Primary 3-Task Macro Success

P0:
0.00%

P1:
3.33%

P2:
3.33%

P3:
0.00%

95% CIs:
P0 [0.00%, 0.00%]; P1 [0.00%, 9.63%]; P2 [0.00%, 10.74%]; P3 [0.00%, 0.00%].

## 12. Primary Ablations

### P1-P0 — Raw tactile

Delta:
+3.33 percentage points

95% CI:
[0.00, +10.00] percentage points

Classification:
NO_MATERIAL_DIFFERENCE

### P2-P1 — Contact-State vs raw

Delta:
0.00 percentage points

95% CI:
[-5.56, +5.93] percentage points

Classification:
NO_MATERIAL_DIFFERENCE

### P3-P2 — Tactile-UniT auxiliary

Delta:
-3.33 percentage points

95% CI:
[-10.37, 0.00] percentage points

Classification:
NO_MATERIAL_DIFFERENCE

### P3-P0 — Full method

Delta:
0.00 percentage points

95% CI:
[0.00, 0.00] percentage points

Classification:
NO_MATERIAL_DIFFERENCE

## 13. Tactile-Active Secondary Analysis

Tasks:
pinch_tongs + click_mouse

P0:
0.00%

P1:
4.44%

P2:
5.00%

P3:
0.00%

P1-P0:
+4.44 pp, 95% CI [0.00, +12.79] pp, NO_MATERIAL_DIFFERENCE.

P2-P1:
+0.56 pp, 95% CI [-7.22, +8.89] pp, NO_MATERIAL_DIFFERENCE.

P3-P2:
-5.00 pp, 95% CI [-13.89, 0.00] pp, NO_MATERIAL_DIFFERENCE.

P3-P0:
0.00 pp, 95% CI [0.00, 0.00] pp, NO_MATERIAL_DIFFERENCE.

Interpretation:
The pre-registered secondary analysis does not establish a tactile benefit. Pinch alone shows P1-P0 improvement and P3-P2 harm, but click_mouse has zero learning and the overall ACT competence floor is weak.

## 14. Hammer-Nail Control Analysis

Mapped tactile active:
NO

P0:
0.00%

P1:
1.11%

P2:
0.00%

P3:
0.00%

Contrasts:
P1-P0 +1.11 pp [0.00, +4.44]; P2-P1 -1.11 pp [-4.44, 0.00]; P3-P2 0.00 pp [0.00, 0.00]; P3-P0 0.00 pp [0.00, 0.00]. All are NO_MATERIAL_DIFFERENCE.

Interpretation:
The mapped-tactile-inactive control remains extremely weak and supplies no evidence of a representation effect; warning `HAMMER_NAIL_MAPPED_TACTILE_INACTIVE` remains active.

## 15. Training-Seed Robustness

Per task / variant:
The exact task-level seed rates are reported in Section 10 and `training_seed_robustness.json`.

seed0:
Primary macro P0/P1/P2/P3 = 0.00/5.56/1.11/0.00%.

seed1:
Primary macro P0/P1/P2/P3 = 0.00/3.33/3.33/0.00%.

seed2:
Primary macro P0/P1/P2/P3 = 0.00/1.11/5.56/0.00%.

Mean/std/range:
P0 0.00/0.00/[0.00,0.00]%; P1 3.33/1.81/[1.11,5.56]%; P2 3.33/1.81/[1.11,5.56]%; P3 0.00/0.00/[0.00,0.00]%.

Main effect sign consistency:
P1-P0 positive in all seeds; P2-P1 mixed (-4.44/0/+4.44 pp); P3-P2 negative in all seeds; P3-P0 zero in all seeds.

Effect dominated by one seed:
YES only for P2-P1; NO for P1-P0, P3-P2, and P3-P0.

## 16. Secondary Metrics

All values below are paired all-task deltas in P1-P0 / P2-P1 / P3-P2 / P3-P0 order; they remain secondary and do not reclassify the primary endpoint.

Time-to-success:
Timeout-imputed completion-time deltas: -0.1776 / +0.0224 / +0.1552 / 0.0000 s. Among actual successes, pinch P1 averaged 11.6525 s, pinch P2 11.8044 s, and hammer P1 3.2200 s; other cells were unavailable.

Timeout rate:
-3.33 / 0.00 / +3.33 / 0.00 pp.

Peak normal force:
+4.2838 / +2.3600 / -0.0321 / +6.6117.

Integrated force:
+18.9544 / +9.6355 / -1.5892 / +27.0007.

Tangential force:
+1.5828 / +0.9204 / +0.3180 / +2.8211.

Contact transition counts:
free-to-contact +0.4630 / +0.7889 / -0.5074 / +0.7444; contact-to-free +0.3481 / +0.7333 / -0.4778 / +0.6037.

Action step norm:
+0.011953 / -0.003190 / +0.189108 / +0.197871.

Action acceleration:
+0.014544 / -0.005947 / +0.358457 / +0.367054.

TCP jerk:
+0.003294 / -0.002090 / +0.004790 / +0.005994.

Hand variation:
+19.6615 / +0.2094 / +3.2263 / +23.0973.

## 17. Offline vs Closed-Loop

Offline Action L1 ranking:
P1 (0.128709) < P0 (0.129083) < P2 (0.129113) < P3 (0.139726); lower is better.

Closed-loop success ranking:
P1 = P2 (3.33%) > P0 = P3 (0.00%).

P3 offline warning:
P3 had worse normalized full-chunk Action L1 than P0/P1/P2 on every task, so `OFFLINE_ACTION_ERROR_WARNING` remains active.

Does offline L1 predict closed-loop outcome:
MIXED — the P3 warning was directionally consistent with zero closed-loop success, but the small P0/P1/P2 ordering was not monotonic.

## 18. Policy Learning Sanity

Classification:

WEAK

Evidence:
Only 18/1080 rollouts succeeded. No task reached the registered healthy threshold of at least 20% success for any variant, whereas HEALTHY requires that threshold on at least two tasks.

Task ceilings:
pinch_tongs 10.00%; hammer_nail 1.11%; click_mouse 0.00%. No ceiling-effect warning applies.

## 19. Uncertainty Status

Runtime uncertainty:
NOT_AVAILABLE_CAUSALLY_FOR_S4_3_2

Future Vision used:
NO

Intervention:
NO

Scientific consequence:
DIAGNOSTIC_UNAVAILABLE_ONLY

## 20. Experiment Validity

Policy data:
PASS

Same reset specs:
PASS

Same success:
PASS

Same timeout:
PASS

Same warm-up:
PASS — frozen 0.5 s warm-up, implemented as 25 actions at 20 ms and represented by 26 samples including the initial sample.

Same stride:
PASS

Same Action adapter:
PASS

No future leakage:
PASS

No expert Action inference:
PASS

S4.2 immutable:
PASS

All 36 checkpoints unchanged:
PASS

Overall:
PASS

## 21. Final S4.3-2 Decision

S4_3_2_ACT_BENCHMARK_WEAK

Reasons:

1. The 1080-rollout benchmark is structurally complete and valid, but no task reached the registered 20% policy-competence threshold.
2. Only 18 rollouts succeeded: 17 on pinch_tongs, one on hammer_nail, and zero on click_mouse.
3. All four primary contrasts are NO_MATERIAL_DIFFERENCE, so reliable representation-effect inference is not supported under the WEAK ACT competence floor.

## 22. S4.3-3 Diffusion Policy Readiness

Decision:

NOT_READY

ACT positive Tactile-UniT effect required:
NO

Reason:
The complete and valid ACT benchmark is too weak to establish a reliable policy-competence floor for a Diffusion Policy comparison.

Warnings:
`ACT_WEAK`, `CLICK_MOUSE_ZERO_SUCCESS`, `P0_ZERO_SUCCESS`, `P3_ZERO_SUCCESS`, `OFFLINE_ACTION_ERROR_WARNING`, `HAMMER_NAIL_MAPPED_TACTILE_INACTIVE`, `FROZEN_CAUSAL_UNCERTAINTY_UNAVAILABLE`, and `CONTROLLER_RESTART_RECORD_RECONSTRUCTED_AFTER_QUEUE_COMPLETION`.

Recommended next:
Pre-register a separate ACT competence remediation/re-baseline with fresh evaluation resets; do not tune on or reuse the exposed `POLICY_EVAL_V1` results for model selection.

Do NOT implement S4.3-3.

## 23. Environment / Regression

unit:
Package hash `3a119880cd4d661259d9476b0d3224302e086ae899ca77250497b5f42fbf6f9b`, Python 3.10.20, unchanged.

DexJoCo:
Package hash `7406008d77c52571b64f2c7fdf36ed35a62da160e2eba4b9d091ac9f85d82f78`, Python 3.11.16, unchanged.

S4.2 hashes:
PASS; Contact-State, C3, A0, Vision, B3, A+H, fallback, shared/private, uncertainty, and tracked S4.2 configs are byte-identical.

36 ACT checkpoints:
PASS, byte-identical.

RoboCasa:
EGL reset/step/RGB/Contact/named-region smoke PASS.

DexJoCo submodule:
Clean at `8d23b0fab23b17a58c4b55f3942e17013aaf8267`.

Nested Diffusion Policy:
UNINITIALIZED

Unit tests:
565 passed, 1 skipped, 0 failed.

DexJoCo tests:
23 passed, 0 failed.

Failures:
0

Overall:
PASS

## 24. Milestone Status

M3:
UNCHANGED

S4.2:
COMPLETE

S4.3-PD:
COMPLETE

S4.3-0:
COMPLETE

S4.3-1:
COMPLETE

Historical first S4.3-2 rollout attempt:
ENVIRONMENT_FAIL — PRESERVED

S4.3-RR:
COMPLETE

S4.3-2:
COMPLETE

S4.3:
IN PROGRESS

M4:
NOT ESTABLISHED

## 25. Stop Point

STOP AFTER S4.3-RR / S4.3-2 DECISION.

DO NOT TRAIN DIFFUSION POLICY.
DO NOT START S4.3-3.
DO NOT TRAIN π0.5.
DO NOT TRAIN GR00T.
DO NOT PORT RH56DFTP.
DO NOT PORT FLEXIV.
DO NOT PUSH.
DO NOT CREATE PR.
