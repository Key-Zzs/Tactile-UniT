# Restarted S4.3-0 to S4.3-2 Causal ACT Policy Benchmark Result

## 1. Git

Branch:
develop/sim-benchmark

Starting HEAD:
ab359b63518baaec374d22852b2d16863ea4842a

Final HEAD:
SELF — the local commit containing this report

Commits:

- f86cf49297462173bbc6e85c635b12f2316ad7b8 — test(sim): freeze restarted S4.3 ACT benchmark protocol
- dbf6b769054d49692c3ff704ac88077bf9de99a7 — feat(sim): establish causal Tactile-UniT ACT integration
- f7dc99c9293dd7c238caef34a0707176041df076 — feat(sim): add ACT tactile policy baselines
- final local commit — eval(sim): record restarted S4.3 R12 failure

Push:
NOT PERFORMED

PR:
NOT CREATED

Working tree:
Clean after the final local commit.

## 2. Historical Failure Preservation

Previous S4.3-0:

S4_3_0_DEMONSTRATION_CONTRACT_FAIL

Modified:
NO

Old 210 probing episodes used for BC:
NO

Restart basis:

S4_3_PD_COMPLETE_POLICY_DATA_READY

## 3. Frozen Policy Dataset

TRAIN:

pinch_tongs:
90 BC-eligible episodes

hammer_nail:
100 BC-eligible episodes

click_mouse:
100 BC-eligible episodes

DEV:

pinch_tongs:
25 BC-eligible episodes

hammer_nail:
20 BC-eligible episodes

click_mouse:
25 BC-eligible episodes

BC-eligible totals:
290 TRAIN / 70 DEV. Fifteen native-failure attempts remain audit-only.

Valid BC windows:
115,575 total: pinch_tongs 33,905 TRAIN / 8,815 DEV; hammer_nail 19,150 / 4,015; click_mouse 43,105 / 6,585.

Manifest SHA:
576080b8c19265e965ab593f74be053d5646917d1752a967c438e76f83157efa

Dataset content SHA:
a334af12ab2c5bbfbd46b47e1167f51fe194c97bbd69fb945d760f96b648d7f4

Mutation:
NO

## 4. Policy Benchmark Protocol

Training seeds:
0, 1, 2

Evaluation resets/task:
30

Timeouts:
pinch_tongs 823; hammer_nail 635; click_mouse 1000 control steps. The common 0.5 s / 26-sample warm-up is excluded.

Success predicates:

- pinch_tongs: tongs height at least table + 0.10 m and pinch count at least 3 continuously for 30 control steps.
- hammer_nail: nail depth at least 0.04 m.
- click_mouse: mouse remains inside mousepad and display is blue for 10 consecutive control steps.

Primary endpoint:
3-task macro success

Secondary tactile-active tasks:
pinch_tongs + click_mouse

Hammer warning:
HAMMER_NAIL_MAPPED_TACTILE_INACTIVE

Protocol SHA:
d08eda3a7005981b88f4623f4b4b85f759efd0ff95d32e2f21073469c7edfd20

## 5. GPU Execution

Eligible GPUs:
0,1,2,3 under the frozen scheduler; only GPU1 was idle at R0.

Jobs:
36 ACT training jobs; 36 checkpoint offline evaluations; 1 rollout worker launch attempt.

Per-GPU counts:
GPU1: all 36 training jobs, all 36 offline checkpoint records, and the single failed rollout worker attempt. GPU0/GPU2/GPU3: zero benchmark jobs.

Busy conflicts:
GPU0, GPU2, and GPU3 contained unrelated allocations and were not used.

Lock violations:
0

Oversubscription:
NO

## 6. Causal Observation Contract

Vision:
I_t only; frozen current-frame DINO, shape [8,32]

Proprio:
22D

P1 tactile:
[26,30]

P2/P3 Contact-State:
[256]

Action:
[27,22]

Replan stride:
5

Future Vision:
NO

Future actual Contact at inference:
NO

Gate:
PASS

## 7. P3 Tactile-UniT Auxiliary

A0:
frozen

B3:
frozen

A+H:
frozen

Target u_c:
training only, exact t to t+27 target path

lambda:
0.1

S4.2 parameter gradients:
none

ACT gradient:
Nonzero on every task; L1 totals were 3,832.8285 (pinch_tongs), 32,140.1071 (hammer_nail), and 769.2404 (click_mouse).

Causal gate:
PASS

## 8. Runtime Smoke

Tasks:
3

Resets:
9 total, 3 per task

Steps:
900 policy control steps

EGL:
PASS

Action adapter:
PASS, central 22D-to-23D adapter

Queue/replan:
180 replans, stride 5, PASS

Causal trace:
No future read; current RGB, 22D proprio, [26,30] tactile history and 256D Contact-State all PASS.

Gate:
PASS — S4_3_1_CAUSAL_POLICY_INTERFACE_READY

## 9. ACT Implementation

Provenance:
Minimal repository-owned ACT reproduction using the standard action-chunking CVAE Transformer structure; no network code fetched.

Architecture:
4-layer encoder + 4-layer decoder, 8 heads, 1024 FFN, learned positions, 27 queries, dropout 0.1.

Vision:
Frozen current-frame DINO representation

Hidden:
256

CVAE:
latent 32, beta 10, deterministic inference prior mean z=0

Parameters:

P0:
7,440,470

P1:
7,504,278; 63,552 tactile-specific parameters

P2:
7,506,518

P3:
7,506,518

Fairness:
PASS; P2/P3 inference architecture, parameter count, and initialization are identical.

## 10. ACT Training

Expected jobs:
36

Completed:
36

Infrastructure retries:
1 exact retry: hammer_nail/P2/seed1 after an external process kill. Step-1000 repeat was exact and partial evidence was preserved.

Scientific-performance retries:
0

Failures:
0 canonical training failures

Per task/variant/seed, selected POLICY_DEV normalized full-chunk Action L1:

| Task | Variant | Seed 0 | Seed 1 | Seed 2 | Selected steps |
|---|---|---:|---:|---:|---|
| pinch_tongs | P0 | 0.118005 | 0.119306 | 0.119309 | 16000 / 20000 / 19000 |
| pinch_tongs | P1 | 0.116421 | 0.117257 | 0.119260 | 17000 / 20000 / 14000 |
| pinch_tongs | P2 | 0.119734 | 0.120301 | 0.120340 | 14000 / 14000 / 20000 |
| pinch_tongs | P3 | 0.128967 | 0.129246 | 0.128695 | 9000 / 13000 / 12000 |
| hammer_nail | P0 | 0.099459 | 0.099701 | 0.099664 | 13000 / 17000 / 19000 |
| hammer_nail | P1 | 0.097249 | 0.101102 | 0.102567 | 12000 / 16000 / 5000 |
| hammer_nail | P2 | 0.100251 | 0.097907 | 0.096991 | 17000 / 17000 / 16000 |
| hammer_nail | P3 | 0.111814 | 0.113860 | 0.112558 | 6000 / 1000 / 7000 |
| click_mouse | P0 | 0.169770 | 0.166947 | 0.169588 | 11000 / 13000 / 12000 |
| click_mouse | P1 | 0.166137 | 0.169919 | 0.168467 | 12000 / 11000 / 16000 |
| click_mouse | P2 | 0.171619 | 0.167358 | 0.167512 | 13000 / 17000 / 12000 |
| click_mouse | P3 | 0.181469 | 0.175640 | 0.175284 | 8000 / 9000 / 15000 |

## 11. Offline POLICY_DEV

Values below are mean normalized full-chunk Action L1, with seed values in parentheses.

### pinch_tongs

P0:
0.118873 (0.118005, 0.119306, 0.119309)

P1:
0.117646 (0.116421, 0.117257, 0.119260)

P2:
0.120125 (0.119734, 0.120301, 0.120340)

P3:
0.128970 (0.128967, 0.129246, 0.128695)

### hammer_nail

P0:
0.099608 (0.099459, 0.099701, 0.099664)

P1:
0.100306 (0.097249, 0.101102, 0.102567)

P2:
0.098383 (0.100251, 0.097907, 0.096991)

P3:
0.112744 (0.111814, 0.113860, 0.112558)

### click_mouse

P0:
0.168768 (0.169770, 0.166947, 0.169588)

P1:
0.168174 (0.166137, 0.169919, 0.168467)

P2:
0.168829 (0.171619, 0.167358, 0.167512)

P3:
0.177464 (0.181469, 0.175640, 0.175284)

Cold reload:
36/36 PASS

Determinism:
36/36 PASS with zero repeat difference; all [27,22] predictions finite and bounded. All P3 [8,32] auxiliary predictions were finite.

Gate:
PASS

Offline values are checkpoint-selection and sanity evidence only; they are not closed-loop effect estimates.

## 12. Closed-Loop Rollouts

Expected:
1080

Completed:
0

Invalid:
0; no scientific rollout began.

Same reset specifications:
PASS at R11 freeze, not exercised in R12.

Failure:
The first click_mouse/P0/seed0 worker launched on GPU1, but the policy server failed while binding its AF_UNIX listener. The production endpoint was 118 bytes versus the 107-byte payload limit, raising `OSError: AF_UNIX path too long` before socket readiness and before the DexJoCo client started.

Protocol disposition:
STRUCTURAL HARD FAILURE. All dependent stages stopped; rollout code remained byte-identical to the R11 freeze and no retuning was performed.

## 13. Success by Task

| Task | P0 | P1 | P2 | P3 |
|---|---:|---:|---:|---:|
| pinch_tongs | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| hammer_nail | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| click_mouse | NOT RUN | NOT RUN | NOT RUN | NOT RUN |

Training-seed CIs / variation:
NOT ESTIMABLE — zero scientific rollouts; no value was imputed from POLICY_DEV.

## 14. Primary 3-Task Macro Success

P0:
NOT RUN

P1:
NOT RUN

P2:
NOT RUN

P3:
NOT RUN

95% CIs:
NOT ESTIMABLE

## 15. Primary Ablations

### Raw tactile: P1-P0

Delta:
NOT ESTIMABLE

CI:
NOT ESTIMABLE

Classification:
NOT RUN — R12 structural failure

### Contact-State vs raw: P2-P1

Delta / CI / classification:
NOT RUN — R12 structural failure

### Tactile-UniT auxiliary: P3-P2

Delta / CI / classification:
NOT RUN — R12 structural failure

### Full method: P3-P0

Delta / CI / classification:
NOT RUN — R12 structural failure

## 16. Tactile-Active Secondary Analysis

Tasks:
pinch_tongs + click_mouse

P0:
NOT RUN

P1:
NOT RUN

P2:
NOT RUN

P3:
NOT RUN

P1-P0:
NOT ESTIMABLE

P2-P1:
NOT ESTIMABLE

P3-P2:
NOT ESTIMABLE

P3-P0:
NOT ESTIMABLE

Interpretation:
Withheld because the structural failure occurred before the first rollout. No offline surrogate was substituted.

## 17. Hammer-Nail Control Analysis

Mapped tactile active:
NO

P0:
NOT RUN

P1:
NOT RUN

P2:
NOT RUN

P3:
NOT RUN

Interpretation:
HAMMER_NAIL_MAPPED_TACTILE_INACTIVE remains the frozen warning; no control-task rollout result exists.

## 18. Secondary Metrics

Time-to-success:
NOT RUN

Timeout:
NOT RUN

Peak normal force:
NOT RUN

Integrated force:
NOT RUN

Tangential:
NOT RUN

Action smoothness:
NOT RUN

TCP jerk:
NOT RUN

Hand variation:
NOT RUN

## 19. Training-Seed Robustness

P0:
NOT ESTIMABLE

P1:
NOT ESTIMABLE

P2:
NOT ESTIMABLE

P3:
NOT ESTIMABLE

Effect dominated by one seed:
NOT ASSESSED

All three training seeds and all 36 checkpoints were retained; no rollout seed was discarded.

## 20. Uncertainty Diagnostics

Success vs failure:
NOT RUN

Contact onset:
NOT RUN

High-force:
NOT RUN

Timeout:
NOT RUN

Runtime availability:
UNAVAILABLE_CAUSAL_INPUT_MISMATCH; the frozen full estimator requires a forbidden future-derived Vision transition representation. Invocations 0; interventions 0.

Scientific status:
DIAGNOSTIC ONLY

## 21. Experiment Validity

Policy data:
PASS

Same resets:
PASS at freeze; not exercised

Same success:
PASS

Same timeout:
PASS

Same Action adapter:
PASS

Same stride:
PASS

No future leakage:
PASS in data, training, offline evaluation, and runtime smoke; R12 did not reach inference.

No expert Action inference:
PASS

S4.2 immutable:
PASS — checkpoints, Vision files, and tracked S4.2 configs are byte-identical.

Learning sanity:
NOT ASSESSED — none of HEALTHY / WEAK / ZERO_LEARNING can be assigned without a scientific rollout.

Overall:
FAIL — frozen runtime execution failure prevented the benchmark endpoint.

## 22. Final S4.3-2 Decision

S4_3_2_ENVIRONMENT_FAIL

Reasons:

1. The frozen production policy server raised `OSError: AF_UNIX path too long` while binding its 118-byte endpoint.
2. The server never became ready and the DexJoCo client never started, so zero of 1080 scientific rollouts began or completed.
3. Protocol rules require stopping dependent analyses on a structural hard failure; no scientific effect was classified or fabricated.

## 23. S4.3-3 Readiness

Diffusion Policy:

NOT_READY

Reason:
Runtime execution failure is an explicit structural blocker even though policy data, causality, training, offline evaluation, environment integrity, regressions, and S4.2 immutability otherwise passed.

ACT positive Tactile-UniT result required:
NO

Recommended next:
Shorten the production AF_UNIX socket endpoint below the platform limit, repeat the production-path runtime integration smoke, re-freeze R11 with corrected rollout code hashes, and start a new clean S4.3-2 rollout execution.

Do NOT implement it in this run.

## 24. Environment / Regression

unit:
Python 3.10.20; package-set hash unchanged (`3a119880cd4d661259d9476b0d3224302e086ae899ca77250497b5f42fbf6f9b`).

DexJoCo:
Python 3.11.16; package-set hash unchanged (`7406008d77c52571b64f2c7fdf36ed35a62da160e2eba4b9d091ac9f85d82f78`).

S4.2 hashes:
PASS; all nine accepted checkpoint identities, Vision files, and tracked configs are byte-identical.

RoboCasa:
EGL smoke PASS.

DexJoCo submodule:
Clean at `8d23b0fab23b17a58c4b55f3942e17013aaf8267`; nested Diffusion Policy remains UNINITIALIZED.

Unit tests:
560 passed, 1 skipped, 0 failed.

DexJoCo tests:
18 passed, 0 failed.

Failures:
One R12 environment failure: AF_UNIX endpoint exceeded the platform limit. Regression failures: 0.

## 25. Milestone Status

M3:
UNCHANGED

S4.2:
COMPLETE

Historical first S4.3-0:
FAILED — PRESERVED

S4.3-PD:
COMPLETE

Restarted S4.3-0:
COMPLETE

S4.3-1:
COMPLETE

S4.3-2:
FAILED — S4_3_2_ENVIRONMENT_FAIL

S4.3:
IN PROGRESS

M4:
NOT ESTABLISHED

## 26. Stop Point

STOP AFTER S4.3-2.

DO NOT TRAIN DIFFUSION POLICY.
DO NOT START S4.3-3.
DO NOT TRAIN π0.5.
DO NOT TRAIN GR00T.
DO NOT PORT RH56DFTP.
DO NOT PORT FLEXIV.
DO NOT PUSH.
DO NOT CREATE PR.
