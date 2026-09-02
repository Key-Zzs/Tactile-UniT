# Restarted S4.3 Causal ACT Policy Benchmark

## Historical boundary and restart basis

The first S4.3-0 result remains `S4_3_0_DEMONSTRATION_CONTRACT_FAIL`. Its 210
S4.2 Contact-probing episodes produced zero native task successes and remain
invalid as behavior-cloning experts. This restart does not alter that result and
does not admit those episodes into policy training.

The restart basis is the separately completed
`S4_3_PD_COMPLETE_POLICY_DATA_READY` acquisition. The frozen policy corpus has
290 successful TRAIN episodes (90 pinch-tongs, 100 hammer-nail, 100
click-mouse) and 70 successful DEV episodes (25, 20, and 25 respectively).
Fresh R1 content verification rehashed all 375 formal attempts, including the
15 failed acquisition attempts retained for audit only. The combined frozen
content identity is
`a334af12ab2c5bbfbd46b47e1167f51fe194c97bbd69fb945d760f96b648d7f4`.

## Frozen policy protocol

The primary endpoint is the equal-weight three-task macro native success rate.
The four and only four variants are P0 (vision/proprio), P1 (plus raw causal
tactile history), P2 (plus frozen Contact-State), and P3 (the P2 inference
architecture with a training-only frozen causal Tactile-UniT auxiliary).
Training seeds are 0, 1, and 2. POLICY_EVAL_V1 freezes 30 fresh random reset
specifications per task before implementation or training. Replanning predicts
27 policy Actions and executes the first five.

Successful TRAIN episode lengths fix timeouts at 823 control steps for
pinch-tongs, 635 for hammer-nail, and the clamped maximum 1000 for click-mouse.
The common 26-sample warm-up is excluded from these timers. Native DexJoCo
success sources are byte-identical to S4.3-PD and are not redefined.

Hammer-nail is preregistered as
`HAMMER_NAIL_MAPPED_TACTILE_INACTIVE`: its frozen mapped tactile diagnostic has
zero boundary events and zero peak force despite valid native task success.
Pinch-tongs and click-mouse therefore form the preregistered, secondary
tactile-active macro analysis. That secondary result cannot replace a negative
three-task primary endpoint.

Success contrasts use a paired hierarchical bootstrap over tasks, matched
training seeds, and matched reset IDs. A material improvement requires at least
+5 percentage points and a strictly positive 95% CI; material hurt requires at
least -5 points and a strictly negative CI. These gates are frozen before any
rollout performance is observed.

## Strict causal integration

Vision is the already accepted frozen current-frame Original-UniT DINO boundary
on `I_t` only. The S4.2 two-frame transition representation and every future
frame are forbidden. P1 sees only `T[t-25:t]`; P2/P3 use the exact frozen
`E_T^sim(T[t-25:t])`. P3 passes its own bounded, physical planned Action through
frozen A0, the B3 Action path, and the frozen causal A+H predictor. Its target is
the same expert transition encoded through frozen Contact-State, C3, and B3 and
is training-only. The planned Action cannot be detached; every S4.2 parameter
remains frozen.

No compatible ACT implementation exists in the repository. The allowed path is
therefore a minimal repository-owned action-chunking CVAE Transformer with the
preregistered 256 hidden width, 32 latent width, four encoder and decoder
layers, eight heads, and deterministic zero-latent evaluation. P0--P3 share the
same vision boundary, core, decoder, optimizer, training budget, checkpoint
selection, and execution contract.

The tracked freeze files are
`configs/simulation/s4_3_restart_policy_protocol.json`,
`configs/simulation/s4_3_restart_act_protocol.json`, and
`configs/simulation/s4_3_policy_eval_v1.json`. The executable R0--R3 audit
records their combined protocol identity in
`.local/artifacts/simulation/s4_3_restart/s4_3_restart_protocol_freeze.json`.

S4.3-3 Diffusion Policy and all later stages remain out of scope until the ACT
benchmark is complete and a readiness decision has been recorded.
