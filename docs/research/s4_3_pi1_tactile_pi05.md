# S4.3-PI1 Tactile-UniT conditioning for official pi0.5

S4.3-PI1 compares two preregistered tactile interventions against the completed
local official DexJoCo pi0.5 `pinch_tongs` reproduction. The study is restricted
to the official 100-episode `rand_obj` dataset and preserves the pinned DexJoCo
submodule at `8d23b0fab23b17a58c4b55f3942e17013aaf8267` byte-for-byte.

## PI1A augmentation result

The raw `pinch_tongs` subtree was selectively mirrored at revision
`125df5c1019e97503929ef1a0ad8f90373436afa`. The official conversion's leading
static-frame rule was reproduced for all 100 episodes, yielding exactly 40,065
rows. Every official state and action value, episode/frame index, task prompt,
and generated 30 Hz timestamp matched the official LeRobot dataset at revision
`5a57c54e55dc5858dd9fb949c5f67c0c9716e6b3`.

The raw demonstrations use one 20 ms simulator control tick per recorded action.
The official 30 Hz timestamp is a dataset convention; it does not change that
one-to-one action/control-tick mapping. Tactile histories therefore contain the
current and preceding 25 simulator control ticks. At the post-trim episode
boundary they use the frozen `LEFT_REPEAT_FIRST` rule and never cross episodes.

Contact extraction uses an isolated read-only MuJoCo data snapshot for every
frame. The live replay follows the untouched raw quaternion action with no state
feedback or action correction; the extraction snapshot restores recorded hand
and object state so contact geometry is aligned without modifying the live
simulation. All 100 episodes passed the frozen TCP, quaternion, hand, object,
table, and finiteness gates.

The augmented dataset is a strict-superset sidecar rather than a rewritten
LeRobot dataset. It stores 30D tactile, 26×30 history, frozen 256D Contact-State,
and the frozen shared 8×32 `t→t+27` Contact transition target. The last 27 rows
of each episode are invalid for the auxiliary target and are explicitly masked.
The original official dataset remains the sole source of RGB, state, action,
prompt, indices, and timestamp.

## Frozen model modes

`NONE` returns the official `openpi.models.pi0.Pi0` class and has no tactile
parameters. It passed fixed-batch zero-tolerance parity for transformed images,
normalized state and actions, prompt tokens, pi0.5 loss, and sampled actions.

`CONTACT_STATE_TOKENS` maps the current precomputed Contact-State through:

```text
LayerNorm(256) → Linear(256,256) → GELU → Linear(256,256)
  → reshape [8,32] → shared Linear(32,2048)
```

The resulting eight valid non-autoregressive tokens are appended to the
observation prefix after the unchanged image/language tokens. They are not part
of the action suffix. The adapter has 199,680 trainable parameters. With the
exact official `pi05_base` loaded, the fixed validation loss was 0.51124 and
the adapter gradient norm was 14.40356, establishing end-to-end conditioning
of the official action loss.

`CONTACT_STATE_TOKENS_PHYSICAL_AUX` has exactly the same inference inputs and
action-generation path. During training only, it averages the final official
action-expert hidden states for the first 27 action positions and applies
`LayerNorm(1024) → Linear(1024,512) → GELU → Linear(512,256) → reshape [8,32]`.
Its 658,176 parameters are below the frozen 1M budget. Masked MSE is computed
against the stop-gradient precomputed shared Contact target. Structural tests
showed finite nonzero gradients into both this head and the action expert, exact
B1/B2 inference equality at common initialization, and a hard inference error
if a future target or validity mask is supplied.

Only two minimal hooks are applied to an ignored writable copy of official
OpenPI: the existing loss returns its already-computed final action hidden
states, and the trainer accepts auxiliary metrics/losses. The versioned patch
does not change the flow-matching operations or mode `NONE`; the frozen
submodule is never patched in place.

## Training and evaluation freeze

B1 and B2 both initialize independently from the same `pi05_base` with seed 42
and use the official 30,000 steps, global batch 32, AdamW, learning-rate
schedule, normalization, and LoRA freeze policy. B2 is not initialized from B1
or B0. Its scalar auxiliary weight will be calculated once, after B1 completes,
from exactly four deterministic seed-42 training batches using
`clamp(0.1 × mean(L_pi05) / mean(L_phys), 1e-3, 1e-1)`; rollout performance is
not involved.

PI1D is preregistered but has not been run. Official evaluator seed 1 is the
smallest unused official DexJoCo evaluator seed in this project; the 50 ordered
reset conditions will be shared by R0, B0, B1, and B2. The primary paired
contrasts are B1−B0, B2−B1, and B2−B0 with Wilson intervals, a fixed-seed
100,000-resample paired bootstrap, paired outcome tables, and exact McNemar
tests. A material effect requires at least 10 percentage points and a paired
95% interval excluding zero in the corresponding direction.

Machine-readable locks are in
`configs/simulation/s4_3_pi1_dataset_tactile_augmentation.json`,
`configs/simulation/s4_3_pi1_modes.json`,
`configs/simulation/s4_3_pi1_training_protocol.json`, and
`configs/simulation/s4_3_pi1d_eval_protocol.json`. Datasets, caches, validation
artifacts, checkpoints, logs, and figures remain under ignored `.local/` roots.
