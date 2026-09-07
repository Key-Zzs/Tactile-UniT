# S4.3-PI0 official DexJoCo pi0.5 reproduction

This stage reproduces the official DexJoCo pi0.5 baseline for only the
single-arm `pinch_tongs` task under `rand_obj`. The frozen official source is
`third_party/dexjoco` at commit
`8d23b0fab23b17a58c4b55f3942e17013aaf8267`; the submodule is read-only.

## Locked scope

- Model: pi0.5 with `gemma_2b_lora` and `gemma_300m_lora`.
- Inputs: front RGB, wrist RGB, 23-dimensional robot state, and the official
  task prompt. No tactile or Tactile-UniT input is used.
- Output: 30-step action chunks. The dataset stores 22-dimensional actions;
  the model pads to its official 32-dimensional internal action width and the
  official single-arm output transform returns the first 22 dimensions.
- Data: only the official `pinch_tongs` subtree from
  `DexJoCo/DexJoCo-Datasets-LeRobot`, revision
  `5a57c54e55dc5858dd9fb949c5f67c0c9716e6b3`.
- Initialization: only `pi05_base`, mirrored from the official OpenPI object
  tree. The 44-dimensional dual-arm base model is excluded.
- Training: one run, seed 42, batch size 32, 30,000 steps, AdamW, 10,000-step
  warmup, constant-endpoint 5e-5 cosine schedule, no EMA, and checkpoints every
  10,000 steps. No hyperparameter sweep or checkpoint selection by rollout.
- Runtime parallelism: two replicated workers with `fsdp_devices=1`, so the
  official global batch of 32 is data-sharded to 16 samples on each GPU. A
  local startup comparison measured approximately 4.4 seconds/step on one GPU
  and 2.6 seconds/step on two GPUs (about 1.69x faster). No algorithmic
  hyperparameter changed.
- Evaluation: the frozen official DexJoCo client/server pipeline, 20 episodes,
  seed 0, `rand_full=false`, and dynamics randomization disabled.

The machine-readable lock is
`configs/simulation/s4_3_pi0_official_pi05_protocol.json`. Local datasets,
weights, normalization statistics, logs, checkpoints, videos, and manifests
remain under `.local/` and are not committed.

## Dataflow

```text
official LeRobot pinch_tongs
  front RGB + wrist RGB + state[23] + official prompt + action[22]
      -> official SingleArmDataConfig
      -> resize 224x224 + quantile normalization + action padding
      -> pi0.5 Gemma 2B LoRA + 300M action-expert LoRA
      -> action chunk [30, 32]
      -> official SingleArmOutputs -> [30, 22]
      -> official DexJoCo wrapper -> environment action [23]
```

## State and action contract

```text
policy state [23]       = TCP xyz [3] + scalar-first quaternion [4] + hand [16]
policy action [22]      = TCP xyz [3] + rotation vector [3] + hand [16]
environment action [23] = TCP xyz [3] + scalar-first quaternion [4] + hand [16]
```

The policy-to-environment rotation conversion is owned by the official
`DexJoCoOpenPIEnv._process_action` implementation.

## Baseline checkpoint result

The official released `pinch_tongs` checkpoint passed cold-load, shape,
finiteness, and rollout gates. Under the locked local 20-episode seed-0
protocol it succeeded in 4/20 episodes (20.0%; Wilson 95% CI approximately
8.1% to 41.6%). The paper reports 24.0% +/- 6.9% for `rand-obj`; that paper
number aggregates training seeds and uses 50 evaluations per task, so it is a
reference rather than an identical statistical protocol.

## Local training completion

The one canonical run completed all 30,000 optimizer steps using data
parallelism on two GPUs, with the official global batch size of 32 split into
16 samples per device. Runtime was 21:54:58. All 300 logged metric records
were finite; the final logged record at step 29,900 had loss 0.0097 and
gradient norm 0.0630. Checkpoints were created only at steps 10,000, 20,000,
and the canonical final directory `29999`. The final train state restores to
step 30,000, its 28-file manifest totals 9,562,228,955 bytes, and its official
pi0.5 LoRA parameter tree cold-loads with all 71 array leaves finite.

The training curve is diagnostic evidence, not a checkpoint-selection
criterion. Only the frozen final checkpoint was evaluated.

## Local checkpoint result

The final local checkpoint passed the official server/runtime contract and
completed the same seed-0 20-episode sequence. It succeeded in 2/20 episodes
(10.0%; Wilson 95% CI 2.79% to 30.10%). Successes occurred on paired episode
IDs 07 and 19. The official released checkpoint succeeded on IDs 00, 05, 06,
and 16, so the paired table contains 0 both-success, 4 official-only success,
2 local-only success, and 14 both-failure episodes. The exact two-sided
McNemar p-value is 0.6875.

The official and local Wilson intervals overlap, and the local interval also
overlaps the paper's 24.0% +/- 6.9% reference band. This is statistically and
operationally compatible under the predeclared reduced-cost criterion; it is
not evidence that the two policies have identical performance.

## Decision and boundary

The PI0 decision is
`S4_3_PI0_FULL_OFFICIAL_BASELINE_REPRODUCTION`. The corresponding S4.3-PI1
readiness is `READY_WITH_WARNINGS`: the official checkpoint, official 30k
training, checkpoint load/serve path, state/action/camera/prompt contract, and
official evaluator all passed, but only one task and 20 fixed-seed episodes
were evaluated. `rand_full`, dynamics randomization, other tasks, and the full
publication table were not reproduced.

The allowed claim is therefore limited to: "Official DexJoCo pi0.5 rand-obj
pipeline reproduced on pinch_tongs under a reduced-cost one-task, 20-episode
protocol." PI0 does not add Tactile-UniT to pi0.5 and does not start PI1.
