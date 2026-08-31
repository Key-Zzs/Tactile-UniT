# S4.2 Sim Contact Representation and Cross-Modal Bridge Protocol

## Freeze boundary

This protocol was frozen at starting HEAD `8629e7b` before formal S4.2 data
generation, representation training, candidate selection, or formal test-data
access. The pinned DexJoCo commit is `8d23b0f`. The S4.1 5-region, 30-D
right-hand tactile schema, 22-D policy action, 50 Hz control rate, 26-sample
history, and 27-step transition offset remain unchanged.

The executable sources of truth are:

- `configs/simulation/s4_2_protocol.json`;
- `configs/simulation/s4_2_dataset_contract.json`;
- `configs/simulation/s4_2_dataset_split_manifest.json`;
- `configs/simulation/s4_2_model_candidate_registry.json`;
- `configs/simulation/s4_2_selection_contract.json`.

The formal test split is forbidden to training and selection code until every
candidate and checkpoint is frozen and the ignored `pretest_freeze.json` has
been written and hashed. One locked evaluation is allowed. A second run may
only verify deterministic equality.

## Task and dataset decision

The runtime audit checks all six registered single-arm tasks and confirms the
shared right-Allegro hand bodies, 23-D environment action, at-least-23-D state,
and 0.02 s control period. S4.2 selects `pinch_tongs`, `hammer_nail`, and
`click_mouse`. This preserves the frozen 5-region/22-D contract and avoids any
bimanual schema expansion.

The formal dataset contains 20 source trajectories per task and five bounded
perturbations per source, for 300 episodes of 160 control steps. The source
group, not the augmented episode, is the split unit. Groups 00--13 are train,
14--16 are validation, and 17--19 are test independently within each task,
giving an exact 70/15/15 group split. Every episode records its source type,
source trajectory, task, seed, randomization identity and parameters, and
perturbation identity. RGB, NPZ, checkpoints, caches, and logs remain ignored.

The repository-owned interaction family uses only reset-time current robot
state and reset-time object pose. It contains approach, compression, stable and
tangential contact, release, and final free-space phases. It does not use a
future state, reward, success, or trained policy. Formal data must pass the
minimum size, leakage, schema, boundary-count, dynamic-fraction, finite-value,
decode, timing, duplicate, and deterministic-rebuild gates before training.

## Randomization and OOD decision

Seeded native reset ranges provide table height and object position/orientation
variation. The pinned per-task dynamics randomization is enabled for ID data.
Visual randomization is disabled so the frozen Original UniT front-camera
preprocessing remains one documented family.

`OOD_DYNAMICS_DEFERRED` is preregistered. The three selected pinned task
classes do not expose a common externally parameterized held-out range across
mass, friction, and pose. Inventing a new range-injection API would no longer
be an audit of the pinned task API. This is a warning that limits an external
generalization claim, not a core S4.2 blocker.

## Representation candidates

The Contact teacher maps normalized `[B,26,30]` histories (plus deterministic
first differences internally) to 256 dimensions. Three fixed baselines are
compared with four preregistered residual-dilated-TCN teacher trials spanning
only N0/N1 force normalization and two reconstruction weights. Its target is
strictly future tactile `t+1:t+13`, a 0.26 s span.

The frozen teacher supplies current and future Contact state for the 0.54 s
transition. Contact dynamics maps a state pair to `[B,8,32]` and compares two
delta-loss weights against persistence, current-only, and delta-MLP baselines.
The Action representation consumes current action-aligned 22-D state and
`a_t:a_t+26`, constructs absolute/relative/difference features, and compares a
flat MLP, TCN, and small query Transformer. It must prove exact temporal order.
Vision uses the immutable Original UniT DINO/M-Former/pre-RQ path on `I_t` and
`I_t+27`; it is an offline future-derived transition teacher, not a causal
runtime input.

## Bridge and downstream hypotheses

The bridge ordering is fixed: native diagnostic, frozen-M3 zero shot, small
residual adapters into frozen M3, then a bounded sim-specific shared-slot
space. All modality encoders are independently callable. The frozen M3 shared
projector checkpoint is the manifest-accepted C2 checkpoint with SHA-256
`454d7a33df20e5329e2be4804760dad211462e3eb405c16141f061f0c1ef113a`.
Only validation may select among the five bridge training trials.

If a bridge passes, S4.2 compares continuous Contact with a same-capacity
two-stage 128-code-per-stage RQ, audits shared/private recovery, evaluates
V/A/H source sufficiency and invalid-H/Action controls, and calibrates scalar
uncertainty for full and missing-H modes. A three-stage RQ and formal OOD split
are not canonical candidates.

## Failure and stop policy

Stages execute in the hard order S4.2-0 through S4.2-8. The first hard failure
stops every dependent stage. Test thresholds, splits, normalization, and
candidates cannot be changed to repair a failure. Passed stages may be
committed locally; a useful failed stage may receive an honestly labeled
diagnostic commit. No push, PR, merge, tag, branch, worktree, policy training,
S4.3 work, RH56DFTP port, or Flexiv port is permitted.

## S4.2-1 dataset result

The frozen collector produced 300 episodes, 100 per task, and 32,400 valid
transition anchors. The source-group split contains 210/45/45 episodes and
22,680/4,860/4,860 train/validation/test anchors. The full two-pass audit
decoded and checksummed all 48,000 frames, found no nonfinite data, duplicate
episode or pair ID, timing discontinuity, schema error, or source-group
leakage, and rebuilt the manifest deterministically. The test quality audit
counted 1,139 free-to-contact and 1,286 contact-to-free anchors. Both train-fit
dynamic-threshold candidates exceed 25% dynamic coverage on every split.

The dataset decision is `DATASET_READY_WITH_WARNINGS`. Right-thumb activity is
zero in this bounded scripted corpus, while palm, index, middle, and ring are
active. The thumb remains present in the frozen 30-D schema; subsequent
per-region probes must report its unusable class balance rather than dropping
it. Formal OOD dynamics also remains deferred as preregistered. Neither item
is a frozen S4.2-1 hard-gate failure.
