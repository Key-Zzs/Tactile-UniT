# S4.2-DS Contact-Dynamics Representation Sufficiency

S4.2-DS is a new representation-selection question. It does not reopen the
historical reconstruction-superiority hypothesis. Original S4.2-3 remains
`S4_2_3_CONTACT_DYNAMICS_FAIL`, and bounded S4.2-DR remains
`S4_2DR_DYNAMICS_REMEDIATION_FAIL`. The historical 10% dynamic-MSE threshold
is unchanged and is not a canonical-selection gate here.

## DS data boundary

Selection uses only original TRAIN source groups, deterministically divided by
`(task, source_trajectory_id)` into DS-TRAIN and DS-DEV. The realized split is
33/9 groups (17,820/4,860 pairs), with three DS-DEV source trajectories per
task. Original formal validation is reserved for one no-retuning follow-up
confirmation after selection. Formal test model metrics remain unopened.

## Frozen latent contracts

C2 is a representation candidate, not only a predictor baseline. Its
`DeltaMLPEncoder` deterministically maps
`concat(h_current, h_future - h_current)` to an explicit `[B,8,32]` bottleneck.
The shared `LatentTransitionDecoder` consumes this bottleneck after it has been
independently produced. It has no batch-peer, task, reward, success, label, or
decoder-hidden-state dependency. C3 exposes the same stable native shape from
explicit current, future, and delta projections. Both use identity common
adapters with zero trainable parameters.

On DS-DEV, C2 contact-transition/force-trend macro-F1 is 0.975715/0.954531;
C3 is 0.976762/0.954150. Both pass finite, deterministic, no-collapse,
transition-specificity, and information-necessity gates. C3's semantic gains
are not material: contact F1 difference is 0.001047 with CI crossing zero, and
force F1 difference is -0.000381 with CI crossing zero. C3 does have a frozen
future-recovery advantage of 7.017%, with a positive paired bootstrap CI;
whether the selected canonical representation is sufficient for VAC still
depends on the preregistered equal-budget bridge gates.

## Frozen pilot and selection protocol

The full protocol is
`configs/simulation/s4_2ds_representation_selection.json`. It freezes one small
27x22 Action TCN/query autoencoder, the immutable Original UniT Vision path on
`I_t, I_t+27`, and one identical independent residual shared-slot bridge family
for each eligible Contact candidate. No architecture or hyperparameter sweep is
allowed. The bridge is trained on DS-TRAIN and evaluated on DS-DEV. If both
candidates pass, simpler C2 is preferred only when C3 has no material semantic,
bridge, or reconstruction advantage. If neither passes, S4.2-DS stops with
`CONTACT_REPRESENTATION_INSUFFICIENT`.

The DS Action/Vision and bridge components are pilot infrastructure only. They
do not complete formal S4.2-4 or S4.2-5.

## Final result

Both equal-budget 25,344-parameter bridge pilots passed all DS-DEV gates. C3
did not establish a material semantic or bridge advantage: its mean
Contact-involving R@10 and MRR were only 1.0247x and 1.0212x C2. C3 did,
however, satisfy the preregistered material reconstruction clause through a
7.017% reduction in frozen future-recovery MSE with a positive paired
bootstrap interval. The frozen selection rule therefore chooses C3 as
`C3_REPRESENTATION_UTILITY_SELECTED`.

The selected C3 and frozen pilot bridge then passed the single no-retuning
follow-up formal-validation confirmation. Native contact-transition and
force-trend macro-F1 were 0.977384 and 0.949815; all V-C/A-C retrieval gates,
retention gates, information-necessity controls, temporal-specificity checks,
and collapse checks passed. Formal test model metrics were not loaded.

Final S4.2-DS decision:
`S4_2DS_C3_REPRESENTATION_UTILITY_SELECTED`. This does not change the failed
historical 10% reconstruction-superiority result. Formal S4.2-4 is ready;
formal S4.2-5 is ready only after S4.2-4. Neither formal stage was implemented
here.
