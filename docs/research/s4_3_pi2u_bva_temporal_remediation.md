# S4.3-PI2U BVA temporal remediation

The completion audit found a unit error in the first BVA target cache. The
canonical S4.1/S4.2 representation transition is 27 simulator control steps at
50 Hz, or 0.54 seconds. The official `pinch_tongs` policy dataset is natively
sampled at 30 Hz. Treating 27 dataset rows as 27 simulator control steps made
the first target horizon 0.9 seconds.

That first seed-42 checkpoint and its seed-5 paired evaluation are therefore
excluded from the formal PI2U claim. They remain preserved as superseded local
evidence; no metric from them may select or tune the correction.

The corrected mapping keeps the frozen physical horizon and native video
observations. For a policy row at time `i / 30`, the future image is row
`i + round(0.54 * 30) = i + 16`, at 0.533333 seconds. Its 0.006667-second
deviation from 0.54 seconds is the exact sampling-rounding error already frozen
in the S4.1 timing contract. Pixel interpolation is not used. In canonical
simulator units the transition remains `t_control -> t_control + 27`; in source
dataset units it is `i_frame -> i_frame + 16`.

Only the future-Vision target timing changes. The VA-only bridge checkpoint,
official pi0.5 base, seed 42, 30,000-step budget, optimizer, loss, auxiliary
head, absence of Contact/tactile inputs, and four-batch lambda calibration rule
remain unchanged. After retraining, evaluator seed 6 is the first unexposed
sequence and is frozen before any corrected-checkpoint rollout. BUniT and PI2B
remain outside this remediation.
