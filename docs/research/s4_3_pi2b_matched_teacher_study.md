# S4.3-PI2B-T matched VA/VAC teacher study

## Scope and preregistration

This branch isolates the effect of adding Contact supervision to a common
Vision/Action shared representation. It does not repeat Track A, train a
policy, train pi0.5, replace a historical teacher, or write the main
`PAPER_CORE`. The only formal training budget is the fixed-seed pair
`T_VA_match` and `T_VAC_match`, one run each.

Both teachers reuse the selected S4.2 B3 slot-projector and native-recovery
recipe. Their common parameters are the shared physical slots, Vision and
Action projectors, and Vision and Action recovery heads. Component-derived RNG
streams make these tensors byte-identical before the first optimizer step.
The two runs use the same 22,680 ordered TRAIN pairs, 800 updates, batch size
512, batch RNG schedule, AdamW learning rate and weight decay, and global
gradient clipping rule. The VA loader exposes no Contact tensor or label.

The loss is factorized as an unchanged VA block plus an additive Contact block:

```text
L_VA  = L_align(V,A) + 5 L_rec(V,A)
        + 0.25 L_rel(V,A) + 0.05 L_var(V,A)
L_VAC = L_VA + L_align({V,A},C) + 5 L_rec(C)
        + 0.25 L_rel(C) + 0.05 L_var(C)
```

The temperature is 0.07. There is no Contact-derived resampling or dynamic
weight. In particular, adding two Contact-involving modality pairs does not
divide or otherwise weaken the VA term. The global clip definition is common;
any difference in whether it activates is a measured mechanism of the Contact
treatment.

## Matching gates

Formal training is forbidden until the audit proves common structure and
byte-identical initialization, exact C-disabled common forward/loss/gradient
and one-step AdamW parity, identical pair ordering and update budgets, denial
of Contact fields to the VA loader, independent encodability, and frozen input
identity. The VAC Contact loss must reach the common path, while the VA block
must not reach Contact-private parameters.

This supports the labels
`COMMON_PATH_INITIALIZATION_MATCHED`,
`COMMON_DATA_AND_UPDATE_BUDGET_MATCHED`, and
`CONTACT_SUPERVISION_IS_TREATMENT`. Total parameters and compute necessarily
differ and are reported. Any scientific result remains
`FIXED_TEACHER_TRAINING_SEED_ONLY`.

## Data and confirmation boundary

TRAIN and DEV reuse the accepted S4.2 paired native caches with episode-local
raw/control `t -> t+27` transitions. DEV is historically exposed and is used
only for diagnostics. The independent confirmation cohort is preregistered as
the next unused source groups 23--25 for each of the three tasks, five episodes
per group: 45 episodes, nine source groups, and 4,860 expected legal pairs.
Its identities are reserved before training. Generation uses the accepted
S4.2 generator and the three frozen FORMAL_TEST_V2 parameter profiles without
teacher-performance filtering.

Confirmation performance is inaccessible until both final checkpoint hashes,
the data-integrity manifest, probe protocol, and evaluator hash are bound in a
pretest freeze. Statistical resampling uses source groups with all overlapping
windows kept together. The study does not preregister an equivalence or
noninferiority margin, so a nonsignificant difference cannot prove
preservation.

## Result status

`COMPLETE_VALID`. The frozen analysis decision is
`VAC_CONTACT_CAPABILITY_WITH_NO_DETECTED_VA_REGRESSION_LIMITED_PRECISION`.
Engineering, matching, noncollapse, and the independent statistics audit all
passed. The result remains a fixed-teacher-training-seed comparison and does
not establish equivalence, noninferiority, or policy utility.

The two canonical runs both completed 800 optimizer updates and 409,600 sample
exposures. `T_VA_match` has 165,888 trainable parameters and used 0.00440
measured GPU-hours; `T_VAC_match` has 248,704 trainable parameters and used
0.00610 measured GPU-hours. Their batch schedule digest is identical. The
final checkpoint SHA-256 values are:

```text
T_VA_match:  91ae74d970917535e199822b3fb415997d88b36342dc7178b61efcd7a746c142
T_VAC_match: 65ba855fda2175c51b23e58cd36e7889fa891da56a697ad442c3ea926951f698
```

The independent confirmation cohort contains 4,860 legal `t -> t+27` pairs
from 45 episodes and nine new source groups, with zero pair or source-group
overlap with prior splits. On this cohort, VAC reduced Vision native-recovery
MSE by 0.000697 (95% stratified source-group bootstrap CI
[-0.000906, -0.000461]) and increased Action native-recovery MSE by 0.000618
([0.000084, 0.001201]); the latter did not survive Holm correction
(`p_holm=0.0624`). Mean bidirectional V/A R@10 changed by -0.000823
([-0.013066, 0.012243]), and mean bidirectional MRR changed by -0.002000
([-0.009300, 0.005619]). VAC improved the true raw-Action reversal margin by
0.039518 ([0.033757, 0.045948], Holm-significant). These mixed effects support
only `NO_DETECTED_VA_REGRESSION_WITH_LIMITED_PRECISION`, not a preservation or
equivalence claim.

The VAC-only Contact path was reliable. Vision-Contact and Action-Contact
paired margins were 0.596750 ([0.584791, 0.608408]) and 0.645059
([0.630299, 0.659259]); their temporal reversal margins were 0.564512
([0.553764, 0.574864]) and 0.594671 ([0.585072, 0.604751]). Contact native
recovery reached MSE 0.124960, R2 0.847829, and cosine 0.936130. In contrast,
the paired effect on the preregistered mean common-V/A Contact-probe macro-F1
was 0.008293 ([-0.001979, 0.019886], `p_holm=0.3012`), so no reliable common
V/A Contact-readout increment was detected.

Boundary and task effects are heterogeneous: boundary Vision recovery favored
VAC, while boundary Action recovery was worse; V/A retrieval at boundaries
was inconclusive. All reported latent geometries had zero near-zero-variance
fraction, but effective and source-relative ranks remained modest and the
fresh set has only three source groups per task. Region-transition probes are
N/A because the accepted TRAIN cache lacks the corresponding training labels.

New teachers remain research candidates. They were not connected to Track A,
no pi0.5 or policy was trained, and policy utility is `NOT_TESTED`. The main
`PAPER_CORE` was not edited; integration is limited to the separately generated
paper delta and handoff artifacts.
