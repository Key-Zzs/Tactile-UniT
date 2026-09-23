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

`NOT_RUN`. This section is updated only from frozen Track B artifacts. New
teachers remain research candidates and are never connected to Track A by this
study.
