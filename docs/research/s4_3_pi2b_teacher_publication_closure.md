# S4.3-PI2B-T-C matched VA/VAC teacher publication closure

## 1. Scope and frozen decision

This document is the publication-facing closure for the completed matched
teacher study. It adds no teacher run, checkpoint selection, inference pass,
policy experiment, scientific candidate, or Track A integration authority.
The frozen study status remains `COMPLETE_VALID`, and its scientific decision
remains
`VAC_CONTACT_CAPABILITY_WITH_NO_DETECTED_VA_REGRESSION_LIMITED_PRECISION`.

The closure uses only the frozen artifacts under
`$PI2B_TEACHER_ROOT/artifacts/` and writes supplemental artifacts under
`$PI2B_TEACHER_ROOT/artifacts/posthoc_readonly_closure/`. The historical
`plots/study_summary.png` and the read-only `PAPER_CORE` snapshot are not
modified.

## 2. Research question

Under matched common Vision/Action initialization, data exposure, update
budget, and loss scaling, what changes when Contact supervision is added?

This is a representation study, not a policy-utility study. It does not test
whether either teacher improves policy success, and it does not authorize a
teacher replacement or connection to Track A.

## 3. Read-only statistical-scope clarification

### 3.1 Shared retrieval kernel

For query (q_i), its paired positive (c_i), and candidate bank

\[
  C = \{c_j\}_{j=1}^{N},
\]

the frozen implementation computes

\[
  r_i = 1 + \sum_{j=1}^{N}
  \mathbb{1}[\cos(q_i,c_j) > \cos(q_i,c_i)].
\]

The positive remains in the bank. No same-pair, same-episode, or
overlapping-window candidate is removed. A candidate tied with the positive
does not increase its rank because the comparison is strict. Directional
metrics are

\[
  R@10 = \frac{1}{N}\sum_i \mathbb{1}[r_i \leq 10], \qquad
  MRR = \frac{1}{N}\sum_i \frac{1}{r_i},
\]

and the reported bidirectional value is the equal mean of V-to-A and A-to-V.

### 3.2 Pooled descriptive estimand

Pooled retrieval uses all 4,860 fresh pairs as queries and all 4,860 fresh
candidates as the bank. Every query has equal weight inside a direction and
the two directions have equal weight. It is a descriptive measurement without
a preregistered confidence interval.

| Metric | VA | VAC | VAC - VA |
| --- | ---: | ---: | ---: |
| Bidirectional R@10 | 0.186111 | 0.194239 | +0.008128 |
| Bidirectional MRR | 0.083103 | 0.085116 | +0.002013 |

### 3.3 Registered source-group estimand

The formal analysis reruns retrieval separately inside each of nine fresh
source groups. Each group has 540 rows, so a query is ranked against a
540-candidate bank from its own source group. If (m_g^M) is the
bidirectional metric for model (M) in group (g), the registered effect is

\[
  \Delta_{group} = \frac{1}{9}\sum_g
  \left(m_g^{VAC} - m_g^{VA}\right).
\]

All overlapping windows remain together inside their source-group unit.
Uncertainty comes from 10,000 paired bootstrap draws that resample the three
source groups with replacement inside each of the three task strata. The
three-groups-per-task design gives equal group and task weight.

| Metric | VA group macro | VAC group macro | VAC - VA (95% CI) |
| --- | ---: | ---: | ---: |
| Bidirectional R@10 | 0.409465 | 0.408642 | -0.000823 [-0.013066, 0.012243] |
| Bidirectional MRR | 0.163146 | 0.161146 | -0.002000 [-0.009300, 0.005619] |

### 3.4 Task-restricted descriptive macro

As a read-only bridge between those scopes, retrieval was also summarized
from the already-saved task endpoints. Each query is ranked against the 1,620
candidates in its task; the three task values are then averaged equally.

| Metric | VA | VAC | VAC - VA |
| --- | ---: | ---: | ---: |
| Bidirectional R@10 | 0.186111 | 0.194239 | +0.008128 |
| Bidirectional MRR | 0.083145 | 0.085135 | +0.001990 |

These task-restricted values are descriptive and have no registered
confidence interval.

### 3.5 Classification

The closure classification is `MULTIPLE_DEFINED_ESTIMANDS`; the operative
mechanism is `CANDIDATE_POOL_DIFFERENCE`. Rank metrics change when candidates
from other source groups or tasks enter the bank. Balanced group sizes, task
sizes, and group counts rule out macro-versus-micro weighting as the source of
the sign change here. The pooled and group-unit values answer different
questions and are not contradictory measurements.

No implementation mismatch or bug was found. The independent audit exactly
recomputed every saved paired group effect. The preregistered source-group
estimand retains its formal role, and the historical scientific decision does
not change.

## 4. Claim scope

### 4.1 Contact path capability

The allowed conclusion is
`RELIABLE_CONTACT_CROSS_MODAL_CAPABILITY`. VAC Contact native recovery reached
MSE 0.124960, R2 0.847829, and cosine 0.936130. Source-group estimates were:

| Effect | Estimate | 95% CI |
| --- | ---: | ---: |
| Vision-Contact paired margin | 0.596750 | [0.584791, 0.608408] |
| Action-Contact paired margin | 0.645059 | [0.630299, 0.659259] |
| Vision-Contact temporal margin | 0.564512 | [0.553764, 0.574864] |
| Action-Contact temporal margin | 0.594671 | [0.585072, 0.604751] |

These results establish reliable cross-modal pairing and one reversal-based
temporal-discrimination capability. They do not establish full physical-state
recovery, complete Contact semantics, or causal dynamics identification.

### 4.2 Common V/A recovery

For a common presentation direction, define improvement as

\[
  \mathrm{Improvement} = MSE_{VA} - MSE_{VAC}.
\]

Vision improvement was +0.000697 with 95% CI [0.000461, 0.000906]. Action
improvement was -0.000618 [-0.001201, -0.000084], an unfavorable point
estimate. The underlying Action test had unadjusted bootstrap p=0.0208 but
Holm-adjusted p=0.0624, so it was not confirmed under the frozen primary
family. Both effects must be reported together.

### 4.3 Common V/A retrieval and semantic readout

Formal group-level R@10 and MRR effects were inconclusive. The paired mean
common-V/A Contact-probe macro-F1 increment was +0.008293 with 95% CI
[-0.001979, 0.019886] and Holm-adjusted p=0.3012, also inconclusive. Pooled
positive retrieval values do not replace these formal results.

### 4.4 Temporal Action structure

The raw-Action reversal-margin effect was +0.039518 [0.033757, 0.045948] and
was Holm-confirmed. This supports improvement in one temporal-discrimination
metric; it does not mean that VAC universally improves the Action
representation.

## 5. Method draft

### 5.1 Native inputs and frozen encoders

The native Vision (`z_v`), Action (`z_a`), and Contact (`z_c`) inputs are
float32 tensors of shape `[B, 8, 32]`. TRAIN contains 22,680 rows and the fresh
confirmation set contains 4,860. The Original UniT Vision tokenizer, A0
Action encoder, C3 Contact-dynamics encoder, and Contact-State model are
frozen. Teacher fitting consumes cached native tensors; it does not update
those encoders.

### 5.2 Matched teacher architecture

Both teachers contain eight learned shared slots of width 32. Each active
modality has an independent four-head slot resampler with hidden width 64 and
a native recovery head with hidden width 128 and query mixing. Dropout is
zero. `T_VA_match` contains the shared slots, Vision and Action projectors,
and Vision and Action recovery heads. `T_VAC_match` adds only the analogous
Contact projector and recovery head. The models have 165,888 and 248,704
trainable parameters, respectively.

### 5.3 Common initialization

Every component receives a deterministic RNG stream derived from the base
seed and component name. The common parameter names, shapes, and initial bytes
are identical across teachers; their frozen common-initialization digest is
`7d6f12e65bf4fcce21a2ec4a6f39bb1a9040f1a999dfec3cd3eb349dfc4d4c30`.

### 5.4 Matched data and update budget

Both teachers use the same 22,680 ordered TRAIN pairs, the same without-
replacement batch RNG schedule, batch size 512, 800 optimizer updates, and
409,600 sample exposures. Their frozen batch-schedule digest is identical.
Contact causes no resampling and does not change the V/A rows or their order.

### 5.5 Loss

Let symmetric alignment be the mean of the two directional InfoNCE losses,
native and relational terms be modality means, and the temperature be 0.07:

```text
L_VA = L_align(V,A)
       + 5 mean[L_native(V), L_native(A)]
       + 0.25 mean[L_rel(V), L_rel(A)]
       + 0.05 mean[L_var(V), L_var(A)]

L_C  = mean[L_align(V,C), L_align(A,C)]
       + 5 L_native(C) + 0.25 L_rel(C) + 0.05 L_var(C)

L_VAC = L_VA + L_C
```

The V/A block is computed identically and is not diluted by averaging over
the additional modality pairs.

### 5.6 Optimization

Both teachers use AdamW, learning rate 0.0003, weight decay 0.0001, a constant
schedule, and one global active-gradient norm clip at 1.0. Because Contact
adds parameters and gradients, it can change global clipping when enabled;
that is part of the treatment rather than a compute-matching claim.

### 5.7 Matching grade and compute disclosure

The matching grade is
`COMMON_PATH_INITIALIZATION_AND_DATA_UPDATE_BUDGET_MATCHED`. It is not
`FULLY_COMPUTE_MATCHED` or `PARAMETER_MATCHED`. Measured training used 0.00440
GPU-hours for VA and 0.00610 GPU-hours for VAC; total parameters and compute
differ by construction.

### 5.8 C-disabled parity

The preregistered audit verified exact C-disabled common forward outputs, V/A
loss, common gradients, global pre-clip norm, and one-step AdamW updates. It
also verified invariance to Contact perturbation or deletion when Contact is
disabled.

### 5.9 Independent encodability and leakage controls

Every public encoder consumes exactly one native modality and emits one shared
`[B, 8, 32]` tensor. The VA data store denies `z_c`, and the frozen access log
contains only the permitted identifiers plus `z_v` and `z_a`. V/A loss has no
gradient path to Contact-private parameters, while the Contact block reaches
the common shared slots. Fresh pair and source-group identities have zero
overlap with TRAIN and DEV.

### 5.10 Fresh confirmation protocol

The independent confirmation set comprises 45 episodes, nine fresh source
groups, three scripted tasks, five episodes per group, and 4,860 legal
`t -> t+27` pairs. Groups 23--25 were reserved before performance access by a
fixed next-unused-group rule without model-performance filtering.

### 5.11 Statistics

The paired source group is the statistical unit. All episode windows remain
inside their source group. Bootstrap sampling is stratified by task with
10,000 draws. Holm correction applies to the frozen primary common-V/A
family. The study preregistered neither an equivalence nor a noninferiority
margin.

### 5.12 Artifact and analysis independence

Teacher checkpoint identities, evaluation implementation, probe protocol,
native data identities, and the single allowed evaluation were bound in the
pretest freeze. A separate direct NumPy implementation recomputed all paired
group point estimates and per-task means exactly. This closure does not rerun
that evaluation or load a teacher checkpoint into a model.

### 5.13 Explicit limitations

The study has one teacher training seed and only three fresh source groups per
task. It does not test policy utility, cross-teacher-seed stability,
equivalence, or noninferiority. Region-transition probes are N/A because the
accepted TRAIN cache lacks region-transition labels. `right_thumb` has zero
fresh positive transitions, so no thumb-specific claim is supported. Total
parameters and compute differ.

## 6. Results draft

### 6.1 Matching and integrity

All engineering and matching gates passed. Both teachers completed 800
updates with identical pair order and sample exposure. Common initialization
was byte-identical before optimization, and C-disabled forward, loss,
gradient, clipping, and one-step-update parity were exact. The closure audit
revalidated the teacher, dataset, protocol, native encoder, evaluation,
analysis, historical figure, and protected paper hashes without altering any
of them.

### 6.2 Contact capability

Adding Contact supervision established a reliable Contact branch. Contact
native recovery reached MSE 0.124960, R2 0.847829, and cosine 0.936130, while
all four Vision/Action-to-Contact paired and temporal margins had lower
confidence bounds above zero. The scoped conclusion is reliable Contact
cross-modal and reversal-based temporal capability.

### 6.3 Common V/A recovery

Vision native recovery improved by 0.000697 MSE [0.000461, 0.000906] in the
positive-is-improvement orientation. Action native recovery instead had an
unfavorable improvement of -0.000618 [-0.001201, -0.000084], but this result
did not pass the frozen Holm correction (`p_holm=0.0624`). The recovery result
is therefore mixed, not lossless preservation.

### 6.4 Retrieval and semantic effects

The formal source-group R@10 effect was -0.000823 [-0.013066, 0.012243], and
the MRR effect was -0.002000 [-0.009300, 0.005619]. Both were inconclusive.
The common-V/A Contact-probe increment was also inconclusive (+0.008293
[-0.001979, 0.019886], `p_holm=0.3012`). Descriptive pooled effects were
positive but measure a different all-candidate-bank estimand and do not
constitute statistical confirmation.

### 6.5 Boundary and task heterogeneity

At Contact boundaries, Vision recovery favored VAC (VAC-VA MSE -0.000546
[-0.000678, -0.000396]), Action recovery was unfavorable (+0.001190
[0.000226, 0.002153]), and group-level R@10 was inconclusive (+0.001032
[-0.021979, 0.025735]). Descriptive task-level group effects differed: click
R@10 was negative (-0.026235), hammer was positive (+0.023457), and pinch was
approximately neutral (+0.000309). With only three groups per task and no
task-specific frozen intervals, these are not stable task laws. No complete
nine-group dynamic/static or high-force paired effect was saved, so none is
imputed.

### 6.6 Geometry and noncollapse

Every reported fresh V/A geometry had zero near-zero-variance fraction. Full-
set effective ranks were 9.89/8.90 for VA Vision/Action and 12.47/10.73 for VAC
Vision/Action; VAC Contact was 20.08. Source-relative mean ranks were lower
(6.07/5.19 for VA, 6.89/5.70 for VAC, and 9.89 for VAC Contact), preserving
the original caution that the representations are noncollapsed but modest in
source-relative dimensionality.

### 6.7 Limitations and primary wording

The publication wording is:

> Adding Contact supervision produced a reliable Contact branch with strong
> Vision-Contact and Action-Contact paired and temporal relations. Effects on
> the common Vision/Action path were mixed: Vision recovery improved, while
> Action recovery showed an unfavorable point estimate that did not survive
> multiplicity correction; group-level retrieval and Contact-probe increments
> remained inconclusive.

This result must not be described as “VAC improves VA while preserving
Action,” “Contact information is successfully injected into Vision and
Action,” “VAC is lossless,” “VAC is equivalent or non-inferior to VA,” or
“VAC improves policy performance.”

## 7. Publication handoff

The proposed paper delta is
`$PI2B_TEACHER_ROOT/artifacts/posthoc_readonly_closure/paper_delta_v2.md`; its
machine-readable counterpart and the integration handoff are in the same
directory. Six publication-facing figures and captions are under
`artifacts/posthoc_readonly_closure/plots/` and
`artifacts/posthoc_readonly_closure/figure_captions.md`.

Track A integration remains `NOT_AUTHORIZED`. The next integration point is
`WAIT_FOR_TRACK_A_FINAL`, followed by a separate `EVIDENCE_INTEGRATION` task.
