# S4.3-PI2B Policy Multi-Seed Confirmation

Status: **PROTOCOL FROZEN; NEW TRAINING NOT YET REPORTED**

This Track A study fixes five recipes—B0, B_VA27, B1, B_HVA, and B2—and
extends the existing seed42 checkpoints with independently initialized seeds43
and 44. It authorizes exactly ten new 30,000-step policy runs, zero teacher
runs, and one shared 200-reset formal cohort for all fifteen checkpoints.

The exact model definitions, lambdas, training order, resource rules, and
statistical hierarchy are machine-readable in
`configs/simulation/pi2b_policy/`. Historical BVA remains a +16-control-tick,
0.32-second result and is not part of this matched VA27 cohort.

The clean VA27 and frozen Contact/VAC targets are training-only supervision.
No new Track B teacher or Track B performance result may enter model selection,
and formal evaluation cannot start until all ten new final checkpoints pass
the completion and cold-load gates. Training uses Prompt1's shared barrier and
GPU locks; formal evaluation will use its exclusive barrier.

Seed42 was selected and exposed during earlier development. Seeds43 and 44 are
the preregistered independent additions. Three training seeds provide only a
limited view of training variance, so the final report must retain per-seed
effects, shared-reset conditional intervals, a two-way sensitivity analysis,
and a seeds43/44-only analysis without claiming population-level certainty.

No result is stated in this document before the frozen runs and 3000 canonical
rollouts are complete. Negative, mixed, or inconclusive results remain valid
completion outcomes and do not authorize replacement runs.
