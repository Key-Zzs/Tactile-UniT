# PI2B Evidence Integration

## Status and provenance

Track A and Track B were fetched from `origin`, pinned to exact commits, and
merged locally into `develop/sim-benchmark` in that order. The integration
preserves both histories rather than squashing them:

- common base: `8f39aed123adf0a8b7241e75472e678355444e5e`;
- Track A final: `ac750461e19ec6f287bbe4ee28dcb751dd268485`;
- Track A merge: `d78108d0cb623a109179c7e862ff8d461a2e9848`;
- Track B final/closure: `d5f52a5ac63e1662820da6855c9177c2abf218e2`;
- Track B merge: `f0af6e14c65d2c799040701a84740d811fe05106`.

The two tracks add disjoint tracked paths and merged without conflicts or
submodule changes. Track A's historical evaluator and Track B's representation
study answer different questions; their results are not combined into a
teacher-to-policy causal claim. This integration has not been pushed.

## Track A: policy stability across training seeds

Track A is `COMPLETE_VALID_THREE_SEEDS_LIMITED`. It reuses five verified seed42
checkpoints and adds exactly ten seed43/44 30k runs. All 15 checkpoints use the
same 200 reset identities from blocks 16--19, yielding 3,000 canonical outcomes.

| Model | seed42 | seed43 | seed44 |
|---|---:|---:|---:|
| B0 | 47/200 | 82/200 | 70/200 |
| B_VA27 | 56/200 | 97/200 | 86/200 |
| B1 | 22/200 | 41/200 | 42/200 |
| B_HVA | 82/200 | 1/200 | 50/200 |
| B2 | 45/200 | 14/200 | 68/200 |

The H increment under VA27 changes from +13 percentage points at seed42 to
-48/-18 at seeds43/44. The B2-minus-B_HVA target contrast changes from -18.5
to +6.5/+9 points. Therefore H conditioning and VA/VAC target ordering are
`MIXED_ACROSS_TRAINING_SEEDS`. B1-minus-B0 is negative at every tested seed
(-12.5/-20.5/-14 points), which is adverse evidence in this task but not a
universal claim that tactile input is useless.

Conditional reset intervals condition on the fifteen trained checkpoints.
The seed-by-reset analysis has only three outer training-seed units. Neither
analysis establishes equivalence, noninferiority, or performance of future
training runs. Seed42 was development-exposed; seeds43/44 were preregistered
additions. The 2,197 failures remain part of the evidence.

## Track B: matched teacher evidence

Track B remains
`VAC_CONTACT_CAPABILITY_WITH_NO_DETECTED_VA_REGRESSION_LIMITED_PRECISION`.
The matched teachers have hashes:

- `T_VA_match`: `91ae74d970917535e199822b3fb415997d88b36342dc7178b61efcd7a746c142`;
- `T_VAC_match`: `65ba855fda2175c51b23e58cd36e7889fa891da56a697ad442c3ea926951f698`.

The fresh study has 4,860 pairs from 45 episodes and nine source groups.
Contact pairing and temporal relations are reliable in the tested setting, and
Vision recovery improves. Action recovery has an unfavorable point estimate,
while a Contact-probe increment in common V/A outputs is inconclusive.

Pooled retrieval uses all 4,860 candidates. The registered group-unit result
repeats retrieval inside each 540-row source group. Their differing directions
are `MULTIPLE_DEFINED_ESTIMANDS / CANDIDATE_POOL_DIFFERENCE`, not a bug. The
study uses one teacher seed and does not establish equivalence, noninferiority,
cross-seed stability, or policy utility.

## Separation and next diagnostic

Track A used historical frozen teachers and never loaded the new Track B
teachers. Consequently, the integrated evidence cannot claim that the matched
VAC teacher helped or harmed policy control.

S4.3-PI2S is a post-hoc development diagnosis of the H interface, physical
time/mask contract, prefix conditioning, optimization signals, and coarse
closed-loop failure location. It has zero training and optimizer-update budget.
Any development rollouts remain separate from the 3,000 formal outcomes, and
an inconclusive diagnosis is a valid endpoint.

Machine-readable claims and counts are in
`configs/simulation/pi2s/evidence_ledger.json`.
