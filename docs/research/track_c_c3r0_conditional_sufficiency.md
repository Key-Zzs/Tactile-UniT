# Full Track C — C3-R0 Conditional Sufficiency and Multimodality Audit

## Status

`COMPLETE — CAUSAL_CONTACT_CONTEXT_REQUIRED`

C3-R0 is a post-C3 diagnostic audit, not an untouched first-look test. It did
not modify the shared space, native encoders, C3-DP predictor, private-residual
contract, or any cached representation. All probe, kNN, and deterministic
ceiling choices were frozen on train plus validation before the locked 17,504
row benchmark was loaded.

The single next-stage recommendation is:

`C3-MS-CC_CAUSAL_CONTEXT_PREDICTION`

That stage was not started. C4, C5, C6/M3, and M3 remain not started or not
established.

## Protocol

- Train: 65,536 immutable C1/C3-DP rows.
- Validation: 8,192 immutable rows.
- Locked benchmark: 17,504 immutable rows.
- Sources: V, A, V+A, H, V+H, A+H, and V+A+H, where H is current-causal
  `h_t^c` and no future Contact state is exposed.
- Probes: the accepted train-standardized balanced ridge family with alpha 10.
- Neighborhoods: train-only reference database, k in {1, 5, 10, 20}, with a
  train-fitted and validation-frozen 64-D PCA per atomic component because both
  permitted GPUs were occupied by unrelated workloads.
- Nonparametric ceilings: 1-NN copy, validation-selected medoid, and explicit
  conditional mean.
- Deterministic ceilings: four bounded trials total (M0/M1 for V+A and
  V+A+H); selected models contain 50,496 and 17,088 trainable parameters.
- Test loading was blocked until `audit_protocol.json`,
  `probe_selection.json`, `knn_protocol.json`, and
  `deterministic_ceiling_selection.json` were written and hashed with
  `test_loaded: false`.

## Locked result

Individual V and A sources remain insufficient. V+A also remains insufficient:
its direct semantic ratios are 0.514450 for Contact and 0.695804 for force, and
its selected deterministic `u_c` ceiling reaches only 0.543068 and 0.697144.

Adding current Contact context changes the result. Direct V+A+H reaches
0.845898 Contact retention and 0.781457 force retention. The selected V+A+H
M1 ceiling reaches 0.869633 Contact retention and 0.792695 force retention,
with free-to-contact recall 0.675439, contact-to-free recall 0.732673, and
future-change macro-F1 0.590058. It passes the pre-registered gate.

The gain over V+A is statistically positive: locked Contact macro-F1 improves
by 0.142975 (bootstrap 95% CI [0.135398, 0.150721]), and force macro-F1 improves
by 0.036256 (95% CI [0.028784, 0.043031]). H is not merely identifying the
current state: V+A+H current-state macro-F1 is 0.936393, while future-change
macro-F1 remains meaningful at 0.572058 in the direct-source confound audit.

Neighborhood evidence does not support genuine conditional multimodality as
the primary cause. At k=10, normalized label entropy falls from 0.397430 for V
and 0.300223 for A to 0.184369 for V+A+H. The oracle `u_c` reference is
0.135969. V+A+H normalized local target variance is 0.264370 of global
variance, compared with 0.075662 in oracle neighborhoods. Mode-preserving
medoids keep more effective rank than means, but do not outperform the mean in
Contact semantics; the strongest legal deterministic V+A+H model passes.

The Action reversal audit reduces direct-A Contact macro-F1 from 0.373361 to
0.359616 and force macro-F1 from 0.460134 to 0.443123. The diagnostic warning
is `ACTION_TEMPORAL_SIGNAL_LOST_BY_C3_SOURCE_USE`; no Action encoder was
changed.

The private residual remains `PRIVATE_RESIDUAL_LARGELY_PRIVATE` and was not a
training target or source. All frozen checkpoint/cache identities match before
and after the audit.

## Decision

Primary: `CAUSAL_CONTACT_CONTEXT_REQUIRED`.

Secondary: `PREDICTOR_OBJECTIVE_BOTTLENECK` (weak V-only direct-to-P2 gap and
rank contraction), but it does not dominate the large, statistically positive
context gain.

Exactly one future path is recommended:
`C3-MS-CC_CAUSAL_CONTEXT_PREDICTION`, targeting shared `u_c` and never the
private residual. C3-R0 stops here.

## Reproduction

```bash
python scripts/tactile_unit/audit_c3r0_source_semantics.py --device cpu
UNIT_FULLDATA_CKPT=/path/to/checkpoint python scripts/tactile_unit/evaluate_c3r0_conditional_sufficiency.py --device cpu
python scripts/tactile_unit/visualize_c3r0_conditional_sufficiency.py
python -m pytest -q tests/tactile_unit/test_c3r0_conditional_sufficiency.py
```
