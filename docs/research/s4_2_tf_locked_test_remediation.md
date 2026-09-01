# S4.2-TF Locked-Test Harness Remediation and Fresh TEST_V2

## Decision

S4.2-TF is complete. The final decisions are:

- **S4.2: COMPLETE**
- **S4.3: READY_WITH_WARNINGS**

The original locked test is classified as `TEST_V1_EXPOSED`. Its original
`STRUCTURAL_FAIL`, 4,860 pair identities, and cache bytes are preserved. It is
not eligible for the scientific decision, and no post-fix TEST_V1 engineering
diagnostic was run.

## Evaluator remediation

The source-of-truth schema audit established that `paired_train.npz` owns the
native `z_v/z_a/z_c` arrays and labels, while `shared_train.npz` owns
`u_v/u_a/u_c`. Their 22,680 `pair_id` rows are exactly aligned. The failed
uncertainty baseline incorrectly referenced `train["u_c"]`; the corrected
evaluator references `shared_train["u_c"]` and now validates field ownership,
row alignment, and `[8,32]` Contact shape before computing performance.

The correction passed a validation-only schema dry run and the complete
simulation regression before the implementation was frozen. No model
training, checkpoint or candidate selection, bridge/predictor/uncertainty
retraining, threshold change, normalization change, or split change occurred.

## Fresh FORMAL_TEST_V2

Before generation, the protocol froze the generator and evaluator hashes,
all checkpoint hashes, all thresholds, the three tasks, 45 episode identities,
45 seeds, 9 source groups, and the expected 4,860 pairs. Groups 20--22 for
each task and their episode seeds are disjoint from TRAIN, DS-DEV, formal
validation, and TEST_V1.

The integrity audit decoded all RGB frames and checked numeric checksums,
schema, finite values, 50 Hz timing, contiguous control steps, exact `t+27`
pairing, counts, identities, and overlap. It found 45 unique episodes, 45
unique seeds, 9 groups, and 4,860 unique pair IDs with zero prior overlap.
No model-performance metric was read before `pretest_v2_freeze.json`.

The first integrity attempt is retained as a structural audit-harness failure:
it incorrectly required control steps to begin at zero instead of enforcing
the established contiguous `+1` contract. The audit correction did not change
the generated dataset, frozen evaluator, checkpoints, or thresholds.

## Locked result

The one completed frozen evaluation passed every section:

| Section | Result | Selected evidence |
|---|---:|---|
| Action | PASS | different-episode/correct MSE ratio 523.8547 |
| Bridge | PASS | minimum Contact retrieval R@10 chance multiplier 33.10; Contact retention 0.99998; force retention 1.00072 |
| Shared/private | PASS | every private-recovery improvement and cross-prediction margin positive |
| Conditional | PASS | full improvement 0.10002; missing-V improvement 0.08510; semantic retention 0.95168 |
| Uncertainty | PASS | full/missing-H NLL improvements 0.07959/0.03025; 90% coverage 0.92984/0.91399 |

The equality-only rerun reproduced metric digest
`7f87740c498ba23a0f31dca7786c91cb004c971661fd0843ea5d815fb6971e93`
exactly. A preceding GPU launch stopped before cache or metric creation because
the deterministic CuBLAS workspace environment variable was absent; that
structural launch failure is retained and the unconsumed first evaluation was
restarted with the required deterministic workspace setting.

## Preserved scientific boundary

The original Contact-State rank gate and original Contact-Dynamics 10%
superiority hypothesis remain historical failures. Contact-State remains
accepted only as `LOW_INTRINSIC_DIMENSION_NOT_COLLAPSE`; bounded dynamics
remediation remains failed; C3 remains the canonical simulated Contact
representation; A0 remains the formal Action; continuous Contact and the
shared/private path remain accepted; the discrete RQ remains rejected.

The result does not establish frozen-M3 zero-shot transfer. It supports the
more limited claim that the shared physical-representation principle
replicates while DexJoCo requires a sim-specific continuous shared mapping.
S4.3 is therefore ready with the existing zero-shot, OOD-dynamics, and inactive
right-thumb warnings.
