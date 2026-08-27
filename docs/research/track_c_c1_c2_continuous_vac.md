# Full Track C — C1 + C2 Continuous VAC

This milestone freezes a deterministic three-modal T-Rex transition dataset and
learns a continuous Vision–Action–Contact shared physical space. It stops after
C2. Cross-prediction, cross-reconstruction, missing-modality training, a causal
student, and M3 evaluation are outside this milestone and remain not started.

## Transition and data contract

Every row describes exactly `t -> t+16`:

- Vision uses `I_t` and `I_t+16`.
- Action uses the normalized current state and `a_t:a_t+15` through the accepted
  R1-P absolute/state-relative/first-difference path.
- Contact uses non-overlapping Teacher windows `[t-15,t]` and `[t+1,t+16]` and
  the native continuous S2 encoder output.

C1 uses 65,536 frozen train rows, a uniform deterministic 8,192-row validation
subset, and all 17,504 test rows. The accepted 960 test IDs remain an exact
identity anchor. Train selection guarantees broad primitive, object, dynamic,
and rare-boundary coverage and includes identity-safe reusable C0 Vision rows.
Validation and test are never label-stratified or changed after selection.

The public cache is array-sharded NPY plus canonical JSON. It contains no RGB,
no arbitrary Python pickle, and no private absolute paths. Every array is
hash-locked and shares one `pair_id` / `source_index` order.

## Independent shared-space architecture

Each representation is independently computable:

```text
z_v [B,8,32] -> P_v -> u_v [B,8,32]
z_a [B,8,32] -> P_a -> u_a [B,8,32]
z_c [B,8,32] -> P_c -> u_c [B,8,32]
```

No projector accepts a paired counterpart. The bounded candidates are native
identity, a residual per-token linear projector, a residual per-token MLP, and
an independent shared-slot resampler. The slot candidate shares eight learned
physical query slots, but each modality has its own key/value attention and
attends only its own native tokens. Retrieval candidates can therefore be
precomputed without knowing a query.

Training uses symmetric three-pair InfoNCE with different-episode negatives,
native recovery heads, within-modality relational preservation, and a minimal
variance floor. Dynamic weighting is bounded and selected only on validation.
The six preregistered trials cover at most three temperatures, three native-loss
weights, and dynamic weights `{1,2}`. The smallest candidate within 0.01 of the
maximum validation utility is selected; test is not loaded during selection.

## Acceptance

C1 requires exact row identity, no episode leakage, finite float32 `[8,32]`
latents, immutable checkpoint provenance, exact canonical 960 identity, cold
reload determinism, and read-only native baselines.

C2 requires positive bootstrap-supported V-A, V-C, and A-C margins; independent
six-direction retrieval evidence; at least 0.90 Contact-transition and force
trend retention above majority; preserved recovered-action reversed/shuffled
dynamic ratios; no modality collapse; independent encodability; and unchanged
native checkpoints. Dynamic and free/contact boundary results are reported
separately even when the overall gate passes.

Runtime outputs live below `.local/cache/tactile_unit/vac_c1`,
`.local/experiments/tactile_unit/vac_c2`, and the corresponding `.local/artifacts`
and `.local/logs` roots. They are intentionally untracked.

M3 remains **NOT ESTABLISHED** regardless of the C2 result.

## Executed result

C1 finished as `C1_READY`. The frozen cache contains 65,536 / 8,192 /
17,504 train / validation / test rows, has no cross-split episode leakage, and
matches all 960 accepted integration rows byte-for-byte for Vision, Action, and
Contact. Deterministic cold recomputation passed for all three native branches.
The Vision cold check uses an explicit `atol=rtol=3e-3` because the frozen
Vision tower runs in float16 and single-sample versus extraction-batch GEMM
reduction order is not bit-exact; the largest observed difference was
0.001760. Action and Contact use `atol=2e-5`, `rtol=1e-5`.

Validation-only selection compared C0 with six preregistered trials. The
selected model is `C2-slot` trial 5: eight shared physical slots, independent
four-head per-modality attention, modality-specific recovery heads, and
248,704 trainable parameters. Its validation utility was 0.707461; test data
was not loaded by the training or selection path.

On the first untouched-test evaluation, all three alignment gates passed. The
V-A, V-C, and A-C paired-minus-shuffled margins were 0.437878, 0.141117, and
0.202614, with strictly positive 95% bootstrap lower bounds. The weakest of
the six full-test R@10 directions was still 9.2 times chance. Action temporal
preservation, non-collapse, independent encodability, and frozen-native
identity checks passed.

The final decision is **`C2_SEMANTIC_PRESERVATION_FAIL`**. Contact-transition
advantage retention was 0.896075, below the preregistered 0.90 hard gate;
force-trend retention was 1.085552. This narrow failure is not overridden by
the strong alignment result, and no post-test tuning or reselection was
performed. A second read-only evaluation was byte-identical to the first.

C3 cross-prediction, C4 missing-modality training, C5 causal-student work, and
C6/M3 remain not started. M3 remains **NOT ESTABLISHED**.
