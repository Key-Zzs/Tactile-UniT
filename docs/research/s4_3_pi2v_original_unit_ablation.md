# S4.3-PI2V Original UniT adapter-only ablation

PI2V evaluates a baseline described strictly as: “Original UniT, using the
official implementation and released tokenizer initialization, with a
DexJoCo-specific embodiment adapter.” BUniT is a baseline, not a Tactile-UniT
contribution, and it is not an untouched released UniT checkpoint.

The representation stage freezes the audited UniT source at commit
`0d762e32180bddd765694ef3846a3a5053f9d37f` and freezes every released tokenizer
parameter. A new category 30 is appended to each official category-specific
state/action encoder and action decoder bank. Only those 32 appended tensors
(23,485,568 parameters) are optimized. The official vision, action, fusion,
two-level RVQ, reconstruction, and multi-scenario cross-reconstruction paths
remain active. The frozen RVQ is placed in evaluation mode only to suppress its
stateful dead-code restart; quantization and VQ loss remain in the objective.

TRAIN/DEV come only from the frozen S4.2 representation split, with 42/9
source-disjoint trajectory groups. The canonical simulator target is `t+27` at
50 Hz (0.54 s). The future policy-dataset target is frame `+16` at 30 Hz
(0.5333 s); RGB is never interpolated. State is converted from quaternion wxyz
to rotation-6D before TRAIN-only normalization. The 27 simulator action commands
are deterministically sampled into the official 16-slot action contract.

The tracked protocol freezes the official source/checkpoint identities, exact
optimizer budget, deterministic resume behavior, and representation-quality
gates before the formal result is observed. The 80k training run uses the
official effective global batch of 256 on two GPUs. Only the final step-80000
adapter is eligible for BUniT target generation after mandatory user approval.

## Formal adapter result

The frozen adapter run completed all 80,000 optimizer steps (20,480,000
samples) with effective global batch 256. Its final checkpoint is
`f507a5ff6e8ac02af22f26360947135b91c9745f23092c73ea9dff346fe260dc`.
All 32 adapter gradient tensors were finite and nonzero, and the frozen
official-parameter digest was identical before and after training.

The final checkpoint was cold-loaded and evaluated once on all 4,860 examples
from the nine-group frozen DEV split. Nine of ten preregistered structural
checks passed: all outputs were finite and nonzero; the RVQ did not collapse;
paired V/A relation exceeded a source-group-disjoint shuffle; all four
cross-reconstruction routes beat their shuffled controls; fused action
reconstruction beat the TRAIN-mean baseline; and official frozen weights and
checkpoint parent remained unchanged.

The single hard failure was future-vision reconstruction. Lower is better, but
the fused representation obtained cosine loss `0.09790460765361786`, compared
with `0.08125562965869904` for the preregistered no-motion (copy-current)
baseline. The learned route was therefore about 20.5% worse than the baseline.
This is a representation-training failure under the frozen protocol, not an
infrastructure failure or codebook-collapse failure.

Decision:
`S4_3_PI2V_UNIT_ADAPTER_REPRESENTATION_FAIL`.

Per protocol, execution stopped at PI2V-8. No BUniT target cache was generated,
no BUniT mode or policy was trained, no five-model policy evaluation was run,
and PI2B was not started. Consequently this result cannot support a policy-level
comparison among BUniT, B0, BVA, B1, and B2; it only rejects this frozen adapter
checkpoint as a valid BUniT teacher. Thresholds were not changed, no alternate
checkpoint was selected, and no rollout result was used.
