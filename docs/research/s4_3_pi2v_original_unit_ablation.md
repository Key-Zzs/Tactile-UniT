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

At this stage B0, BVA, B1, and B2 remain immutable; BUniT policy training,
five-model evaluation, PI2V conclusions, and PI2B are not started.
