# S3.3 T-Rex Action Embodiment Bootstrap

This document is the Track A implementation record. Track B remains independent, and no Track C integration is performed here.

## Frozen starting point

Track A started from `ead93bcb9ec3be3fe781dd734433eb82bb1819fd`. At the start of work, `develop/tactile-action-bootstrap`, `develop/tactile-unit`, and the Track B worktree all pointed to that same commit. The Track A worktree was clean and had no branch-only commits.

The data path reuses the accepted S1/S3.1 artifacts without creating a new split:

- episode membership: frozen S1 train/validation/test partitions;
- normalization: S3.1 mean/std fitted on the train episodes only;
- raw state: `[58]`;
- raw action chunk: `[16,58]`, exactly `a_t:t+15`;
- ordering: left arm 7, left hand 22, right arm 7, right hand 22;
- canonical state/action: `[128]` and `[16,128]`, with the first 58 channels valid;
- T-Rex RGB and Contact representations: not read or used.

The numeric cache is an evenly spaced deterministic sample within each already-frozen episode partition. It is not a new split. Its manifest records the selected `(episode, anchor)` identity, validates zero overlap between episode partitions, and records zero video decodes.

## Source-of-truth Action Branch audit

The authoritative values come from the released nested tokenizer `config.json`, its safetensor shapes, and the current Action encoder/decoder implementation—not from Python dataclass defaults.

1. **Released capacity:** 30 category rows, valid IDs `0..29`. The Python class default of 32 does not describe the released checkpoint.
2. **GR1 ID:** 24, from `EMBODIMENT_TAG_MAPPING`.
3. **Previous T-Rex contract ID:** S3.1 used the generic `new_embodiment` route, ID 31. Track A registers the explicit `trex` tag at the same reserved ID 31 while retaining the legacy generic spelling for compatibility.
4. **ID 30:** unassigned by the registry. It is also absent from the 30-row checkpoint. After 30-to-32 expansion, row 30 is created but remains explicitly unused.
5. **ID 31:** out of bounds for a 30-row tensor because valid indices end at 29. A mapping entry alone cannot create category parameters.
6. **Category-indexed parameters:** 32 tensors in total. The encoder has category-specific stem/body causal-convolution weights and biases, LayerNorm affine rows, and both state-MLP linear layers. The decoder has category-specific input/body/upsample LayerNorm rows and causal-convolution weights and biases.
7. **Not embedding-only:** there is no single embodiment embedding table whose expansion solves the problem. Every category-indexed tensor above must gain rows 30 and 31.
8. **Decoder:** category-indexed structure is present throughout the ResNet decoder and final action projection. Its M-Former is shared, not category-indexed.
9. **128-D contract:** native. The released config declares `state_dim=128`, `action_dim=128`, `action_horizon=16`, `query_num=8`, and pre-RQ dimension 32.
10. **`new_embodiment`:** a registry placeholder in this checkpoint. It maps to ID 31, but the released Action encoder/decoder contain no row 31 and there is no automatic row creation or training mechanism.

The released action-only path is:

```text
[B,16,128] action + [B,1,128] state
    -> category-specific Action encoder + shared M-Former
    -> [B,8,1024] Action L1
    -> frozen action-only alignment/shared projection
    -> frozen 1024->32 down-resampler
    -> z_a^T-Rex [B,8,32] (continuous Action L2, before RQ)
    -> frozen 32->1024 bridge + category-specific Action decoder
    -> [B,16,128] reconstruction
```

## Safe expansion and optimizer isolation

The deployment expansion rule is 30 to 32:

- rows `0..29`: copied byte-for-byte;
- row 30: deterministic initialization, explicitly unused;
- row 31: learned T-Rex overlay;
- no modulo, clamp, ID reordering, GR1 alias, or unrelated embodiment reuse.

The compact training model is deliberately row-isolated. It loads all shared Action-only tensors exactly but instantiates only one local category row, guarded by the global ID 31 at its public boundary. Only the category-specific encoder and decoder row is trainable in A1. This design is stronger than masking gradients on a 32-row parameter: an optimizer cannot allocate state for or apply weight decay to old rows because those rows are absent from the optimization model.

The checkpoint is an overlay containing only T-Rex-owned category values (plus the optional A2 adapter). It includes the released checkpoint identity and refuses cold reload against a different base. Materialization covers all 32 category-indexed encoder/decoder tensors and asserts exact equality for rows `0..29`.

The all-old-row SHA-256 covers tensor names, shapes, dtypes, and bytes for all 32 category-indexed tensors. Before training it was:

```text
e92ced68df2247c19dd99f5be8b165922fbcc06ffa0c27597999ebb4b54d803c
```

Because the overlay contains no old row and no shared parameter, the digest after A1/A2 must remain identical. GR1 selects row 24; therefore exact preservation of row 24 plus every shared action-only tensor proves exact GR1 Action L2 preservation when A3 is not entered.

## Progressive candidates

- **A0:** compare the legal old-row mean against deterministic small random initialization on validation only. The nominal generic/new-embodiment initializer is recorded as unavailable because the released checkpoint has no row 31. A0 requires finite `[B,8,32]` output, valid reconstruction shape, deterministic evaluation, and no training.
- **A1:** train only the isolated T-Rex category tensors. Shared encoder/decoder M-Formers, action-only fusion, the 1024-to-32 down-resampler, the 32-to-1024 bridge, Vision, Contact, and Original-UniT RQ remain frozen.
- **A2:** only if A1 fails either the validation reconstruction threshold or any predeclared validation temporal-control threshold, add a zero-initialized residual `LayerNorm(32)+Linear(32,32)` specific to T-Rex. Candidate escalation is decided before the untouched test pass. A2 increases only the failed temporal-margin weight, focuses its negative schedule on reversed/shuffled chunks, and raises train-derived dynamic weighting from 2 to 4. A2 checkpoints are selected on validation reconstruction plus all three temporal gates; no test metric participates. The adapter cannot affect another embodiment.
- **A3:** disabled. It may only be designed after measured A1 and A2 insufficiency and would require separate GR1 rehearsal plus final T4 non-regression. The canonical T4 held-out 960 is never used for training or model selection.

Training uses masked reconstruction over the 58 valid channels, action-delta reconstruction, rotating reversed/shuffled/different-episode temporal margins, and variance/query-diversity anti-collapse terms. Dynamic weighting uses a deterministic two-cluster threshold fitted only to train normalized RMS action deltas; it does not force a predetermined static/dynamic fraction. Primitive metadata is not a training target.

## Held-out evidence contract

The untouched test partition reports normalized MSE/MAE and inverse-normalized raw-unit aggregate metrics for all and dynamic windows, split into left arm, left hand, right arm, and right hand. Temporal controls compare the correct chunk against reversed, shuffled, and different-episode chunks under the same reconstruction/matching loss.

Frozen linear probes cover:

- action magnitude and trend — **DERIVED** from the canonical action chunk;
- active side — **DERIVED**;
- arm-versus-hand activity — **DERIVED**;
- primitive — **ACTUAL METADATA**, auxiliary only.

Non-collapse diagnostics cover per-dimension variance, effective rank, token norm, query diversity, collapsed-query fraction, and pairwise sample distance. The interface must be finite, deterministic, `[B,8,32]`, and cold reloadable.

The frozen Original-UniT RQ is evaluated read-only. Reported diagnostics include relative distortion, pre/post cosine, active codes, perplexity, top-1/top-5 code mass, quantized reconstruction retention, and frozen-probe retention. A continuous representation may be accepted with a shared-RQ warning; Track A never updates the shared RQ.

## GPU isolation

Every scientific GPU process must be launched under a shared advisory lock in the Git common directory. Selection is atomic: acquire the GPU 3 lock, recheck physical compute occupancy, and only keep the lock if GPU 3 is truly free; otherwise repeat for GPU 2 and then GPU 1. The process exposes exactly one physical GPU through `CUDA_VISIBLE_DEVICES` and uses only logical `cuda:0`. GPU 1 was added as an explicitly user-authorized fallback on 2026-08-26 after repeated GPU 2/3 contention; GPU 0 remains forbidden.

At the first training attempt, GPUs 2/3 had conflicting compute processes, so the state was `GPU_RESOURCE_BUSY`. CPU-only cache, digest, test, and documentation work continued; no process was killed or preempted and no forbidden GPU was used. The final result must replace this provisional observation with the actual locked assignment used for training/evaluation.

## Track-separation note

The only shared core file changed by Track A is `gr00t/data/embodiment_tags.py`. The change is additive: it registers the explicit `trex` enum/tag at the already-reserved ID 31 and leaves every existing tag and ID unchanged. No Contact tokenizer file, Contact semantic API, README, or Track B worktree is modified. The additive registry change is the smallest shared surface needed for an explicit process-wide embodiment identity; Track B conflict risk is limited to a possible textual overlap if it independently edits the same registry.

## Measured result

The user explicitly authorized physical GPU 1 after sustained contention on GPUs 2/3. The final A1/A2 training and untouched evaluation used physical GPU 1 under `tactile3d_unit_gpu1.lock`, exposed only as logical `cuda:0`; isolation passed. A1 took 108.97 seconds and A2 took 46.17 seconds. A2 added 1,120 parameters, or about 0.000728% of the compact Action Branch, and the selected overlay SHA-256 is `8a45df45e4e12e5c702622dbb72d5fbb7f9733b53cd86ae3bbba334191706899`.

A1 passed reconstruction but failed validation reversed/shuffled temporal ratios (1.0037 and 1.0012 versus the 1.05 gate), which triggered A2 without consulting test. A2 improved validation reconstruction to 0.0709 and the reversed ratio to 1.0195, but still failed reversed/shuffled temporal gates. A3 was not executed: it would update shared Action parameters and requires a separately designed GR1 rehearsal/T4 non-regression protocol, whereas this milestone has already measured the complete minimal-bootstrap ceiling requested by A0/A1/A2.

The single frozen A2 checkpoint was then evaluated across all 536,499 untouched test windows. Overall normalized MSE/MAE were 0.06917/0.18693 and dynamic-window normalized MSE/MAE were 0.12254/0.24753. The different-episode control was strong (15.3389 times correct loss), but reversed and shuffled were only 1.0272 and 1.0072 times correct, below the 1.05 gate. Frozen probes remained measurable (magnitude R² 0.2018, trend R² 0.0820, active-side balanced accuracy 0.7884, arm-versus-hand balanced accuracy 0.6560). Effective rank was 12.201, collapsed-query fraction was zero, and mean query cosine distance was 0.8481.

Every old category row remained bit-identical with before/after digest `e92ced68df2247c19dd99f5be8b165922fbcc06ffa0c27597999ebb4b54d803c`; GR1 Action L2 preservation passed by exact equality. The read-only Original-UniT RQ increased reconstruction error by 2.3704 times, with mean pre/post cosine 0.9325, but shared-RQ interpretation is secondary because the continuous representation already failed the temporal acceptance gate.

The final decision is **`ACTION_BOOTSTRAP_INSUFFICIENT`**. The implementation is structurally valid and learns reconstruction plus measurable action semantics, but A1/A2 do not establish sufficiently credible within-chunk temporal ordering. Track A is therefore not ready to unblock Track C.
