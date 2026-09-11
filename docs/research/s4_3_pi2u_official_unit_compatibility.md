# S4.3-PI2U Official UniT Compatibility and Ablation Design Audit

## Decision

The exact compatibility class is:

`UNIT_DEXJOCO_ADAPTER_ONLY_COMPATIBLE`

This is not direct compatibility. The released tokenizer has no DexJoCo data transform, normalization statistics, embodiment identifier, or trained category slice. It is adapter-only compatible because the official implementation confines embodiment-dependent parameters to the state/action front end and action reconstruction back end. The official vision branch, visual-action fusion core, residual VQ codebooks, bridge projector, vision decoder, and shared action M-Formers can remain frozen while a new DexJoCo category slice is trained.

This is an architecture-level minimal-adaptation conclusion, not evidence that the adapter has learned a valid DexJoCo representation. No BUniT training was run.

## 1. Authoritative release identity

Only official upstream materials were used:

- Source: `xpeng-robotics/UniT`, commit `0d762e32180bddd765694ef3846a3a5053f9d37f`, Apache-2.0. The ignored checkout was clean, and the SHA-256 of its complete `git ls-tree` text is `4268e41678c643bb6a5f4e6823cb087707393e9d44ec06035a46e0b6600c8648`.
- Checkpoints: `xpeng-robotics/VLA-UniT-checkpoints`, Hugging Face revision `3ceba29d6eb4b184eb81a49e94d387addf9e584a`, public, ungated, Apache-2.0.
- Releases: `VLA-UniT-3B-fulldata` and `VLA-UniT-3B-fewshot`. Each release contains a four-shard VLA policy and a nested two-shard tokenizer.

At the pinned commit, the README marks tokenizer training/inference, VLA-UniT training and RoboCasa GR1 evaluation, and pretrained checkpoints as available. WM-UniT and real-world deployment remain pending (`README.md:356-370`). The full example is tokenizer 80k steps followed by dual-system pretraining for 160k; the few-shot example is 80k + 20k + 20k. Example scripts default to eight GPUs (`examples/README.md:5-23`).

The metadata-only checkpoint audit found:

| Payload | Tensor count | Index-declared bytes |
| --- | ---: | ---: |
| One VLA policy | 1,151 | 19,369,104,384 |
| One nested tokenizer | 1,045 | 5,494,676,608 |
| Complete HF tree, both releases | 28 files | 49,732,598,629 |

The HF model API reports 49,728,165,984 used bytes; the small difference from the enumerated tree is retained rather than normalized away. The full-data and few-shot config and index JSON files are byte-identical, but their LFS shard object IDs differ. No large weight shard was downloaded.

## 2. Source-backed architecture map

The released tokenizer config fixes a 1,024-wide latent, eight query tokens, a 16-step action horizon, 128-D padded state/action envelopes, and a two-stage residual VQ with 128 entries of width 32 per stage.

```text
current RGB + goal RGB(t+16)
        │
        └─ DINOv2-large layer -2 patches
             └─ shared Vision M-Former ───────────────► z_v [8,1024]

state [128] + action [16,128] + embodiment id
        │
        ├─ category-specific state MLP
        └─ category-specific temporal ResNet
             └─ shared Action M-Former ───────────────► z_a [8,1024]

z_v only ─── align_visual ─┐
z_a only ─── align_action ─┼─ shared projection ─► down to 32
z_v + z_a ─ Fusion M-Former┘                         │
                                                   ▼
                                  shared 2-stage residual VQ
                                  indices [8,2], 128 classes/stage
                                                   │
                    ┌──────────────────────────────┴─────────────────┐
                    ▼                                                ▼
 current DINO patches + shared visual decoder       state feature + shared M-Former
                    │                                  + category action decoder
                    ▼                                                ▼
           goal DINO reconstruction                         16x128 action reconstruction
```

The vision branch applies the same DINO and M-Former pipeline to current and goal frames (`gr00t/model/tokenizer/vision_branch_encoder.py:46-130`). The action branch uses category-indexed convolution, normalization, and state-MLP tensors before a shared M-Former (`gr00t/model/tokenizer/action_branch_encoder.py:69-186,295-439`). The fusion module implements distinct vision-only and action-only alignment paths plus an M-Former whose condition is vision and target is action; all three routes then use one shared projection (`gr00t/model/tokenizer/visual_action_fusion.py:9-175`).

Cross-reconstruction is not a paper-only inference. With multi-scenario training enabled, the implementation expands each example into fused, vision-only, and action-only routes, passes every routed representation through the same quantizer, and supplies the resulting shared latent to both decoders (`gr00t/model/gr00t_n1_tokenizer_unit.py:542-712`). The visual decoder reconstructs future DINO features from current DINO patches; the action decoder reconstructs the action chunk conditioned on the encoded state. The released config does not enable the optional `unit_fusion_distill` term.

The shared/embodiment-specific boundary is therefore:

| Frozen shared component | New DexJoCo-specific component |
| --- | --- |
| DINOv2 and Vision M-Former | State MLP category slice |
| Shared action M-Former | Action-encoder temporal convolution/LN slices |
| Fusion M-Former and alignment projections | Action-decoder ResNet/LN/output slices |
| VQ down projection and both residual codebooks | TRAIN-only semantics and normalization config |
| Bridge projector and vision decoder | Dedicated category id/bank allocation |
| Action-decoder M-Former | — |

The new category slices contain 23,485,568 parameters: 12,334,080 in the action encoder and 11,151,488 in the category-specific action decoder. Because those parameters are stored as indexed banks rather than standalone adapters, an implementation must use extracted parametrization or gradient masking and must verify that every frozen tensor remains byte-identical.

## 3. Official VLA-UniT supervision target

The released VLA is `GR00T_N1_5_UniT`, with eight bridge tokens and one categorical predictor for each residual-VQ stage. Its tokenizer is loaded from the checkpoint-local `tokenizer` directory, forced to evaluation mode, and frozen (`gr00t/model/gr00t_n1_unit.py:718-755`). Ground truth is obtained under `no_grad` by calling the tokenizer with `return_motion_token_ids_only=True` (`gr00t/model/gr00t_n1_unit.py:787-830`).

The most faithful future BUniT target is therefore exactly the fused route's index tensor:

- route: `pv=1, pa=1`;
- target shape: `[8,2]` integer indices;
- prediction shape: `[8,2,128]` logits;
- loss: `CE(stage 0) + 0.5 × CE(stage 1)`;
- extraction inputs: current RGB, goal RGB at `t+16`, state, 16-step action, and embodiment id;
- inference-time target access: none.

A continuous fusion vector, quantized vector, or vision-only token is not an exact substitute. The inference wrapper returns indices, quantized vectors, and pre-quantization vectors separately (`gr00t/model/gr00t_n1_tokenizer_unit_inference.py:350-397`); official VLA-UniT supervises the indices.

## 4. GR1, human, and DexJoCo contracts

The main released GR1-Joints recipe selects one ego camera and five joint groups: right arm 7, right hand 6, left arm 7, left hand 6, and waist 3. The raw 29-D state becomes 58-D after SinCos transformation; the action remains 29-D with TRAIN mean/std normalization. Both are padded to 128 and the action horizon is 16 (`gr00t/experiment/data_config_unit.py:1145-1222`). The repository also provides one-camera GR1 EEF and human EgoDex recipes using wrist/fingertip Cartesian and rotation representations (`gr00t/experiment/data_config_unit.py:948-1142`). Released VLA checkpoint metadata names GR1; the few-shot tokenizer metadata names human EgoDex and GR1. None names DexJoCo.

DexJoCo instead uses:

- state 23D: TCP xyz3 + quaternion wxyz4 + Allegro16;
- policy action 22D: TCP xyz3 + rotation-vector3 + Allegro16;
- environment action 23D after quaternion conversion;
- a 30-step π0.5 policy action chunk;
- front and wrist RGB policy inputs.

Official padding code would accept a 23-D state and 22-D action after choosing 16 steps. State dimensions above 128 are silently truncated, smaller states are padded, actions above 128 are rejected, and smaller actions are padded (`gr00t/model/transforms.py:378-437`). This establishes only a shape envelope.

The category mapping makes direct use even less safe. GR1 is id 24, while the generic `new_embodiment` tag is id 31 (`gr00t/data/embodiment_tags.py:47-57`). The released tokenizer config sets `max_num_embodiments=30`, so valid bank indices are 0–29. Future work must allocate a dedicated DexJoCo id and explicitly expand the banks, or first prove and reserve an unused in-range slot. It must not alias DexJoCo to GR1 or pass out-of-range id 31.

## 5. Safe direct-compatibility probe

One non-scientific TRAIN episode, `pinch_tongs-g00-p00`, was inspected without using its values for model selection. Steps and RGB tree hashes were `181a73674b29d270c815117259748ea77d9c36a14ac49388c1456ebe8e1f80ca` and `b13f1611cc587621ee80e737d59caf75d3f1fb1dc95ababdf1dcd0631760004a`. Frames at `t=0` and `t=16` were finite RGB; the first 23 proprio dimensions and a `16×22` policy-action slice were finite.

- Shape gate: pass after declared resize, horizon selection, masked padding, and no truncation.
- Semantic direct-compatibility gate: fail.

The failure is decisive: GR1 id 24 would select weights trained for bimanual arm/hand/waist joint meanings and statistics, not a Franka TCP plus Allegro hand. Zero-padding cannot supply the absent transform, statistics, or category weights. The official checkpoint therefore cannot ingest the sample *semantically* without adding/training embodiment-specific parameters.

The audit itself initially recorded `NATIVE_SANITY_NOT_RUN_RESOURCE_LIMIT`, because its bounded task did not authorize a multi-gigabyte download and static evidence had already disproved direct semantic compatibility. The concurrent, explicitly authorized BVA target build subsequently downloaded the revision-pinned 5.49 GB tokenizer, verified all four published file hashes, loaded it with zero missing or unexpected keys, and executed the frozen Vision transition path for all 37,365 valid `t→t+27` TRAIN anchors on one GPU. Outputs were finite `[8,32]` tensors after the frozen VA bridge. This upgrades the narrow native result to `PASS_VISION_TRANSITION_ONLY_DURING_BVA_TARGET_BUILD`; it does not test the embodiment-specific Action path and does not change the direct-compatibility failure.

## 6. Future BUniT design — frozen, not executed

BUniT must retain the exact official π0.5 policy backbone, initialization, policy inputs, seed, train steps, and evaluation protocol used by B0/BVA/B1/B2. It is not a comparison between the released GR00T VLA-UniT policy and a π0.5 policy. It contains no tactile or contact input.

### Stage U-Adapt: DexJoCo embodiment adapter

Use DexJoCo TRAIN and source-group-disjoint DEV data only; do not touch TEST or policy success/reset outcomes.

1. Feed the front frame at `t` and front goal frame at `t+16` through the official crop/resize and ImageNet-normalized DINO path. The wrist camera remains an unchanged π0.5 policy input but is excluded from the one-camera tokenizer target path to match the released GR1 view topology.
2. Convert state quaternion wxyz explicitly to rotation6d, giving `xyz3 + rotation6d6 + Allegro16 = 25` real dimensions. Apply TRAIN-only mean/std statistics, mask, and pad to 128.
3. Select the first 16 *absolute target* actions from the 30-step π0.5 chunk. Preserve `xyz3 + rotvec3 + Allegro16 = 22` semantics, apply TRAIN-only mean/std, mask, and pad to 128. This is a declared tokenizer view of the policy target, not silent horizon truncation.
4. Initialize only the dedicated DexJoCo encoder/decoder category slices. Freeze DINO, all shared M-Formers, fusion/alignment, VQ projection and codebooks, bridge projector, vision decoder, and all existing category slices.
5. Train with the official fused/vision-only/action-only multi-scenario cross-reconstruction objective. Do not introduce a custom target or an absent fusion-distillation loss.

Before freezing the adapter, require on source-group-disjoint DEV: finite values; indices in `[0,127]`; non-collapsed code use for all three routes; paired transition/action agreement above a different-episode shuffled control with a positive confidence interval; action reconstruction above a preregistered constant/mean baseline; future-DINO reconstruction above a no-motion baseline; unchanged native GR1 behavior; and byte-identical hashes for every frozen tensor. Failure stops BUniT and triggers reclassification rather than silent shared-component unfreezing.

### Stage BUniT: π0.5 auxiliary

Freeze the accepted adapted tokenizer in eval/no-grad and cache its fused `[8,2]` RVQ indices. A compact categorical head can use the same final π0.5 action-expert hidden-state source as the existing auxiliary ablations:

```text
LayerNorm(1024)
→ Linear(1024,512) → GELU
→ Linear(512,8×32) → reshape [8,32]
→ two shared Linear(32,128) stage classifiers
→ logits [8,2,128]
```

This head has 666,624 parameters, only 8,448 (1.284%) more than the 658,176-parameter continuous B2 reference head. Train with

`L = L_pi05 + λ_unit × (CE_0 + 0.5 × CE_1)`.

Calibrate `λ_unit` once from four fixed deterministic TRAIN-only batches as `clamp(0.1 × mean(L_pi05) / mean(L_unit), 1e-3, 1e-1)`, then freeze it. Do not tune on rollouts. At inference the adapted tokenizer, future image/action targets, and auxiliary head are absent or disabled.

This permits a future `YES_WITH_ADAPTATION` DexJoCo comparison: fresh paired B0/BVA/BUniT/B1/B2 rollouts under one frozen protocol, with identical resets/seeds and no checkpoint shopping. It does not justify training BUniT until Stage U-Adapt passes.

## 7. Resource and claim boundary

The official README says default training hyperparameters assume 120+ GB of GPU memory per device (`README.md:112-114`), while its example pipelines use eight GPUs. Full official tokenizer/VLA training is therefore prohibited on four 24-GB RTX 4090s. The separate statement that rollout was tested on a 24-GB RTX 4090 (`README.md:121-123`) is not evidence that training fits.

One DexJoCo adapter slice needs about 46.97 MB in bf16 weights or 93.94 MB in fp32 weights, but its forward still traverses the frozen tokenizer and DINOv2. The frozen Vision-only transition path completed at batch 16 on one 24-GB GPU during BVA target construction; this is not evidence that the additional embodiment-specific Action adapter or its training optimizer state fits. Once accepted indices are precomputed, target payload is only 16 unpacked uint8 values per sample and the large tokenizer need not be resident during π0.5 training.

Allowed wording after successful future adaptation:

> Official UniT source/checkpoint initialized, DexJoCo embodiment-adapted, exact official fused RVQ-index auxiliary on the same π0.5 backbone.

Disallowed wording:

- directly compatible released UniT checkpoint;
- untouched or “Original UniT” on DexJoCo;
- official VLA-UniT DexJoCo reproduction;
- primary representation ablation that compares the GR00T VLA policy to π0.5.

## 8. Audit artifacts

The machine-readable audit is under `$REPO_ROOT/.local/artifacts/simulation/s4_3_pi2u/`. The tracked freeze is `configs/simulation/s4_3_pi2u_unit_compatibility.json`. No official source, checkpoint shard, dataset byte, or local absolute path is tracked.
