# Track C0 — Continuous Contact Bridge Preflight

## Scope and stop point

Track C0 freezes the interface between the accepted Track B Contact representation and a future multimodal Track C. It does not train Action, does not run V+A+C, does not establish M3, and does not start full Track C.

Track B's accepted decision is `CONTINUOUS_CONTACT_RECOMMENDED`. C0 therefore does not instantiate Original UniT RQ, Contact RQ, whitening, or a semantic/private discrete tokenizer, and it does not retrain the Contact encoder.

## Frozen provenance

The Track B base is `7051f8140239a7e72c51aa0749bac703eb60a923`. C0 records and verifies four Original UniT tokenizer file hashes, the S1 Teacher checkpoint SHA256, and the S2 checkpoint SHA256. The S2 checkpoint identity and module identities are deliberately distinct:

- S2 checkpoint SHA256: `c36c0531bba461875384cebf6bd91c34d43d3f84d2083c15c47ae7dee4e64fa4`
- E_c parameter digest: `1d75189991f1dc557854cfd37c7b7e38e7b915f59b860934567bad3351621436`
- D_c parameter digest: `50ec1fd7639377fbd94bea6a5e5c519dbd3b9cc84b95c6aa4b852f71f7dc19ae`

Original UniT Vision, S1 E_T, S2 E_c, and S2 D_c remain in evaluation mode with gradients disabled. Only C0 projectors, cross-attention, and the causal gate may be optimized.

## Continuous Contact contract

The current tactile history is `T_[t-0.5:t]`. The frozen S1 Teacher produces the current causal state

`h_t^c = E_T(T_[t-0.5:t]) ∈ R^256`.

The frozen S2 encoder produces the observed transition teacher

`z_c = E_c(h_t^c, h_t+16^c) ∈ R^(8×32)`.

Both tensors are float32. `z_c` uses the native frozen E_c output scale; Track B whitening is forbidden. The current Teacher window is `[t-15,t]`, the future Teacher window is `[t+1,t+16]`, and the windows have no raw-sample overlap. At 30 Hz the transition horizon is 16 frames, or 0.533333 seconds.

## Offline teacher versus online context

During representation pretraining, `z_v = E_v(I_t,I_t+16)` and `z_c = E_c(h_t^c,h_t+16^c)` are legal offline teachers because both describe the same observed transition. They may be alignment targets, auxiliary supervision, or offline representations.

At deployment, legal observations are `I_≤t`, robot state up to t, `T_[t-0.5:t]`, and `h_t^c`. A causally predicted `z_hat_c` is also legal. The true future image `I_t+16`, future Contact state `h_t+16^c`, and true future-derived `z_c` are oracle fields and cannot be policy observations.

The public API encodes this distinction with `CurrentContactContext`, `ContactTransitionTarget`, `VisionTransitionTarget`, and `PredictedContactTransition`. Inference mode rejects transition targets. A mapping-level guard also rejects nested oracle fields unless the caller explicitly selects `oracle_evaluation` mode.

## Paired T-Rex contract

C0 reuses the accepted S3.1 `observation.images.head_left` lazy decoder and adapter. Vision and Contact pairs share episode, pair ID, and k=16. Train, validation, and test preserve the exact S1/S2 episode split. C0 never copies or re-encodes the 103 GiB RGB corpus.

The untouched evaluation set is all 960 exact S3.1 pair IDs; no replacement or rebalanced test subset is permitted. The preflight feature extraction uses deterministic, train/validation-only proportional samples for small bridge fitting and validation. This sampling does not alter the canonical test set.

Original UniT processes `(I_t,I_t+16)` through the frozen Vision branch and frozen L2 down-projector to `z_v [B,8,32]`. B3 additionally uses `(I_t,I_t)` to create a current-only visual context; it never consumes transition `z_v`, `I_t+16`, or true `z_c`.

## Baselines and bridge candidates

- B0 is the untrained native `z_v`/`z_c` baseline.
- B1 applies independent shared-over-query residual MLPs to Vision and Contact.
- B2 applies small bidirectional token-set cross-attention, avoiding an assumed query-position correspondence.
- B3 predicts a scalar Contact gate from current visual context and current `h_t^c` only.

B1 and B2 use paired InfoNCE, bidirectional prediction, and within-modality relational preservation. Dynamic and rare-boundary weights come from train labels. There is no strong MMD objective; MMD and sliced Wasserstein distance remain diagnostics only.

Raw and projected evaluation reports paired, different-episode, same-episode wrong-time, reversed-transition, and fixed-seed shuffled controls; bidirectional retrieval; ridge prediction; token norms; effective rank; query diversity; CKA; dynamic results; and rare-boundary results. Validation selects bridge checkpoints and the selected bridge. Test is evaluated after selection.

## Semantic retention and fallback

Train/validation-selected linear probes measure Contact-transition and force-trend macro-F1 on native and projected Contact representations. Advantage retention is measured relative to the train-majority control. The engineering criteria are `R_contact ≥ 0.90` and `R_force ≥ 0.90`; rare free→contact and contact→free recall is reported separately.

Missing, masked, and zero-current Contact cases must be finite and deterministic. For residual fusion, a missing or masked Contact stream produces the exact pure-Vision baseline. B3's function signature has no future field.

## Revised future M3 protocol

Future M3 does not require V/A/C to share a codebook. Its preregistered gates are defined in `configs/tactile_unit/m3_continuous_vac_evaluation.json`: Original UniT V-A non-regression; paired V-C and A-C evidence over shuffled controls; Vision/Action prediction of continuous z_c; Contact benefit for future prediction; dynamic and boundary reporting; causal current h_t^c; teacher-only future z_c; graceful missing Contact; retained M2 Contact semantics; Action Track A-R readiness; and no modality collapse.

Track C0 writes this protocol but does not execute A-C or full V+A+C. Action Track A-R remains an independent prerequisite.

## Reproduction entry points

The contract audit is `scripts/tactile_unit/audit_continuous_contact_contract.py`. Paired feature extraction and small-candidate fitting are performed by `scripts/tactile_unit/train_vision_contact_bridge.py`; evaluation and human-facing plots use `scripts/tactile_unit/evaluate_vision_contact_bridge.py` and `scripts/tactile_unit/visualize_vision_contact_bridge.py`.

All runtime evidence is ignored under `.local/{artifacts,experiments,cache,logs}/tactile_unit/c0`. Machine-local dataset and checkpoint paths are supplied through ignored environment configuration or explicit CLI arguments and never appear in tracked files.

## Preflight result

The final C0 state is `C0_READY_WITH_ALIGNMENT_WARNING`. Both allowed physical GPUs had conflicting compute workloads, so CUDA was hidden and the preflight ran on CPU; GPU0/1 were never exposed. The exact paired extraction contains 4,096 deterministic train pairs, 1,024 validation pairs, and all 960 canonical test pair IDs.

B1, the 8,512-parameter independent two-tower residual projector, is the selected bridge. On untouched test its paired cosine is 0.02376 versus 0.00425 for different-episode negatives, a margin of 0.01951 with bootstrap 95% CI `[0.01628, 0.02287]`. V→C R@5 is 0.00625 versus 0.00521 chance and R@10 is 0.02083 versus 0.01042 chance, but R@1 only equals chance. Paired cosine also only narrowly exceeds reversed Contact (0.02328). These weak retrieval and temporal-direction results require the alignment warning.

The V→C ridge predictor reaches test MSE 0.05857 versus 0.06018 for the train-mean control and 0.06088 for the different-episode-target control. On the dynamic subset it reaches 0.17368 versus 0.17893 and 0.17987. Projected Contact retains 0.9856 of Contact-transition advantage and 1.0112 of force-trend advantage, both above the 0.90 engineering gates. Free→contact/contact→free recall is 0.7368/0.5625. No-collapse and deterministic-reload checks pass.

B2 validates the small `[B,8,32]` bidirectional cross-attention interface but is pair-conditioned; its apparent retrieval is therefore excluded from independent alignment claims. B3 uses only current visual context and current `h_t^c`, reaches 95.94% test contact/free accuracy, suppresses the mean free-space gate to 0.0810, and activates the mean contact-state gate to 0.9301. Missing and masked Contact produce exact finite deterministic Vision fallback.

All provenance, timing, split, causal-rejection, frozen-identity, shape, fallback, and privacy gates pass. Full Track C remains not started, M3 remains a preregistered specification rather than an established milestone, and Action Track A-R is still required.
