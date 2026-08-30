# Full Track C — C3-DP Shared–Private Cross-Modal Prediction

## Scope

C3-DP starts from the accepted C2-R checkpoint with SHA-256
`21dccb8fc7fbe6de2598c18e718bd65f226e220e44352ab3d43246e7f9abdf89`.
It does not reopen C1, C2, or C2-R. All three native encoders, the accepted
shared projectors and slots, recovery heads, S1, S2, and the A-R decoder are
frozen throughout this stage.

The canonical Contact representation is explicitly dual path:

```text
u_c = P_c(z_c)
z_c_shared = R_c(u_c)
r_c_priv = z_c - z_c_shared
z_c = z_c_shared + r_c_priv
```

The private residual is retained for Contact-available downstream use. It is
never an input or target of the canonical six-direction shared predictor.

## Predictor contract

One predictor handles all ordered source/target modality pairs. Its only
sample-dependent input is one source shared representation (`u_v`, `u_a`, or
`u_c`); source and target modality identities are non-sample conditioning.
The predictor cannot accept a paired target sample, native target latent,
pair identifier, or `r_c_priv`.

The bounded validation-only candidate family is:

- P0: source/target-conditioned residual linear map;
- P1: source/target-conditioned residual MLP;
- P2: learned target slots with one or two cross-attention blocks.

The objective combines direct shared prediction, frozen native recovery,
cosine, relational, and variance-floor terms. Dynamic samples receive the
preregistered weight. Six candidates are the complete search budget; no test
metric may create, remove, or reorder candidates.

## Dual-path and private audits

Before predictor training, train and validation caches prove the arithmetic
identity to floating-point tolerance and record residual norm, energy,
effective rank, query diversity, CKA, Contact/force probes, rare-boundary
behavior, and frozen-S2 physics. Separate validation-only ridge diagnostics
measure `V -> r_c_priv` and `A -> r_c_priv`; these results are diagnostic and
cannot train or select the canonical predictor.

The accepted residual is classified as
`PRIVATE_RESIDUAL_LARGELY_PRIVATE`: validation R² is `0.003317` from Vision
and `0.006369` from Action. The residual retains nonzero Contact information,
but that fact does not make it a legal predictor input.

## Selection and locked evaluation

Candidate P2, trial 2, epoch 19 is the unique validation-only selection. It
has 17,664 trainable parameters and uses two target-slot cross-attention
blocks. The frozen checkpoint SHA-256 is
`a4382a776bb5296d2f989ee5adedb49a2800f66e50e9c17ba59101470784ffea`.
The selection artifact SHA-256 is
`2112788322473f4541bfe720cd5913831e383043f9ef645125d38695d1dbaf38`
and records `test_loaded: false`.

Only after verifying both hashes and every pretest audit may the evaluator
build/load the 17,504-row test cache. The locked benchmark reports direct
prediction against mean, shuffled-source, different-episode, and
same-episode-wrong-time controls; paired cosine margins; retrieval; dynamic
and rare-boundary subsets; same-family Contact probes; frozen native
decoders; geometry; and source temporal perturbations. The locked result is
not an untouched first look and cannot trigger tuning.

All six latent prediction gates pass on the locked benchmark. Contact shared
physics, Action-target semantics, Vision-target semantics, non-collapse, and
all frozen identity checks also pass. Contact-transition retention is only
`0.360308` for V→C and `0.453090` for A→C, below the hard `0.75` gate. Force
retention is also below `0.75`. The final decision is therefore
**`C3DP_SHARED_SEMANTIC_LOSS`**.

## Reproduction

The original UniT checkpoint path is supplied only at runtime and is never
written to tracked files.

```bash
python scripts/tactile_unit/audit_c3dp_dual_path.py --device cpu
python scripts/tactile_unit/train_c3dp_cross_prediction.py --device cpu
python scripts/tactile_unit/evaluate_c3dp_cross_prediction.py \
  --device cpu --unit-checkpoint "$UNIT_FULLDATA_CKPT"
python scripts/tactile_unit/visualize_c3dp_cross_prediction.py
python -m pytest -q tests/tactile_unit/test_c3dp_shared_private.py \
  tests/tactile_unit/test_cross_modal_predictor.py
```

Runtime caches, checkpoints, reports, and plots live only below
`.local/{cache,experiments,artifacts,logs,tmp}/tactile_unit/vac_c3dp/`.

## Stop point

This work stops after C3-DP. Because the final state is semantic loss, C4 is
not ready. C4, C5, C6/M3, and M3 are not started; M3 is not established.
