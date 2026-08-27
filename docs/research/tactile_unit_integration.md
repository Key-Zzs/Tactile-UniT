# Continuous VAC Integration (pre-Track-C)

This stage joins the accepted C0 and A-R lineages and freezes one reusable
Vision–Action–Contact transition boundary. It performs no representation
training, projector training, RQ adaptation, missing-modality training, causal
student training, M3 execution, VLA work, robot work, or 3D work.

## Canonical physical transition

Every row is identified by `(pair_id, episode_id, t, t_future)` with
`t_future = t + 16` at 30 Hz. The three native continuous teachers describe
that exact interval:

- Vision: `z_v = E_v(I_t, I_t+16)`, shape `[B,8,32]`.
- Action: `z_a = E_a^R1P(s_t,a_t:t+15)`, shape `[B,8,32]`. Its 16 actions end
  at `a_t+15`; `a_t+16` is not part of the chunk.
- Contact: `z_c = E_c(h_t^c,h_t+16^c)`, shape `[B,8,32]`.
- Current causal Contact: `h_t^c`, shape `[B,256]`, from `T_[t-0.5:t]`.

The Contact Teacher windows are `[t-15,t]` and `[t+1,t+16]`, so they have no
raw sample overlap. The Action path is the accepted continuous pre-RQ R1-P
path. The Contact path is the native frozen S2 output with no RQ, whitening,
or semantic/private tokenizer.

## Offline and online boundary

`OfflineVACTransitionTeachers` carries same-order `z_v`, `z_a`, `z_c`, current
context, state/action tensors, modality masks, anchors, and provenance.
`OnlineCausalContext` carries only current visual/state/tactile context plus
policy-generated values.

`ActionTransitionTarget` is a demonstration/planned teacher. It is not a
sensory observation. A future policy may instead create a
`PredictedOrPlannedActionTransition` from its own planned action chunk.

The recursive online guard rejects top-level or nested real future images,
future Contact state, and true `z_v`, `z_a`, or `z_c` teachers unless the
caller explicitly requests oracle evaluation.

## Frozen provenance

The public contract records immutable identities for the Original UniT
tokenizer, S1 Teacher, S2 checkpoint and its `E_c`/`D_c` parameter digests,
the A-R checkpoint, and Original UniT Action rows 0–29. The integration audit
hashes these before and after the combined smoke, sets loaded models to eval
with gradients disabled, and instantiates no optimizer or backward pass.

The optional C0 B1 bridge checkpoint is only a historical alignment baseline.
It is not the canonical Contact representation and is not required to load the
integrated native teachers.

## Acceptance and read-only baseline

The auditor validates the full canonical 960-row S3.1 manifest structurally,
copies only the identity-checked ignored C0 native cache from the dynamically
discovered C0 worktree, and runs the accepted A-R encoder on the exact same
episode/frame anchors. It writes ignored runtime artifacts under
`.local/{artifacts,cache}/tactile_unit/integration`.

The native V-A, V-C, and A-C cosine, shuffled control, CKA, and bidirectional
retrieval snapshot is an initial condition for future Track C. Weak alignment
is a research baseline warning, not an integration failure. No projector or
cross-modal model is fitted here.

Run from the repository root after loading the existing local runtime
environment:

```bash
python scripts/tactile_unit/audit_tactile_unit_integration.py
pytest -q tests/tactile_unit/test_tactile_unit_integration.py
```

M3 remains `SPEC_ONLY_NOT_EXECUTED`. Full Track C is not started by this stage.
