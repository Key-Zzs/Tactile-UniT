# S4.3-0 demonstration contract failure

S4.3 began from the clean `develop/sim-benchmark` source-of-truth at the
completed S4.2-TF confirmation. The frozen S4.2 representations and historical
decisions remain unchanged.

The first policy-specific hard gate fails. The frozen S4.2-DS source supplies
the required RGB, state, 22D policy Action, 30D simulated tactile, 50 Hz timing,
episode identity, and native success fields. An audit-only inspection confirms
that its 33/9 DS source-group identities are disjoint from formal validation,
exposed TEST_V1, and fresh TEST_V2; S4.3-0B itself was not entered.
However, all 165 `POLICY_TRAIN` episodes and all 45 `POLICY_DEV` episodes have
zero native task success and zero positive reward.

This is consistent with the source generator: it implements a deterministic
TCP approach, compression, tangential contact probe, and release while keeping
the Allegro targets at the initial neutral posture. It does not implement the
task objectives for `pinch_tongs`, `hammer_nail`, or `click_mouse`. Therefore
these episodes are valid representation data but are not expert or scripted
successful interaction demonstrations suitable for behavior cloning.

The frozen rule requires `S4_3_0_DEMONSTRATION_CONTRACT_FAIL` and an immediate
downstream stop. No success predicate was frozen, no evaluation seeds were
selected, no causal policy integration was built, no ACT implementation was
trained, and no closed-loop rollout was run. Formal validation, TEST_V1, and
TEST_V2 were not used for policy decisions.

Final S4.3-2 classification: `STRUCTURAL_FAIL`.

S4.3-3 Diffusion Policy readiness: `NOT_READY`.

The next scientifically valid action is a separately frozen acquisition or
generation protocol for successful task-solving demonstrations. The contact
probe corpus must not be relabeled or used to train ACT merely to see whether
it works.
