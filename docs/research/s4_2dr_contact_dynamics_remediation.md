# S4.2-DR Contact-Dynamics baseline-ceiling audit and bounded remediation

S4.2-DR is an independent follow-up protocol. It preserves the historical
`S4_2_3_CONTACT_DYNAMICS_FAIL`: the frozen C3 dynamic validation MSE
(`0.017156061`) improved over C2 (`0.018156318`) by `5.509144%`, below the
original preregistered `10%` gate. Nothing in this follow-up changes that gate,
deletes that failure, or relabels the original protocol as passing.

DR1 reproduced the frozen checkpoints on the same 22,680-row train and
4,860-row validation pair identities, Contact-State targets, q70 dynamic mask,
normalization, and 0.54 s horizon. C2 and C3 have the same pair-conditioned
information: C2 receives `h_current` and `h_future - h_current`; C3 receives the
equivalent explicit current, future, and delta views. C2 has 954,048 parameters
and C3 has 1,251,520; both ran 16 epochs and approximately 720 optimizer steps.
C2 therefore has no larger parameter, optimization, or search budget.

DR2 uses only `h_current` for the diagnostic linear, ridge, fixed small MLP, and
kNN predictors. Ridge alpha and kNN k are selected on a deterministic,
episode-disjoint 80/20 split inside TRAIN, followed by a full-TRAIN refit and
one formal-validation evaluation. These predictors define an **EMPIRICAL
CONDITIONAL PREDICTABILITY REFERENCE**, not an oracle or a mathematically exact
irreducible floor. Local neighbor variance is likewise only a local conditional
dispersion estimate. The best diagnostic is ridge at dynamic MSE `0.388999679`,
which is not below C2; the formal outcome is
`NO_EMPIRICAL_HEADROOM_REFERENCE`.

DR3 freezes q90 thresholds from TRAIN before validation decomposition. C3 beats
C2 with positive paired-bootstrap lower bounds in all requested tasks and
contact regimes. The relative gains include `8.018%` for free-to-contact,
`5.993%` for contact-to-free, `5.746%` in high force, and `5.894%` in high
tangential. Zero, shuffled, reversed, different-episode, and mismatched-future
controls remain significantly worse in dynamic, boundary, high-force, and
high-tangential validation subsets. Control necessity also holds separately for
pinch_tongs, hammer_nail, and click_mouse.

The follow-up freezes two mutually exclusive acceptance routes before DR5.
Route A is limited to the existing frozen C3 and requires every ceiling,
headroom, critical-regime, control, semantic-probe, noncollapse, and reload gate.
Route B exists only if Route A fails and requires a bounded remediated candidate
to meet the original 10% magnitude threshold plus every semantic and structural
gate. At most R1, R2, and R3 may run, under the original epoch budget and no more
than 1.25 times its optimizer steps. Selection is validation-only.

The formal model-performance test remains unopened. Acceptance can only freeze
`E_c^sim`, `D_c^sim`, and the `[B,8,32]` `z_c^sim` convention and declare
S4.2-4 ready to resume. This task does not execute Action, Vision, VAC Bridge, or
any later stage.

## Final validation-only result

Route A failed only because no `h_current`-only empirical predictor beat the
pair-conditioned C2 transition autoencoder, so the required C2-to-reference gap
and C3 headroom-capture ratios were undefined. The existing C3 continued to
pass its positive paired improvement, critical-regime, control, semantic-probe,
noncollapse, shape, and deterministic-reload gates.

All three authorized remediation trials then ran. R1 reached dynamic MSE
`0.017446430` (3.910% over C2), R2 reached `0.016895035` (6.947%), and R3
reached `0.017224282` (5.133%). Every trial passed the positive-bootstrap,
overall-regression, control, semantic-probe, noncollapse, shape, and reload
gates. Every trial failed the same unchanged magnitude gate: none reached 10%
over frozen C2. The final decision is therefore
`S4_2DR_DYNAMICS_REMEDIATION_FAIL`. No checkpoint is accepted or frozen,
S4.2-4 is not ready, and the formal model-performance test remains unopened.
