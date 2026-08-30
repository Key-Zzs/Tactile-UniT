# Track B — S3.2-Q Semantic-Preserving Contact Tokenizer

## Status and decision

Track B is complete. The preregistered outcome is:

`CONTINUOUS_CONTACT_RECOMMENDED`

No evaluated 112-bit discrete Contact tokenizer satisfies the combined reconstruction, Contact-transition, force-trend, rare-boundary, temporal-control, and non-collapse gates. Track C should preserve the accepted continuous transition latent

`z_c = E_c(h_t^c, h_{t+16}^c) ∈ R^(8×32)`

and integrate it through continuous cross-attention, gating, or continuous latent alignment. This result does not claim information-theoretic losslessness.

## Frozen protocol

- Base SHA: `ead93bcb9ec3be3fe781dd734433eb82bb1819fd`
- Canonical horizon: 16 frames
- Canonical pairs: 279,680 train; 17,504 validation; 17,504 untouched test
- Split identity: accepted S1/S2 episode-disjoint manifest
- S1 Teacher SHA256: `54aedbfe0d72b18822624874ef3724512357c31ea03876513c6dea75d3aae8ac`
- S2 E_c/D_c checkpoint SHA256: `c36c0531bba461875384cebf6bd91c34d43d3f84d2083c15c47ae7dee4e64fa4`
- S2 transition manifest SHA256: `2e9a14d13c80e24464e4e1bb47318ceb0aa8459f9e62cac3506d90c810667c72`
- Q_BASE_2 SHA256: `7cc27e3553d2cfdc605e91ebeebe1982dfc4feb570695e7ef6f3576fdc5b5dd5`
- Q_BASE_3 SHA256: `1f4acede2343ca330ace61e6d310300a6ac034ac01a7f4a2312efb4f88e1abef`

All whitening/PCA statistics, semantic objectives, and codebooks were fitted on train. Candidate and regularization selection used validation only. Test was evaluated after selection. Frozen S2 encoder/decoder parameter digests were identical before and after evaluation.

Track B used physical GPU2 under the shared repository lock, exposed inside the process only as `cuda:0`. Physical GPU3 was occupied by another workload during final evaluation and was not shared. GPU0/1 were never exposed.

## Q0 — semantic error diagnosis

The frozen ordinary two-stage Contact RQ has global test quantization MSE 0.04174. The preregistered flattened-PC rank thirds show:

| Train PCA band | Explained variance | Contact F1 from band | Force F1 from band | Error energy |
|---|---:|---:|---:|---:|
| High, PCs 0–84 | 98.610% | 0.540 | 0.615 | 0.09784 |
| Mid, PCs 85–169 | 1.295% | 0.333 | 0.370 | 0.01584 |
| Low, PCs 170–255 | 0.095% | 0.173 | 0.316 | 0.01189 |

Important semantic information is distributed beyond the highest-variance subspace, but the low-variance third is not the primary carrier of Contact-transition information. The failure is therefore not well described as only low-variance semantic loss.

The stronger result is boundary/dynamic asymmetry:

| Transition | Windows | Ordinary RQ MSE |
|---|---:|---:|
| free→free | 7,052 | 0.00585 |
| free→contact | 342 | 0.13941 |
| contact→contact | 9,807 | 0.06091 |
| contact→free | 303 | 0.14651 |

Boundary MSE is 3.77× non-boundary MSE. A train-only native probe applied before and after quantization drops Contact macro-F1 from 0.676 to 0.407. Its boundary semantic-direction error is 1.657 versus 0.639 on non-boundaries, and its boundary margin loss is 1.022 versus 0.338. Dynamic examples are also substantially more damaged than static examples.

Adding the second residual stage reduces cumulative MSE from 0.05291 to 0.04174, but does not restore task semantics. A deterministic 256-sample output-direction Jacobian estimate found nonzero error projection onto decoder-sensitive directions; it is explicitly a Hutchinson directional estimate rather than a full Jacobian.

Q0 diagnosis: `MIXED`, with evidence for `DYNAMIC_BOUNDARY_UNDERWEIGHTED` and `EUCLIDEAN_OBJECTIVE_MISMATCH`.

## Bitrate accounting

- Ordinary, whitened, predictive Q1: `8 queries × 2 stages × log2(128) = 112 bits/transition`
- Q2 semantic: `8 × 1 × log2(128) = 56 bits/transition`
- Q2 private residual: `8 × 1 × log2(128) = 56 bits/transition`
- Q2 full: 112 bits/transition

No algorithmic improvement is attributed to an unreported rate increase.

## Q1 — single-stream results

Validation selected ZCA whitening with regularization 0.01. Across all six PCA/ZCA candidates, inverse-whitening maximum absolute error was at most `1.02e-5`. Whitening was numerically sound but did not improve the Contact objective.

| Candidate | Bits | Dynamic MSE | R_recon | Contact F1 | R_contact | Force F1 | R_force | Native R² | CKA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ordinary | 112 | 0.02307 | 0.603 | 0.484 | 0.614 | 0.558 | 0.829 | 0.692 | 0.948 |
| Whitened | 112 | 0.02588 | 0.549 | 0.481 | 0.607 | 0.561 | 0.836 | 0.698 | 0.891 |
| Predictive | 112 | 0.02378 | 0.609 | 0.478 | 0.601 | 0.553 | 0.818 | 0.574 | 0.900 |

All Q1 candidates pass correct-vs-reversed, different-episode shuffled, and mismatched-future temporal controls. None collapses; every stage uses 124–128 codes, query collapse is false, and mild-noise index agreement is at least 0.996. These healthy diagnostics do not compensate for the failed task-retention gates.

Rare-boundary recall remains poor. Ordinary free→contact/contact→free recall is 0.114/0.059; whitened is 0.094/0.066; predictive is 0.108/0.066. Continuous z_c reaches 0.319/0.257.

Exact same-episode test links are available for 190 t+24 windows and 644 t+32 windows. The predictive learned heads obtain MSE 0.00911 and 0.01521, respectively. These small deterministic subsets support multi-horizon reporting but are not used to weaken the canonical t+16 gates.

## Q2 — semantic plus private residual

Q2 maintains the 112-bit baseline budget with a 56-bit semantic stream and 56-bit private residual stream.

| Representation | Bits | Dynamic MSE | R_recon | Contact F1 | R_contact | Force F1 | R_force | Native R² | CKA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Semantic only | 56 | 0.03007 | 0.489 | 0.428 | 0.500 | 0.494 | 0.689 | 0.473 | 0.908 |
| Private only | 56 | 0.05988 | 0.191 | 0.443 | 0.530 | 0.565 | 0.845 | — | — |
| Full | 112 | 0.02162 | 0.645 | 0.515 | 0.675 | 0.570 | 0.855 | 0.650 | 0.947 |

The structure does not bypass the semantic stream: semantic-zero dynamic MSE rises to 0.05988, private-zero rises to 0.03007, shuffled-semantic rises to 0.04981, and shuffled-private rises to 0.03190, versus 0.02162 full. The registered bypass flag is false. Both streams use all 128 codes and avoid hard/query collapse.

However, the semantic stream independently fails its required `R_contact ≥ 0.90` and `R_force ≥ 0.90` gates, rare-boundary retention fails, and full `R_recon=0.645` is below 0.80. An honest non-bypassed hierarchy is still insufficient.

The Q2 semantic/full learned heads report exact-link t+24 MSE 0.00873/0.00836 and t+32 MSE 0.01467/0.01441. Canonical conclusions remain based on the full untouched t+16 test set.

## Answers to the scientific questions

1. Ordinary Euclidean RVQ damage is not primarily confined to low-variance directions. It disproportionately damages dynamic and rare-boundary samples and has large projection onto train-probe semantic directions.
2. Contact-transition information is strongest in the high-variance PC third, with useful information also present in the mid band. Boundary-relevant probe errors are distributed and cannot be rescued by equalizing only the lowest-energy tail.
3. Train-only PCA/ZCA whitening is stable but does not improve same-rate task retention.
4. Dynamic-aware predictive training does not overcome the stable-transition-dominated objective at the evaluated capacity and rate.
5. Predictive, temporal, and relational objectives improve the scientific diagnosis but do not outperform ordinary RVQ on the combined gates. Euclidean reconstruction alone is mismatched, yet the tested semantic objectives are still insufficient.
6. A single discrete semantic Contact stream is not sufficient.
7. Semantic plus Contact-private residual is structurally viable and non-bypassed, but neither semantic nor full reconstruction gates pass; it is not ready for Track C.
8. Because continuous z_c retains accepted M2 performance while all bounded discrete candidates damage Contact Dynamics, Track C should retain continuous z_c.

## Track C output contract

- Interface type: `CONTINUOUS`
- Continuous component: `z_c [B,8,32]`, float32
- Normalization: native frozen S2 E_c output; do not apply Q1 whitening
- Checkpoint: frozen S2 E_c/D_c
- Checkpoint SHA256: `c36c0531bba461875384cebf6bd91c34d43d3f84d2083c15c47ae7dee4e64fa4`
- Decoder interface: `(z_c [B,8,32], h_t [B,256]) -> h_future [B,256]`
- Cross-modal use: continuous gated cross-attention or continuous latent alignment is allowed
- Contact-private information: preserve native Contact detail on the Contact side; do not force it into the Original UniT frozen codebook

Track C is not started in this worktree. Track A completion remains independently required.

## Runtime evidence

- `.local/artifacts/tactile_unit/s3_2_q/q0_diagnosis.json`
- `.local/experiments/tactile_unit/s3_2_q/training_summary.json`
- `.local/artifacts/tactile_unit/s3_2_q/evaluation.json`
- `.local/artifacts/tactile_unit/s3_2_q/final_decision.json`
- `.local/artifacts/tactile_unit/s3_2_q/HUMAN_ACCEPTANCE.md`
- `.local/artifacts/tactile_unit/s3_2_q/plots/`
