# S4.2-R Contact-State Intrinsic-Dimension and Coverage Audit

S4.2-R is a separate follow-up to the original S4.2 Contact-State result. The
original result remains `S4_2_CONTACT_STATE_FAIL`: effective rank
`11.255626 < 16.0`. This audit does not change that threshold, result, or its
artifacts.

## Scope and integrity

The audit used only the frozen formal train and validation splits. It did not
load test arrays or test representation metrics and did not train a model. The
selected `B3-N1-R0.25` checkpoint retained SHA-256
`3d4519195e7a0d9f5af63399840a48121d5f614ea587c80b9461fc696a372a19`.
All S4.2-R evidence is written beneath the ignored, repository-relative root
`.local/artifacts/simulation/s4_2r/`; the original `.local/artifacts/simulation/s4_2/`
evidence is not overwritten.

## Intrinsic dimension

The audit computed effective rank, stable rank, participation ratio, PCA
`d90`/`d95`/`d99`, top-PC variance, full eigenvalue spectra, and a deterministic
TwoNN diagnostic. It covered raw current frames, 780-D flattened histories,
180-D temporal summaries, first differences, 390-D future targets, and the
frozen 256-D latent. It reported both the raw schema and a diagnostic
contact-active-masked schema. The latter replaces inactive CoP entries with the
TRAIN active-contact CoP mean so inactive schema zeros contribute no centered
CoP deviation; it never modifies a training tensor.

TRAIN-only source results include:

- active-frame raw effective rank `2.647927` and participation ratio `2.161358`;
- masked temporal-summary effective rank `5.024976` and participation ratio
  `3.595108`;
- masked flattened-history effective rank `7.017312`, with `d95 = 15`;
- train-standardized flattened-history effective rank `19.896041`, showing that
  channel scale and temporal redundancy materially affect the unstandardized
  spectrum;
- masked first-difference effective rank `91.359351`, showing that localized
  temporal changes are not globally collapsed;
- masked future-target effective rank `4.527652`.

The preregistration input `d_src` is `3.121517446651421`, the median of active
raw-frame effective/participation ranks and history-summary
effective/participation ranks. The exact source-reference payload has canonical
SHA-256 `62a33f61000606431bd3b34e6cb7dd3b6a903d3851ad56c0e152f6b6a47ddb7b`.

On validation, the frozen latent has effective rank `11.257566`, participation
ratio `5.641272`, `d95 = 21`, and top-PC explained variance `0.359593`. Its
effective rank is `4.0791x` active raw-frame rank and `2.2262x` temporal-summary
rank. All latent dimensions have non-negligible variance. Different-sample
latent distance has positive fifth percentile while exact duplicate distance is
zero.

TwoNN is explicitly diagnostic-only. Repeated static windows produced tied
nearest neighbors and made most whole-dataset TwoNN estimates unstable. The
active-contact current-frame estimate was stable (`3.179599`), consistent with
the covariance-spectrum source estimates. No hard conclusion relies on an
unstable TwoNN value.

The evidence supports a low-dimensional physical/contact source with strong
temporal redundancy, not a collapsed latent. First differences and standardized
histories retain substantially broader variation, while the predictive latent
uses more diverse factors than the source summaries.

## Thumb and region coverage

All five configured thumb bodies resolve by name to distinct collision geoms in
each task runtime. All 255 train/validation stored region audits report complete
mapping, no overlap, and no missing object bodies. The S4.2-R coverage analysis
does not read test arrays or test metadata.

A deterministic replay of frozen group `00`, perturbation `00` for each task
matched every stored tactile sample exactly. Raw MuJoCo contains thumb-to-table
contacts but zero thumb-to-task-object contacts. The extractor therefore has
zero matched thumb pairs and the frozen thumb channel correctly remains zero.
The classification is `THUMB_INACTIVITY_IS_DATA_COVERAGE`, not
`CONTACT_REGION_MAPPING_BUG`.

Palm, index, middle, and ring are active in both train and validation. Each task
and split supplies at least 50 dynamic-q70 and 50 boundary anchors across at
least two independent source trajectories. Original predictive, temporal,
contact, force, and trend gates remain strong. The frozen dataset is retained
with warning `THUMB_ZERO_ACTIVITY`.

## Decision boundary

R1 and R2 pass and permit a separately preregistered no-collapse protocol. No
existing checkpoint has yet been accepted under that new protocol at this audit
commit. No remediation training or downstream S4.2 stage has started.
