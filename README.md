# Tactile3D-UniT

[English](README.md) | [中文](README_zh-CN.md) | [Original UniT README](README_UniT.md)

## Project Overview

Tactile3D-UniT is a research fork of [UniT](README_UniT.md). It extends the
idea of a Unified Physical Language toward a representation that can eventually
unify 2D visual consequence, 3D geometric consequence, action realization, and
contact dynamics. **M0-R, S1, and S2 are complete: M1 establishes a predictable
continuous contact-state latent, and M2 establishes a predictive continuous
contact-dynamics representation.** S3 is in progress: S3.0 completes the
shared-codebook compatibility audit and S3.1 establishes the paired T-Rex
vision–action–contact data contract. Contact tokens are not yet integrated into
UniT's shared RQ and no shared tactile token is claimed.

## Stage Roadmap

| Stage | Goal | Status |
| --- | --- | --- |
| M0-R | Original UniT representation-ready reproduction | Complete |
| S1 | Predictable Tactile / Contact-State Teacher | Complete (M1 established) |
| S2 | Predictive Contact-Dynamics Branch | Complete (M2 established) |
| S3 | Vision–Action–Contact Unified Token | In progress (S3.0 and S3.1 complete) |
| S4 | RGB Point Cloud / 3D Physical Transition | Planned |
| S5 | Full Tactile3D-UniT Shared Physical Vocabulary | Planned |
| S6 | VLA Integration | Planned |
| S7 | Simulation & Offline Evaluation | Planned |
| S8 | Real-World RGB-D + Contact Bridge Dataset | Planned |
| S9 | Cross-Sensor / Cross-Embodiment Transfer | Planned |

## Current TODOs

- [x] Validate S0 data/model assets
- [x] Validate GR1 data contract
- [x] Load official UniT checkpoint
- [x] Reproduce offline evaluation
- [x] Reproduce RoboCasa headless rollout
- [x] Reproduce official ID evaluation protocol
- [x] Establish canonical local UniT baseline
- [x] Validate official OOD evaluation protocols
- [x] Validate single-task tokenizer training behavior
- [x] Validate single-GPU multi-task tokenizer mixture
- [ ] Validate dual-system DDP training when the required resources are available
- [x] Freeze original UniT representation baseline
- [x] M0-R Original UniT representation-ready reproduction
- [x] S1 Predictable Contact-State Teacher
- [x] S2 Predictive Contact-Dynamics Branch
- [ ] S3 Vision–Action–Contact Shared Physical Tokenizer — in progress
- [x] S3.0 Shared-Codebook Compatibility Audit
- [x] S3.1 Paired Vision–Action–Contact Data Contract
- [ ] S3.2 Contact Adaptor

Multi-GPU DDP validation remains a deferred M0-R resource item; it is not a
prerequisite for the completed single-GPU representation milestone.

The released `VLA-UniT-3B-fulldata` checkpoint has been validated on the full
GR1 ID protocol, and the resulting local reproduction is frozen as the
project's canonical UniT baseline. See
[`configs/reproduction/baselines/unit_gr1_fulldata.json`](configs/reproduction/baselines/unit_gr1_fulldata.json)
for the public protocol and baseline metadata.

The canonical Original UniT representation benchmark is now established. T4
freezes the official nested tokenizer, deterministic held-out GR1 sampling, and
the shared L1–L5 alignment metric protocol for future UniT, Tactile-UniT,
UniT-3D, and Tactile3D-UniT comparisons. See
[`configs/reproduction/baselines/unit_representation_gr1.json`](configs/reproduction/baselines/unit_representation_gr1.json)
for the tracked benchmark specification; generated tensors and results remain
local under `.local/artifacts/reproduction/t4/`.

S1 freezes an episode-disjoint, physical-time benchmark over the public T-Rex
60-D wrench history and a continuous 256-D contact-state representation. The
tracked protocol is
[`configs/tactile_teacher/s1_contact_state_teacher.json`](configs/tactile_teacher/s1_contact_state_teacher.json);
datasets, checkpoints, latent tensors, metrics, and plots remain local under
`.local/`. Image/deformation modalities and UniT integration are deferred.

S2 freezes the accepted S1 Teacher and models contact-state transitions at
`k=16` frames. The current `[t-15,t]` and future `[t+1,t+16]` Teacher windows
share zero raw wrench samples; their anchors are separated by `16/30 = 0.533333`
seconds while each history spans `0.500` seconds. The resulting continuous
transition code has shape `[B,8,32]`, matching Original UniT's T4 VQ-input
geometry without quantizing or connecting it to the shared RQ. See
[`configs/contact_dynamics/s2_contact_dynamics.json`](configs/contact_dynamics/s2_contact_dynamics.json);
checkpoints, cached latents, metrics, and plots remain local under `.local/`.

S3.0 audits the continuous contact transition code directly against the frozen
Original UniT residual VQ and establishes the shared-codebook compatibility
decision. The result recommends evaluating a lightweight 32-to-32 contact
adaptor before the shared RQ in the next integration stage; no adaptor or
shared codebook is trained in S3.0. See
[`configs/tactile_unit/s3_0_codebook_compatibility.json`](configs/tactile_unit/s3_0_codebook_compatibility.json).
S3.1 then freezes the same-transition T-Rex `head_left` RGB pair, 16-step
action chunk, anchored state, and accepted S1/S2 contact representations under
the existing episode-disjoint split and S2 pair identities. The public data and
interface definition is
[`configs/tactile_unit/s3_1_paired_vac_contract.json`](configs/tactile_unit/s3_1_paired_vac_contract.json).
The released action branch still requires later T-Rex-specific category
parameters; no model or adaptor is trained in S3.1. S3 remains in progress.

## Environment

M0-R reproduction, S1, and S2 use the existing UniT environment:

```bash
conda activate unit
```

For the upstream installation and simulator setup, see the [original UniT
installation guide](README_UniT.md#installation) and
[`examples/environment_setup.sh`](examples/environment_setup.sh). Keep machine
paths outside tracked files and configure them in the shell or a local ignored
file:

```bash
export UNIT_STORAGE=/path/to/unit_storage
export GR1_DATASET_DIR=/path/to/gr1_lerobot
export UNIT_FULLDATA_CKPT=/path/to/VLA-UniT-3B-fulldata
```

## Quick Start

```bash
git clone https://github.com/Key-Zzs/Tactile-UniT.git
cd Tactile-UniT

conda activate unit
export UNIT_STORAGE=/path/to/unit_storage
export GR1_DATASET_DIR=/path/to/gr1_lerobot

python scripts/reproduce/check_s0_assets.py --unit-storage "$UNIT_STORAGE"
python scripts/reproduce/check_gr1_data_contract.py \
  --dataset-root "$GR1_DATASET_DIR" --mode full
```

These commands validate S0 assets and the GR1 data contract only. They do not
load a full policy checkpoint, train a model, or run an evaluation rollout.

### View a GR1 Episode

The current GR1 source data uses the LeRobot v2.0 schema. The UniT-pinned
official viewer cannot directly read this layout because it expects a
`frame_index` field that is not present. Use the standalone local data viewer
instead; it embeds RGB video, a UniT-aligned goal frame (+16 steps), and raw
joint state/action trajectories without claiming a simulator replay:

```bash
python scripts/reproduce/visualize_gr1_data.py \
  --dataset-root "$GR1_DATASET_DIR" \
  --task gr1_unified.PnPWineToCabinetClose \
  --episodes 0 500 999

xdg-open .local/artifacts/visualization/\
gr1_unified.PnPWineToCabinetClose_episodes_0-500-999.html
```

The generated HTML is standalone and opens directly in a desktop browser. This
is a dataset viewer, not a simulator replay or physical-3D trajectory viewer.

## Core Workflow

```text
Environment
   ↓
M0-R Asset Validation
   ↓
GR1 Data Contract
   ↓
Official Checkpoint
   ↓
Offline Eval
   ↓
RoboCasa Eval
   ↓
S1 Wrench History Contract
   ↓
Continuous Contact-State Teacher (M1)
   ↓
Non-overlapping Contact Transition Contract (k=16)
   ↓
Continuous 8×32 Contact-Dynamics Code (M2)
```

S2 remains tactile-only and continuous. S3.1 adds the paired
Vision–Action–Contact data contract; shared quantization, learned fusion, and
UniT integration remain for S3.2 or later.

## Local Runtime Artifact Policy

Runtime logs, generated visualizations, local experiments, caches,
machine-specific configs, and temporary artifacts must be stored under
`.local/`, which is intentionally ignored by Git.

## Documentation Index

- [Original UniT README](README_UniT.md) — upstream reference, setup, and citation.
- [中文 README](README_zh-CN.md) — Chinese project overview and S0 entry points.
- [Example pipelines](examples/README.md) — upstream training/evaluation recipes.
- [ID evaluation notes](docs/evaluation_id_results.md) — existing evaluation documentation.
- [`scripts/reproduce/`](scripts/reproduce/) — reusable S0 validation and visualization tools.
- [`scripts/tactile/`](scripts/tactile/) — S1 data, training, evaluation, visualization, and M1 audit tools.
- [`scripts/contact_dynamics/`](scripts/contact_dynamics/) — S2 transition-cache, training, evaluation, visualization, and M2 audit tools.
- [`scripts/tactile_unit/`](scripts/tactile_unit/) — S3 compatibility, paired-contract, frozen-branch, and synchronization audit tools.

## License

This repository retains the [Apache-2.0 License](LICENSE). The included
[NOTICE](NOTICE.txt) records third-party notices and licenses that continue to
apply to their respective components.

## Acknowledgements

Tactile3D-UniT builds on UniT by XPENG Robotics. The fork also retains code and
interfaces derived from NVIDIA Isaac GR00T where documented in
[NOTICE](NOTICE.txt). The isolated S1 VQ baseline adapts the MIT-licensed T-Rex
tactile VQ-VAE design; attribution and license terms are recorded in
[NOTICE](NOTICE.txt). The T-Rex repository itself is not vendored.

## Citation

Tactile3D-UniT citation: TBD.

For UniT, please cite:

```bibtex
@article{chen2026unit,
  title={UniT: Toward a Unified Physical Language for Human-to-Humanoid Policy Learning and World Modeling},
  author={Chen, Boyu and Chen, Yi and Qiu, Lu and Bai, Jerry and Ge, Yuying and Ge, Yixiao},
  journal={arXiv preprint arXiv:2604.19734},
  year={2026}
}
```
