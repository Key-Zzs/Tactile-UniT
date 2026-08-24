# Tactile3D-UniT

[English](README.md) | [中文](README_zh-CN.md) | [Original UniT README](README_UniT.md)

## Project Overview

Tactile3D-UniT is a research fork of [UniT](README_UniT.md). It extends the
idea of a Unified Physical Language toward a representation that can eventually
unify 2D visual consequence, 3D geometric consequence, action realization, and
contact dynamics. This repository is at **S0 reproduction in progress**:
upstream UniT is being reproduced and validated before any tactile or 3D branch
is claimed as implemented.

## Stage Roadmap

| Stage | Goal | Status |
| --- | --- | --- |
| S0 | Original UniT reproduction closure | In Progress |
| S1 | Predictable Tactile / Contact-State Teacher | Planned |
| S2 | Contact-Dynamics Branch | Planned |
| S3 | Vision–Action–Contact Unified Token | Planned |
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
- [ ] Smoke-test dual-system training
- [x] Freeze original UniT representation baseline
- [ ] Close M0

Multi-GPU DDP validation is deferred because GPU availability is currently
restricted to a single project GPU.

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

## Environment

S0 reproduction uses the existing UniT environment:

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
S0 Asset Validation
   ↓
GR1 Data Contract
   ↓
Official Checkpoint
   ↓
Offline Eval
   ↓
RoboCasa Eval
```

The current repository commands cover the environment, asset validation, and
data contract checks. The upstream checkpoint and evaluation entry points are
documented in [`examples/README.md`](examples/README.md) and are deliberately
not presented here as completed S0 work.

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

## License

This repository retains the [Apache-2.0 License](LICENSE). The included
[NOTICE](NOTICE.txt) records third-party notices and licenses that continue to
apply to their respective components.

## Acknowledgements

Tactile3D-UniT builds on UniT by XPENG Robotics. The fork also retains code and
interfaces derived from NVIDIA Isaac GR00T where documented in
[NOTICE](NOTICE.txt). T-Rex and related tactile research are research
inspiration only; no T-Rex implementation is claimed in this repository.

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
