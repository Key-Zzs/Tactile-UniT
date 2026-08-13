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

- [ ] Validate S0 data/model assets
- [ ] Validate GR1 data contract
- [ ] Load official UniT checkpoint
- [ ] Reproduce offline evaluation
- [ ] Reproduce RoboCasa headless rollout
- [ ] Reproduce official ID metric
- [ ] Smoke-test tokenizer training
- [ ] Smoke-test dual-system training
- [ ] Close M0

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
