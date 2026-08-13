# Tactile3D-UniT

[English](README.md) | [中文](README_zh-CN.md) | [Original UniT README](README_UniT.md)

## 项目概览

Tactile3D-UniT 是基于 [UniT](README_UniT.md) 的研究 fork。项目旨在将
Unified Physical Language 扩展为未来可统一二维视觉后果、三维几何后果、动作实现和接触动力学的表征。
当前处于 **S0 原始 UniT 复现进行中**：在完成上游 UniT 的复现和验证前，不声明触觉或三维分支已经实现。

## 阶段路线图

| 阶段 | 目标 | 状态 |
| --- | --- | --- |
| S0 | 原始 UniT 复现收口 | 进行中 |
| S1 | 可预测触觉 / 接触状态教师 | 计划中 |
| S2 | 接触动力学分支 | 计划中 |
| S3 | 视觉–动作–接触统一 token | 计划中 |
| S4 | RGB 点云 / 三维物理转移 | 计划中 |
| S5 | 完整 Tactile3D-UniT 共享物理词表 | 计划中 |
| S6 | VLA 集成 | 计划中 |
| S7 | 仿真与离线评估 | 计划中 |
| S8 | 真实 RGB-D + 接触桥接数据集 | 计划中 |
| S9 | 跨传感器 / 跨具身迁移 | 计划中 |

## 当前 TODO

- [x] 验证 S0 数据与模型资产
- [x] 验证 GR1 数据契约
- [ ] 加载官方 UniT checkpoint
- [ ] 复现离线评估
- [ ] 复现 RoboCasa 无头 rollout
- [ ] 复现官方 ID 指标
- [ ] 冒烟测试 tokenizer 训练
- [ ] 冒烟测试双系统训练
- [ ] 完成 M0

## 环境

S0 复现使用已有的 UniT 环境：

```bash
conda activate unit
```

上游安装与仿真配置请参见[原始 UniT 安装说明](README_UniT.md#installation)和
[`examples/environment_setup.sh`](examples/environment_setup.sh)。机器相关路径不要写入受 Git 跟踪的文件；请在 shell 或本地忽略文件中配置：

```bash
export UNIT_STORAGE=/path/to/unit_storage
export GR1_DATASET_DIR=/path/to/gr1_lerobot
export UNIT_FULLDATA_CKPT=/path/to/VLA-UniT-3B-fulldata
```

## 快速开始

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

这些命令只验证 S0 资产和 GR1 数据契约；不会加载完整策略 checkpoint、训练模型或运行评估 rollout。

### 查看 GR1 Episode

当前 GR1 源数据采用 LeRobot v2.0 schema；UniT 固定的官方 viewer 要求该布局中不存在的
`frame_index` 字段，无法直接读取。请改用 standalone 本地数据 viewer：它嵌入 RGB 视频、与
UniT 对齐的 goal frame（+16 steps）和原始关节 state/action 轨迹，不会伪装成仿真 replay：

```bash
python scripts/reproduce/visualize_gr1_data.py \
  --dataset-root "$GR1_DATASET_DIR" \
  --task gr1_unified.PnPWineToCabinetClose \
  --episodes 0 500 999

xdg-open .local/artifacts/visualization/\
gr1_unified.PnPWineToCabinetClose_episodes_0-500-999.html
```

生成的 HTML 是 standalone 文件，可由桌面浏览器直接打开。该工具是数据集 viewer，
并非仿真 replay 或物理三维轨迹 viewer。

## 核心流程

```text
环境
   ↓
S0 资产验证
   ↓
GR1 数据契约
   ↓
官方 Checkpoint
   ↓
离线评估
   ↓
RoboCasa 评估
```

当前仓库命令覆盖环境、资产验证和数据契约检查。上游 checkpoint 与评估入口请见
[`examples/README.md`](examples/README.md)，此处不会将其表述为已完成的 S0 工作。

## 本地运行产物策略

运行日志、生成的可视化、本地实验、缓存、机器专有配置和临时产物必须存放在
`.local/` 下；该目录被 Git 有意忽略。

## 文档索引

- [原始 UniT README](README_UniT.md) — 上游参考、安装说明和引用信息。
- [English README](README.md) — 英文项目概览和 S0 入口。
- [示例流水线](examples/README.md) — 上游训练与评估配方。
- [ID 评估说明](docs/evaluation_id_results.md) — 仓库现有评估文档。
- [`scripts/reproduce/`](scripts/reproduce/) — 可复用的 S0 验证和可视化工具。

## 许可证

本仓库保留 [Apache-2.0 License](LICENSE)。[NOTICE](NOTICE.txt) 记录了第三方组件的声明与许可证；相应条款仍适用于各自组件。

## 致谢

Tactile3D-UniT 基于 XPENG Robotics 的 UniT。fork 还保留了 [NOTICE](NOTICE.txt) 中说明的 NVIDIA Isaac GR00T 派生代码与接口。T-Rex 及相关触觉研究仅作为研究灵感；本仓库不声称移植了 T-Rex 实现。

## 引用

Tactile3D-UniT citation: TBD。

如使用 UniT，请引用：

```bibtex
@article{chen2026unit,
  title={UniT: Toward a Unified Physical Language for Human-to-Humanoid Policy Learning and World Modeling},
  author={Chen, Boyu and Chen, Yi and Qiu, Lu and Bai, Jerry and Ge, Yuying and Ge, Yixiao},
  journal={arXiv preprint arXiv:2604.19734},
  year={2026}
}
```
