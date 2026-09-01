# S4.3-PD task-solving policy demonstration acquisition

This protocol replaces no historical result. The earlier S4.3-0 decision remains
`S4_3_0_DEMONSTRATION_CONTRACT_FAIL`: its 210 Contact-probing episodes contain zero
native task successes and are excluded from policy behavior cloning.

## Expert and task contracts

All three tasks use official raw teleoperation demonstrations from the pinned
`DexJoCo/DexJoCo-Datasets-Raw` revision recorded in the expert contract. Official
23D targets are converted to the policy-facing 22D Action and then passed through
the repository's single central 22D-to-23D adapter. The student record contains
only current RGB, current 22D proprio, current 30D tactile, and the 22D expert
Action. Native task state and the 23D environment Action are audit-only fields.

The pinch task is numerically sensitive at its third open threshold after the
float32 policy boundary. The frozen adapter therefore reuses the official late
open hand target for 30 control steps, retains the official final lifted TCP
target, and then returns to the official final close target. This is physical
Action replay: it does not set an object pose, task counter, reward, or success.

## Pilot evidence

The fixed pilot manifest contains four source groups and five visual
perturbations per group for each task. The first adaptation produced 15/20 native
pinch successes and exposed the float32 threshold issue; that complete iteration
is retained locally. After the bounded PD4 adaptation, a clean rerun of the same
fixed manifest produced 20/20 native successes for each task, with zero corrupt
episodes, invalid Actions, or non-finite values. The pilot audit also confirms
non-degenerate hand motion and produces representative videos, screenshots, and
Action/task-state/Contact traces.

## Frozen formal acquisition

Each task has 25 new official source groups, disjoint from the four pilot groups.
Groups 0–19 are assigned to `POLICY_TRAIN`; groups 20–24 are assigned to
`POLICY_DEV`. Every group has exactly five attempts, yielding 100 train attempts,
25 dev attempts, and 125 attempts per task (375 total). Exact group IDs and the
seed formula are in `configs/simulation/s4_3_pd_formal_acquisition.json`; the
expanded 375-row manifest and every source checksum are frozen under `.local`.

Formal generation must run all 375 identities. It never stops after collecting a
success count, never replaces a failed seed, and retains native expert failures in
the acquisition audit. Only all schema-valid native successes are eligible for
the policy dataset. Infrastructure failures may be rerun only with the exact same
identity and seeds, while preserving the failure record.

The per-task hard gates are at least 80/100 train successes, 20/25 dev successes,
100/125 total successes, and 80% overall success. A failure freezes the protocol
as failed; it does not reopen expert tuning.

## Commands

Freeze and verify the protocol before formal acquisition:

```bash
python scripts/simulation/generate_s4_3_policy_expert_data.py --freeze-only
python scripts/simulation/freeze_s4_3_pd_protocol.py
```

Formal acquisition is executed task-wise on independently locked, genuinely idle
GPUs with `MUJOCO_GL=egl`, `DISPLAY` unset, and the selected physical GPU masked
to logical device 0:

```bash
python scripts/simulation/generate_s4_3_policy_expert_data.py --task TASK
```

No ACT, Diffusion Policy, or later S4.3 stage is trained or started by this work.
