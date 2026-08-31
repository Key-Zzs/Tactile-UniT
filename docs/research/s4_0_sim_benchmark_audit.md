# S4.0 Simulation Benchmark Technical Audit

## Scope and protocol freeze

S4.0 is a technical audit and benchmark-selection stage. It does not port a
robot, implement a tactile proxy, train a policy, collect demonstrations, or
start S4.1. The paper protocol explicitly permits the simulation embodiment to
differ from Flexiv + RH56DFTP: the required invariant is the Vision / Action /
Contact contract, not robot appearance.

The decision is:

- **Primary:** DexJoCo.
- **Secondary / regression:** the repository-existing RoboCasa / robosuite /
  GR1 stack.
- **Optional scale-up:** Isaac Lab / Isaac Sim.
- **Data expansion:** DexMimicGen.
- **Readiness:** `S4_0_READY_WITH_ENVIRONMENT_WARNING` because the selected
  primary should run in a separate pinned environment rather than changing the
  accepted `unit` environment.

Scores below are technical-fit scores on a 0--4 scale. Their unweighted sums
are not a deployment ranking: availability, licensing, environment isolation,
and engineering cost are separate decision gates. This is why Isaac Lab's raw
sum does not override DexJoCo's lower-cost and immediately relevant task fit.

## Evidence basis

The audit inspected the pinned source, dependency metadata, task definitions,
observation/action paths, contact calls, rendering paths, and policy examples.
The existing stack was also exercised on the headless server. Primary upstream
references are [DexJoCo](https://github.com/brave-eai/dexjoco),
[DexMimicGen](https://github.com/NVlabs/dexmimicgen),
[Isaac Lab](https://github.com/isaac-sim/IsaacLab), and the
[RoboCasa GR1 tasks](https://github.com/robocasa/robocasa-gr1-tabletop-tasks).
Isaac Lab's sensor semantics were checked against its official
[contact-sensor documentation](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/contact_sensor.html).

Audited identities:

| Candidate | Audited commit | License summary |
|---|---|---|
| RoboCasa GR1 | `4840e671596f93ca03651524b9f72ffb1aadfeff` | MIT; included third-party components retain their terms |
| robosuite backend | `a071383d53568ab798eb315c0e95357911be922d` | MIT |
| DexJoCo | `8d23b0fab23b17a58c4b55f3942e17013aaf8267` | MIT; bundled assets/components retain their terms |
| DexMimicGen | `940e8a1b3ad70eb1925ada6b364b197de6bb2af9` | NVIDIA Source Code License, non-commercial research/evaluation; datasets CC-BY-4.0 |
| Isaac Lab | `bffdce9d7467f349bfc8ab111fe633a0bb234851` | BSD-3-Clause framework; Isaac Sim and assets have additional terms |

**External benchmarks retain their original licenses.** Adding a submodule
does not relicense its contents under Tactile-UniT.

## Existing RoboCasa / robosuite / GR1 stack

1. **Version.** The installed editable sources resolve to RoboCasa GR1 commit
   `4840e671596f93ca03651524b9f72ffb1aadfeff` and robosuite commit
   `a071383d53568ab798eb315c0e95357911be922d` (`robocasa==0.2.0`,
   `robosuite==1.5.1`).
2. **Packaging.** They are editable package installs backed by ignored local
   checkouts, not tracked vendored files or superproject submodules.
3. **Embodiment.** `GR1ArmsAndWaistFourierHands`.
4. **Observation.** Left/right arm (7 each), left/right hand (6 each), waist
   (3), two 256x256 RGB preprocessing views derived from egoview, and a coarse
   language annotation. The proprioceptive state is 29-D.
5. **Action.** Absolute joint targets through the BASIC composite controller:
   left arm 7 + right arm 7 + left hand 6 + right hand 6 + waist 3 = 29-D.
6. **Cameras.** The backend supports RGB, depth, and multiple cameras; the
   current policy contract consumes one egoview stream through two transforms.
7. **Contact access.** Direct MuJoCo `data.contact[0:ncon]` access is available.
8. **Physics fields.** Geom/body pairs, world contact point, contact frame,
   penetration/distance, and the 6-D local wrench from `mj_contactForce` are
   accessible. robosuite also exposes contact helpers and robot end-effector
   force/torque readings. MuJoCo does not expose a ready-made taxel signal;
   impulse can be estimated later by force integration under a frozen timestep.
9. **Headless.** Import, reset, ten zero-action steps, raw contact extraction,
   and offscreen RGB passed with EGL and no `DISPLAY`.
10. **Rollout reuse.** `SimulationInferenceClient`, Gym vectorization,
    `MultiStepWrapper`, success aggregation, and video output can be preserved
    behind a benchmark environment adapter.
11. **Role.** Use as upstream regression, not the primary paper benchmark. It
    is the cheapest runtime check and the native GR00T path, but its task/contact
    diversity is weaker than DexJoCo for the proposed paper questions.

The smoke observed a raw 800x1280 RGB frame and up to 122 simultaneous contacts.
Large forces during initial/static penetration show that the later proxy must
freeze region masks, units, clipping/outlier handling, and temporal integration.
Gym's passive checker also warned that returned observations did not satisfy the
declared observation space. This did not invalidate physics/rendering, but S4.1
must correct or explicitly disable that checker only after a schema test.

## Candidate matrix

| Criterion (0--4) | RoboCasa | DexJoCo | DexMimicGen | Isaac Lab |
|---|---:|---:|---:|---:|
| Task coverage | 2 | 4 | 4 | 4 |
| Dexterous | 2 | 4 | 4 | 4 |
| Bimanual | 2 | 4 | 4 | 3 |
| Contact rich | 2 | 4 | 4 | 4 |
| Tactile proxy | 4 | 4 | 4 | 4 |
| Headless | 4 | 3 | 3 | 4 |
| Data | 4 | 4 | 4 | 3 |
| ACT | 3 | 3 | 3 | 3 |
| Diffusion Policy | 3 | 2 | 3 | 3 |
| GR00T | 4 | 2 | 2 | 3 |
| pi0.5 | 2 | 4 | 2 | 2 |
| Randomization | 3 | 4 | 3 | 4 |
| Scale | 2 | 2 | 2 | 4 |
| ICRA relevance | 3 | 4 | 4 | 4 |
| ICLR relevance | 3 | 4 | 3 | 4 |
| **Unweighted total** | **43** | **52** | **49** | **53** |
| Engineering cost | LOW | MEDIUM | HIGH | VERY HIGH |
| Environment | Compatible | Split recommended | Split recommended | Split required |

### Interpretation by candidate

- **DexJoCo:** eleven stock single- and bimanual tasks cover tool use,
  insertion, articulated interaction, pinch, impact, and long horizons. It has
  direct MuJoCo contact access, ego/wrist RGB, demonstrations, dynamics and
  visual randomization, and an upstream pi0.5 path. This is the best compact
  ICRA protocol and still supports ICLR representation/generalization studies.
- **RoboCasa:** already reproducible and integrated with GR00T; ideal as a
  low-cost regression and adapter sanity check.
- **DexMimicGen:** nine bimanual environments and large-scale demonstration
  generation make it valuable for data expansion. Its research-only licensing,
  dependency downgrades, and larger integration surface argue against primary
  status.
- **Isaac Lab:** strongest vectorized scale, contact sensor, and randomization
  path. The separate Isaac Sim/Python stack and asset/controller work make it an
  optional scale-up after the core result, not the first implementation target.

Stock benchmark robots/hands are acceptable. None of these choices requires an
RH56DFTP asset to test the representation hypothesis.

## Headless and environment audit

| Candidate | Headless conclusion | S4.0 execution |
|---|---|---|
| RoboCasa | EGL works without `DISPLAY`; reset, step, contacts, RGB passed | Actual bounded smoke, PASS |
| DexJoCo | Official `rgb_array` MuJoCo path; source verified | Runtime smoke skipped to protect `unit` from MuJoCo 3.4/Python 3.11 conflict |
| DexMimicGen | robosuite offscreen mode with interactive render omitted; source verified | Runtime smoke skipped because requirements would downgrade accepted packages |
| Isaac Lab | Official headless and vectorized execution supported | Not installed; separate Python 3.12/Isaac Sim 6.0.x stack required |

The server was genuinely headless (`DISPLAY` unset). EGL physics/rendering ran
on a safely locked free RTX 4090; no workload was killed or shared. Dependency
resolution was dry-run only. The canonical `unit` environment was not changed.

## Contact and simulated tactile feasibility

| Candidate | Pair / IDs | Point / distance | Normal + tangent | Sensor route | Feasibility |
|---|---|---|---|---|---|
| RoboCasa | MuJoCo geom/body IDs | `pos`, `frame`, `dist` | `mj_contactForce` | robosuite EEF F/T | Direct; verified at runtime |
| DexJoCo | MuJoCo geom pairs; tasks already read `ncon` | Native MuJoCo fields | `mj_contactForce` via adapter | Custom site/touch sensors possible | Direct backend access |
| DexMimicGen | robosuite/MuJoCo geom/body pairs | Native MuJoCo fields | Native MuJoCo wrench | robosuite sensor hooks | Direct backend access |
| Isaac Lab | Contact sensor body/filter identities | Averaged contact positions | Net/filtered normal and friction force, with history | Native `ContactSensorData` | Best native sensor abstraction |

The benchmark-independent draft contract, intentionally **not implemented**, is
per finger/fingertip/named hand region:

`[contact_occupancy, normal_force, tangential_force,
contact_position_or_center_of_pressure, optional_contact_impulse_or_force_integral]`.

This is sufficient for contact onset/release, force trends, slip-like tangent
dynamics, missing tactile, and noise/dropout/latency studies. Simulation uses
`E_T^sim`; the real sensor later uses `E_T^RH`; both align in the shared Contact
space. Exact RH56DFTP taxel electronics are neither necessary nor claimed.

## Observation, action, and policy compatibility

All candidates can produce `o_t = {Vision, Proprio, Contact}` after adapters.
DexJoCo provides ego/wrist RGB and flattened state. Its environment target is
23-D for one arm or 46-D bimanual with quaternion rotations; the policy-facing
rotvec actions are 22-D or 44-D and already support chunked execution. S4.1 must
freeze this conversion, controller frequency, history, and temporal ordering in
`PlannedActionChunk`; it must never infer them from array length alone.

RoboCasa's 29-D absolute joint target path maps cleanly through an existing
adapter. DexMimicGen needs per-environment robosuite action/state normalization.
Isaac Lab needs a task-specific manager/controller and tensor/device adapter.
Contacts must be grouped by named geom/body regions and aligned to observation
timestamps in every case.

Recommended policy order:

1. **ACT** as the first compact, architecture-neutral chunked-action baseline.
2. **Diffusion Policy** after the observation/action contract is frozen; use the
   DexJoCo converter only after auditing its uninitialized pinned nested module.
3. **pi0.5** as the native DexJoCo foundation baseline.
4. **GR00T** as an optional cross-embodiment comparison and retained RoboCasa
   regression, not a prerequisite for the primary result.

No policy was trained in S4.0.

## Paper task shortlist

| Exact DexJoCo task | Contact regime | Paper value | Required S4.1+ work |
|---|---|---|---|
| `bimanual_assembly` | Two-hand stabilization and terminal peg/socket contact | Precision transitions and bimanual coordination | 46-D adapter, contact-region map, episode logger |
| `hammer_nail` | Intermittent impact/high force | Contact-event prediction and calibrated risk | 23-D adapter, hammer/hand groups, force clipping |
| `pinch_tongs` | Sustained pinch and tangential load | Force trends, slip-like dynamics, missing tactile | 23-D adapter, fingertip/tongs regions |
| `bimanual_unlock_ipad` | Ordered localized fingertip presses | Exact action order and uncertainty | 46-D adapter, fingertip/screen regions, prompt handling |
| `bimanual_microwave_cook` | Grasp, articulated door, placement, button | Long-horizon composition and recovery | Multi-region mapping and stage logger |
| `bimanual_hanoi` | Repeated grasp/release/support and coordination | Generalization across recurring contact phases | Disk/peg regions and subgoal logger |

## Paper-question coverage

- **Q1 continuous VAC:** compare continuous Vision/Action/Contact against simple
  concatenation on identical trajectories.
- **Q2 shared/private:** ablate shared versus private Contact factors across
  tasks and contact regimes.
- **Q3 action + current contact:** predict future Contact with/without current
  Contact context under fixed actions.
- **Q4 exact action temporal order:** permute or delay actions within the same
  chunk and measure Contact/rollout change.
- **Q5 missing tactile:** drop Contact at evaluation and measure graceful
  degradation relative to vision/action-only models.
- **Q6 uncertainty:** test whether predictive uncertainty identifies impact,
  slip-like, failed-grasp, or unsafe-contact events.
- **Q7 tactile corruption:** sweep force noise, region dropout, delay, and
  temporal jitter.
- **Q8 unseen dynamics:** hold out mass, friction, pose, appearance, and camera
  randomization ranges.
- **Q9 policy benefit:** compare success, contact safety, and recovery with and
  without Tactile-UniT features.

DexJoCo supports all nine through native task diversity, contacts, action
chunks, demonstrations, and randomization; no morphological match is required.

## S4.1 recommended execution plan (not started)

1. Create and pin a separate DexJoCo runtime; retain `unit` unchanged.
2. Reproduce one official headless task with exact pinned assets/dependencies.
3. Implement a thin benchmark environment adapter in the superproject.
4. Freeze the unified Vision / Action / Contact observation schema.
5. Freeze 22-D/44-D policy action to 23-D/46-D environment conversions,
   frequency, chunking, and exact temporal ordering.
6. Implement and test the named-region MuJoCo contact proxy.
7. Implement `E_T^sim` into the shared Contact space.
8. Add deterministic episode/dataset logging with unit and frame conventions.
9. Add ACT, then Diffusion Policy adapters; retain pi0.5 as the foundation
   baseline.
10. Add a bounded six-task headless evaluator and success/safety/recovery
    metrics.
11. Freeze dynamics and tactile noise/dropout/latency protocols and deterministic
    acceptance tests.

Estimated effort is **MEDIUM** for the primary adapter/proxy/evaluator, excluding
training. Dependencies are the pinned DexJoCo runtime/assets, a contact-region
map, explicit action conversion, and dataset/policy boundaries. DexMimicGen and
Isaac Lab remain later independent environments. Do not create any of them in
S4.0.

## Limitations and acceptance boundary

- Only the repository-existing stack received an actual runtime smoke. External
  conclusions combine pinned-source inspection and dependency dry-runs.
- Contact availability establishes proxy feasibility, not sensor realism or
  sim-to-real transfer.
- The smoke used zero actions and does not establish task success, long-rollout
  stability, determinism, or policy quality.
- Raw MuJoCo contact forces require a frozen timestep, units, region mapping,
  aggregation, and outlier protocol before scientific use.
- Licenses and asset terms require a fresh release-time review.
- S4.0 selects the protocol; it makes no M4 claim.

The machine-readable decision is in
`configs/simulation/s4_0_benchmark_audit.json`. Local-only runtime evidence is
emitted under `.local/artifacts/simulation/s4_0/`; it is deliberately untracked.
