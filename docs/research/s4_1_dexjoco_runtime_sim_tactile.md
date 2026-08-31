# S4.1 DexJoCo Runtime and Simulated Tactile Contract

## Scope and decision boundary

S4.1 is runtime infrastructure. It establishes a reproducible DexJoCo process,
a repository-owned adapter, physical-time contracts, a physics-derived contact
proxy, and a small debug dataset. It does not train `E_T^sim`, a Contact
teacher, a policy, ACT, Diffusion Policy, pi0.5, GR00T, RL, BC, or DAgger.

## Why DexJoCo

S4.0 selected the pinned DexJoCo source because its official task set combines
dexterous hands, tool contact, bimanual tasks, direct MuJoCo state, offscreen
rendering, and a documented policy interface. S4.1 uses the official
`pinch_tongs` task: it is single-arm, contact-rich, exposes Allegro-hand/tool
contacts directly, and is inexpensive enough for deterministic headless debug
runs. No benchmark task is invented here.

## Environment split

The accepted M3 environment remains `unit`. DexJoCo runs only in
`tactile-unit-dexjoco`, built from
`configs/simulation/s4_1_dexjoco_environment.yml`. The tracked spec pins Python
3.11.16, MuJoCo 3.4.0, NumPy 1.26.4, Gymnasium 1.0.0, OpenCV 4.11.0.86, and the
pinned editable DexJoCo submodule. PyTorch is not installed because S4.1 does
not train or serve a model. The long-term boundary is files: DexJoCo writes raw
Vision/Action/Contact episodes and `unit` may consume them in S4.2. No RPC
service is introduced.

## Runtime adapter and observation contract

`DexJoCoRuntimeAdapter` owns lifecycle, reset/step, rendering, action
conversion, physical timestamps, success/termination, and contact extraction.
It lazily imports DexJoCo and MuJoCo, so importing pure contracts in `unit` does
not cross the environment boundary.

The canonical `SimObservation` contains simulation time in seconds, control
step, episode/task identity, raw front-camera RGB, exact proprioception,
simulated tactile, termination/truncation, and optional success. RGB remains
the official 640 x 640 `uint8` RGB frame. The 31-D proprio order is frozen in
`s4_1_dexjoco_contract.json`: TCP position/quaternion, 16 named Allegro joint
positions, tongs position/quaternion, and table-height delta, with meters,
radians, and unitless quaternion components named explicitly.

## Action contract

The policy-facing action is 22-D:

`[target_xyz(3), target_rotvec(3), Allegro_joint_targets(16)]`.

The environment-facing action is 23-D:

`[target_xyz(3), target_quaternion_wxyz(4), Allegro_joint_targets(16)]`.

`policy_action_to_env_action` is the only canonical conversion. The neutral
action is reconstructed from the current TCP quaternion and current hand joint
positions, matching DexJoCo's official stay convention. Inputs must be finite;
the official absolute-pose policy wrapper otherwise exposes an unbounded Box.

## Timing contract

The pinned task uses a 0.002 s MuJoCo physics step and 10 substeps per 0.02 s
control step, or 50 Hz in physical simulation time. The upstream task also
paces wall-clock execution toward 30 Hz; this does not alter `data.time` or the
physical-time indexing contract. Every control step emits one observation and
render.

The current tactile history is 0.5 s, which is 26 endpoint-inclusive samples
at 50 Hz. The canonical transition target remains `16 / 30 = 0.533333...` s.
Nearest-step indexing gives 27 control steps, or 0.54 s (6.67 ms error). A
future history ending at that anchor starts 0.04 s after the current anchor, so
the two raw 0.5 s histories do not overlap. Indexing rejects anchors that need
future data.

## Named contact regions

The right Allegro hand is divided into five semantic regions: palm, index,
middle, ring, and thumb. Every relevant collision geom attached to these named
bodies is resolved to a runtime ID. The config contains no numeric geom IDs;
this matters because the pinned assets leave hand collision geoms unnamed.
Only contacts whose other body is one of the tongs bodies (`link_0`, `link_1`)
enter the tactile vector. Table, robot self-contact, and unrelated scene
contacts are excluded.

## Force convention and simulated tactile equation

MuJoCo's `mj_contactForce` returns a 3-D force followed by 3-D torque in the
contact frame. The contact-frame X axis is the normal and the Y/Z axes are
tangents; `mjContact.frame` stores those axes by rows. This is specified in the
[MuJoCo 3.4 API](https://mujoco.readthedocs.io/en/3.4.0/APIreference/APIfunctions.html#mj-contactforce)
and [simulation documentation](https://mujoco.readthedocs.io/en/3.4.0/programming/simulation.html#contacts).
Contact positions are world-frame meters. Normal direction follows geom order,
but equal-and-opposite physics makes that order arbitrary; the canonical proxy
therefore uses a nonnegative normal scalar and tangential magnitude rather than
an unstable contact-local tangential vector.

For region `r` and its matched positive-force contacts `C_r(t)`:

```
o_r(t) = 1[|C_r(t)| > 0]
F_n^r(t) = sum_i max(F_n,i(t), 0)
F_t^r(t) = sum_i sqrt(F_t1,i(t)^2 + F_t2,i(t)^2)
w_i = max(F_n,i(t), 0)
p_r(t) = sum_i w_i p_i / (sum_i w_i + epsilon)
```

The canonical per-region feature is

`[occupancy, normal_force, tangential_force_magnitude, cop_x, cop_y, cop_z]`,

and `T_t^sim` concatenates regions in the tracked order for 30 total values.
Units are binary, newtons, newtons, and meters. An empty region is all zeros.

## Multiple contacts, normalization, and impulse

Normal forces and tangential magnitudes are summed across contacts. CoP is the
normal-force-weighted world position. Contact count is diagnostic metadata.
Raw physical values are retained; no learning normalization is fit in S4.1.
`force * control_dt` may be inspected as a diagnostic force integral, but
impulse is not part of the canonical feature vector.

## Debug dataset and logger

`DexJoCoEpisodeLogger` writes numeric step arrays to compressed NPZ, RGB frames
to referenced JPEG files, and deterministic metadata to JSON. Every step has
episode/task/seed identity, physical timestamp, control step, RGB reference,
proprioception, policy and environment actions, simulated tactile, contact
count, reward, terminal flags, and success. Ten 100-step episodes use a
deterministic scripted approach/contact/tangential/release probe. They are
debug evidence, not successful demonstrations or paper-scale data.

LeRobot conversion is mechanical: decode each referenced JPEG and map the NPZ
columns to named features while retaining episode metadata and timestamps.

## Sanity and determinism checks

The acceptance probes cover free-space zeros, occupancy and positive normal
force at onset, increased total loading during controlled descent, nonzero
tangential response during lateral motion, release back to zero, and distinct
signals across regions. A second seeded replay compares proprioception,
rewards, terminal flags, occupancy, and tactile values with floating tolerance.
The offline visualization combines an RGB frame, region force bars, and the
normal-force timeline.

## Known limitations

- This is a contact proxy, not a faithful electronic simulation of RH56DFTP
  taxels, compliance, hysteresis, noise, bandwidth, or calibration.
- Contact-local tangent directions are not stable world axes, so direction is
  deliberately reduced to magnitude.
- The 0.533333 s target rounds to 0.54 s at the task's 50 Hz control rate.
- `pinch_tongs` validates the single-arm contract; bimanual action/region
  expansion belongs to later benchmark work.
- The scripted episodes diagnose physics and storage; they do not measure task
  success or policy quality.

## S4.2 boundary

S4.2 may freeze this raw schema, generate train/validation/test episodes, fit
train-split normalization, train `E_T^sim`, construct simulated Contact state
and transition representations, and align them with shared `u_c`. Real tactile
continues to use `E_T^RH`; simulation uses `E_T^sim`. Alignment is claimed only
at the shared Contact-representation level. None of that training is performed
in S4.1.
