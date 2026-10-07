# S4.3-PI2S-R: Canonical Contact Runtime & Stepwise Telemetry

## Scope and frozen history

PI2S-R is a prospective engineering qualification. It does not replace Track A (`COMPLETE_VALID_THREE_SEEDS_LIMITED`), Track B (`VAC_CONTACT_CAPABILITY_WITH_NO_DETECTED_VA_REGRESSION_LIMITED_PRECISION`), or PI2S (`COMPLETE_VALID_DIAGNOSIS_INCONCLUSIVE`). The historical cached/direct equality and historical live Unix-service maximum absolute discrepancy of approximately `2.7418e-6` remain unchanged. No policy or teacher is trained, no formal success-rate cohort is rerun, and no real hardware is used.

The versioned freezes are:

- `configs/simulation/pi2sr/runtime_contract_v2.json`
- `configs/simulation/pi2sr/telemetry_contract_v1.json`
- `configs/simulation/pi2sr/provenance_contract_v1.json`

## Prospective runtime qualification

The preselected canonical candidate is direct, in-process `E_T` on CPU with float32 batch shape `[1,26,30]`, eval and inference mode, one compute/inter-op thread, MKLDNN/autocast/TF32 disabled, deterministic algorithms enabled, and the accepted checkpoint and N1 normalization. Selection is by identity, causality, reproducibility, operational simplicity, latency, and scoped consistency—not policy success.

Transport-only qualification sends an already-computed float32 `[256]` H tensor through the same length-prefixed AF_UNIX framing without invoking `E_T`. Compute-only qualification uses sixteen frozen PI2S histories selected before measurement from free-space, contact, and dynamic/boundary strata. Same-path repeatability establishes a prospective v2 numeric envelope by the formula in the runtime contract. That envelope is scoped only to the matched PI2S-R configuration and cannot reclassify historical PI2S.

## Shadow telemetry

Telemetry is asynchronous and shadow-only. It records step identity/timing, policy-facing and environment-facing states, command/applied actions, tactile regional summaries, causal history state, canonical H identity/value or reference, action-queue semantics, and available native task predicates. Privileged fields are marked `evaluation_only=true` and `policy_input=false`; missing native fields are unavailable rather than synthesized.

`CONTACT`, `LIFT`, `NATIVE_TRIGGER`, and `SUCCESS` preserve native/canonical meanings. `STABLE_GRASP` is a prospective diagnostic only: at least two occupied regions and total normal force at least `0.5` for five consecutive control steps. It is not applied retroactively to historical outcomes. `APPROACH` remains undefined.

## Provenance boundary

The prospective manifest binds source, submodules, configs, checkpoints, environment, data/reset identities, initialization namespaces, and all available PRNG seeds or request counters. Unavailable historical per-step PRNG state is explicitly missing; it is never reconstructed. The manifest distinguishes same-source/config seed changes from source/config changes.

## Completion boundary

PI2S-R may run at most six non-scientific smoke episodes. Smoke validates runtime, reset, contact, queue, end-of-episode, telemetry, and failure isolation only. It cannot support a policy-effect, success-rate, checkpoint-selection, teacher-replacement, or real-hardware claim.

## Qualified result

Status: `PI2SR_CANONICAL_RUNTIME_TELEMETRY_READY`.

The selected runtime is `DIRECT_IN_PROCESS` under the exact CPU/float32/batch-one scope above. The accepted `E_T` checkpoint SHA-256 remains `3d4519195e7a0d9f5af63399840a48121d5f614ea587c80b9461fc696a372a19`. Sixty-four transport-only echo requests were byte-exact (`TRANSPORT_EXACT`; maximum and mean error `0`). Direct and Unix-service computation were each byte-repeatable across eight repetitions of sixteen frozen histories; their matched cross-path outputs were also byte-exact (`COMPUTE_REPEATABLE_CANONICAL`; maximum and mean error `0`). This isolates the historical discrepancy away from float32 AF_UNIX transport and to historical compute execution conditions such as batch/runtime settings. It does not establish that the discrepancy caused any rollout failure.

Canonical direct latency over 64 measured CPU requests was p50 `3.919 ms`, p95 `4.218 ms`, and p99 `5.862 ms`. GPU, accelerator kernels, autocast/float16/bfloat16, batch sizes other than one, MKLDNN-enabled execution, and cross-host transport remain unsupported.

Fixed-observation binding covered four H-conditioned checkpoints and eight positions each. Matched canonical/alternate H was byte-identical, so the bound frozen action was also byte-identical (maximum/mean difference `0`). Historical lag-5 H changed 12/32 action samples and zero-H changed 32/32, preserving the historical sensitivity finding without a causal rollout claim.

The six-run engineering smoke paired telemetry off/on for three development task fixtures and 90 total control steps. It exercised reset, contact, the prospective `STABLE_GRASP` diagnostic, stride-five action queues, writer failure, and end-of-episode handling. All 45 telemetry-on records were written to `$UNIT_EXPERIMENT_ROOT/simulation/s4_3_pi2sr/telemetry/`, with zero drops and maximum queue backlog one. Worst per-task submit overhead was p50 `0.159 ms`, p95 `0.166 ms`, and p99 `0.170 ms`; observation/action traces, transforms, predicates, and queue semantics matched telemetry-off traces exactly. The smoke used a deterministic engineering stub, not a trained policy, and is not a scientific benchmark.

The prospective provenance manifest binds a clean source commit, submodules, three config hashes, the contact encoder, CPU environment, development data/reset identities, reset seed, and policy request counter. Training/initialization/data-loader/augmentation/noise seeds and policy key indices that do not exist for this smoke are explicitly null. No historical per-step PRNG reconstruction is claimed.

The sibling `develop/real-dualflexiv` worktree was created from the same merged main base and remains clean. Real-hardware execution still requires separate authorization. The recommended next stage is `S5.0 — DualFlexiv + RH56DFTP Sensor / Control / Data Contract`.
