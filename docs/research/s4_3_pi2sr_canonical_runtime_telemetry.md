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
