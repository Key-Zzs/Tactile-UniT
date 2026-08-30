# Full Track C — C2-R Contact Preservation Remediation

## Scope

C2-R is a bounded, post-C2 remediation of the accepted continuous Vision/Action/Contact shared space. The accepted C2 point estimate remains `R_contact = 0.896075` and remains a failure of the preregistered `R_contact >= 0.90` hard gate. C2-R does not reinterpret that result and is not an untouched first-look test.

The remediation starts from the accepted C2-slot checkpoint with SHA-256 `454d7a33df20e5329e2be4804760dad211462e3eb405c16141f061f0c1ef113a`. Vision, Action, shared slots, native encoders, and the frozen S2 decoder are immutable. Only the modality-specific Contact projector and Contact recovery head are trainable.

## Fixed protocol

The accepted C2 alignment, native recovery, relational, variance, temperature, optimizer family, and dynamic weighting settings remain fixed. The only grid is:

- `lambda_future` in `{0.5, 1.0, 2.0}`;
- `lambda_delta = lambda_future`;
- train-derived Contact boundary weight in `{1.0, 2.0}`.

Dynamic and boundary sample weights apply consistently to Contact alignment, native-code recovery, frozen-decoder future prediction, and delta prediction. The fixed Contact relational term preserves global pairwise cosine, native local neighborhoods, and native transition-distance ordering; it introduces no searched setting.

There are at most six trials, at most ten epochs per trial, and patience is three. Training and checkpoint selection load only the frozen C1 train and validation caches. The validation utility prioritizes Contact-transition retention, future/delta physics, V-C and A-C alignment, and Contact non-collapse, in that order. Effective ties prefer the smallest `lambda_future`, followed by boundary weight one.

Before training, the C2-R0 audit reproduces the canonical native/shared Contact probes, audits identical protocols, runs a three-seed diagnostic, and bootstraps the already-inspected C2 test predictions. An implementation discrepancy stops the study as `C2R_METRIC_IMPLEMENTATION_INVALID`.

## Locked evaluation and stop point

After validation selection, `selection.json` is written with `test_loaded: false`, hashed, and verified before the 17,504-row locked test can be loaded. The evaluation is labeled **LOCKED RE-EVALUATION AFTER POST-C2 REMEDIATION**. The locked result cannot trigger more tuning.

The final state is exactly one of the registered C2-R decisions. C3, C4, C5, C6/M3, VLA, 3D, and deployment are outside this work. M3 remains not established. The task stops after C2-R.

Physical GPU1 was added to the allowed runtime set only after explicit user authorization during this C2-R goal on 2026-08-27. GPU0 remains forbidden.

## Result

The six preregistered trials completed using train and validation only. Trial 4 was frozen before test with `lambda_future = lambda_delta = 2.0` and boundary weight `1.0`; its checkpoint SHA-256 is `21dccb8fc7fbe6de2598c18e718bd65f226e220e44352ab3d43246e7f9abdf89`. The locked evaluation was repeated only for deterministic verification and produced byte-identical JSON.

The remediation improved Contact future MSE from `0.005611` to `0.003538` and dynamic MSE from `0.017815` to `0.011015`. V-C and A-C alignment remained above their hard gates, frozen Vision/Action outputs were identical, all native identities passed, and Contact did not collapse. Force retention passed at `1.065784`.

Contact-transition retention improved from `0.896075` to `0.899283`, but remained below the preregistered `0.90` hard gate. The final decision is therefore **`C2R_DUAL_PATH_RECOMMENDED`**. No additional tuning is permitted from this locked-test result. A future C3 must be redesigned and preregistered around shared `u_v/u_a/u_c` plus Contact-native `z_c`; C3 was not implemented here.
