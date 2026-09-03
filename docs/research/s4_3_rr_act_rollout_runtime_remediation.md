# S4.3-RR ACT rollout runtime remediation

## Scope and preserved history

The first S4.3-2 production rollout attempt remains classified as
`S4_3_2_ENVIRONMENT_FAIL`. The `click_mouse/P0/train-seed-0` policy server failed
while binding its AF_UNIX listener: the repository-coupled pathname was 118 encoded
bytes while the active Linux runtime accepted at most 107 pathname payload bytes.
The server never became ready, the DexJoCo client never started, and no scientific
rollout or policy-performance result was produced. The original
`POLICY_EVAL_V1` reset specifications therefore remain frozen and eligible.

S4.3-RR changes only IPC endpoint construction, endpoint ownership/cleanup,
transport validation, and the tests and audit scaffolding needed to exercise that
transport. It does not change checkpoints, model inputs, inference, serialization,
Action or observation values, the 22-to-23 Action adapter, stride, resets, warm-up,
timeouts, success predicates, or statistics.

## Root cause and endpoint V2

The old endpoint embedded the repository path and full task/variant/seed identity:

```text
$REPOSITORY/.local/tmp/simulation/s4_3_restart/rollout_sockets/<task>_<variant>_seed<seed>.sock
```

V2 selects a short private runtime root through standard runtime/temp facilities
and uses:

```text
$RUNTIME_TMP/tu3d_<short_hash>_<pid>_<nonce>.sock
```

The 12-hex digest covers experiment identity, task, variant, training seed, and
worker identity. Full provenance stays in the local endpoint manifest and rollout
metadata; it is not expanded into the pathname and it does not affect scientific
seeds. Endpoint construction enforces a project ceiling of 80 encoded bytes before
`bind()`.

The production launcher writes one versioned endpoint manifest. Both the ACT
server and DexJoCo client load and validate that same manifest, so no duplicated
path construction or fallback address exists. A server may reclaim an existing
socket only when the private-runtime namespace, uid, registered PID liveness,
socket type, and inode all prove that it is stale. Normal and failed startup paths
perform best-effort cleanup of their own validated endpoint.

## RR0--RR4 evidence

- RR0 revalidated the clean starting branch/HEAD, DexJoCo revision, uninitialized
  nested Diffusion Policy, both package sets, all 36 ACT checkpoint SHA256 values,
  frozen S4.2 model/vision/config identities, and the 90-reset contract.
- RR1 reproduced the 118-byte bind failure as a transport-only fixture without
  opening any `POLICY_EVAL_V1` reset.
- RR2 produced 36 unique simultaneous job identities with a maximum encoded
  endpoint length below the 80-byte ceiling.
- RR3 passed long-path, four-worker bind/connect/no-cross-talk, active/stale/
  arbitrary endpoint cleanup, failed-connect, normal-exit, and RPC-payload-parity
  tests.
- RR4 ran P0 and P3 for all three tasks through the exact production worker on
  disjoint `RR_SMOKE` resets. Each run completed 100 post-warm-up control steps and
  passed checkpoint/Vision load, EGL, RGB, proprio, tactile, P3 Contact diagnostics,
  `[27,22]` output, 22-to-23 adaptation, stride 5, causal-read, endpoint-length,
  cleanup, and process-leak gates. These are infrastructure results only.

The machine-readable evidence and plots are stored under the ignored local
`s4_3_rr` artifact/log roots. The historical S4.3-2 failure artifacts are not
overwritten.

## Frozen resume contract

After RR4, rollout runtime V2 and the final benchmark contract are frozen in
`configs/simulation/s4_3_rr_rollout_runtime_v2.json` and
`configs/simulation/s4_3_rr_final_benchmark_contract.json`. The resumed benchmark
uses the same 36 checkpoints, 30 task-specific resets per checkpoint, 0.5-second
26-sample warm-up, stride 5, task timeouts, native success contracts, Action
adapter, paired hierarchical bootstrap, seed 43020, and material-effect thresholds.
No training, checkpoint selection, hyperparameter change, or reset substitution is
permitted after scientific rollout begins.
