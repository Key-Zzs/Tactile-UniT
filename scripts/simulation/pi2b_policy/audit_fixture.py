#!/usr/bin/env python3
"""Cold-load and record the bounded three-step NAS engineering fixture."""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import re
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.pi2b_policy.contract import Workspace
from gr00t.simulation.pi2b_policy.integrity import file_manifest, sha256_file, tree_sha256


ARTIFACT = ROOT / ".local/artifacts/simulation/s4_3_pi2b_policy/engineering_fixture.json"
METRIC = re.compile(r"Step (?P<step>\d+): (?P<metrics>[^\r\n]+)")
VALUE = re.compile(r"(?P<key>[a-z0-9_]+)=(?P<value>[-+0-9.eE]+)")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    if ARTIFACT.exists():
        raise SystemExit("refusing to overwrite engineering fixture audit")
    workspace = Workspace.load(ROOT)
    fixture_root = workspace.write_root / "tmp/fixtures/b_hva_seed43"
    checkpoint = fixture_root / "pinch_tongs/s43_pi2b_b_hva_seed43/2"
    log = workspace.write_root / "logs/fixtures/b_hva_seed43_3step.log"
    if not checkpoint.is_dir() or not log.is_file():
        raise SystemExit("bounded fixture checkpoint or log is missing")
    rows = []
    for match in METRIC.finditer(log.read_text(errors="replace")):
        rows.append(
            {
                "step": int(match.group("step")),
                **{
                    value.group("key"): float(value.group("value"))
                    for value in VALUE.finditer(match.group("metrics"))
                },
            }
        )
    started = perf_counter()
    manifest = file_manifest(checkpoint)
    checkpoint_hash = tree_sha256(manifest)
    hash_seconds = perf_counter() - started

    import jax
    import numpy as np
    import orbax.checkpoint as ocp

    started = perf_counter()
    with ocp.PyTreeCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(checkpoint / "train_state")
        restore_args = jax.tree.map(
            lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), metadata
        )
        state = checkpointer.restore(
            checkpoint / "train_state",
            ocp.args.PyTreeRestore(item=metadata, restore_args=restore_args),
        )
    restored_step = int(np.asarray(state["step"]).item())
    finite = all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(state))
    del state
    gc.collect()
    restore_seconds = perf_counter() - started
    gates = {
        "bounded_exactly_three_steps": restored_step == 3,
        "metric_steps_exact": [row["step"] for row in rows] == [0, 1, 2],
        "metrics_finite": bool(rows)
        and all(math.isfinite(float(value)) for row in rows for key, value in row.items() if key != "step"),
        "expected_gradients_present": all(
            key in rows[-1] and rows[-1][key] > 0
            for key in (
                "pi05_trainable_grad_norm",
                "contact_adapter_grad_norm",
                "physical_auxiliary_grad_norm",
            )
        ),
        "lambda_exact": bool(rows)
        and abs(rows[-1].get("lambda_phys", -1) - 0.026468189597253295) <= 5e-5,
        "checkpoint_arrays_finite": finite,
        "checkpoint_persisted_to_policy_nas": workspace.write_root in checkpoint.parents,
        "checkpoint_has_params_and_train_state": (checkpoint / "params").is_dir()
        and (checkpoint / "train_state").is_dir(),
        "no_canonical_checkpoint_promotion": "tmp/fixtures" in checkpoint.as_posix(),
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2b-policy-engineering-fixture.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scientific_role": "NONE_ENGINEERING_ONLY",
        "model_id": "B_HVA",
        "mode": "CONTACT_STATE_TOKENS_VA_PHYSICAL_AUX",
        "seed": 43,
        "optimizer_steps": 3,
        "global_batch_size": 32,
        "checkpoint": str(checkpoint),
        "checkpoint_tree_sha256": checkpoint_hash,
        "checkpoint_files": len(manifest),
        "checkpoint_bytes": sum(int(row["bytes"]) for row in manifest),
        "hash_seconds": hash_seconds,
        "restored_train_state_step": restored_step,
        "restore_seconds": restore_seconds,
        "log": str(log),
        "last_metrics": rows[-1] if rows else None,
        "gates": {key: "PASS" if value else "FAIL" for key, value in gates.items()},
        "promoted_to_candidate": False,
    }
    atomic_json(ARTIFACT, payload)
    print(json.dumps({"status": payload["status"], "gates": payload["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
