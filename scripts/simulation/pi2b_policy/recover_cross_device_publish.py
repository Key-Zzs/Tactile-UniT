#!/usr/bin/env python3
"""Recover first-complete PI2B blocks after the cross-mount publish failure.

This script does not run policies and does not inspect performance fields.  It
validates the three first-attempt 50-reset artifacts structurally, durably
publishes their unchanged bytes to NAS, enriches the canonical metadata, and
records the narrowly scoped infrastructure amendment used by resume mode.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.pi2b_policy.integrity import sha256_file
from scripts.simulation.pi2b_policy import evaluate as runtime


EXPECTED_PRE_FREEZE_SHA256 = "01c96ab15f01572c564b586a06038d52bc918907ed254b7c82210cefa1e1c54d"
SOURCES = {
    "B0": (
        Path("/tmp/pi2ba_final/r16_b0_s42/attempt_1/worker_artifacts/pi1d_b0_eval.json"),
        "860f2f3ab2af71fe3ae2a58612f041b466892ea70a94327d1fc18a4071016a63",
        1,
        8801,
    ),
    "B_VA27": (
        Path("/tmp/pi2ba_final/r16_b_va27_s42/attempt_1/worker_artifacts/pi1d_b_va27_eval.json"),
        "b7c0392bc2e7ca1350784424bb98ba85d0f5993215540284d675d9cdf3a80473",
        2,
        8802,
    ),
    "B1": (
        Path("/tmp/pi2ba_final/r16_b1_s42/attempt_1/worker_artifacts/pi1d_b1_eval.json"),
        "e45f98bd57abbd2b6e6571f822228627a056ea443058190eb0c418983829a188",
        3,
        8803,
    ),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def structural_gate(payload: dict[str, Any], model: str, expected: list[str]) -> dict[str, bool]:
    rows = payload.get("episode_results", [])
    observed = [row.get("reset_identity") for row in rows]
    return {
        "status_pass": payload.get("status") == "PASS",
        "model_exact": payload.get("model") == model,
        "mode_exact": payload.get("mode") == runtime.RUNTIME_MODES[model],
        "reset_seed_exact": payload.get("evaluator_seed") == 16,
        "episodes_exact": payload.get("episodes") == 50 and len(rows) == 50,
        "episode_indices_exact": [row.get("episode_index") for row in rows] == list(range(50)),
        "reset_order_exact": observed == expected,
        "resets_unique": len(set(observed)) == 50,
        "contact_delivery_exact": payload.get("contact_state_sent_to_policy") == (runtime.RUNTIME_MODES[model] != "NONE"),
        "frozen_integrity_gates_pass": bool(payload.get("gates")) and all(value == "PASS" for value in payload["gates"].values()),
    }


def main() -> None:
    if runtime.AMENDMENT.exists():
        raise SystemExit("refusing to overwrite runtime amendment")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise SystemExit("tracked worktree must be clean before recovery")
    if sha256_file(runtime.PRE_FREEZE) != EXPECTED_PRE_FREEZE_SHA256:
        raise SystemExit("pre-FINAL freeze identity changed")
    if runtime.EXECUTION.exists() or runtime.COMPLETENESS.exists() or runtime.PROGRESS.exists():
        raise SystemExit("final execution/completeness/progress already exists")

    freeze = runtime.read_json(runtime.PRE_FREEZE)
    manifest = runtime.read_json(runtime.RESET_MANIFEST)
    expected_resets = runtime.expected_resets(manifest, 16)
    recovered_workers = []
    recovered_blocks = []
    for model, (source, expected_sha, gpu, port) in SOURCES.items():
        if sha256_file(source) != expected_sha:
            raise SystemExit(f"first-attempt source drifted: {source}")
        payload = runtime.read_json(source)
        gates = structural_gate(payload, model, expected_resets)
        if not all(gates.values()):
            raise SystemExit(f"first-attempt structural gate failed: {model} {gates}")
        paths = runtime.attempt_paths(model, 42, 16, 1)
        published = paths.ARTIFACTS / f"{model.lower()}_raw_rollouts.json"
        canonical = runtime.raw_path(model, 42, 16)
        if published.exists() or canonical.exists():
            raise SystemExit(f"refusing to overwrite recovered block: {model}")
        publication = runtime.publish_file(source, published)
        if sha256_file(published) != expected_sha:
            raise SystemExit(f"published attempt bytes drifted: {model}")

        bound = runtime.checkpoint_row(freeze, model, 42)
        payload.update(
            {
                "training_seed": 42,
                "reset_seed": 16,
                "checkpoint_path": bound["path"],
                "checkpoint_tree_sha256": bound["tree_sha256"],
                "server_sampling_contract": "PI2N_ACCEPTED_FRESH_SERVER_REQUEST_ADVANCE",
                "determinism_env": {
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "XLA_FLAGS": list(runtime.XLA_FLAGS),
                },
                "infrastructure_attempt": 1,
                "recovered_after_cross_device_publish_failure": True,
            }
        )
        for row in payload["episode_results"]:
            row.update(
                {
                    "model": model,
                    "training_seed": 42,
                    "reset_seed": 16,
                    "checkpoint_tree_sha256": bound["tree_sha256"],
                    "runtime_mode": runtime.RUNTIME_MODES[model],
                }
            )
        runtime.atomic_json(canonical, payload)
        canonical_sha = sha256_file(canonical)
        worker = {
            "model": model,
            "training_seed": 42,
            "reset_seed": 16,
            "gpu": gpu,
            "port": port,
            "attempt": 1,
            "episodes": 50,
            "elapsed_seconds": None,
            "status": "PASS",
            "raw_sha256": canonical_sha,
            "prior_infrastructure_errors": [],
            "recovered_first_complete_attempt": True,
        }
        recovered_workers.append(worker)
        recovered_blocks.append(
            {
                "model": model,
                "training_seed": 42,
                "reset_seed": 16,
                "source_attempt": 1,
                "source_sha256": expected_sha,
                "published_attempt_sha256": sha256_file(published),
                "canonical_sha256": canonical_sha,
                "publication": publication,
                "structural_gates": {key: "PASS" for key in gates},
                "performance_fields_read": False,
            }
        )

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    amendment = {
        "schema": "tactile3d-unit.s4-3-pi2b-policy-runtime-amendment.v1",
        "created_at_utc": now(),
        "status": "PASS",
        "reason": "Cross-filesystem rename from /tmp to NAS failed after complete evaluator artifacts were emitted.",
        "scope": "ARTIFACT_PUBLICATION_ONLY",
        "pre_final_freeze_sha256": EXPECTED_PRE_FREEZE_SHA256,
        "pre_final_git_head": freeze["git_head"],
        "amended_git_head": head,
        "original_frozen_evaluate_sha256": freeze["sources_sha256"]["scripts/simulation/pi2b_policy/evaluate.py"],
        "amended_evaluate_sha256": sha256_file(Path(runtime.__file__).resolve()),
        "recovery_script_sha256": sha256_file(Path(__file__).resolve()),
        "recovered_workers": recovered_workers,
        "recovered_blocks": recovered_blocks,
        "first_complete_attempt_is_canonical": True,
        "later_attempts_are_noncanonical": True,
        "source_attempt_artifacts_preserved": True,
        "policy_or_evaluator_semantics_changed": False,
        "reset_or_sampling_semantics_changed": False,
        "success_or_timeout_rules_changed": False,
        "performance_seen": False,
        "track_b_new_teacher_read": False,
    }
    runtime.atomic_json(runtime.AMENDMENT, amendment)
    print(
        json.dumps(
            {
                "status": amendment["status"],
                "recovered_blocks": len(recovered_blocks),
                "canonical_rollouts": len(recovered_blocks) * 50,
                "performance_seen": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
