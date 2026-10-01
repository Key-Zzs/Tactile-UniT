#!/usr/bin/env python3
"""Create the sole Track-A pre-FINAL freeze before policy performance access."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.pi2b_policy.contract import MODEL_ORDER, SEED42_CHECKPOINT_HASHES, Workspace
from gr00t.simulation.pi2b_policy.integrity import sha256_file


ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2b_policy"
OUTPUT = ARTIFACTS / "pre_final_freeze.json"
RESET_MANIFEST = ARTIFACTS / "reset_manifest.json"
STARTING_INTEGRITY = ARTIFACTS / "starting_integrity.json"
TRAINING_COMPLETION = ARTIFACTS / "training_completion.json"
EXPECTED_NEW = {
    (43, "B0"): "33888318e7dcb56ef8262784589ae9caf29dbd3b7568e722258e89bb13ec4a87",
    (43, "B_VA27"): "347f0f40a660ee8a1627374fcaeeab2edc1fecd4ff3840ae293c48fdfbfd53c8",
    (43, "B1"): "cb15ab1d2b9f3dd738359bd1b65d81cfc562f03b3bc752dd15cdfef0f7e7fdae",
    (43, "B_HVA"): "528ba1888d094dc903f32818d6eea5d57c847a2c73d90057fb1ce679f8814d2a",
    (43, "B2"): "b3a143e3262f47c61d0dfa14dacda62ad6ba1f7574239d488d886db9f56a50a8",
    (44, "B0"): "2355c820e8e0a003ee744426f4b375e40b492d808ab4239186b06d63552591f3",
    (44, "B_VA27"): "86adfaf983c4fcd6a39f2fb4413c246f94baa161f7e8cfe24754a1209baecb98",
    (44, "B1"): "63b9038261ea2e5d3559c7ffaa0d59c117dcbe3914f16890ad23703b075a1bbc",
    (44, "B_HVA"): "b342e4c0a4303f338906ef0030e3d86730730ed443698d0aa3392ecaaf3f594c",
    (44, "B2"): "9f46aae6eed9f1523168fc83fe1236f2b0413e5fc38b488c12b84eed3d4ce203",
}
SOURCES = (
    "scripts/simulation/pi2b_policy/serve.py",
    "scripts/simulation/pi2b_policy/evaluate.py",
    "scripts/simulation/pi2b_policy/freeze_final.py",
    "scripts/simulation/pi2b_policy/status_evaluation.py",
    "scripts/simulation/run_s4_3_pi2u_eval.py",
    "scripts/simulation/evaluate_s4_3_pi1d_augmented.py",
    "scripts/simulation/serve_s4_3_pi1_contact_state.py",
    "gr00t/simulation/pi2b_policy/contract.py",
    "gr00t/simulation/pi2b_policy/statistics.py",
    "gr00t/simulation/pi05_tactile_unit.py",
    "gr00t/simulation/s4_3_pi1.py",
    "configs/simulation/pi2b_policy/statistics.json",
)


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def bind_checkpoint(seed: int, model: str, path: Path, expected: str, actual: str) -> dict[str, Any]:
    return {
        "model": model,
        "training_seed": seed,
        "path": str(path),
        "tree_sha256": actual,
        "expected_tree_sha256": expected,
        "params_present": (path / "params").is_dir(),
        "train_state_present": (path / "train_state").is_dir(),
        "hash_source": "starting_integrity.json" if seed == 42 else "training_completion.json",
        "status": "PASS" if actual == expected else "FAIL",
    }


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit("refusing to overwrite pre_final_freeze.json")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise SystemExit("tracked worktree must be clean before pre-FINAL freeze")
    workspace = Workspace.load(ROOT)
    reset = json.loads(RESET_MANIFEST.read_text())
    starting = json.loads(STARTING_INTEGRITY.read_text())
    completion = json.loads(TRAINING_COMPLETION.read_text())
    if reset.get("policy_performance_seen") is not False or reset.get("episodes") != 200:
        raise SystemExit("frozen reset manifest is not eligible")
    cached = {
        (int(row["training_seed"]), str(row["model"])): row
        for row in completion.get("runs", [])
    }
    checkpoints = []
    for seed in (42, 43, 44):
        for model in MODEL_ORDER:
            path = workspace.checkpoint_seed42(model) if seed == 42 else workspace.final_checkpoint(model, seed)
            expected = SEED42_CHECKPOINT_HASHES[model] if seed == 42 else EXPECTED_NEW[(seed, model)]
            if seed == 42:
                actual = starting["seed42_checkpoints"][model]["tree_sha256"]
            else:
                row = cached[(seed, model)]
                if row.get("cold_restore") != "PASS" or row.get("restored_train_state_step") != 30000 or row.get("all_train_state_arrays_finite") is not True:
                    raise SystemExit(f"completion evidence failed for {model}/seed{seed}")
                actual = row["tree_sha256"]
            checkpoints.append(bind_checkpoint(seed, model, path, expected, actual))
    sources = {relative: sha256_file(ROOT / relative) for relative in SOURCES}
    external_sources = {
        str(workspace.main_root / "third_party/dexjoco/configs/rand_obj/pinch_tongs.yaml"):
        sha256_file(workspace.main_root / "third_party/dexjoco/configs/rand_obj/pinch_tongs.yaml")
    }
    gates = {
        "fifteen_checkpoints_exact": len(checkpoints) == 15 and all(row["status"] == "PASS" and row["params_present"] and row["train_state_present"] for row in checkpoints),
        "seed42_live_hash_cache_pass": starting.get("status") == "PASS" and all(starting["seed42_checkpoints"][model].get("status") == "PASS" for model in MODEL_ORDER),
        "new_training_completion_pass": completion.get("status") == "PASS" and len(cached) == 10 and all(value == "PASS" for value in completion.get("gates", {}).values()),
        "shared_200_resets_unseen": len(reset.get("ordered_reset_identities", [])) == 200 and len(set(reset.get("ordered_reset_identities", []))) == 200 and reset.get("policy_performance_seen") is False,
        "reset_blocks_exact": [int(row["seed"]) for row in reset.get("blocks", [])] == [16, 17, 18, 19],
        "formal_outputs_absent": not any((ARTIFACTS / name).exists() for name in ("attempts", "final_raw", "final_progress.json", "rollout_completeness.json", "final_gpu_execution.json", "paired_statistics.json")),
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2b-policy-pre-final.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "formal_models": list(MODEL_ORDER),
        "training_seeds": [42, 43, 44],
        "checkpoints": checkpoints,
        "reset_manifest": str(RESET_MANIFEST),
        "reset_manifest_sha256": sha256_file(RESET_MANIFEST),
        "starting_integrity_sha256": sha256_file(STARTING_INTEGRITY),
        "training_completion_sha256": sha256_file(TRAINING_COMPLETION),
        "ordered_reset_sequence_sha256": reset["ordered_reset_sequence_sha256"],
        "sources_sha256": sources,
        "external_sources_sha256": external_sources,
        "runtime": {
            "official_async_dexjoco_openpi": True,
            "replan_ratio": 0.8,
            "episodes_per_block": 50,
            "reset_seed_blocks": [16, 17, 18, 19],
            "cublas_workspace_config": ":4096:8",
            "xla_flags": ["--xla_gpu_deterministic_ops=true", "--xla_gpu_exclude_nondeterministic_ops=true", "--xla_gpu_autotune_level=0"],
            "block_retry_limit": 2,
            "retry_only_infrastructure_failure": True,
        },
        "statistics": {"bootstrap_repetitions": 100000, "bootstrap_seed": 4317},
        "seed42_prior_performance_seen": True,
        "new_seed43_44_performance_seen": False,
        "cohort_performance_seen": False,
        "track_b_new_teacher_read": False,
        "gates": {key: "PASS" if value else "FAIL" for key, value in gates.items()},
    }
    atomic_json(OUTPUT, payload)
    print(json.dumps({"status": payload["status"], "checkpoints": len(checkpoints), "gates": payload["gates"]}, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
