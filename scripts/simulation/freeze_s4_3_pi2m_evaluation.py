#!/usr/bin/env python3
"""Bind the frozen PI2M evaluation protocol to final checkpoints before performance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2m"
OUTPUT = ARTIFACTS / "pre_eval_freeze.json"
CHECKPOINTS = {
    "B1": ROOT
    / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/"
    "s43_pi1b_contact_tokens_seed42/29999",
    "B_HVA": ROOT
    / ".local/experiments/simulation/s4_3_pi2m/bhva/pinch_tongs/"
    "s43_pi2m_bhva_seed42/29999",
    "B2": ROOT
    / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/"
    "s43_pi1c_contact_tokens_physical_aux_seed42/29999",
}
EXPECTED = {
    "B1": "04b59cbc3491bf4e88dc75558a08a94e5588e2fbe398d413e1766a87c7ab6f4f",
    "B_HVA": "82469f1d48b09f1c6152019512dba82f0f899f07bb0aab76bb58c0e0ba1e3a8b",
    "B2": "86c4908533ca24da06ae66f943e3605b5ccbae455441290ca8fb35514359e949",
}
SOURCES = (
    "configs/simulation/s4_3_pi2m_evaluation_protocol.json",
    "configs/simulation/s4_3_pi2m_statistical_protocol.json",
    "gr00t/simulation/pi05_tactile_unit.py",
    "gr00t/simulation/s4_3_pi1.py",
    "scripts/simulation/analyze_s4_3_pi2m.py",
    "scripts/simulation/evaluate_s4_3_pi1d_augmented.py",
    "scripts/simulation/freeze_s4_3_pi2m_evaluation.py",
    "scripts/simulation/run_s4_3_pi2m_eval.py",
    "scripts/simulation/run_s4_3_pi2m_production_smoke.py",
    "scripts/simulation/serve_s4_3_pi1_contact_state.py",
    "scripts/simulation/serve_s4_3_pi2m_policy.py",
    "third_party/dexjoco/configs/rand_obj/pinch_tongs.yaml",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(32 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256_file(file).encode())
        digest.update(b"\0")
        digest.update(str(file.stat().st_size).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def symbolic(path: Path) -> str:
    return "$REPO_ROOT/" + path.resolve().relative_to(ROOT).as_posix()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit("refusing to overwrite PI2M pre-evaluation freeze")
    if command("git", "status", "--short"):
        raise SystemExit("tracked worktree must be clean before PI2M final evaluation freeze")
    if command("git", "branch", "--show-current") != "develop/sim-benchmark":
        raise SystemExit("PI2M evaluation freeze requires develop/sim-benchmark")
    initial = json.loads((ARTIFACTS / "evaluation_protocol_freeze.json").read_text())
    reset = json.loads((ARTIFACTS / "fresh_reset_manifest.json").read_text())
    seed = json.loads((ARTIFACTS / "fresh_seed_audit.json").read_text())
    smoke = json.loads((ARTIFACTS / "production_smoke.json").read_text())
    completion = json.loads((ARTIFACTS / "training_completion.json").read_text())
    manifest = json.loads((ARTIFACTS / "bhva_checkpoint_manifest.json").read_text())
    evaluation = json.loads(
        (ROOT / "configs/simulation/s4_3_pi2m_evaluation_protocol.json").read_text()
    )
    statistics = json.loads(
        (ROOT / "configs/simulation/s4_3_pi2m_statistical_protocol.json").read_text()
    )
    checkpoint_hashes = {name: tree_hash(path) for name, path in CHECKPOINTS.items()}
    initial_code_unchanged = all(
        sha256_file(ROOT / relative) == expected
        for relative, expected in initial["code_sha256"].items()
    )
    formal_outputs = [
        ARTIFACTS / f"{model.lower()}_raw_rollouts.json" for model in ("B1", "B_HVA", "B2")
    ]
    formal_outputs.extend(
        [
            ROOT / ".local/logs/simulation/s4_3_pi2m/evaluation_seed7",
            ROOT / ".local/cache/simulation/s4_3_pi2m/evaluation/seed7",
            ROOT / ".local/tmp/s43m7",
        ]
    )
    gates = {
        "initial_training_time_evaluation_freeze_unchanged": initial_code_unchanged,
        "B_HVA_training_completion_PASS": completion.get("status") == "PASS"
        and completion.get("optimizer_steps") == 30_000,
        "B_HVA_checkpoint_manifest_PASS": manifest.get("status") == "PASS"
        and manifest.get("frozen") is True,
        "checkpoint_hashes_exact": checkpoint_hashes == EXPECTED,
        "production_smoke_PASS": smoke.get("status") == "PASS"
        and smoke.get("scientific_result") is False,
        "no_formal_performance_seen": initial.get("performance_seen") is False
        and not any(path.exists() for path in formal_outputs),
        "models_exact": evaluation.get("models") == ["B1", "B_HVA", "B2"],
        "200_each_600_total": evaluation.get("episodes_per_model") == 200
        and evaluation.get("total_rollouts") == 600,
        "fresh_seed7": evaluation.get("evaluator_seed") == 7
        and seed.get("selected_seed") == 7
        and seed.get("selected_seed_performance_inspected_before_freeze") is False,
        "reset_manifest_exact": reset.get("status") == "PASS"
        and reset.get("episodes") == 200
        and reset.get("policy_performance_seen") is False
        and sha256_file(ARTIFACTS / "fresh_reset_manifest.json")
        == initial["reset_manifest_sha256"]
        and reset.get("reset_sequence_sha256") == initial["reset_sequence_sha256"],
        "same_online_runtime": evaluation.get("runtime_mode_all_models")
        == "CONTACT_STATE_TOKENS",
        "targets_future_reward_success_forbidden": set(
            evaluation.get("training_only_fields_forbidden", [])
        )
        >= {
            "va_shared_target",
            "va_aux_valid",
            "contact_shared_target",
            "physical_aux_valid",
            "reward",
            "success",
            "future action",
        },
        "fixed_checkpoint_selection": evaluation.get("checkpoint_selection")
        == "final step only",
        "statistics_frozen": statistics.get("status") == "FROZEN_BEFORE_BHVA_TRAINING"
        and statistics.get("bootstrap_resamples") == 100_000
        and statistics.get("bootstrap_seed") == 4317,
        "dexjoco_revision_exact": command(
            "git", "-C", "third_party/dexjoco", "rev-parse", "HEAD"
        )
        == "8d23b0fab23b17a58c4b55f3942e17013aaf8267",
        "PI2B_not_started": completion.get("model_id") == "B_HVA",
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2m-pre-evaluation-freeze.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "git_head": command("git", "rev-parse", "HEAD"),
        "models": ["B1", "B_HVA", "B2"],
        "checkpoint_paths": {name: symbolic(path) for name, path in CHECKPOINTS.items()},
        "checkpoint_tree_sha256": checkpoint_hashes,
        "episodes_per_model": 200,
        "total_rollouts": 600,
        "evaluator_seed": 7,
        "reset_manifest": symbolic(ARTIFACTS / "fresh_reset_manifest.json"),
        "reset_manifest_sha256": sha256_file(ARTIFACTS / "fresh_reset_manifest.json"),
        "reset_sequence_sha256": reset["reset_sequence_sha256"],
        "runtime_mode_all_models": "CONTACT_STATE_TOKENS",
        "dynamics": {
            "config": "rand_obj/pinch_tongs",
            "rand_full": False,
            "randomize_dynamics": False,
            "replan_ratio": 0.8,
            "control_dt_seconds": 0.02,
        },
        "success_definition": "native DexJoCo pinch_tongs success, unchanged",
        "timeout_policy": "native max_steps; all native timeouts count in denominator",
        "checkpoint_selection": "final step only",
        "production_smoke_sha256": sha256_file(ARTIFACTS / "production_smoke.json"),
        "training_completion_sha256": sha256_file(ARTIFACTS / "training_completion.json"),
        "bhva_checkpoint_manifest_sha256": sha256_file(
            ARTIFACTS / "bhva_checkpoint_manifest.json"
        ),
        "initial_evaluation_freeze_sha256": sha256_file(
            ARTIFACTS / "evaluation_protocol_freeze.json"
        ),
        "statistical_protocol_sha256": sha256_file(
            ARTIFACTS / "statistical_protocol.json"
        ),
        "sources_sha256": {symbolic(ROOT / path): sha256_file(ROOT / path) for path in SOURCES},
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "formal_performance_seen": False,
        "PI2B_started": False,
    }
    atomic_json(OUTPUT, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "checkpoint_tree_sha256": checkpoint_hashes,
                "reset_sequence_sha256": payload["reset_sequence_sha256"],
            },
            sort_keys=True,
        )
    )
    if payload["status"] != "PASS":
        failed = [name for name, value in gates.items() if not value]
        raise SystemExit("PI2M_PRE_EVAL_FREEZE_FAIL: " + ",".join(failed))


if __name__ == "__main__":
    main()
