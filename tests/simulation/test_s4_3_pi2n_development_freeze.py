from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/simulation/freeze_s4_3_pi2n_development.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "freeze_s4_3_pi2n_development", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_development_scope_is_exact_and_does_not_replace_x() -> None:
    module = load_module()
    assert module.MODELS == ("B_HVA", "B2", "B_VAC_V")
    assert module.NEW_VAC_CANDIDATES == ("B_VAC_V",)
    assert module.SEED_BLOCKS == (9, 10, 11)
    assert module.EPISODES_PER_BLOCK == 10
    assert set(module.EXECUTION_ORDER) == set(module.SEED_BLOCKS)
    assert all(set(order) == set(module.MODELS) for order in module.EXECUTION_ORDER.values())
    assert [module.EXECUTION_ORDER[seed][0] for seed in module.SEED_BLOCKS] == [
        "B_HVA",
        "B2",
        "B_VAC_V",
    ]


def test_freeze_binds_every_scientific_runtime_source() -> None:
    module = load_module()
    assert (
        "scripts/simulation/audit_s4_3_pi2n_candidate_completion.py"
        in module.SOURCE_FILES
    )
    assert "scripts/simulation/run_s4_3_pi2n_development.py" in module.SOURCE_FILES
    assert "scripts/simulation/serve_s4_3_pi2n_policy.py" in module.SOURCE_FILES
    assert "scripts/simulation/evaluate_s4_3_pi1d_augmented.py" in module.SOURCE_FILES
    assert "gr00t/simulation/pi05_tactile_unit.py" in module.SOURCE_FILES
    assert set(module.CHECKPOINT_MANIFESTS) == set(module.MODELS)
    assert set(module.CHECKPOINT_PATHS) == set(module.MODELS)


def test_pre_dev_freeze_is_non_overwriting_and_performance_blind() -> None:
    source = SCRIPT.read_text()
    assert "refusing to overwrite PI2N pre-DEV freeze" in source
    assert "development performance already exists" in source
    assert '"development_performance_seen_before_freeze": False' in source
    assert '"final_performance_seen_before_freeze": False' in source
    assert '"candidate_selection_performed_before_freeze": False' in source
    assert "fresh_policy_server_per_model_seed_block" in source
    assert "no_best_of_retry_splicing" in source
    assert "all_live_checkpoint_trees_match_manifests" in source
    assert "all_training_completion_artifacts_exact" in source


def test_completion_chain_requires_exact_hash_and_all_pass_gates(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    completion_path = tmp_path / "completion.json"
    completion = {
        "status": "PASS",
        "optimizer_steps": 30_000,
        "gates": {"cold_load": "PASS", "finite": "PASS"},
    }
    completion_path.write_text(json.dumps(completion))
    manifest = {
        "training_completion": "$REPO_ROOT/completion.json",
        "training_completion_sha256": hashlib.sha256(
            completion_path.read_bytes()
        ).hexdigest(),
        "gates": {"checkpoint": "PASS"},
    }
    assert module.completion_chain_valid(manifest)
    manifest["training_completion_sha256"] = "0" * 64
    assert not module.completion_chain_valid(manifest)
    manifest["training_completion_sha256"] = hashlib.sha256(
        completion_path.read_bytes()
    ).hexdigest()
    completion["gates"]["finite"] = "FAIL"
    completion_path.write_text(json.dumps(completion))
    manifest["training_completion_sha256"] = hashlib.sha256(
        completion_path.read_bytes()
    ).hexdigest()
    assert not module.completion_chain_valid(manifest)
