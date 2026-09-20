from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import pytest
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/simulation/run_s4_3_pi2n_development.py"
MANIFEST = ROOT / ".local/artifacts/simulation/s4_3_pi2n/development_manifest.json"


def load_module():
    spec = importlib.util.spec_from_file_location("run_s4_3_pi2n_development", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_development_scope_matches_frozen_manifest() -> None:
    module = load_module()
    manifest = json.loads(MANIFEST.read_text())
    assert module.MODELS == ("B_HVA", "B2", "B_VAC_V")
    assert module.SEED_BLOCKS == (9, 10, 11)
    assert module.EPISODES_PER_BLOCK == 10
    assert manifest["episodes"] == 30
    expected = module.expected_resets_by_seed(manifest)
    assert all(len(expected[seed]) == 10 for seed in module.SEED_BLOCKS)
    assert len({item for values in expected.values() for item in values}) == 30


def test_execution_order_balances_first_position_without_changing_models() -> None:
    module = load_module()
    assert all(
        set(module.EXECUTION_ORDER[seed]) == set(module.MODELS)
        for seed in module.SEED_BLOCKS
    )
    assert [module.EXECUTION_ORDER[seed][0] for seed in module.SEED_BLOCKS] == [
        "B_HVA",
        "B2",
        "B_VAC_V",
    ]


def test_runtime_v2_environment_is_exact() -> None:
    module = load_module()
    runtime = json.loads(
        (
            ROOT
            / ".local/artifacts/simulation/s4_3_pi2n/runtime_protocol_v2.json"
        ).read_text()
    )
    assert list(module.XLA_FLAGS) == runtime["required_server_environment"][
        "XLA_FLAGS"
    ]
    source = SCRIPT.read_text()
    assert '"CUBLAS_WORKSPACE_CONFIG": ":4096:8"' in source
    assert "serve_s4_3_pi2n_policy.py" in source
    assert "fresh_policy_server" in source


def test_selection_is_single_candidate_only_after_complete_integrity() -> None:
    source = SCRIPT.read_text()
    assert "PI2N DEV raw cohort integrity failed; selection is forbidden" in source
    assert '"eligible_new_vac_candidates": ["B_VAC_V"]' in source
    assert '"anchors_not_eligible_for_vac_star": ["B_HVA", "B2"]' in source
    assert '"development_result_is_confirmatory_evidence": False' in source
    assert '"final_performance_accessed": False' in source
    assert "refusing to overwrite or resume canonical PI2N DEV outcomes" in source


def test_consolidation_requires_all_nine_exact_reset_cohorts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    raw_root = tmp_path / "raw"
    identities = {
        seed: [f"{seed:02d}{index:062d}" for index in range(10)]
        for seed in module.SEED_BLOCKS
    }
    manifest = {
        "ordered_reset_sequence_sha256": "f" * 64,
        "reset_specs": [
            {"seed": seed, "reset_identity": identity}
            for seed in module.SEED_BLOCKS
            for identity in identities[seed]
        ],
    }
    manifest_path = tmp_path / "development_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    for seed in module.SEED_BLOCKS:
        block = raw_root / f"seed_{seed}"
        block.mkdir(parents=True)
        for model in module.MODELS:
            payload = {
                "status": "PASS",
                "model": model,
                "mode": module.RUNTIME_MODES[model],
                "evaluator_seed": seed,
                "episodes": 10,
                "gates": {"complete": "PASS"},
                "episode_results": [
                    {
                        "episode_index": index,
                        "reset_identity": identity,
                        "success": index % 2 == 0,
                    }
                    for index, identity in enumerate(identities[seed])
                ],
            }
            (block / f"{model.lower()}_raw_rollouts.json").write_text(
                json.dumps(payload)
            )
    monkeypatch.setattr(module, "RAW", raw_root)
    monkeypatch.setattr(module, "RESET_MANIFEST", manifest_path)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    freeze = {
        "selection": {"rule": "frozen"},
        "checkpoint_manifests": {
            "B_VAC_V": {"checkpoint_tree_sha256": "a" * 64}
        },
    }
    results, selection, execution = module.consolidate_and_select(
        freeze, [], []
    )
    assert results["total_canonical_outcomes"] == 90
    assert results["all_models_exact_same_ordered_resets"] is True
    assert selection["selected_vac_star"] == "B_VAC_V"
    assert selection["selected_checkpoint_tree_sha256"] == "a" * 64
    assert execution["performance_interpreted_before_all_cohorts_complete"] is False
    broken = raw_root / "seed_10/b2_raw_rollouts.json"
    payload = json.loads(broken.read_text())
    payload["episode_results"][0]["reset_identity"] = "b" * 64
    broken.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="selection is forbidden"):
        module.consolidate_and_select(freeze, [], [])
