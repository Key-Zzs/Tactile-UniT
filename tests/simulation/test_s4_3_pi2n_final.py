from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.simulation import analyze_s4_3_pi2n_final as analysis
from scripts.simulation import freeze_s4_3_pi2n_final as freeze
from scripts.simulation import launch_s4_3_pi2n_final as launch
from scripts.simulation import run_s4_3_pi2n_final as final


def _raw(model: str, seed: int, identities: list[str]) -> dict:
    contact = final.RUNTIME_MODES[model] != "NONE"
    return {
        "status": "PASS",
        "model": model,
        "mode": final.RUNTIME_MODES[model],
        "evaluator_seed": seed,
        "episodes": 50,
        "contact_state_sent_to_policy": contact,
        "gates": {
            "training_only_targets_never_sent": "PASS",
            "all_actions_finite": "PASS",
        },
        "episode_results": [
            {
                "episode_index": index,
                "reset_identity": identity,
                "success": False,
                "termination": "timeout",
                "steps": 1000,
            }
            for index, identity in enumerate(identities)
        ],
    }


def test_final_completeness_requires_all_1200_exact_aligned_outcomes(
    tmp_path: Path, monkeypatch
) -> None:
    raw_root = tmp_path / "raw"
    manifest_path = tmp_path / "final_reset_manifest.json"
    identities = {
        seed: [f"{seed}-{index:02d}" for index in range(50)]
        for seed in final.SEED_BLOCKS
    }
    manifest = {
        "ordered_reset_identities": [
            identity for seed in final.SEED_BLOCKS for identity in identities[seed]
        ],
        "reset_specs": [
            {"seed": seed, "reset_identity": identity}
            for seed in final.SEED_BLOCKS
            for identity in identities[seed]
        ],
    }
    manifest_path.write_text(json.dumps(manifest))
    for seed in final.SEED_BLOCKS:
        block = raw_root / f"seed_{seed}"
        block.mkdir(parents=True)
        for model in final.MODELS:
            (block / f"{model.lower()}_raw_rollouts.json").write_text(
                json.dumps(_raw(model, seed, identities[seed]))
            )
    monkeypatch.setattr(final, "RAW", raw_root)
    monkeypatch.setattr(final, "RESET_MANIFEST", manifest_path)
    monkeypatch.setattr(final, "ROOT", tmp_path)
    workers = [
        {"model": model, "seed": seed, "status": "PASS", "physical_gpu": 0}
        for seed in final.SEED_BLOCKS
        for model in final.MODELS
    ]
    completeness, execution = final.audit_completeness(
        {"vac_star": "B_VAC_V"}, workers, []
    )
    assert completeness["status"] == "PASS"
    assert completeness["total_canonical_outcomes"] == 1200
    assert completeness["statistics_computed"] is False
    assert execution["status"] == "PASS"

    bad_path = raw_root / "seed_12" / "b2_raw_rollouts.json"
    bad = json.loads(bad_path.read_text())
    bad["episode_results"][0]["reset_identity"] = "modified"
    bad_path.write_text(json.dumps(bad))
    completeness, _ = final.audit_completeness(
        {"vac_star": "B_VAC_V"}, workers, []
    )
    assert completeness["status"] == "FAIL"
    assert completeness["per_job_integrity_gates"]["seed_12/B2"] == "FAIL"


def _analysis_rows(successes: int) -> list[dict]:
    return [
        {
            "reset_identity": f"reset-{index:03d}",
            "success": index < successes,
            "termination": "success" if index < successes else "timeout",
            "steps": 500 if index < successes else 1000,
            "task_progress": {
                "final_pinch_count": 3 if index < successes else 1,
                "max_pinch_count": 3 if index < successes else 2,
            },
            "tactile_diagnostics": {
                "max_normal_force": 4.0,
                "mean_l2": 2.0,
                "active_samples": 100,
                "matched_contact_count_sum": 300,
            },
            "executed_action_stats": {"mean_l2": 1.5},
        }
        for index in range(200)
    ]


def test_final_statistics_cover_frozen_six_comparison_family(
    tmp_path: Path, monkeypatch
) -> None:
    protocol_path = tmp_path / "protocol.json"
    freeze_path = tmp_path / "freeze.json"
    completeness_path = tmp_path / "completeness.json"
    for path in (protocol_path, freeze_path, completeness_path):
        path.write_text("{}")
    monkeypatch.setattr(analysis, "PROTOCOL", protocol_path)
    monkeypatch.setattr(analysis, "PRE_FREEZE", freeze_path)
    monkeypatch.setattr(analysis, "COMPLETENESS", completeness_path)
    rows = {
        "B0": _analysis_rows(40),
        "B_VA27": _analysis_rows(60),
        "B1": _analysis_rows(50),
        "B_HVA": _analysis_rows(100),
        "B2": _analysis_rows(80),
        "B_VAC_V": _analysis_rows(150),
    }
    protocol = {
        "formal_statistics": {
            "paired_bootstrap_samples": 2_000,
            "paired_bootstrap_seed": 4317,
            "material_threshold_pp": 10,
        }
    }
    statistics, process, claim, decision = analysis.analyze(
        rows, protocol, {"vac_star": "B_VAC_V"}
    )
    assert statistics["status"] == "PASS"
    assert list(statistics["paired_comparisons"]) == [
        name for name, _, _ in analysis.COMPARISONS
    ]
    assert len(statistics["paired_comparisons"]) == 6
    assert statistics["paired_comparisons"]["VAC_STAR-B_HVA"][
        "classification"
    ] == "POSITIVE_CONFIRMED"
    assert claim["VAC_VS_STRONG_VA"] == "VAC_ADVANTAGE_CONFIRMED"
    assert claim["CLAIM_LEVEL"] == "FIXED_SEED_ONLY"
    assert process["models"]["B_VAC_V"]["max_normal_force_proxy"][
        "available"
    ] is True
    assert process["tangential_force"] == "NA_NOT_EMITTED_BY_FROZEN_RUNTIME"
    assert decision["PI2B_started"] is False


def test_holm_adjust_is_monotone_and_family_complete() -> None:
    raw = {f"c{index}": value for index, value in enumerate([0.001, 0.02, 0.03, 0.2, 0.4, 1.0])}
    adjusted = analysis.holm_adjust(raw)
    ordered = sorted(raw, key=raw.get)
    values = [adjusted[name] for name in ordered]
    assert len(adjusted) == 6
    assert values == sorted(values)
    assert all(raw[name] <= adjusted[name] <= 1.0 for name in raw)


def test_final_freeze_and_launcher_are_manual_non_overwriting_gates() -> None:
    assert freeze.MODELS == final.MODELS
    source = Path(freeze.__file__).read_text()
    launch_source = Path(launch.__file__).read_text()
    runner_source = Path(final.__file__).read_text()
    assert "refusing to overwrite PI2N pre-FINAL freeze" in source
    assert "def no_final_outputs" in source
    assert "final_performance_seen_before_freeze" in source
    assert '"automatic_statistics": False' in launch_source
    assert '"automatic_PI2B": False' in launch_source
    assert "statistics_computed" in runner_source
    assert "performance_interpreted_during_execution" in runner_source
    assert len(final.MODELS) == 6
    assert len(final.SEED_BLOCKS) == 4
    assert np.prod([len(final.MODELS), len(final.SEED_BLOCKS), 50]) == 1200
