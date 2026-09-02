import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gr00t.simulation.s4_3_training import (
    CACHE_ARRAYS,
    PolicyCacheDataset,
    cache_paths,
    valid_anchors,
)
from scripts.simulation.analyze_s4_3_closed_loop import classify, hierarchical_bootstrap
from scripts.simulation.finalize_s4_3_restart import scientific_decision


def write_cache(root: Path, variant: str = "P3") -> PolicyCacheDataset:
    task = "pinch_tongs"
    split = "dev"
    rows = 3
    for name, (tail, dtype) in CACHE_ARRAYS.items():
        path = cache_paths(root, task, split)[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        value = np.zeros((rows, *tail), dtype=dtype)
        np.save(path, value, allow_pickle=False)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "immutable": True,
                "tasks": {task: {split: {"rows": rows}}},
            }
        ),
        encoding="utf-8",
    )
    return PolicyCacheDataset(root, task, split, variant)


def test_policy_cache_dataset_loads_only_variant_required_arrays(tmp_path: Path) -> None:
    dataset = write_cache(tmp_path)
    assert len(dataset) == 3
    assert set(dataset[0]) == {
        "vision",
        "proprio",
        "action",
        "contact_state",
        "contact_target",
    }
    assert dataset[0]["action"].shape == (27, 22)
    assert all(value.dtype == torch.float32 for value in dataset[0].values())


def test_policy_cache_refuses_unfrozen_manifest(tmp_path: Path) -> None:
    write_cache(tmp_path, "P0")
    manifest = tmp_path / "manifest.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["immutable"] = False
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not frozen"):
        PolicyCacheDataset(tmp_path, "pinch_tongs", "dev", "P0")


def test_full_transition_anchor_excludes_action_only_final_row() -> None:
    assert len(valid_anchors(446)) == 394
    assert max(valid_anchors(446)) + 27 == 445


def test_hierarchical_bootstrap_is_deterministic_and_paired() -> None:
    cube = np.ones((3, 3, 30), dtype=np.float64) * 0.1
    first = hierarchical_bootstrap(cube, samples=200, seed=7)
    second = hierarchical_bootstrap(cube, samples=200, seed=7)
    assert first == second
    assert first[0] == pytest.approx(0.1)
    assert first[1] == pytest.approx([0.1, 0.1])
    assert classify(*first) == "MATERIAL_IMPROVEMENT"
    assert classify(-first[0], [-0.1, -0.1]) == "MATERIAL_HURT"
    assert classify(0.049, [0.01, 0.08]) == "NO_MATERIAL_DIFFERENCE"


def comparison(classification: str, deltas: tuple[float, float, float]) -> dict:
    return {
        "classification": classification,
        "per_task": {
            task: {
                "delta": delta,
                "classification": (
                    "MATERIAL_IMPROVEMENT"
                    if delta >= 0.05
                    else "MATERIAL_HURT" if delta <= -0.05 else "NO_MATERIAL_DIFFERENCE"
                ),
            }
            for task, delta in zip(("pinch_tongs", "hammer_nail", "click_mouse"), deltas)
        },
    }


def decision_primary(
    p1: tuple[str, tuple[float, float, float]],
    p2: tuple[str, tuple[float, float, float]],
    p3p2: tuple[str, tuple[float, float, float]],
    p3p0: tuple[str, tuple[float, float, float]],
) -> dict:
    return {
        "comparisons": {
            "P1-P0": comparison(*p1),
            "P2-P1": comparison(*p2),
            "P3-P2": comparison(*p3p2),
            "P3-P0": comparison(*p3p0),
        }
    }


def test_scientific_decision_uses_frozen_mixed_hurt_and_full_precedence() -> None:
    neutral = ("NO_MATERIAL_DIFFERENCE", (0.0, 0.0, 0.0))
    validity = {"status": "PASS"}
    summary = {"benchmark_learning_floor": "HEALTHY"}
    mixed = decision_primary(
        neutral,
        neutral,
        neutral,
        ("NO_MATERIAL_DIFFERENCE", (0.10, 0.0, -0.10)),
    )
    assert scientific_decision(validity, summary, mixed) == "S4_3_2_MIXED_TASK_DEPENDENT_RESULT"
    hurt = decision_primary(
        neutral,
        neutral,
        ("MATERIAL_HURT", (-0.12, -0.04, -0.03)),
        neutral,
    )
    assert scientific_decision(validity, summary, hurt) == "S4_3_2_TACTILE_UNIT_HURTS_ACT"
    full = decision_primary(
        neutral,
        neutral,
        ("MATERIAL_IMPROVEMENT", (0.10, 0.02, 0.08)),
        ("MATERIAL_IMPROVEMENT", (0.12, 0.01, 0.09)),
    )
    assert scientific_decision(validity, summary, full) == ("S4_3_2_FULL_TACTILE_UNIT_IMPROVES_ACT")


def test_scientific_decision_covers_frozen_non_p3_and_benchmark_outcomes() -> None:
    neutral = ("NO_MATERIAL_DIFFERENCE", (0.0, 0.0, 0.0))
    valid = {"status": "PASS"}
    healthy = {"benchmark_learning_floor": "HEALTHY"}
    baseline = decision_primary(neutral, neutral, neutral, neutral)
    assert (
        scientific_decision({"status": "FAIL"}, healthy, baseline) == "S4_3_2_ACT_BENCHMARK_INVALID"
    )
    assert (
        scientific_decision(valid, {"benchmark_learning_floor": "WEAK"}, baseline)
        == "S4_3_2_ACT_BENCHMARK_WEAK"
    )
    assert scientific_decision(valid, healthy, baseline) == "S4_3_2_NO_MATERIAL_ACT_GAIN"
    contact = decision_primary(
        neutral,
        ("MATERIAL_IMPROVEMENT", (0.08, 0.03, 0.06)),
        neutral,
        neutral,
    )
    assert scientific_decision(valid, healthy, contact) == "S4_3_2_CONTACT_STATE_IMPROVES_ACT"
    raw = decision_primary(
        ("MATERIAL_IMPROVEMENT", (0.08, 0.03, 0.06)),
        neutral,
        neutral,
        neutral,
    )
    assert scientific_decision(valid, healthy, raw) == "S4_3_2_RAW_TACTILE_IMPROVES_ACT_ONLY"
