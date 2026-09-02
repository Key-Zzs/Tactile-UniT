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
