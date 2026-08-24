import json

import numpy as np

from gr00t.contact_dynamics.cache import ARRAY_NAMES, ContactTransitionDataset


def test_transition_cache_pair_shapes_and_metadata(tmp_path):
    cache = tmp_path / "cache"
    split_dir = cache / "train"
    split_dir.mkdir(parents=True)
    shapes = {
        "current": (3, 256),
        "future": (3, 256),
        "current_finger_force": (3, 10),
        "future_finger_force": (3, 10),
        "finger_change": (3, 10),
    }
    for name in ARRAY_NAMES:
        shape = shapes.get(name, (3,))
        np.save(split_dir / f"{name}.npy", np.zeros(shape, dtype=np.float32))
    (cache / "manifest.json").write_text(
        json.dumps({"splits": {"train": {"pairs": 3}}})
    )
    dataset = ContactTransitionDataset(cache, "train")
    item = dataset[0]
    assert len(dataset) == 3
    assert item["current"].shape == (256,)
    assert item["future"].shape == (256,)
    assert item["finger_change"].shape == (10,)
    assert item["task_id"].ndim == 0
