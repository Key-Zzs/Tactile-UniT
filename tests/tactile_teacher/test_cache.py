import json

import numpy as np

from gr00t.tactile_teacher.cache import WrenchWindowDataset


def test_window_cache_loader(tmp_path):
    split = tmp_path / "train"
    split.mkdir()
    np.save(split / "history.npy", np.zeros((2, 16, 60), dtype=np.float32))
    np.save(split / "future.npy", np.zeros((2, 8, 60), dtype=np.float32))
    np.save(split / "episode_id.npy", np.array([1, 2], dtype=np.int32))
    np.save(split / "primitive_id.npy", np.array([3, 4], dtype=np.int16))
    np.save(split / "object_id.npy", np.array([5, 6], dtype=np.int16))
    (tmp_path / "manifest.json").write_text(
        json.dumps({"splits": {"train": {"windows": 2}}})
    )
    dataset = WrenchWindowDataset(tmp_path, "train")
    sample = dataset[1]
    assert len(dataset) == 2
    assert sample["history"].shape == (16, 60)
    assert sample["future"].shape == (8, 60)
    assert sample["episode_id"].item() == 2
