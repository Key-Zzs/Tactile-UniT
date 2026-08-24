import json

import numpy as np
import pandas as pd

from gr00t.tactile_teacher.dataset import TactileEpisodeStore


def test_episode_store_reads_contiguous_numeric_episode(tmp_path):
    root = tmp_path / "trex"
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    features = {
        "observation.tactile_force": {"dtype": "float32", "shape": [60]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": 30,
                "total_episodes": 1,
                "total_frames": 4,
                "features": features,
            }
        )
    )
    pd.DataFrame(
        {
            "episode_index": [0],
            "length": [4],
            "data/chunk_index": [0],
            "data/file_index": [0],
            "dataset_from_index": [0],
            "dataset_to_index": [4],
            "motor_primitive": ["press"],
            "object": ["foam"],
            "target": [None],
        }
    ).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    wrench = np.arange(4 * 60, dtype=np.float32).reshape(4, 60)
    pd.DataFrame(
        {
            "observation.tactile_force": list(wrench),
            "timestamp": np.arange(4, dtype=np.float32) / 30,
            "frame_index": np.arange(4, dtype=np.int64),
            "episode_index": np.zeros(4, dtype=np.int64),
            "index": np.arange(4, dtype=np.int64),
            "task_index": np.zeros(4, dtype=np.int64),
        }
    ).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")

    store = TactileEpisodeStore(root, dataset_revision="test")
    episode = store.get_episode(0)
    np.testing.assert_array_equal(episode.wrench, wrench)
    assert episode.record.motor_primitive == "press"
    assert store.contract.tactile_dim == 60
