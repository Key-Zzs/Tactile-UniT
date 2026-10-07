from __future__ import annotations

import numpy as np

from scripts.simulation.summarize_pi2sr_action_sensitivity import (
    CONDITIONS,
    MODELS,
    POSITIONS,
    summarize,
)


def test_summary_requires_and_aggregates_frozen_conditions() -> None:
    records = []
    for model in MODELS:
        for position in POSITIONS:
            for condition in CONDITIONS:
                delta = 0.0 if condition == "correct" else float(position + 1)
                records.append(
                    {
                        "checkpoint_id": model,
                        "fixed_position": position,
                        "condition": condition,
                        "global_row": 100 + position,
                        "action": np.zeros((30, 22), dtype=np.float32).tolist(),
                        "action_sha256": f"{model}-{position}-{condition}",
                        "delta": {"chunk_l2": {"all": delta}},
                    }
                )
    result = summarize(records)
    assert len(result["canonical_vs_alternate"]) == len(MODELS) * len(POSITIONS)
    assert result["historical_perturbations"]["zero"]["nonzero_count"] == 32
    assert result["historical_perturbations"]["same_episode_lag5"]["max_chunk_l2"] == 8.0
