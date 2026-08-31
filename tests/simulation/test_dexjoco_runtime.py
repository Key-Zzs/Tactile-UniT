from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

if importlib.util.find_spec("dexjoco") is None:
    pytest.skip("DexJoCo integration tests require tactile-unit-dexjoco", allow_module_level=True)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.pop("DISPLAY", None)

from gr00t.simulation.dexjoco_adapter import DexJoCoRuntimeAdapter, SimPolicyAction


def test_real_pinch_tongs_headless_reset_step_rgb_contact_and_named_regions():
    adapter = DexJoCoRuntimeAdapter(task_name="pinch_tongs", seed=7, episode_id="pytest")
    audit = adapter.start()
    observation = adapter.reset()
    initial = observation.proprio.copy()
    try:
        assert audit["status"] == "PASS"
        assert audit["mapped_geom_count"] == 21
        assert audit["unmapped_relevant_geoms"] == []
        assert observation.rgb.shape == (640, 640, 3)
        assert observation.rgb.dtype == np.uint8
        assert observation.proprio.shape == (31,)
        assert observation.sim_tactile.shape == (30,)
        assert np.count_nonzero(observation.sim_tactile) == 0
        tongs = initial[23:26]
        target = np.asarray([tongs[0] - 0.159, tongs[1], tongs[2] - 0.029])
        neutral = adapter.neutral_policy_action().values
        maximum_normal = 0.0
        maximum_tangent = 0.0
        occupied_regions = set()
        for step in range(100):
            alpha = min(1.0, (step + 1) / 30.0)
            xyz = neutral[:3] * (1.0 - alpha) + target * alpha
            action = SimPolicyAction(np.concatenate([xyz, neutral[3:]]))
            observation, _, _, env_action = adapter.step(action)
            matrix = observation.sim_tactile.reshape(5, 6)
            maximum_normal = max(maximum_normal, float(matrix[:, 1].sum()))
            maximum_tangent = max(maximum_tangent, float(matrix[:, 2].sum()))
            occupied_regions.update(np.flatnonzero(matrix[:, 0] > 0).tolist())
            assert env_action.values.shape == (23,)
        assert observation.timestamp_sec == pytest.approx(2.0)
        assert maximum_normal > 0
        assert maximum_tangent > 0
        assert occupied_regions
        assert adapter.raw_env.data.ncon >= 0
    finally:
        adapter.close()
