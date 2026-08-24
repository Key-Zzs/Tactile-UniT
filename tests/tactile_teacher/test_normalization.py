import numpy as np

from gr00t.tactile_teacher.normalization import RunningFeatureStats


def test_running_feature_stats_merge_and_roundtrip():
    rng = np.random.default_rng(4)
    values = rng.normal(size=(1000, 60)).astype(np.float32)
    running = RunningFeatureStats()
    running.update(values[:400], quantile_sample=values[:200])
    running.update(values[400:], quantile_sample=values[400:600])
    stats = running.finalize()

    assert stats.count == 1000
    assert stats.quantile_sample_count == 400
    np.testing.assert_allclose(stats.mean, values.mean(axis=0), atol=1e-6)
    probe = values[:10]
    np.testing.assert_allclose(stats.denormalize(stats.normalize(probe, clip=100)), probe, atol=1e-5)
