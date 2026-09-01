from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gr00t.simulation.intrinsic_dimension import (
    dimension_metrics,
    fit_active_cop_means,
    mask_inactive_cop,
    temporal_summary,
)
from scripts.simulation import audit_s4_2r_intrinsic_dimension as intrinsic_audit
from scripts.simulation import audit_s4_2r_region_coverage as region_audit


ROOT = Path(__file__).resolve().parents[2]


def test_intrinsic_dimension_estimators_are_deterministic_and_rank_ordered() -> None:
    rng = np.random.default_rng(7)
    factors = rng.normal(size=(400, 3))
    projection = rng.normal(size=(3, 12))
    values = factors @ projection
    first = dimension_metrics(values, twonn=False)
    second = dimension_metrics(values, twonn=False)
    assert first == second
    assert 1.0 <= first["stable_rank"] <= first["participation_ratio"]
    assert first["participation_ratio"] <= first["effective_rank"] <= 3.000001
    assert first["d99"] <= 3


def test_contact_mask_neutralizes_only_inactive_cop() -> None:
    tactile = np.zeros((4, 30), dtype=np.float64)
    matrix = tactile.reshape(4, 5, 6)
    matrix[0, 0] = [1.0, 2.0, 0.5, 1.0, 2.0, 3.0]
    matrix[1, 0] = [1.0, 3.0, 0.7, 3.0, 4.0, 5.0]
    matrix[2, 0, 3:6] = [99.0, 99.0, 99.0]
    means = fit_active_cop_means(tactile)
    masked = mask_inactive_cop(tactile, means).reshape(4, 5, 6)
    np.testing.assert_allclose(means[0], [2.0, 3.0, 4.0])
    np.testing.assert_allclose(masked[2, 0, 3:6], means[0])
    np.testing.assert_allclose(masked[0, 0, 3:6], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(masked[:, :, :3], matrix[:, :, :3])


def test_temporal_summary_has_six_statistics_per_channel() -> None:
    history = np.arange(2 * 26 * 30, dtype=np.float64).reshape(2, 26, 30)
    summary = temporal_summary(history)
    assert summary.shape == (2, 180)
    np.testing.assert_array_equal(summary[:, 60:90], history[:, -1])
    np.testing.assert_array_equal(summary[:, 150:180], history[:, -1] - history[:, 0])


def test_s4_2r_audits_are_train_validation_only() -> None:
    assert intrinsic_audit.SPLITS == ("train", "validation")
    assert region_audit.SPLITS == ("train", "validation")


def test_region_contract_has_no_duplicate_body_or_explicit_geom_assignment() -> None:
    config = json.loads(
        (ROOT / "configs/simulation/s4_2_dexjoco_contact_regions.json").read_text()
    )
    bodies = [body for row in config["regions"] for body in row["body_names"]]
    geoms = [geom for row in config["regions"] for geom in row["geom_names"]]
    assert len(bodies) == len(set(bodies))
    assert len(geoms) == len(set(geoms))
    assert next(
        row for row in config["regions"] if row["region_name"] == "right_thumb"
    )["body_names"] == ["th_base", "th_proximal", "th_medial", "th_distal", "th_tip"]
