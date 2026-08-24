from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/reproduce"))

from unit_representation_metrics import (
    ami_score,
    code_agreement,
    codebook_stats,
    linear_cka,
    mean_query_pool,
    mmd_rbf,
    nmi_score,
    retrieval_metrics,
    sliced_wasserstein,
)


def test_identical_embeddings_have_zero_distribution_distance_and_perfect_retrieval():
    rng = np.random.default_rng(7)
    values = rng.normal(size=(12, 3, 4, 8))
    pooled = mean_query_pool(values)
    result = retrieval_metrics(pooled[:, 0], pooled[:, 0])
    assert result["recall_at_1"] == 1.0
    assert result["recall_at_5"] == 1.0
    assert result["recall_at_10"] == 1.0
    assert mmd_rbf(pooled[:, 0], pooled[:, 0])["mmd"] < 1e-10
    assert sliced_wasserstein(pooled[:, 0], pooled[:, 0], projections=16)["swd"] < 1e-10
    assert abs(linear_cka(pooled[:, 0], pooled[:, 0]) - 1.0) < 1e-10


def test_discrete_identical_and_permuted_controls():
    first = np.asarray([0, 1, 0, 2, 1, 2, 0, 1])
    assert abs(nmi_score(first, first) - 1.0) < 1e-10
    assert abs(ami_score(first, first) - 1.0) < 1e-10
    permuted = first[::-1]
    assert nmi_score(first, permuted) < 1.0
    assert ami_score(first, permuted) < 1.0


def test_code_usage_and_pair_agreement_are_well_formed():
    codes = np.zeros((5, 3, 4, 2), dtype=np.int64)
    codes[:, :, :, 0] = np.arange(4)[None, None, :]
    codes[:, :, :, 1] = 1
    usage = codebook_stats(codes, codebook_size=8)
    assert len(usage) == 6
    assert all(0.0 <= row["active_ratio"] <= 1.0 for row in usage)
    agreements = code_agreement(codes)
    exact = [row for row in agreements if "stage_exact_match" in row]
    assert all(row["stage_exact_match"] == 1.0 for row in exact)


def test_random_independent_embeddings_are_not_perfectly_aligned():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(64, 16))
    y = rng.normal(size=(64, 16))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    assert retrieval_metrics(x, y)["recall_at_1"] < 1.0
