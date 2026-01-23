from __future__ import annotations

import math
import pytest

from src.eval.metrics import (
    compute_metrics_for_run,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_recall_at_k_basic() -> None:
    relevant = {"d1", "d3"}
    ranked = ["d1", "d2", "d3"]
    assert recall_at_k(relevant, ranked, 1) == pytest.approx(0.5)
    assert recall_at_k(relevant, ranked, 3) == pytest.approx(1.0)


def test_precision_at_k_basic() -> None:
    relevant = {"d1", "d3"}
    ranked = ["d1", "d2", "d3", "d4"]
    # At k=1: 1 hit / 1 retrieved = 1.0
    assert precision_at_k(relevant, ranked, 1) == pytest.approx(1.0)
    # At k=2: 1 hit / 2 retrieved = 0.5
    assert precision_at_k(relevant, ranked, 2) == pytest.approx(0.5)
    # At k=3: 2 hits / 3 retrieved = 0.667
    assert precision_at_k(relevant, ranked, 3) == pytest.approx(2.0 / 3.0)
    # At k=4: 2 hits / 4 retrieved = 0.5
    assert precision_at_k(relevant, ranked, 4) == pytest.approx(0.5)


def test_mrr_at_k_basic() -> None:
    relevant = {"d3"}
    ranked = ["d1", "d2", "d3"]
    assert mrr_at_k(relevant, ranked, 10) == pytest.approx(1.0 / 3.0)
    assert mrr_at_k(relevant, ranked, 2) == pytest.approx(0.0)


def test_ndcg_at_k_basic() -> None:
    relevant = {"d1", "d3"}
    ranked = ["d1", "d2", "d3"]
    # DCG@3: 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5 = 1.5
    # IDCG@3: 1/log2(2) + 1/log2(3) = 1.0 + 0.631 = 1.631
    # NDCG@3 = 1.5 / 1.631 ≈ 0.92
    dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    expected_ndcg = dcg / idcg
    assert ndcg_at_k(relevant, ranked, 3) == pytest.approx(expected_ndcg)


def test_ndcg_at_k_perfect_ranking() -> None:
    relevant = {"d1", "d2"}
    ranked = ["d1", "d2", "d3"]  # Perfect ranking
    # DCG = IDCG, so NDCG = 1.0
    assert ndcg_at_k(relevant, ranked, 3) == pytest.approx(1.0)


def test_ndcg_at_k_no_relevant() -> None:
    relevant: set[str] = set()
    ranked = ["d1", "d2", "d3"]
    assert ndcg_at_k(relevant, ranked, 3) == pytest.approx(0.0)


def test_compute_metrics_for_run_means_over_queries() -> None:
    qrels = {"q1": {"d1"}, "q2": {"d2"}}
    run = {"q1": ["d1", "d3"], "q2": ["d9", "d2"]}

    metrics = compute_metrics_for_run(qrels, run, ks=(1, 3))
    assert metrics["recall@1"] == pytest.approx((1.0 + 0.0) / 2.0)
    assert metrics["mrr@1"] == pytest.approx((1.0 + 0.0) / 2.0)
    assert metrics["mrr@3"] == pytest.approx((1.0 + 0.5) / 2.0)
    # Check new metrics are present
    assert "precision@1" in metrics
    assert "ndcg@1" in metrics


def test_metrics_reject_non_positive_k() -> None:
    with pytest.raises(ValueError):
        recall_at_k({"d1"}, ["d1"], 0)
    with pytest.raises(ValueError):
        mrr_at_k({"d1"}, ["d1"], -1)
    with pytest.raises(ValueError):
        precision_at_k({"d1"}, ["d1"], 0)
    with pytest.raises(ValueError):
        ndcg_at_k({"d1"}, ["d1"], -1)

