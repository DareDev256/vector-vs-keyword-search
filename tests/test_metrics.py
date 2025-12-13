from __future__ import annotations

import pytest

from src.eval.metrics import compute_metrics_for_run, mrr_at_k, recall_at_k


def test_recall_at_k_basic() -> None:
    relevant = {"d1", "d3"}
    ranked = ["d1", "d2", "d3"]
    assert recall_at_k(relevant, ranked, 1) == pytest.approx(0.5)
    assert recall_at_k(relevant, ranked, 3) == pytest.approx(1.0)


def test_mrr_at_k_basic() -> None:
    relevant = {"d3"}
    ranked = ["d1", "d2", "d3"]
    assert mrr_at_k(relevant, ranked, 10) == pytest.approx(1.0 / 3.0)
    assert mrr_at_k(relevant, ranked, 2) == pytest.approx(0.0)


def test_compute_metrics_for_run_means_over_queries() -> None:
    qrels = {"q1": {"d1"}, "q2": {"d2"}}
    run = {"q1": ["d1", "d3"], "q2": ["d9", "d2"]}

    metrics = compute_metrics_for_run(qrels, run, ks=(1, 3))
    assert metrics["recall@1"] == pytest.approx((1.0 + 0.0) / 2.0)
    assert metrics["mrr@1"] == pytest.approx((1.0 + 0.0) / 2.0)
    assert metrics["mrr@3"] == pytest.approx((1.0 + 0.5) / 2.0)


def test_metrics_reject_non_positive_k() -> None:
    with pytest.raises(ValueError):
        recall_at_k({"d1"}, ["d1"], 0)
    with pytest.raises(ValueError):
        mrr_at_k({"d1"}, ["d1"], -1)

