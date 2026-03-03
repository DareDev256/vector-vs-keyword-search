from __future__ import annotations

import math
import pytest

from src.eval.metrics import (
    compute_metrics_for_run,
    dcg_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


# ── Recall@k ──────────────────────────────────────────

def test_recall_at_k_basic() -> None:
    relevant = {"d1", "d3"}
    ranked = ["d1", "d2", "d3"]
    assert recall_at_k(relevant, ranked, 1) == pytest.approx(0.5)
    assert recall_at_k(relevant, ranked, 3) == pytest.approx(1.0)


def test_recall_at_k_no_relevant() -> None:
    assert recall_at_k(set(), ["d1", "d2"], 5) == pytest.approx(0.0)


def test_recall_at_k_no_hits() -> None:
    assert recall_at_k({"d9"}, ["d1", "d2", "d3"], 3) == pytest.approx(0.0)


def test_recall_at_k_k_exceeds_list() -> None:
    """k larger than ranked list should still work (just checks what's there)."""
    relevant = {"d1"}
    ranked = ["d1"]
    assert recall_at_k(relevant, ranked, 100) == pytest.approx(1.0)


# ── Precision@k ───────────────────────────────────────

def test_precision_at_k_basic() -> None:
    relevant = {"d1", "d3"}
    ranked = ["d1", "d2", "d3", "d4"]
    assert precision_at_k(relevant, ranked, 1) == pytest.approx(1.0)
    assert precision_at_k(relevant, ranked, 2) == pytest.approx(0.5)
    assert precision_at_k(relevant, ranked, 3) == pytest.approx(2.0 / 3.0)
    assert precision_at_k(relevant, ranked, 4) == pytest.approx(0.5)


def test_precision_at_k_empty_ranked() -> None:
    assert precision_at_k({"d1"}, [], 5) == pytest.approx(0.0)


def test_precision_at_k_all_relevant() -> None:
    relevant = {"d1", "d2", "d3"}
    ranked = ["d1", "d2", "d3"]
    assert precision_at_k(relevant, ranked, 3) == pytest.approx(1.0)


# ── MRR@k ─────────────────────────────────────────────

def test_mrr_at_k_basic() -> None:
    relevant = {"d3"}
    ranked = ["d1", "d2", "d3"]
    assert mrr_at_k(relevant, ranked, 10) == pytest.approx(1.0 / 3.0)
    assert mrr_at_k(relevant, ranked, 2) == pytest.approx(0.0)


def test_mrr_at_k_first_position() -> None:
    assert mrr_at_k({"d1"}, ["d1", "d2"], 5) == pytest.approx(1.0)


def test_mrr_at_k_no_relevant() -> None:
    assert mrr_at_k(set(), ["d1", "d2"], 5) == pytest.approx(0.0)


# ── DCG@k ─────────────────────────────────────────────

def test_dcg_at_k_single_hit() -> None:
    relevant = {"d1"}
    ranked = ["d1", "d2"]
    # DCG = 1/log2(2) = 1.0
    assert dcg_at_k(relevant, ranked, 2) == pytest.approx(1.0)


def test_dcg_at_k_no_hits() -> None:
    assert dcg_at_k({"d9"}, ["d1", "d2"], 2) == pytest.approx(0.0)


# ── NDCG@k ────────────────────────────────────────────

def test_ndcg_at_k_basic() -> None:
    relevant = {"d1", "d3"}
    ranked = ["d1", "d2", "d3"]
    dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    expected_ndcg = dcg / idcg
    assert ndcg_at_k(relevant, ranked, 3) == pytest.approx(expected_ndcg)


def test_ndcg_at_k_perfect_ranking() -> None:
    relevant = {"d1", "d2"}
    ranked = ["d1", "d2", "d3"]
    assert ndcg_at_k(relevant, ranked, 3) == pytest.approx(1.0)


def test_ndcg_at_k_no_relevant() -> None:
    assert ndcg_at_k(set(), ["d1", "d2", "d3"], 3) == pytest.approx(0.0)


def test_ndcg_at_k_worst_ranking() -> None:
    """Relevant doc at last position should give NDCG < 1."""
    relevant = {"d3"}
    ranked = ["d1", "d2", "d3"]
    ndcg = ndcg_at_k(relevant, ranked, 3)
    assert 0.0 < ndcg < 1.0


# ── compute_metrics_for_run ───────────────────────────

def test_compute_metrics_for_run_means_over_queries() -> None:
    qrels = {"q1": {"d1"}, "q2": {"d2"}}
    run = {"q1": ["d1", "d3"], "q2": ["d9", "d2"]}

    metrics = compute_metrics_for_run(qrels, run, ks=(1, 3))
    assert metrics["recall@1"] == pytest.approx((1.0 + 0.0) / 2.0)
    assert metrics["mrr@1"] == pytest.approx((1.0 + 0.0) / 2.0)
    assert metrics["mrr@3"] == pytest.approx((1.0 + 0.5) / 2.0)
    assert "precision@1" in metrics
    assert "ndcg@1" in metrics


def test_compute_metrics_for_run_empty_run() -> None:
    """No overlapping queries between run and qrels should return zeros."""
    qrels = {"q1": {"d1"}}
    run = {"q99": ["d1"]}
    metrics = compute_metrics_for_run(qrels, run, ks=(1,))
    assert metrics["recall@1"] == pytest.approx(0.0)
    assert metrics["mrr@1"] == pytest.approx(0.0)


def test_compute_metrics_for_run_single_query() -> None:
    qrels = {"q1": {"d1", "d2"}}
    run = {"q1": ["d2", "d1", "d3"]}
    metrics = compute_metrics_for_run(qrels, run, ks=(1, 3))
    assert metrics["recall@1"] == pytest.approx(0.5)
    assert metrics["recall@3"] == pytest.approx(1.0)
    assert metrics["precision@3"] == pytest.approx(2.0 / 3.0)


# ── Validation ────────────────────────────────────────

def test_metrics_reject_non_positive_k() -> None:
    with pytest.raises(ValueError):
        recall_at_k({"d1"}, ["d1"], 0)
    with pytest.raises(ValueError):
        mrr_at_k({"d1"}, ["d1"], -1)
    with pytest.raises(ValueError):
        precision_at_k({"d1"}, ["d1"], 0)
    with pytest.raises(ValueError):
        ndcg_at_k({"d1"}, ["d1"], -1)


def test_dcg_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError):
        dcg_at_k({"d1"}, ["d1"], 0)


def test_compute_metrics_rejects_non_positive_ks() -> None:
    with pytest.raises(ValueError):
        compute_metrics_for_run({"q1": {"d1"}}, {"q1": ["d1"]}, ks=(0,))
