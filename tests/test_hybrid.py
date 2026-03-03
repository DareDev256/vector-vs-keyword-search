from __future__ import annotations

from src.retrieval.common import RetrievalResult, Retriever
from src.retrieval.hybrid import HybridConfig, HybridRetriever


class FakeRetriever(Retriever):
    """Deterministic retriever for testing."""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        return self._results[:k]


def test_hybrid_merges_results() -> None:
    bm25 = FakeRetriever([
        RetrievalResult(doc_id="d1", score=10.0),
        RetrievalResult(doc_id="d2", score=8.0),
    ])
    dense = FakeRetriever([
        RetrievalResult(doc_id="d2", score=0.9),
        RetrievalResult(doc_id="d3", score=0.8),
    ])
    hybrid = HybridRetriever(bm25, dense, HybridConfig())
    results = hybrid.retrieve("test query", k=3)

    doc_ids = [r.doc_id for r in results]
    # d2 appears in both → highest RRF score
    assert doc_ids[0] == "d2"
    assert len(results) == 3
    assert set(doc_ids) == {"d1", "d2", "d3"}


def test_hybrid_respects_k_limit() -> None:
    bm25 = FakeRetriever([
        RetrievalResult(doc_id=f"d{i}", score=float(10 - i))
        for i in range(5)
    ])
    dense = FakeRetriever([
        RetrievalResult(doc_id=f"d{i}", score=float(5 - i) / 5)
        for i in range(5)
    ])
    hybrid = HybridRetriever(bm25, dense, HybridConfig())
    results = hybrid.retrieve("test", k=2)
    assert len(results) == 2


def test_hybrid_weights_affect_ranking() -> None:
    """Higher weight for one retriever should boost its results."""
    bm25 = FakeRetriever([RetrievalResult(doc_id="bm25_only", score=10.0)])
    dense = FakeRetriever([RetrievalResult(doc_id="dense_only", score=0.9)])

    # Heavily weight BM25
    config = HybridConfig(bm25_weight=10.0, dense_weight=0.1)
    hybrid = HybridRetriever(bm25, dense, config)
    results = hybrid.retrieve("test", k=2)
    assert results[0].doc_id == "bm25_only"

    # Heavily weight Dense
    config2 = HybridConfig(bm25_weight=0.1, dense_weight=10.0)
    hybrid2 = HybridRetriever(bm25, dense, config2)
    results2 = hybrid2.retrieve("test", k=2)
    assert results2[0].doc_id == "dense_only"
