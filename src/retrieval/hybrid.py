from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from src.retrieval.common import RetrievalResult, Retriever
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class HybridConfig:
    """Configuration for hybrid retrieval."""
    rrf_k: int = 60  # RRF constant (typical values: 60)
    bm25_weight: float = 1.0
    dense_weight: float = 1.0


class HybridRetriever(Retriever):
    """
    Combines BM25 and Dense retrieval using Reciprocal Rank Fusion (RRF).

    RRF Score = sum(weight / (k + rank)) for each retriever

    This is a simple but effective fusion method that:
    - Doesn't require score normalization
    - Is robust to different score scales
    - Often outperforms either method alone
    """

    def __init__(
        self,
        bm25_retriever: Retriever,
        dense_retriever: Retriever,
        config: HybridConfig | None = None,
    ) -> None:
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.config = config or HybridConfig()

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        if k <= 0:
            raise ValueError("k must be positive")

        # Retrieve from both systems (fetch more than k to ensure good fusion)
        fetch_k = min(k * 2, 100)
        bm25_results = self.bm25.retrieve(query, fetch_k)
        dense_results = self.dense.retrieve(query, fetch_k)

        # Compute RRF scores
        rrf_scores: dict[str, float] = defaultdict(float)
        rrf_k = self.config.rrf_k

        # BM25 contribution
        for rank, result in enumerate(bm25_results, start=1):
            rrf_scores[result.doc_id] += self.config.bm25_weight / (rrf_k + rank)

        # Dense contribution
        for rank, result in enumerate(dense_results, start=1):
            rrf_scores[result.doc_id] += self.config.dense_weight / (rrf_k + rank)

        # Sort by fused score (descending)
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Return top-k
        results: list[RetrievalResult] = []
        for doc_id, score in sorted_docs[:k]:
            results.append(RetrievalResult(doc_id=doc_id, score=score))

        return results
