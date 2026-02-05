from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from src.retrieval.common import RetrievalResult, Retriever
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class HybridConfig:
    """Configuration for hybrid Reciprocal Rank Fusion (RRF) retrieval.

    Attributes:
        rrf_k: Smoothing constant for RRF scoring. Controls how much lower-ranked
            documents are penalized. Higher values flatten the rank curve, giving
            more weight to lower-ranked results. The standard default is 60.
            Formula per document: score += weight / (rrf_k + rank).
        bm25_weight: Multiplicative weight applied to BM25 retriever scores in
            the RRF fusion. Increase to favor keyword-based matches.
        dense_weight: Multiplicative weight applied to dense retriever scores in
            the RRF fusion. Increase to favor semantic/embedding-based matches.
    """
    rrf_k: int = 60
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
        logger.debug("Hybrid retrieve: query=%r, k=%d, fetch_k=%d", query, k, fetch_k)

        bm25_results = self.bm25.retrieve(query, fetch_k)
        logger.debug("BM25 returned %d results", len(bm25_results))

        dense_results = self.dense.retrieve(query, fetch_k)
        logger.debug("Dense returned %d results", len(dense_results))

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
        logger.debug(
            "RRF fusion produced %d unique docs, returning top %d",
            len(sorted_docs),
            k,
        )

        # Return top-k
        results: list[RetrievalResult] = []
        for doc_id, score in sorted_docs[:k]:
            results.append(RetrievalResult(doc_id=doc_id, score=score))

        return results
