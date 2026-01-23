from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Metrics:
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr_at_1: float
    mrr_at_3: float
    mrr_at_5: float
    mrr_at_10: float


def recall_at_k(relevant: set[str], ranked_doc_ids: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    hits = 0
    for doc_id in ranked_doc_ids[:k]:
        if doc_id in relevant:
            hits += 1
    return hits / float(len(relevant))


def precision_at_k(relevant: set[str], ranked_doc_ids: list[str], k: int) -> float:
    """Precision@k: fraction of retrieved documents that are relevant."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not ranked_doc_ids:
        return 0.0
    hits = sum(1 for doc_id in ranked_doc_ids[:k] if doc_id in relevant)
    return hits / float(min(k, len(ranked_doc_ids)))


def mrr_at_k(relevant: set[str], ranked_doc_ids: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / float(rank)
    return 0.0


def dcg_at_k(relevant: set[str], ranked_doc_ids: list[str], k: int) -> float:
    """Discounted Cumulative Gain at k (binary relevance)."""
    if k <= 0:
        raise ValueError("k must be positive")
    dcg = 0.0
    for i, doc_id in enumerate(ranked_doc_ids[:k]):
        if doc_id in relevant:
            # Binary relevance: rel=1 if relevant, else 0
            # DCG formula: rel / log2(rank + 1), where rank is 1-indexed
            dcg += 1.0 / math.log2(i + 2)  # i+2 because i is 0-indexed
    return dcg


def ndcg_at_k(relevant: set[str], ranked_doc_ids: list[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at k.
    NDCG = DCG / IDCG, where IDCG is the ideal DCG (all relevant docs ranked first).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0

    dcg = dcg_at_k(relevant, ranked_doc_ids, k)

    # Ideal DCG: all relevant documents ranked first
    num_relevant = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(num_relevant))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def compute_metrics_for_run(
    qrels: dict[str, set[str]],
    run: dict[str, list[str]],
    *,
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, float]:
    """
    Compute mean metrics over queries.
    `run` maps qid -> ranked doc_ids (best-first).
    """
    for k in ks:
        if k <= 0:
            raise ValueError("ks must be positive")

    qids = [qid for qid in run.keys() if qid in qrels]
    if not qids:
        empty: dict[str, float] = {}
        for k in ks:
            empty[f"recall@{k}"] = 0.0
            empty[f"precision@{k}"] = 0.0
            empty[f"mrr@{k}"] = 0.0
            empty[f"ndcg@{k}"] = 0.0
        return empty

    sums: dict[str, float] = {}
    for k in ks:
        sums[f"recall@{k}"] = 0.0
        sums[f"precision@{k}"] = 0.0
        sums[f"mrr@{k}"] = 0.0
        sums[f"ndcg@{k}"] = 0.0

    for qid in qids:
        relevant = qrels[qid]
        ranked = run[qid]
        for k in ks:
            sums[f"recall@{k}"] += recall_at_k(relevant, ranked, k)
            sums[f"precision@{k}"] += precision_at_k(relevant, ranked, k)
            sums[f"mrr@{k}"] += mrr_at_k(relevant, ranked, k)
            sums[f"ndcg@{k}"] += ndcg_at_k(relevant, ranked, k)

    denom = float(len(qids))
    return {name: value / denom for name, value in sums.items()}

