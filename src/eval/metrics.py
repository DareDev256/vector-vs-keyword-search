from __future__ import annotations

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


def mrr_at_k(relevant: set[str], ranked_doc_ids: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / float(rank)
    return 0.0


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
        return {f"recall@{k}": 0.0 for k in ks} | {f"mrr@{k}": 0.0 for k in ks}

    sums: dict[str, float] = {f"recall@{k}": 0.0 for k in ks} | {f"mrr@{k}": 0.0 for k in ks}
    for qid in qids:
        relevant = qrels[qid]
        ranked = run[qid]
        for k in ks:
            sums[f"recall@{k}"] += recall_at_k(relevant, ranked, k)
            sums[f"mrr@{k}"] += mrr_at_k(relevant, ranked, k)

    denom = float(len(qids))
    return {name: value / denom for name, value in sums.items()}

