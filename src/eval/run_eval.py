from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, REPORTS_DIR, DEFAULT_SEED
from src.data.download import download_beir_dataset
from src.data.preprocess import load_beir_queries
from src.data.qrels import load_beir_qrels_tsv
from src.eval.metrics import compute_metrics_for_run
from src.retrieval.bm25_es import ElasticsearchBM25Retriever, ElasticsearchConfig, wait_for_elasticsearch, create_client
from src.retrieval.dense_faiss import DenseFaissRetriever, DenseConfig
from src.utils.io import write_json
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


@dataclass(frozen=True)
class EvalConfig:
    dataset: str = "scifact"
    split: str = "test"
    limit_queries: int = 200
    seed: int = DEFAULT_SEED
    ks: tuple[int, ...] = (1, 3, 5, 10)


def _sample_qids(qids: list[str], limit: int, seed: int) -> list[str]:
    if limit <= 0 or limit >= len(qids):
        return qids
    import random

    rng = random.Random(seed)
    return rng.sample(qids, k=limit)


def _avg_ms(times_s: list[float]) -> float:
    if not times_s:
        return 0.0
    return 1000.0 * (sum(times_s) / float(len(times_s)))


def _render_summary(method_rows: list[dict]) -> str:
    df = pd.DataFrame(method_rows).set_index("method")
    metric_cols = [c for c in df.columns if c.startswith("recall@") or c.startswith("mrr@")]
    latency_col = "avg_latency_ms"

    lines: list[str] = []
    lines.append("# Retrieval Evaluation Summary")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append(df[metric_cols + [latency_col]].to_markdown(floatfmt=".4f"))
    lines.append("")
    lines.append("## Analysis")
    lines.append("")
    if len(df.index) >= 2:
        best = {}
        for col in metric_cols:
            best_method = df[col].astype(float).idxmax()
            best[col] = best_method
        bm25 = "bm25_es" if "bm25_es" in df.index else df.index[0]
        dense = "dense_faiss" if "dense_faiss" in df.index else df.index[-1]
        wins_bm25 = sum(1 for m in best.values() if m == bm25)
        wins_dense = sum(1 for m in best.values() if m == dense)
        faster = df[latency_col].astype(float).idxmin()

        lines.append(f"- Metric wins: `{dense}`={wins_dense}, `{bm25}`={wins_bm25} (higher is better).")
        lines.append(f"- Latency: `{faster}` is faster on average for retrieval (lower is better).")
        lines.append("- Dense retrieval often improves semantic matching; BM25 can be strong on exact/lexical matches.")
        lines.append("- This evaluation is retrieval-only (no reranking); results will vary by dataset and index settings.")
    else:
        lines.append("- Run with `--method all` to compare BM25 vs Dense side-by-side.")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BM25 (ES) vs Dense (FAISS) retrieval on BEIR.")
    parser.add_argument("--dataset", type=str, default="scifact")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--method", type=str, default="all", choices=["bm25", "dense", "all"])
    parser.add_argument("--k", type=int, default=10, help="Max retrieval depth (must be >= 10).")
    parser.add_argument("--limit_queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--es_host", type=str, default="http://localhost:9200")
    parser.add_argument("--es_index", type=str, default="beir_docs")

    parser.add_argument("--dense_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--dense_index_path", type=Path, default=None)
    parser.add_argument("--dense_mapping_path", type=Path, default=None)
    args = parser.parse_args()

    if args.k < 10:
        raise SystemExit("--k must be >= 10 (metrics are computed at k in {1,3,5,10}).")

    set_seed(args.seed)
    cfg = EvalConfig(dataset=args.dataset, split=args.split, limit_queries=args.limit_queries, seed=args.seed)
    max_k = max(cfg.ks)

    paths = download_beir_dataset(dataset=cfg.dataset, split=cfg.split)
    queries_all = load_beir_queries(paths.queries_path)
    qrels = load_beir_qrels_tsv(paths.qrels_path, min_relevance=1).qrels

    eligible_qids = [qid for qid in queries_all.keys() if qid in qrels and qrels[qid]]
    selected_qids = _sample_qids(eligible_qids, cfg.limit_queries, cfg.seed)
    queries = {qid: queries_all[qid] for qid in selected_qids}
    qrels_selected = {qid: qrels[qid] for qid in selected_qids}
    logger.info("Evaluating %d queries (dataset=%s split=%s)", len(selected_qids), cfg.dataset, cfg.split)

    method_rows: list[dict] = []

    if args.method in ("bm25", "all"):
        es_config = ElasticsearchConfig(host=args.es_host, index_name=args.es_index)
        es_client = create_client(es_config)
        wait_for_elasticsearch(es_client, timeout_s=60)
        bm25 = ElasticsearchBM25Retriever(es_config)

        run: dict[str, list[str]] = {}
        times: list[float] = []
        for qid, text in queries.items():
            start = time.perf_counter()
            results = bm25.retrieve(text, k=max_k)
            times.append(time.perf_counter() - start)
            run[qid] = [r.doc_id for r in results]

        metrics = compute_metrics_for_run(qrels_selected, run, ks=cfg.ks)
        method_rows.append({"method": "bm25_es", **metrics, "avg_latency_ms": _avg_ms(times)})

    if args.method in ("dense", "all"):
        default_dense_dir = DATA_DIR / "indexes" / "dense" / cfg.dataset
        dense_index_path = args.dense_index_path or (default_dense_dir / "faiss.index")
        dense_mapping_path = args.dense_mapping_path or (default_dense_dir / "doc_ids.json")
        dense_config = DenseConfig(
            model_name=args.dense_model,
            index_path=dense_index_path,
            mapping_path=dense_mapping_path,
        )
        dense = DenseFaissRetriever(dense_config)

        run = {}
        times = []
        for qid, text in queries.items():
            start = time.perf_counter()
            results = dense.retrieve(text, k=max_k)
            times.append(time.perf_counter() - start)
            run[qid] = [r.doc_id for r in results]

        metrics = compute_metrics_for_run(qrels_selected, run, ks=cfg.ks)
        method_rows.append({"method": "dense_faiss", **metrics, "avg_latency_ms": _avg_ms(times)})

    if not method_rows:
        raise SystemExit("No methods selected.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(method_rows).sort_values("method")
    csv_path = REPORTS_DIR / "metrics.csv"
    json_path = REPORTS_DIR / "metrics.json"
    md_path = REPORTS_DIR / "summary.md"

    df.to_csv(csv_path, index=False)
    write_json(json_path, {"config": cfg.__dict__, "rows": method_rows})
    summary = _render_summary(method_rows)
    md_path.write_text(summary, encoding="utf-8")

    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n" + summary)


if __name__ == "__main__":
    main()
