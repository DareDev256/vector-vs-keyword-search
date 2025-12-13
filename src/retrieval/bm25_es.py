from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Iterable

from elasticsearch import Elasticsearch
from elasticsearch import helpers

from src.data.download import download_beir_dataset
from src.data.preprocess import Document, load_beir_corpus
from src.retrieval.common import RetrievalResult, Retriever
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ElasticsearchConfig:
    host: str = "http://localhost:9200"
    index_name: str = "beir_docs"
    request_timeout_s: int = 60


def create_client(config: ElasticsearchConfig) -> Elasticsearch:
    return Elasticsearch(
        hosts=[config.host],
        request_timeout=config.request_timeout_s,
    )


def wait_for_elasticsearch(client: Elasticsearch, *, timeout_s: int = 60) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if client.ping():
                return
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError("Elasticsearch did not become ready within timeout.")


def ensure_index(client: Elasticsearch, index_name: str) -> None:
    if client.indices.exists(index=index_name):
        return
    body = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "text": {"type": "text"},
            }
        },
    }
    client.indices.create(index=index_name, body=body)


def iter_bulk_actions(index_name: str, docs: Iterable[Document]) -> Iterable[dict]:
    for doc in docs:
        yield {
            "_op_type": "index",
            "_index": index_name,
            "_id": doc.doc_id,
            "_source": {"title": doc.title, "text": doc.text},
        }


def index_corpus(
    client: Elasticsearch,
    *,
    index_name: str,
    corpus: dict[str, Document],
    recreate: bool = False,
    chunk_size: int = 500,
) -> None:
    if recreate and client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
    ensure_index(client, index_name)

    actions = iter_bulk_actions(index_name, corpus.values())
    success, errors = helpers.bulk(client, actions, chunk_size=chunk_size, raise_on_error=False)
    if errors:
        raise RuntimeError(f"Bulk indexing had errors (success={success}): {errors[:3]}")
    client.indices.refresh(index=index_name)
    logger.info("Indexed %d documents into index=%s", len(corpus), index_name)


class ElasticsearchBM25Retriever(Retriever):
    def __init__(self, config: ElasticsearchConfig) -> None:
        self.config = config
        self.client = create_client(config)

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        if k <= 0:
            raise ValueError("k must be positive")
        resp = self.client.search(
            index=self.config.index_name,
            size=k,
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^2", "text"],
                    }
                }
            },
        )
        hits = resp.get("hits", {}).get("hits", [])
        results: list[RetrievalResult] = []
        for hit in hits:
            doc_id = str(hit.get("_id"))
            score = float(hit.get("_score", 0.0))
            results.append(RetrievalResult(doc_id=doc_id, score=score))
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Index and query BM25 via Elasticsearch.")
    parser.add_argument("--dataset", type=str, default="scifact")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--es_host", type=str, default="http://localhost:9200")
    parser.add_argument("--index_name", type=str, default="beir_docs")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=500)
    args = parser.parse_args()

    config = ElasticsearchConfig(host=args.es_host, index_name=args.index_name)
    client = create_client(config)
    wait_for_elasticsearch(client)

    paths = download_beir_dataset(dataset=args.dataset, split=args.split)
    corpus = load_beir_corpus(paths.corpus_path)

    index_corpus(
        client,
        index_name=config.index_name,
        corpus=corpus,
        recreate=args.recreate,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
