from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import DATA_DIR
from src.data.download import download_beir_dataset
from src.data.preprocess import Document, load_beir_corpus
from src.retrieval.common import RetrievalResult, Retriever
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


@dataclass(frozen=True)
class DenseConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    index_path: Path = Path("data") / "indexes" / "dense" / "faiss.index"
    mapping_path: Path = Path("data") / "indexes" / "dense" / "doc_ids.json"
    batch_size: int = 64
    normalize: bool = True  # cosine similarity via inner product


def _doc_to_text(doc: Document) -> str:
    if doc.title:
        return f"{doc.title}\n{doc.text}".strip()
    return doc.text


def build_faiss_index(
    *,
    model: SentenceTransformer,
    corpus: dict[str, Document],
    batch_size: int = 64,
    normalize: bool = True,
) -> tuple[faiss.Index, list[str]]:
    doc_ids = list(corpus.keys())
    texts = [_doc_to_text(corpus[doc_id]) for doc_id in doc_ids]

    embeddings: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding corpus"):
        batch = texts[start : start + batch_size]
        embs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embs = embs.astype(np.float32)
        if normalize:
            faiss.normalize_L2(embs)
        embeddings.append(embs)

    matrix = np.vstack(embeddings) if embeddings else np.zeros((0, model.get_sentence_embedding_dimension()), np.float32)
    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    return index, doc_ids


def save_index(index: faiss.Index, doc_ids: list[str], *, index_path: Path, mapping_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path.as_posix())
    mapping_path.write_text(json.dumps(doc_ids, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index(*, index_path: Path, mapping_path: Path) -> tuple[faiss.Index, list[str]]:
    index = faiss.read_index(index_path.as_posix())
    doc_ids = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(doc_ids, list):
        raise ValueError("Invalid doc_id mapping; expected a JSON list.")
    return index, [str(x) for x in doc_ids]


class DenseFaissRetriever(Retriever):
    def __init__(self, config: DenseConfig) -> None:
        self.config = config
        self.model = SentenceTransformer(config.model_name)
        if not config.index_path.exists() or not config.mapping_path.exists():
            raise FileNotFoundError(
                f"Dense index not found. Build it first: missing {config.index_path} or {config.mapping_path}"
            )
        self.index, self.doc_ids = load_index(index_path=config.index_path, mapping_path=config.mapping_path)

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        if k <= 0:
            raise ValueError("k must be positive")
        q = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        if self.config.normalize:
            faiss.normalize_L2(q)
        scores, indices = self.index.search(q, k)
        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0].tolist(), indices[0].tolist(), strict=True):
            if idx < 0:
                continue
            results.append(RetrievalResult(doc_id=self.doc_ids[idx], score=float(score)))
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a dense FAISS index from a BEIR dataset.")
    parser.add_argument("--dataset", type=str, default="scifact")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--index_path", type=Path, default=None)
    parser.add_argument("--mapping_path", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    paths = download_beir_dataset(dataset=args.dataset, split=args.split)
    corpus = load_beir_corpus(paths.corpus_path)

    model = SentenceTransformer(args.model_name)
    out_dir = DATA_DIR / "indexes" / "dense" / args.dataset
    index_path = args.index_path or (out_dir / "faiss.index")
    mapping_path = args.mapping_path or (out_dir / "doc_ids.json")
    index, doc_ids = build_faiss_index(
        model=model,
        corpus=corpus,
        batch_size=args.batch_size,
        normalize=True,
    )
    save_index(index, doc_ids, index_path=index_path, mapping_path=mapping_path)
    logger.info("Saved FAISS index to %s and mapping to %s", index_path, mapping_path)


if __name__ == "__main__":
    main()
