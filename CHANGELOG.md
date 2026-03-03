# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-03-02

### Added
- GitHub Actions CI pipeline (pytest on push/PR, Python 3.10-3.12)
- Expanded test suite: 27 tests covering metrics, qrels, hybrid retrieval, edge cases, and integration
- Architecture diagram and code structure walkthrough in README
- CHANGELOG.md

### Changed
- README rewritten as portfolio-grade documentation with architecture diagram, methods table, and project structure

## [0.2.0] - 2026-02-25

### Added
- **Hybrid Retrieval (RRF)**: Reciprocal Rank Fusion combining BM25 + Dense results
- **NDCG@k metric**: Normalized Discounted Cumulative Gain with binary relevance
- **Precision@k metric**: Fraction of retrieved documents that are relevant
- Visualization script (`visualize.py`) generating comparison bar charts
- Makefile and helper scripts (`setup.sh`, `run_all.sh`)

## [0.1.0] - 2026-02-24

### Added
- Initial search-retrieval evaluation pipeline
- BM25 retrieval via Elasticsearch with `multi_match` queries
- Dense retrieval via SentenceTransformers + FAISS `IndexFlatIP`
- BEIR dataset downloader with automatic caching
- Evaluation runner with Recall@k and MRR@k metrics
- Docker Compose for single-node Elasticsearch
- Report generation (CSV, JSON, Markdown)
- 9 unit tests for core metric functions
