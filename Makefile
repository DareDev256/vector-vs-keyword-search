.PHONY: up down data index-bm25 index-dense eval test help clean

DATASET ?= scifact
SPLIT ?= test
LIMIT_QUERIES ?= 200
ES_HOST ?= http://localhost:9200
ES_INDEX ?= beir_$(DATASET)

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  help          Show this help message"
	@echo "  up            Start Elasticsearch via Docker Compose"
	@echo "  down          Stop Docker Compose services"
	@echo "  data          Download BEIR dataset (DATASET=$(DATASET), SPLIT=$(SPLIT))"
	@echo "  index-bm25    Build BM25 index in Elasticsearch"
	@echo "  index-dense   Build dense FAISS index"
	@echo "  eval          Run evaluation across retrieval methods"
	@echo "  test          Run pytest test suite"
	@echo "  clean         Remove generated data, indexes, and reports"

up: ## Start Elasticsearch via Docker Compose
	docker compose up -d

down: ## Stop Docker Compose services
	docker compose down

data: ## Download BEIR dataset
	python -m src.data.download --dataset $(DATASET) --split $(SPLIT)

index-bm25: ## Build BM25 index in Elasticsearch
	python -m src.retrieval.bm25_es --dataset $(DATASET) --split $(SPLIT) --es_host $(ES_HOST) --index_name $(ES_INDEX) --recreate

index-dense: ## Build dense FAISS index
	python -m src.retrieval.dense_faiss --dataset $(DATASET) --split $(SPLIT)

eval: ## Run evaluation across retrieval methods
	python -m src.eval.run_eval --dataset $(DATASET) --split $(SPLIT) --method all --k 10 --limit_queries $(LIMIT_QUERIES) --es_host $(ES_HOST) --es_index $(ES_INDEX)

test: ## Run pytest test suite
	pytest -q

clean: ## Remove generated data, indexes, and reports
	rm -rf data/indexes data/datasets reports/*.json reports/*.png
