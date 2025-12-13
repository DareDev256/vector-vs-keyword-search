.PHONY: up down data index-bm25 index-dense eval test

DATASET ?= scifact
SPLIT ?= test
LIMIT_QUERIES ?= 200
ES_HOST ?= http://localhost:9200
ES_INDEX ?= beir_$(DATASET)

up:
	docker compose up -d

down:
	docker compose down

data:
	python -m src.data.download --dataset $(DATASET) --split $(SPLIT)

index-bm25:
	python -m src.retrieval.bm25_es --dataset $(DATASET) --split $(SPLIT) --es_host $(ES_HOST) --index_name $(ES_INDEX) --recreate

index-dense:
	python -m src.retrieval.dense_faiss --dataset $(DATASET) --split $(SPLIT)

eval:
	python -m src.eval.run_eval --dataset $(DATASET) --split $(SPLIT) --method all --k 10 --limit_queries $(LIMIT_QUERIES) --es_host $(ES_HOST) --es_index $(ES_INDEX)

test:
	pytest -q
