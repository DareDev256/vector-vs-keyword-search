#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-scifact}"
SPLIT="${SPLIT:-test}"
LIMIT_QUERIES="${LIMIT_QUERIES:-200}"

echo "Starting Elasticsearch..."
docker compose up -d

echo "Waiting for Elasticsearch to be ready..."
for i in {1..60}; do
  if curl -fsS http://localhost:9200 >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "Downloading dataset..."
make data DATASET="$DATASET" SPLIT="$SPLIT"

echo "Building BM25 (Elasticsearch) index..."
make index-bm25 DATASET="$DATASET" SPLIT="$SPLIT"

echo "Building dense (FAISS) index..."
make index-dense DATASET="$DATASET" SPLIT="$SPLIT"

echo "Running evaluation..."
make eval DATASET="$DATASET" SPLIT="$SPLIT" LIMIT_QUERIES="$LIMIT_QUERIES"

echo "Done. Reports written under ./reports/"
