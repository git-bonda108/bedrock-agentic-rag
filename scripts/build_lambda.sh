#!/usr/bin/env bash
# Builds the ingest Lambda package (arm64, Python 3.12) into build/ingest.
# boto3 is bundled because the Lambda runtime's copy may predate the S3 Vectors API.
set -euo pipefail

cd "$(dirname "$0")/.."
OUT=build/ingest

rm -rf "$OUT"
mkdir -p "$OUT"
uv pip install --quiet --target "$OUT" \
  --python-platform aarch64-manylinux_2_28 --python-version 3.12 \
  "boto3>=1.43" "pypdf>=5.0" "aws-lambda-powertools[tracer]>=3.35"

# Only the modules the ingest path imports (no LangChain / FastAPI).
mkdir -p "$OUT/agentic_rag/ingest"
cp src/agentic_rag/__init__.py src/agentic_rag/config.py src/agentic_rag/documents.py \
   src/agentic_rag/embeddings.py src/agentic_rag/text.py src/agentic_rag/vectorstore.py \
   "$OUT/agentic_rag/"
cp src/agentic_rag/ingest/__init__.py src/agentic_rag/ingest/pipeline.py \
   src/agentic_rag/ingest/handler.py "$OUT/agentic_rag/ingest/"
find "$OUT" -name "__pycache__" -type d -prune -exec rm -rf {} +

echo "Lambda package: $(du -sh "$OUT" | cut -f1) in $OUT"
