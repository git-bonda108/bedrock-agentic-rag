"""Ingestion pipeline: S3 object -> text -> chunks -> Titan embeddings -> S3 Vectors.

Idempotent by content hash, so SQS redeliveries and re-uploads of unchanged files are no-ops.
Framework-free so it can be unit tested without Lambda.
"""

from __future__ import annotations

import hashlib
import logging
import posixpath
from dataclasses import dataclass
from typing import Any

from agentic_rag.documents import (
    DocumentRecord,
    DocumentRegistry,
    chunk_key,
    doc_id_for,
    now_iso,
)
from agentic_rag.embeddings import Embedder
from agentic_rag.text import SUPPORTED_SUFFIXES, chunk_text, extract_text, title_from_text
from agentic_rag.vectorstore import VectorRecord, VectorStore

logger = logging.getLogger(__name__)

# S3 Vectors metadata is capped per vector; keep stored text comfortably below it.
_MAX_TEXT_METADATA = 6000


@dataclass
class IngestResult:
    doc_id: str
    action: str  # indexed | unchanged | deleted | skipped
    chunks: int = 0


class IngestionPipeline:
    def __init__(
        self,
        s3: Any,
        bucket: str,
        embedder: Embedder,
        store: VectorStore,
        registry: DocumentRegistry,
        chunk_size: int = 1800,
        chunk_overlap: int = 250,
    ) -> None:
        self._s3 = s3
        self._bucket = bucket
        self._embedder = embedder
        self._store = store
        self._registry = registry
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def index_object(self, key: str) -> IngestResult:
        doc_id = doc_id_for(key)
        if not key.lower().endswith(SUPPORTED_SUFFIXES):
            logger.info("skipping unsupported object %s", key)
            return IngestResult(doc_id, "skipped")

        data = self._s3.get_object(Bucket=self._bucket, Key=key)["Body"].read()
        digest = hashlib.sha256(data).hexdigest()
        existing = self._registry.get(doc_id)
        if existing and existing.sha256 == digest and existing.status == "INDEXED":
            return IngestResult(doc_id, "unchanged", existing.chunk_count)

        text = extract_text(key, data)
        title = title_from_text(text, posixpath.basename(key))
        chunks = chunk_text(text, self._chunk_size, self._chunk_overlap)
        vectors = self._embedder.embed([c.text for c in chunks]) if chunks else []

        records = [
            VectorRecord(
                key=chunk_key(doc_id, c.index),
                vector=vec,
                metadata={
                    "doc_id": doc_id,
                    "source": key,
                    "title": title,
                    "section": c.section,
                    "chunk_index": c.index,
                    "text": c.text[:_MAX_TEXT_METADATA],
                },
            )
            for c, vec in zip(chunks, vectors, strict=True)
        ]
        self._store.upsert(records)

        # A shorter new version leaves stale tail chunks behind; remove them.
        if existing and existing.chunk_count > len(chunks):
            self._store.delete(
                [chunk_key(doc_id, i) for i in range(len(chunks), existing.chunk_count)]
            )

        self._registry.put(
            DocumentRecord(
                doc_id=doc_id,
                source_key=key,
                title=title,
                sha256=digest,
                status="INDEXED",
                chunk_count=len(chunks),
                indexed_at=now_iso(),
            )
        )
        return IngestResult(doc_id, "indexed", len(chunks))

    def remove_object(self, key: str) -> IngestResult:
        doc_id = doc_id_for(key)
        existing = self._registry.get(doc_id)
        if existing:
            self._store.delete([chunk_key(doc_id, i) for i in range(existing.chunk_count)])
            self._registry.delete(doc_id)
        return IngestResult(doc_id, "deleted", existing.chunk_count if existing else 0)
