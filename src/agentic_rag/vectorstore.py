"""Vector storage on Amazon S3 Vectors, with an in-memory twin for tests and local runs."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

# S3 Vectors limits: 500 vectors per PutVectors and per DeleteVectors call.
_PUT_BATCH = 500
_DELETE_BATCH = 500


@dataclass(frozen=True)
class VectorRecord:
    key: str
    vector: list[float]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SearchHit:
    key: str
    score: float  # cosine similarity, higher is better
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(Protocol):
    def upsert(self, records: Sequence[VectorRecord]) -> None: ...

    def delete(self, keys: Sequence[str]) -> None: ...

    def query(
        self, vector: Sequence[float], top_k: int, filter: dict[str, Any] | None = None
    ) -> list[SearchHit]: ...


def _batched(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class S3VectorStore:
    def __init__(self, client: Any, vector_bucket: str, index_name: str) -> None:
        self._client = client
        self._bucket = vector_bucket
        self._index = index_name

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        for batch in _batched(records, _PUT_BATCH):
            self._client.put_vectors(
                vectorBucketName=self._bucket,
                indexName=self._index,
                vectors=[
                    {"key": r.key, "data": {"float32": r.vector}, "metadata": r.metadata}
                    for r in batch
                ],
            )

    def delete(self, keys: Sequence[str]) -> None:
        for batch in _batched(list(keys), _DELETE_BATCH):
            self._client.delete_vectors(
                vectorBucketName=self._bucket, indexName=self._index, keys=list(batch)
            )

    def query(
        self, vector: Sequence[float], top_k: int, filter: dict[str, Any] | None = None
    ) -> list[SearchHit]:
        params: dict[str, Any] = {
            "vectorBucketName": self._bucket,
            "indexName": self._index,
            "topK": top_k,
            "queryVector": {"float32": list(vector)},
            "returnMetadata": True,
            "returnDistance": True,
        }
        if filter:
            params["filter"] = filter
        response = self._client.query_vectors(**params)
        # The index uses cosine distance; convert to similarity so callers can sort descending.
        return [
            SearchHit(
                key=v["key"],
                score=1.0 - float(v.get("distance", 1.0)),
                metadata=v.get("metadata", {}),
            )
            for v in response.get("vectors", [])
        ]


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        for record in records:
            self._records[record.key] = record

    def delete(self, keys: Sequence[str]) -> None:
        for key in keys:
            self._records.pop(key, None)

    def query(
        self, vector: Sequence[float], top_k: int, filter: dict[str, Any] | None = None
    ) -> list[SearchHit]:
        hits = [
            SearchHit(key=r.key, score=_cosine(vector, r.vector), metadata=dict(r.metadata))
            for r in self._records.values()
            if _matches(r.metadata, filter)
        ]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def __len__(self) -> int:
        return len(self._records)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _matches(metadata: dict[str, Any], filter: dict[str, Any] | None) -> bool:
    """Equality subset of the S3 Vectors filter language: {"k": v} or {"k": {"$eq": v}}."""
    if not filter:
        return True
    for key, expected in filter.items():
        value = expected.get("$eq") if isinstance(expected, dict) else expected
        if metadata.get(key) != value:
            return False
    return True


def build_vector_store(settings: Any, client: Any | None = None) -> VectorStore:
    if settings.vector_store == "memory":
        return InMemoryVectorStore()
    if client is None:
        import boto3

        client = boto3.client("s3vectors", region_name=settings.region)
    return S3VectorStore(client, settings.vector_bucket, settings.vector_index)
