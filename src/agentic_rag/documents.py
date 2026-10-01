"""Document registry in DynamoDB: tracks what is indexed, so ingestion is idempotent
and deletes can remove exactly the vectors a document produced."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any


def doc_id_for(source_key: str) -> str:
    return hashlib.sha256(source_key.encode()).hexdigest()[:24]


def chunk_key(doc_id: str, index: int) -> str:
    return f"{doc_id}#{index:05d}"


@dataclass
class DocumentRecord:
    doc_id: str
    source_key: str
    title: str
    sha256: str
    status: str  # INDEXED | FAILED
    chunk_count: int
    indexed_at: str
    error: str | None = None


class DocumentRegistry:
    def __init__(self, table: Any) -> None:
        self._table = table

    @classmethod
    def from_name(cls, table_name: str, region: str) -> DocumentRegistry:
        import boto3

        return cls(boto3.resource("dynamodb", region_name=region).Table(table_name))

    def get(self, doc_id: str) -> DocumentRecord | None:
        item = self._table.get_item(Key={"doc_id": doc_id}).get("Item")
        if not item:
            return None
        item["chunk_count"] = int(item.get("chunk_count", 0))
        return DocumentRecord(**item)

    def put(self, record: DocumentRecord) -> None:
        item = {k: v for k, v in asdict(record).items() if v is not None}
        self._table.put_item(Item=item)

    def delete(self, doc_id: str) -> None:
        self._table.delete_item(Key={"doc_id": doc_id})

    def list(self, limit: int = 100) -> list[DocumentRecord]:
        items: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {"Limit": limit}
        while len(items) < limit:
            page = self._table.scan(**kwargs)
            items.extend(page.get("Items", []))
            if "LastEvaluatedKey" not in page:
                break
            kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]
        records = []
        for item in items[:limit]:
            item["chunk_count"] = int(item.get("chunk_count", 0))
            records.append(DocumentRecord(**item))
        return sorted(records, key=lambda r: r.indexed_at, reverse=True)


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
