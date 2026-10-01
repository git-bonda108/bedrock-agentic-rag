"""Text embeddings: Amazon Titan Text Embeddings V2 on Bedrock, plus a local hashing
embedder used by tests and the offline eval so they run without AWS.

Uses boto3 directly (no LangChain) so the ingest Lambda can share it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dimensions: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class BedrockTitanEmbedder:
    """Titan V2 takes one input per InvokeModel call, so batches fan out on a small pool."""

    def __init__(
        self,
        client: Any,
        model_id: str = "amazon.titan-embed-text-v2:0",
        dimensions: int = 1024,
        max_workers: int = 8,
    ) -> None:
        self._client = client
        self._model_id = model_id
        self.dimensions = dimensions
        self._max_workers = max_workers

    def _embed_one(self, text: str) -> list[float]:
        body = json.dumps({"inputText": text, "dimensions": self.dimensions, "normalize": True})
        response = self._client.invoke_model(
            modelId=self._model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        return json.loads(response["body"].read())["embedding"]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if len(texts) <= 1:
            return [self._embed_one(t) for t in texts]
        with ThreadPoolExecutor(max_workers=min(self._max_workers, len(texts))) as pool:
            return list(pool.map(self._embed_one, texts))


class HashingEmbedder:
    """Deterministic bag-of-words feature hashing. No semantics, but stable and free."""

    def __init__(self, dimensions: int = 512) -> None:
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def build_embedder(settings: Any, bedrock_runtime: Any | None = None) -> Embedder:
    if settings.embeddings_provider == "hashing":
        return HashingEmbedder()
    if bedrock_runtime is None:
        import boto3

        bedrock_runtime = boto3.client("bedrock-runtime", region_name=settings.region)
    return BedrockTitanEmbedder(
        bedrock_runtime,
        model_id=settings.embedding_model_id,
        dimensions=settings.embedding_dimensions,
    )
