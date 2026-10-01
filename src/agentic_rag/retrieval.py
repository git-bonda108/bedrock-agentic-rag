"""Two-stage retrieval: dense recall from S3 Vectors, lexical BM25 re-scoring fused with
Reciprocal Rank Fusion, then an optional Amazon Bedrock Rerank cross-encoder pass."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from rank_bm25 import BM25Okapi

from agentic_rag.embeddings import Embedder
from agentic_rag.vectorstore import SearchHit, VectorStore

_TOKEN = re.compile(r"[a-z0-9]+")
_RRF_K = 60


@dataclass
class RetrievedChunk:
    key: str
    text: str
    source: str
    title: str
    section: str
    dense_score: float
    fused_score: float = 0.0
    rerank_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.fused_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "text": self.text,
            "source": self.source,
            "title": self.title,
            "section": self.section,
            "score": round(self.score, 4),
        }


class Reranker(Protocol):
    def rerank(self, query: str, texts: list[str], top_n: int) -> list[tuple[int, float]]: ...


class BedrockReranker:
    """Wraps the bedrock-agent-runtime Rerank API (Amazon Rerank / Cohere Rerank models)."""

    def __init__(self, client: Any, model_id: str, region: str) -> None:
        self._client = client
        self._model_arn = f"arn:aws:bedrock:{region}::foundation-model/{model_id}"

    def rerank(self, query: str, texts: list[str], top_n: int) -> list[tuple[int, float]]:
        if not texts:
            return []
        response = self._client.rerank(
            queries=[{"type": "TEXT", "textQuery": {"text": query}}],
            sources=[
                {
                    "type": "INLINE",
                    "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": t}},
                }
                for t in texts
            ],
            rerankingConfiguration={
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "modelConfiguration": {"modelArn": self._model_arn},
                    "numberOfResults": min(top_n, len(texts)),
                },
            },
        )
        return [(r["index"], float(r["relevanceScore"])) for r in response["results"]]


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _to_chunk(hit: SearchHit) -> RetrievedChunk:
    meta = hit.metadata
    return RetrievedChunk(
        key=hit.key,
        text=str(meta.get("text", "")),
        source=str(meta.get("source", "")),
        title=str(meta.get("title", "")),
        section=str(meta.get("section", "")),
        dense_score=hit.score,
        metadata=meta,
    )


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = _RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    return scores


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        reranker: Reranker | None = None,
        candidate_k: int = 24,
        final_k: int = 5,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._reranker = reranker
        self.candidate_k = candidate_k
        self.final_k = final_k

    def retrieve(
        self, query: str, k: int | None = None, filter: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        k = k or self.final_k
        [vector] = self._embedder.embed([query])
        candidates = [_to_chunk(h) for h in self._store.query(vector, self.candidate_k, filter)]
        candidates = [c for c in candidates if c.text]
        if not candidates:
            return []

        dense_rank = [c.key for c in candidates]
        bm25 = BM25Okapi([_tokens(c.text) for c in candidates])
        lexical = bm25.get_scores(_tokens(query))
        lexical_rank = [
            candidates[i].key
            for i in sorted(range(len(candidates)), key=lambda i: lexical[i], reverse=True)
        ]
        fused = reciprocal_rank_fusion([dense_rank, lexical_rank])
        for c in candidates:
            c.fused_score = fused[c.key]
        candidates.sort(key=lambda c: c.fused_score, reverse=True)

        if self._reranker is None:
            return candidates[:k]

        pool = candidates[: max(k * 3, 10)]
        ranked = self._reranker.rerank(query, [c.text for c in pool], top_n=k)
        result = []
        for index, relevance in ranked:
            chunk = pool[index]
            chunk.rerank_score = relevance
            result.append(chunk)
        return result
