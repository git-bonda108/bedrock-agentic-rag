from __future__ import annotations

import os
from collections.abc import Sequence

import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("POWERTOOLS_TRACE_DISABLED", "1")
os.environ.setdefault("POWERTOOLS_METRICS_NAMESPACE", "AgenticRagTest")

from agentic_rag.agent.llm import RouteDecision, VerifyDecision
from agentic_rag.embeddings import HashingEmbedder
from agentic_rag.retrieval import HybridRetriever
from agentic_rag.vectorstore import InMemoryVectorStore, VectorRecord

DOCS = {
    "corpus/leave-policy.md": "Paid time off\n\nEmployees receive 25 days of PTO per year.",
    "corpus/on-call.md": "Compensation\n\nThe primary on-call stipend is $400 per week.",
    "corpus/expenses.md": "Meals\n\nThe international daily meal limit is $100.",
}


@pytest.fixture
def retriever() -> HybridRetriever:
    embedder = HashingEmbedder()
    store = InMemoryVectorStore()
    records = []
    for i, (source, text) in enumerate(DOCS.items()):
        [vec] = embedder.embed([text])
        records.append(
            VectorRecord(
                key=f"doc{i}#00000",
                vector=vec,
                metadata={"source": source, "title": source, "section": "", "text": text},
            )
        )
    store.upsert(records)
    return HybridRetriever(embedder, store, candidate_k=10, final_k=2)


class FakeJudge:
    """Scriptable judge. `grades` and `verdicts` are consumed one per call."""

    def __init__(
        self,
        route: str = "knowledge_base",
        grades: Sequence[list[int]] | None = None,
        verdicts: Sequence[bool] | None = None,
    ) -> None:
        self._route = route
        self._grades = list(grades) if grades is not None else None
        self._verdicts = list(verdicts) if verdicts is not None else None
        self.route_histories: list[int] = []
        self.rewrites: list[str] = []

    def route(self, question, history):
        self.route_histories.append(len(history))
        query = question if self._route == "knowledge_base" else ""
        return RouteDecision(route=self._route, standalone_query=query)

    def grade(self, query, passages):
        if self._grades is None:
            return list(range(1, len(passages) + 1))
        return self._grades.pop(0) if self._grades else []

    def rewrite(self, query):
        self.rewrites.append(query)
        return f"{query} policy"

    def verify(self, answer, passages):
        grounded = self._verdicts.pop(0) if self._verdicts else True
        return VerifyDecision(
            grounded=grounded, unsupported_claims=[] if grounded else ["made-up claim"]
        )


class FakeGenerator:
    def __init__(self) -> None:
        self.feedback: list[str | None] = []

    def answer(self, question, passages, history, feedback=None):
        self.feedback.append(feedback)
        return f"Per the handbook, {passages[0][:40]} [1]."

    def converse(self, question, history):
        return "Happy to help. What would you like to know?"
