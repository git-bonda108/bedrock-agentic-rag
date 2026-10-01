from agentic_rag.embeddings import HashingEmbedder
from agentic_rag.retrieval import HybridRetriever, reciprocal_rank_fusion
from agentic_rag.vectorstore import InMemoryVectorStore


def test_rrf_rewards_agreement_between_rankings():
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "c"]])
    assert scores["a"] == scores["b"] > scores["c"]


def test_hybrid_retriever_finds_exact_term(retriever):
    results = retriever.retrieve("on-call stipend per week")
    assert results[0].source == "corpus/on-call.md"
    assert len(results) == 2


def test_reranker_order_is_final(retriever):
    class ReverseReranker:
        def rerank(self, query, texts, top_n):
            return [(i, 1.0 - i * 0.1) for i in reversed(range(len(texts)))][:top_n]

    retriever._reranker = ReverseReranker()
    results = retriever.retrieve("meal limit", k=2)
    assert len(results) == 2
    assert all(r.rerank_score is not None for r in results)
    assert results[0].score == results[0].rerank_score


def test_empty_store_returns_nothing():
    assert HybridRetriever(HashingEmbedder(), InMemoryVectorStore()).retrieve("anything") == []
