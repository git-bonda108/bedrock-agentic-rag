"""Self-correcting RAG agent as a LangGraph StateGraph.

    route ── knowledge_base ─> retrieve ─> grade ─> generate ─> verify ─> finalize
      │                           ^           │        ^          │
      │                           └ rewrite <─┤        └ retry <──┤ unsupported claims
      │                     (no relevant docs)│                   │ (attempts exhausted)
      │                                       └> not_found        └> caveat ─> finalize
      ├─ conversational ─> converse
      └─ out_of_scope ───> decline

Conversation history is persisted per thread_id by the checkpointer (DynamoDB in AWS).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agentic_rag.agent import prompts
from agentic_rag.agent.llm import Generator, Judge
from agentic_rag.retrieval import HybridRetriever

_CITATION = re.compile(r"\[(\d+)\]")


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    route: str
    query: str
    rewrites: int
    documents: list[dict[str, Any]]
    answer: str
    attempts: int
    feedback: str | None
    grounded: bool
    citations: list[dict[str, Any]]
    trace: list[str]


@dataclass
class AgentDeps:
    judge: Judge
    generator: Generator
    retriever: HybridRetriever
    max_query_rewrites: int = 2
    max_generation_attempts: int = 2


def _history(state: AgentState) -> list[BaseMessage]:
    # The latest HumanMessage is the current question; history is everything before it.
    return list(state.get("messages", []))[:-1]


def build_graph(deps: AgentDeps, checkpointer: Any = None):
    def route(state: AgentState) -> dict[str, Any]:
        question = state["question"]
        decision = deps.judge.route(question, _history(state))
        return {
            "route": decision.route,
            "query": decision.standalone_query or question,
            "rewrites": 0,
            "attempts": 0,
            "feedback": None,
            "documents": [],
            "citations": [],
            "grounded": False,
            "trace": [f"route:{decision.route}"],
        }

    def retrieve(state: AgentState) -> dict[str, Any]:
        chunks = deps.retriever.retrieve(state["query"])
        return {
            "documents": [c.to_dict() for c in chunks],
            "trace": [*state["trace"], f"retrieve:{len(chunks)}"],
        }

    def grade(state: AgentState) -> dict[str, Any]:
        docs = state["documents"]
        if not docs:
            return {"trace": [*state["trace"], "grade:0"]}
        keep = deps.judge.grade(state["query"], [d["text"] for d in docs])
        relevant = [docs[i - 1] for i in keep]
        return {"documents": relevant, "trace": [*state["trace"], f"grade:{len(relevant)}"]}

    def after_grade(state: AgentState) -> Literal["generate", "rewrite", "not_found"]:
        if state["documents"]:
            return "generate"
        if state["rewrites"] < deps.max_query_rewrites:
            return "rewrite"
        return "not_found"

    def rewrite(state: AgentState) -> dict[str, Any]:
        query = deps.judge.rewrite(state["query"])
        return {
            "query": query,
            "rewrites": state["rewrites"] + 1,
            "trace": [*state["trace"], "rewrite"],
        }

    def generate(state: AgentState) -> dict[str, Any]:
        answer = deps.generator.answer(
            state["question"],
            [d["text"] for d in state["documents"]],
            _history(state),
            feedback=state.get("feedback"),
        )
        return {
            "answer": answer,
            "attempts": state["attempts"] + 1,
            "trace": [*state["trace"], "generate"],
        }

    def verify(state: AgentState) -> dict[str, Any]:
        verdict = deps.judge.verify(state["answer"], [d["text"] for d in state["documents"]])
        feedback = "\n".join(f"- {c}" for c in verdict.unsupported_claims) or None
        return {
            "grounded": verdict.grounded,
            "feedback": None if verdict.grounded else feedback,
            "trace": [*state["trace"], f"verify:{'ok' if verdict.grounded else 'fail'}"],
        }

    def after_verify(state: AgentState) -> Literal["finalize", "generate", "caveat"]:
        if state["grounded"]:
            return "finalize"
        if state["attempts"] < deps.max_generation_attempts:
            return "generate"
        return "caveat"

    def caveat(state: AgentState) -> dict[str, Any]:
        return {
            "answer": state["answer"] + prompts.UNVERIFIED_CAVEAT,
            "trace": [*state["trace"], "caveat"],
        }

    def finalize(state: AgentState) -> dict[str, Any]:
        docs = state["documents"]
        cited = sorted(
            {int(n) for n in _CITATION.findall(state["answer"]) if 0 < int(n) <= len(docs)}
        )
        citations = [
            {
                "n": n,
                "source": docs[n - 1]["source"],
                "title": docs[n - 1]["title"],
                "section": docs[n - 1]["section"],
                "key": docs[n - 1]["key"],
            }
            for n in cited
        ]
        return {
            "citations": citations,
            "messages": [AIMessage(state["answer"])],
            "trace": [*state["trace"], "finalize"],
        }

    def not_found(state: AgentState) -> dict[str, Any]:
        return {
            "answer": prompts.NOT_FOUND_REPLY,
            "grounded": True,
            "messages": [AIMessage(prompts.NOT_FOUND_REPLY)],
            "trace": [*state["trace"], "not_found"],
        }

    def converse(state: AgentState) -> dict[str, Any]:
        reply = deps.generator.converse(state["question"], _history(state))
        return {
            "answer": reply,
            "grounded": True,
            "messages": [AIMessage(reply)],
            "trace": [*state["trace"], "converse"],
        }

    def decline(state: AgentState) -> dict[str, Any]:
        return {
            "answer": prompts.OUT_OF_SCOPE_REPLY,
            "grounded": True,
            "messages": [AIMessage(prompts.OUT_OF_SCOPE_REPLY)],
            "trace": [*state["trace"], "decline"],
        }

    graph = StateGraph(AgentState)
    for name, fn in [
        ("route", route),
        ("retrieve", retrieve),
        ("grade", grade),
        ("rewrite", rewrite),
        ("generate", generate),
        ("verify", verify),
        ("caveat", caveat),
        ("finalize", finalize),
        ("not_found", not_found),
        ("converse", converse),
        ("decline", decline),
    ]:
        graph.add_node(name, fn)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        lambda s: s["route"],
        {"knowledge_base": "retrieve", "conversational": "converse", "out_of_scope": "decline"},
    )
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", after_grade)
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", after_verify)
    graph.add_edge("caveat", "finalize")
    for terminal in ("finalize", "not_found", "converse", "decline"):
        graph.add_edge(terminal, END)

    return graph.compile(checkpointer=checkpointer)


def run_turn(graph: Any, question: str, thread_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    result = graph.invoke(
        {"question": question, "messages": [HumanMessage(question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return {
        "answer": result["answer"],
        "citations": result.get("citations", []),
        "route": result["route"],
        "grounded": result.get("grounded", False),
        "trace": result.get("trace", []),
        "thread_id": thread_id,
        "latency_ms": round((time.perf_counter() - started) * 1000),
    }
