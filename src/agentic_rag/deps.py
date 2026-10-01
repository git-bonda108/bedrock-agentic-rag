"""Wires settings into concrete AWS-backed components for the API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import boto3
from botocore.config import Config

from agentic_rag.agent import AgentDeps, build_graph
from agentic_rag.agent.llm import build_bedrock_models
from agentic_rag.config import Settings
from agentic_rag.documents import DocumentRegistry
from agentic_rag.embeddings import build_embedder
from agentic_rag.retrieval import BedrockReranker, HybridRetriever
from agentic_rag.vectorstore import build_vector_store

_BOTO = Config(retries={"mode": "adaptive", "max_attempts": 6}, connect_timeout=5, read_timeout=60)


@dataclass
class Container:
    settings: Settings
    retriever: HybridRetriever
    graph: Any
    registry: DocumentRegistry | None
    s3: Any


def build_checkpointer(settings: Settings) -> Any:
    if settings.checkpoints_table:
        from langgraph_checkpoint_aws import DynamoDBSaver

        return DynamoDBSaver(
            table_name=settings.checkpoints_table,
            region_name=settings.region,
            ttl_seconds=settings.checkpoint_ttl_seconds,
            enable_checkpoint_compression=True,
        )
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


def build_container(settings: Settings) -> Container:
    reranker = None
    if settings.rerank_enabled:
        reranker = BedrockReranker(
            boto3.client("bedrock-agent-runtime", region_name=settings.region, config=_BOTO),
            settings.rerank_model_id,
            settings.region,
        )
    retriever = HybridRetriever(
        embedder=build_embedder(
            settings, boto3.client("bedrock-runtime", region_name=settings.region, config=_BOTO)
        ),
        store=build_vector_store(settings),
        reranker=reranker,
        candidate_k=settings.candidate_k,
        final_k=settings.final_k,
    )
    judge, generator = build_bedrock_models(settings)
    graph = build_graph(
        AgentDeps(
            judge=judge,
            generator=generator,
            retriever=retriever,
            max_query_rewrites=settings.max_query_rewrites,
            max_generation_attempts=settings.max_generation_attempts,
        ),
        checkpointer=build_checkpointer(settings),
    )
    registry = (
        DocumentRegistry.from_name(settings.documents_table, settings.region)
        if settings.documents_table
        else None
    )
    s3 = boto3.client("s3", region_name=settings.region, config=Config(signature_version="s3v4"))
    return Container(settings, retriever, graph, registry, s3)
