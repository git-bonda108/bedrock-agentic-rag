"""Runtime configuration, read once from environment variables.

Kept dependency-free (no pydantic) so the ingest Lambda package stays small.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    region: str
    # Storage
    documents_bucket: str
    documents_table: str
    checkpoints_table: str
    vector_bucket: str
    vector_index: str
    # Backends: "s3vectors" | "memory" and "bedrock" | "hashing"
    vector_store: str
    embeddings_provider: str
    # Models
    answer_model_id: str
    judge_model_id: str
    embedding_model_id: str
    embedding_dimensions: int
    rerank_model_id: str
    rerank_enabled: bool
    # Retrieval
    candidate_k: int
    final_k: int
    # Agent loop limits
    max_query_rewrites: int
    max_generation_attempts: int
    # Ingestion
    chunk_size: int
    chunk_overlap: int
    # API
    api_key: str | None
    api_key_secret_arn: str | None
    checkpoint_ttl_seconds: int

    @classmethod
    def from_env(cls) -> Settings:
        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-west-2"
        return cls(
            region=region,
            documents_bucket=os.getenv("DOCUMENTS_BUCKET", ""),
            documents_table=os.getenv("DOCUMENTS_TABLE", ""),
            checkpoints_table=os.getenv("CHECKPOINTS_TABLE", ""),
            vector_bucket=os.getenv("VECTOR_BUCKET", ""),
            vector_index=os.getenv("VECTOR_INDEX", "chunks"),
            vector_store=os.getenv("VECTOR_STORE", "s3vectors"),
            embeddings_provider=os.getenv("EMBEDDINGS_PROVIDER", "bedrock"),
            answer_model_id=os.getenv("ANSWER_MODEL_ID", "anthropic.claude-opus-5-5"),
            judge_model_id=os.getenv("JUDGE_MODEL_ID", "anthropic.claude-haiku-4-5"),
            embedding_model_id=os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"),
            embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1024")),
            rerank_model_id=os.getenv("RERANK_MODEL_ID", "amazon.rerank-v1:0"),
            rerank_enabled=_bool("RERANK_ENABLED", True),
            candidate_k=int(os.getenv("CANDIDATE_K", "24")),
            final_k=int(os.getenv("FINAL_K", "5")),
            max_query_rewrites=int(os.getenv("MAX_QUERY_REWRITES", "2")),
            max_generation_attempts=int(os.getenv("MAX_GENERATION_ATTEMPTS", "2")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1800")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "250")),
            api_key=os.getenv("API_KEY") or None,
            api_key_secret_arn=os.getenv("API_KEY_SECRET_ARN") or None,
            checkpoint_ttl_seconds=int(os.getenv("CHECKPOINT_TTL_SECONDS", str(7 * 86400))),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
