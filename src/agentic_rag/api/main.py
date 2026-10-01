"""FastAPI service for the agentic RAG assistant (runs on ECS Fargate behind ALB + CloudFront).

No `from __future__ import annotations` here: FastAPI must evaluate the `Depends(...)`
annotations, which reference functions local to `create_app`.
"""

import logging
import re
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from agentic_rag.agent import run_turn
from agentic_rag.api.auth import ApiKeyVerifier
from agentic_rag.config import Settings, get_settings
from agentic_rag.deps import Container, build_container
from agentic_rag.telemetry import configure_logging, emit_metrics
from agentic_rag.text import SUPPORTED_SUFFIXES

logger = logging.getLogger("agentic_rag.api")

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
UPLOAD_PREFIX = "uploads/"


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(default=None, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")


class Citation(BaseModel):
    n: int
    source: str
    title: str
    section: str
    key: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    route: str
    grounded: bool
    thread_id: str
    latency_ms: int
    trace: list[str]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)


class UploadRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=200)
    content_type: str = Field(default="text/markdown", max_length=100)


def create_app(
    settings: Settings | None = None,
    container_factory: Callable[[Settings], Container] = build_container,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging()
        app.state.container = container_factory(settings)
        app.state.verifier = ApiKeyVerifier(settings.api_key, settings.api_key_secret_arn)
        if not app.state.verifier.enabled:
            logger.warning("API key auth disabled: set API_KEY or API_KEY_SECRET_ARN")
        yield

    app = FastAPI(title="Bedrock Agentic RAG", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        if request.url.path != "/healthz":
            logger.info(
                "request",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000),
                },
            )
        return response

    def container(request: Request) -> Container:
        return request.app.state.container

    def authorize(request: Request, x_api_key: Annotated[str | None, Header()] = None) -> None:
        if not request.app.state.verifier.verify(x_api_key):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing x-api-key")

    auth = [Depends(authorize)]

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    # Sync handlers: FastAPI runs them in its threadpool, so blocking boto3/LangGraph
    # calls don't stall the event loop.
    @app.post("/v1/chat", response_model=ChatResponse, dependencies=auth)
    def chat(body: ChatRequest, c: Annotated[Container, Depends(container)]) -> dict[str, Any]:
        thread_id = body.thread_id or uuid.uuid4().hex
        result = run_turn(c.graph, body.question, thread_id)
        emit_metrics(
            {
                "ChatLatencyMs": result["latency_ms"],
                "Grounded": 1.0 if result["grounded"] else 0.0,
                "Citations": len(result["citations"]),
            },
            units={"ChatLatencyMs": "Milliseconds", "Citations": "Count"},
            properties={"route": result["route"], "thread_id": thread_id},
        )
        return result

    @app.post("/v1/search", dependencies=auth)
    def search(body: SearchRequest, c: Annotated[Container, Depends(container)]) -> dict[str, Any]:
        chunks = c.retriever.retrieve(body.query, k=body.k)
        return {"results": [ch.to_dict() for ch in chunks]}

    @app.post("/v1/documents/upload-url", dependencies=auth)
    def upload_url(
        body: UploadRequest, c: Annotated[Container, Depends(container)]
    ) -> dict[str, Any]:
        name = _SAFE_NAME.sub("-", body.filename).strip("-.")
        if not name.lower().endswith(SUPPORTED_SUFFIXES):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, f"supported types: {', '.join(SUPPORTED_SUFFIXES)}"
            )
        key = f"{UPLOAD_PREFIX}{uuid.uuid4().hex[:8]}-{name}"
        url = c.s3.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": c.settings.documents_bucket,
                "Key": key,
                "ContentType": body.content_type,
            },
            ExpiresIn=900,
        )
        return {"key": key, "upload_url": url, "method": "PUT", "expires_in": 900}

    @app.get("/v1/documents", dependencies=auth)
    def list_documents(c: Annotated[Container, Depends(container)]) -> dict[str, Any]:
        if c.registry is None:
            return {"documents": []}
        return {"documents": [vars(r) for r in c.registry.list()]}

    @app.delete("/v1/documents/{doc_id}", status_code=status.HTTP_202_ACCEPTED, dependencies=auth)
    def delete_document(doc_id: str, c: Annotated[Container, Depends(container)]) -> dict[str, str]:
        record = c.registry.get(doc_id) if c.registry else None
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
        # Deleting the object emits s3:ObjectRemoved, and the ingest Lambda removes its vectors.
        c.s3.delete_object(Bucket=c.settings.documents_bucket, Key=record.source_key)
        return {"doc_id": doc_id, "status": "deleting"}

    return app


app = create_app()
