"""Lambda entry point: SQS batch of S3 event notifications -> IngestionPipeline.

Uses Powertools batch processing so one bad document fails only its own message (which
SQS retries and eventually moves to the DLQ) instead of the whole batch.
"""

from __future__ import annotations

import json
import urllib.parse
from typing import Any

import boto3
from aws_lambda_powertools import Logger, Metrics, Tracer
from aws_lambda_powertools.metrics import MetricUnit
from aws_lambda_powertools.utilities.batch import (
    BatchProcessor,
    EventType,
    process_partial_response,
)
from aws_lambda_powertools.utilities.data_classes.sqs_event import SQSRecord

from agentic_rag.config import get_settings
from agentic_rag.documents import DocumentRegistry
from agentic_rag.embeddings import build_embedder
from agentic_rag.ingest.pipeline import IngestionPipeline
from agentic_rag.vectorstore import build_vector_store

logger = Logger(service="ingest")
tracer = Tracer(service="ingest")
metrics = Metrics(namespace="AgenticRag", service="ingest")
processor = BatchProcessor(event_type=EventType.SQS)

_pipeline: IngestionPipeline | None = None


def _get_pipeline() -> IngestionPipeline:
    global _pipeline
    if _pipeline is None:
        settings = get_settings()
        _pipeline = IngestionPipeline(
            s3=boto3.client("s3", region_name=settings.region),
            bucket=settings.documents_bucket,
            embedder=build_embedder(settings),
            store=build_vector_store(settings),
            registry=DocumentRegistry.from_name(settings.documents_table, settings.region),
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    return _pipeline


def s3_events(body: dict[str, Any]) -> list[tuple[str, str]]:
    """Returns (event_name, object_key) pairs; s3:TestEvent messages have no Records."""
    events = []
    for record in body.get("Records", []):
        name = record.get("eventName", "")
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        events.append((name, key))
    return events


@tracer.capture_method
def record_handler(record: SQSRecord) -> None:
    pipeline = _get_pipeline()
    for event_name, key in s3_events(json.loads(record.body)):
        if event_name.startswith("ObjectRemoved"):
            result = pipeline.remove_object(key)
        else:
            result = pipeline.index_object(key)
        logger.info(
            "processed", extra={"key": key, "action": result.action, "chunks": result.chunks}
        )
        if result.action == "indexed":
            metrics.add_metric(name="DocumentsIndexed", unit=MetricUnit.Count, value=1)
            metrics.add_metric(name="ChunksIndexed", unit=MetricUnit.Count, value=result.chunks)


@logger.inject_lambda_context
@tracer.capture_lambda_handler
@metrics.log_metrics
def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    return process_partial_response(
        event=event, record_handler=record_handler, processor=processor, context=context
    )
