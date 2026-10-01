import json
from types import SimpleNamespace

import boto3
import pytest
from moto import mock_aws

from agentic_rag.documents import DocumentRegistry, chunk_key, doc_id_for
from agentic_rag.embeddings import HashingEmbedder
from agentic_rag.ingest.pipeline import IngestionPipeline
from agentic_rag.vectorstore import InMemoryVectorStore

BUCKET = "docs"
TABLE = "documents"
LONG_DOC = "# Policy\n\n" + "\n\n".join(f"Rule {i}: " + "detail " * 40 for i in range(12))


@pytest.fixture
def aws():
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-west-2")
        s3.create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "us-west-2"}
        )
        ddb = boto3.resource("dynamodb", region_name="us-west-2")
        table = ddb.create_table(
            TableName=TABLE,
            KeySchema=[{"AttributeName": "doc_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "doc_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        yield s3, table


@pytest.fixture
def pipeline(aws):
    s3, table = aws
    store = InMemoryVectorStore()
    registry = DocumentRegistry(table)
    p = IngestionPipeline(
        s3, BUCKET, HashingEmbedder(), store, registry, chunk_size=400, chunk_overlap=50
    )
    return p, s3, store, registry


def test_index_is_idempotent_and_shrinks_cleanly(pipeline):
    p, s3, store, registry = pipeline
    key = "uploads/policy.md"
    s3.put_object(Bucket=BUCKET, Key=key, Body=LONG_DOC.encode())

    first = p.index_object(key)
    assert first.action == "indexed" and first.chunks > 3
    assert len(store) == first.chunks
    record = registry.get(doc_id_for(key))
    assert record.status == "INDEXED" and record.title == "Policy"

    assert p.index_object(key).action == "unchanged"

    s3.put_object(Bucket=BUCKET, Key=key, Body=b"# Policy\n\nOnly one rule now.")
    second = p.index_object(key)
    assert second.chunks == 1
    assert len(store) == 1
    [hit] = store.query(HashingEmbedder().embed(["rule"])[0], 5)
    assert hit.key == chunk_key(doc_id_for(key), 0)


def test_remove_deletes_vectors_and_registry(pipeline):
    p, s3, store, registry = pipeline
    key = "uploads/policy.md"
    s3.put_object(Bucket=BUCKET, Key=key, Body=LONG_DOC.encode())
    p.index_object(key)
    result = p.remove_object(key)
    assert result.action == "deleted"
    assert len(store) == 0
    assert registry.get(doc_id_for(key)) is None


def test_unsupported_files_are_skipped(pipeline):
    p, *_ = pipeline
    assert p.index_object("uploads/image.png").action == "skipped"


def _sqs_event(*bodies):
    return {
        "Records": [
            {
                "messageId": f"m{i}",
                "receiptHandle": "r",
                "body": json.dumps(body),
                "attributes": {},
                "messageAttributes": {},
                "md5OfBody": "",
                "eventSource": "aws:sqs",
                "eventSourceARN": "arn:aws:sqs:us-west-2:123456789012:q",
                "awsRegion": "us-west-2",
            }
            for i, body in enumerate(bodies)
        ]
    }


def _s3_body(event, key):
    return {"Records": [{"eventName": event, "s3": {"object": {"key": key}}}]}


def test_lambda_handler_reports_partial_batch_failures(pipeline, monkeypatch):
    from agentic_rag.ingest import handler

    p, s3, store, _ = pipeline
    s3.put_object(Bucket=BUCKET, Key="uploads/my policy.md", Body=LONG_DOC.encode())
    monkeypatch.setattr(handler, "_pipeline", p)

    event = _sqs_event(
        _s3_body("ObjectCreated:Put", "uploads/my+policy.md"),  # URL-encoded key
        _s3_body("ObjectCreated:Put", "uploads/missing.md"),  # NoSuchKey -> fails
        {"Event": "s3:TestEvent"},
    )
    context = SimpleNamespace(
        function_name="ingest",
        memory_limit_in_mb=1024,
        invoked_function_arn="arn:aws:lambda:us-west-2:123456789012:function:ingest",
        aws_request_id="req-1",
    )
    response = handler.handler(event, context)
    assert response == {"batchItemFailures": [{"itemIdentifier": "m1"}]}
    assert len(store) > 0
