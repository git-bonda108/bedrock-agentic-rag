import boto3
from botocore.stub import ANY, Stubber

from agentic_rag.vectorstore import InMemoryVectorStore, S3VectorStore, VectorRecord


def _client():
    return boto3.client("s3vectors", region_name="us-west-2")


def test_s3_vectors_upsert_batches_at_500():
    client = _client()
    store = S3VectorStore(client, "vectors", "chunks")
    records = [VectorRecord(f"k{i}", [0.1, 0.2], {"text": "t"}) for i in range(501)]
    with Stubber(client) as stub:
        for _ in range(2):
            stub.add_response(
                "put_vectors",
                {},
                {"vectorBucketName": "vectors", "indexName": "chunks", "vectors": ANY},
            )
        store.upsert(records)
        stub.assert_no_pending_responses()


def test_s3_vectors_query_converts_distance_to_similarity():
    client = _client()
    store = S3VectorStore(client, "vectors", "chunks")
    with Stubber(client) as stub:
        stub.add_response(
            "query_vectors",
            {
                "vectors": [
                    {"key": "a", "distance": 0.1, "metadata": {"text": "alpha"}},
                    {"key": "b", "distance": 0.4, "metadata": {"text": "beta"}},
                ],
                "distanceMetric": "cosine",
            },
            {
                "vectorBucketName": "vectors",
                "indexName": "chunks",
                "topK": 2,
                "queryVector": {"float32": [1.0, 0.0]},
                "returnMetadata": True,
                "returnDistance": True,
            },
        )
        hits = store.query([1.0, 0.0], top_k=2)
    assert [h.key for h in hits] == ["a", "b"]
    assert round(hits[0].score, 2) == 0.9
    assert hits[0].metadata["text"] == "alpha"


def test_in_memory_store_ranks_and_filters():
    store = InMemoryVectorStore()
    store.upsert(
        [
            VectorRecord("a", [1.0, 0.0], {"doc_id": "x"}),
            VectorRecord("b", [0.0, 1.0], {"doc_id": "y"}),
        ]
    )
    assert [h.key for h in store.query([1.0, 0.1], 2)] == ["a", "b"]
    assert [h.key for h in store.query([1.0, 0.1], 2, {"doc_id": {"$eq": "y"}})] == ["b"]
    store.delete(["a"])
    assert len(store) == 1
