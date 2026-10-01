import dataclasses

import boto3
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from moto import mock_aws

from agentic_rag.agent import AgentDeps, build_graph
from agentic_rag.api.auth import ApiKeyVerifier
from agentic_rag.api.main import create_app
from agentic_rag.config import Settings
from agentic_rag.deps import Container
from tests.conftest import FakeGenerator, FakeJudge

KEY = {"x-api-key": "secret"}


@pytest.fixture
def client(retriever):
    with mock_aws():
        settings = dataclasses.replace(
            Settings.from_env(), api_key="secret", documents_bucket="docs", region="us-west-2"
        )
        s3 = boto3.client("s3", region_name="us-west-2")

        def factory(_settings):
            graph = build_graph(
                AgentDeps(FakeJudge(), FakeGenerator(), retriever), checkpointer=InMemorySaver()
            )
            return Container(settings, retriever, graph, None, s3)

        with TestClient(create_app(settings, factory)) as test_client:
            yield test_client


def test_healthz_is_public(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_requires_api_key(client):
    assert client.post("/v1/chat", json={"question": "hi"}).status_code == 401
    bad = client.post("/v1/chat", json={"question": "hi"}, headers={"x-api-key": "nope"})
    assert bad.status_code == 401


def test_chat_returns_answer_and_thread(client):
    response = client.post("/v1/chat", json={"question": "on-call stipend?"}, headers=KEY)
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert body["citations"][0]["n"] == 1
    assert len(body["thread_id"]) == 32
    follow_up = client.post(
        "/v1/chat",
        json={"question": "and secondary?", "thread_id": body["thread_id"]},
        headers=KEY,
    )
    assert follow_up.json()["thread_id"] == body["thread_id"]


def test_chat_validates_input(client):
    assert client.post("/v1/chat", json={"question": ""}, headers=KEY).status_code == 422
    bad_thread = {"question": "x", "thread_id": "../etc"}
    assert client.post("/v1/chat", json=bad_thread, headers=KEY).status_code == 422


def test_search(client):
    response = client.post("/v1/search", json={"query": "meal limit", "k": 1}, headers=KEY)
    assert response.status_code == 200
    assert len(response.json()["results"]) == 1


def test_upload_url_sanitises_and_validates(client):
    ok = client.post(
        "/v1/documents/upload-url", json={"filename": "../My Policy.md"}, headers=KEY
    ).json()
    assert ok["key"].startswith("uploads/") and ok["key"].endswith("-My-Policy.md")
    assert ".." not in ok["key"]
    assert "Signature=" in ok["upload_url"]
    bad = client.post("/v1/documents/upload-url", json={"filename": "x.exe"}, headers=KEY)
    assert bad.status_code == 400


def test_verifier_uses_secrets_manager_and_caches():
    class Secrets:
        calls = 0

        def get_secret_value(self, SecretId):
            Secrets.calls += 1
            return {"SecretString": "from-secret"}

    verifier = ApiKeyVerifier(None, "arn:secret", Secrets())
    assert verifier.verify("from-secret")
    assert not verifier.verify("wrong")
    assert not verifier.verify(None)
    assert Secrets.calls == 1
