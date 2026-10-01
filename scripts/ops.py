"""Operational helpers for a deployed stack.

python scripts/ops.py outputs          # KEY=VALUE lines (append to $GITHUB_ENV in CI)
python scripts/ops.py seed             # upload data/corpus and wait until it is indexed
python scripts/ops.py ask "question"   # one chat turn against the deployed API
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import boto3
import httpx

STACK = os.getenv("STACK_NAME", "AgenticRagStack")
ROOT = Path(__file__).resolve().parent.parent
CORPUS_PREFIX = "uploads/corpus/"


def outputs() -> dict[str, str]:
    cfn = boto3.client("cloudformation")
    stack = cfn.describe_stacks(StackName=STACK)["Stacks"][0]
    return {o["OutputKey"]: o["OutputValue"] for o in stack.get("Outputs", [])}


def api_key(secret_arn: str) -> str:
    return boto3.client("secretsmanager").get_secret_value(SecretId=secret_arn)["SecretString"]


def _client(out: dict[str, str]) -> httpx.Client:
    return httpx.Client(
        base_url=out["ApiUrl"],
        headers={"x-api-key": api_key(out["ApiKeySecretArn"])},
        timeout=120,
    )


def seed(timeout_s: int = 600) -> None:
    out = outputs()
    s3 = boto3.client("s3")
    files = sorted((ROOT / "data" / "corpus").glob("*.md"))
    for path in files:
        s3.upload_file(
            str(path),
            out["DocumentsBucketName"],
            CORPUS_PREFIX + path.name,
            ExtraArgs={"ContentType": "text/markdown"},
        )
    print(f"uploaded {len(files)} documents; waiting for ingestion")

    expected = {CORPUS_PREFIX + p.name for p in files}
    indexed: set[str] = set()
    deadline = time.monotonic() + timeout_s
    with _client(out) as client:
        while time.monotonic() < deadline:
            docs = client.get("/v1/documents").raise_for_status().json()["documents"]
            indexed = {d["source_key"] for d in docs if d["status"] == "INDEXED"}
            if expected <= indexed:
                print(f"indexed {len(expected)} documents")
                return
            time.sleep(10)
    sys.exit(f"timed out; indexed so far: {sorted(indexed & expected)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("outputs")
    sub.add_parser("seed")
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--thread-id")
    args = parser.parse_args()

    if args.cmd == "outputs":
        out = outputs()
        print(f"API_URL={out['ApiUrl']}")
        print(f"API_KEY_SECRET_ARN={out['ApiKeySecretArn']}")
        print(f"DOCUMENTS_BUCKET={out['DocumentsBucketName']}")
    elif args.cmd == "seed":
        seed()
    else:
        out = outputs()
        with _client(out) as client:
            body = {"question": args.question, "thread_id": args.thread_id}
            response = client.post("/v1/chat", json=body).raise_for_status()
            print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
