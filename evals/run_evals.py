"""Evaluation gate for the agentic RAG system.

Two modes, both exit non-zero when any metric misses its threshold in evals/thresholds.json:

  offline  Retrieval regression test with no AWS calls: indexes data/corpus with the local
           hashing embedder and in-memory store, then measures hit@k and MRR of the
           dense + BM25 fusion pipeline. Runs on every pull request.

  live     End-to-end test of a deployed stack over HTTPS: source hit rate, fact recall,
           abstention accuracy (not-in-handbook and out-of-scope questions), grounded
           rate and p95 latency. Runs after every deploy.

Usage:
  python -m evals.run_evals offline
  python -m evals.run_evals live --url https://dxxxx.cloudfront.net --api-key $KEY
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "evals" / "golden.jsonl"
THRESHOLDS = json.loads((ROOT / "evals" / "thresholds.json").read_text())
CORPUS = ROOT / "data" / "corpus"
REPORTS = ROOT / "evals" / "reports"

ABSTAIN_MARKERS = ("couldn't find", "could not find", "not in the handbook", "can only help")


def load_golden() -> list[dict[str, Any]]:
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]


def _source_name(source: str) -> str:
    return source.rsplit("/", 1)[-1]


# ---------------------------------------------------------------- offline


def build_offline_retriever():
    from agentic_rag.documents import chunk_key, doc_id_for
    from agentic_rag.embeddings import HashingEmbedder
    from agentic_rag.retrieval import HybridRetriever
    from agentic_rag.text import chunk_text, title_from_text
    from agentic_rag.vectorstore import InMemoryVectorStore, VectorRecord

    embedder = HashingEmbedder()
    store = InMemoryVectorStore()
    for path in sorted(CORPUS.glob("*.md")):
        text = path.read_text()
        key = f"corpus/{path.name}"
        doc_id = doc_id_for(key)
        chunks = chunk_text(text, size=1800, overlap=250)
        vectors = embedder.embed([c.text for c in chunks])
        store.upsert(
            [
                VectorRecord(
                    key=chunk_key(doc_id, c.index),
                    vector=v,
                    metadata={
                        "source": key,
                        "title": title_from_text(text, path.name),
                        "section": c.section,
                        "text": c.text,
                    },
                )
                for c, v in zip(chunks, vectors, strict=True)
            ]
        )
    return HybridRetriever(embedder, store, reranker=None, candidate_k=24, final_k=5)


def run_offline(k: int = 5) -> dict[str, Any]:
    retriever = build_offline_retriever()
    cases = [c for c in load_golden() if c["expected_sources"]]
    hits, reciprocal_ranks, rows = 0, [], []
    for case in cases:
        results = retriever.retrieve(case["question"], k=k)
        names = [_source_name(r.source) for r in results]
        rank = next((i + 1 for i, n in enumerate(names) if n in case["expected_sources"]), None)
        hits += rank is not None
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        rows.append({"id": case["id"], "rank": rank, "retrieved": names})
    return {
        "mode": "offline",
        "metrics": {
            "hit_at_k": round(hits / len(cases), 3),
            "mrr": round(statistics.mean(reciprocal_ranks), 3),
        },
        "cases": rows,
    }


# ---------------------------------------------------------------- live


def _ask(client: Any, case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    response = client.post("/v1/chat", json={"question": case["question"]})
    elapsed = round((time.perf_counter() - started) * 1000)
    response.raise_for_status()
    body = response.json()
    answer = body["answer"]
    sources = {_source_name(c["source"]) for c in body["citations"]}
    row: dict[str, Any] = {
        "id": case["id"],
        "category": case["category"],
        "route": body["route"],
        "grounded": body["grounded"],
        "latency_ms": elapsed,
        "cited_sources": sorted(sources),
        "answer": answer,
    }
    if case["category"] in {"factual", "multi_hop"}:
        expected = set(case["expected_sources"])
        row["source_hit"] = bool(expected & sources)
        found = [p for p in case["must_include"] if p.lower() in answer.lower()]
        row["fact_recall"] = len(found) / len(case["must_include"]) if case["must_include"] else 1.0
    else:
        abstained = any(m in answer.lower() for m in ABSTAIN_MARKERS)
        if case["category"] == "out_of_scope":
            row["abstained"] = body["route"] == "out_of_scope" or abstained
        else:
            row["abstained"] = abstained
    return row


def run_live(url: str, api_key: str, concurrency: int = 4) -> dict[str, Any]:
    import httpx

    cases = load_golden()
    with (
        httpx.Client(
            base_url=url.rstrip("/"), headers={"x-api-key": api_key}, timeout=180
        ) as client,
        ThreadPoolExecutor(max_workers=concurrency) as pool,
    ):
        rows = list(pool.map(lambda c: _ask(client, c), cases))

    answerable = [r for r in rows if "source_hit" in r]
    abstain = [r for r in rows if "abstained" in r]
    latencies = sorted(r["latency_ms"] for r in rows)
    p95 = latencies[min(len(latencies) - 1, round(0.95 * (len(latencies) - 1)))]
    return {
        "mode": "live",
        "metrics": {
            "source_hit_rate": round(sum(r["source_hit"] for r in answerable) / len(answerable), 3),
            "fact_recall": round(statistics.mean(r["fact_recall"] for r in answerable), 3),
            "abstain_accuracy": round(sum(r["abstained"] for r in abstain) / len(abstain), 3),
            "grounded_rate": round(sum(r["grounded"] for r in rows) / len(rows), 3),
            "p50_latency_ms": latencies[len(latencies) // 2],
            "p95_latency_ms": p95,
        },
        "cases": rows,
    }


# ---------------------------------------------------------------- gate


def check(report: dict[str, Any]) -> list[str]:
    failures = []
    for name, threshold in THRESHOLDS[report["mode"]].items():
        value = report["metrics"][name]
        ok = value <= threshold if name.endswith("latency_ms") else value >= threshold
        if not ok:
            failures.append(f"{name}={value} (threshold {threshold})")
    return failures


def write_summary(report: dict[str, Any], failures: list[str]) -> None:
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"{report['mode']}-{time.strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2))
    lines = [f"## Eval gate ({report['mode']}): {'FAILED' if failures else 'passed'}", ""]
    lines += ["| metric | value | threshold |", "|---|---|---|"]
    for name, value in report["metrics"].items():
        lines.append(f"| {name} | {value} | {THRESHOLDS[report['mode']].get(name, '')} |")
    summary = "\n".join(lines)
    print(summary)
    print(f"\nFull report: {out.relative_to(ROOT)}")
    if path := os.getenv("GITHUB_STEP_SUMMARY"):
        with open(path, "a") as fh:
            fh.write(summary + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("offline")
    live = sub.add_parser("live")
    live.add_argument("--url", default=os.getenv("API_URL"))
    live.add_argument("--api-key", default=os.getenv("API_KEY"))
    args = parser.parse_args()

    if args.mode == "offline":
        report = run_offline()
    else:
        if not args.url or not args.api_key:
            parser.error("live mode needs --url and --api-key (or API_URL / API_KEY)")
        report = run_live(args.url, args.api_key)

    failures = check(report)
    write_summary(report, failures)
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
