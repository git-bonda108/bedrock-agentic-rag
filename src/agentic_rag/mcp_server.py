"""MCP server exposing the deployed assistant to MCP clients (Claude Desktop, Claude Code, IDEs).

It is a thin client of the REST API, so it needs only the API URL and key:

    AGENTIC_RAG_URL=https://dxxxx.cloudfront.net AGENTIC_RAG_API_KEY=... \\
        python -m agentic_rag.mcp_server
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from mcp.server.mcpserver import MCPServer

server = MCPServer(
    name="northwind-handbook",
    instructions=(
        "Answers questions about the Northwind Cloud employee handbook with citations. "
        "Use `ask` for questions and `search_handbook` to inspect raw passages."
    ),
)


def _client() -> httpx.Client:
    return httpx.Client(
        base_url=os.environ["AGENTIC_RAG_URL"].rstrip("/"),
        headers={"x-api-key": os.environ["AGENTIC_RAG_API_KEY"]},
        timeout=120,
    )


@server.tool()
def ask(question: str, thread_id: str | None = None) -> dict[str, Any]:
    """Ask the handbook assistant a question. Pass the returned thread_id to follow up."""
    with _client() as client:
        response = client.post("/v1/chat", json={"question": question, "thread_id": thread_id})
        response.raise_for_status()
        return response.json()


@server.tool()
def search_handbook(query: str, k: int = 5) -> dict[str, Any]:
    """Return the top-k handbook passages for a query, after hybrid ranking and reranking."""
    with _client() as client:
        response = client.post("/v1/search", json={"query": query, "k": k})
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    server.run("stdio")
