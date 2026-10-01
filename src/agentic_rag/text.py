"""Document text extraction and structure-aware chunking."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")

SUPPORTED_SUFFIXES = (".md", ".markdown", ".txt", ".pdf")


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    section: str


def extract_text(key: str, data: bytes) -> str:
    lowered = key.lower()
    if lowered.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if lowered.endswith((".md", ".markdown", ".txt")):
        return data.decode("utf-8", errors="replace")
    raise ValueError(f"Unsupported document type: {key}")


def title_from_text(text: str, fallback: str) -> str:
    for line in text.splitlines():
        match = _HEADING.match(line.strip())
        if match:
            return match.group(2).strip()
    return fallback


def _sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading path, body) pairs so chunks never straddle sections."""
    sections: list[tuple[str, str]] = []
    path: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(path), body))
        buf.clear()

    for line in text.splitlines():
        match = _HEADING.match(line.strip())
        if match:
            flush()
            level = len(match.group(1))
            path[:] = [*path[: level - 1], match.group(2).strip()]
            continue
        buf.append(line)
    flush()
    return sections


def _split_long(body: str, size: int, overlap: int) -> list[str]:
    """Pack paragraphs up to `size` chars; hard-wrap any single paragraph that is too long."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    pieces: list[str] = []
    for para in paragraphs:
        if len(para) <= size:
            pieces.append(para)
            continue
        step = max(1, size - overlap)
        pieces.extend(para[i : i + size] for i in range(0, len(para), step))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{piece}" if tail and len(tail) + len(piece) <= size else piece
        else:
            current = piece
    if current:
        chunks.append(current)
    return chunks


def chunk_text(text: str, size: int = 1800, overlap: int = 250) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks: list[Chunk] = []
    for section, body in _sections(text):
        for piece in _split_long(body, size, overlap):
            # The heading path is prepended so each chunk is self-describing for retrieval.
            content = f"{section}\n\n{piece}" if section else piece
            chunks.append(Chunk(index=len(chunks), text=content, section=section))
    return chunks
