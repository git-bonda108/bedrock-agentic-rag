import pytest

from agentic_rag.text import chunk_text, extract_text, title_from_text

DOC = """# Leave Policy

Intro paragraph.

## Paid time off

Employees receive 25 days.

## Sick leave

Ten sick days per year.
"""


def test_chunks_follow_sections_and_carry_heading_path():
    chunks = chunk_text(DOC, size=500, overlap=50)
    assert [c.section for c in chunks] == [
        "Leave Policy",
        "Leave Policy > Paid time off",
        "Leave Policy > Sick leave",
    ]
    assert chunks[1].text.startswith("Leave Policy > Paid time off")
    assert "25 days" in chunks[1].text
    assert [c.index for c in chunks] == [0, 1, 2]


def test_long_sections_are_split_within_size():
    body = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(20))
    chunks = chunk_text(f"# Big\n\n{body}", size=600, overlap=80)
    assert len(chunks) > 3
    assert all(len(c.text) <= 600 + len("Big\n\n") for c in chunks)


def test_single_huge_paragraph_is_hard_wrapped():
    chunks = chunk_text("x" * 5000, size=1000, overlap=100)
    assert len(chunks) >= 5
    assert all(len(c.text) <= 1000 for c in chunks)


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_text("text", size=100, overlap=100)


def test_extract_and_title():
    assert extract_text("a.md", DOC.encode()) == DOC
    assert title_from_text(DOC, "fallback") == "Leave Policy"
    assert title_from_text("no heading", "fallback") == "fallback"
    with pytest.raises(ValueError):
        extract_text("a.exe", b"")
