"""Tests for document ingestion, loading, and chunking."""

import json
from pathlib import Path

from app.ingestion.chunker import Chunk, chunk_text
from app.ingestion.loaders import load_document
from app.ingestion.pipeline import ingest_directory


def test_chunk_text_respects_chunk_size_and_overlap_boundaries() -> None:
    """Assert chunking keeps chunks within size and overlaps adjacent chunks."""
    text = " ".join(
        [
            "apple",
            "banana",
            "carrot",
            "delta",
            "ember",
            "forest",
            "garden",
            "harbor",
            "island",
            "jungle",
            "kitten",
            "lemon",
            "magnet",
            "nectar",
            "orange",
            "planet",
            "quartz",
            "river",
            "silver",
            "timber",
            "umbra",
            "velvet",
            "window",
            "yellow",
            "zephyr",
        ]
    )

    chunks = chunk_text(
        text=text,
        doc_id="doc",
        source_path="memory.md",
        chunk_size=10,
        chunk_overlap=2,
    )

    assert len(chunks) >= 3
    assert all(chunk.token_count <= 10 for chunk in chunks)
    for previous, current in zip(chunks, chunks[1:]):
        assert previous.text.split()[-2:] == current.text.split()[:2]
    assert chunks[0].chunk_id == "doc_0"
    assert chunks[1].chunk_id == "doc_1"


def test_load_document_dispatches_by_extension(tmp_path: Path) -> None:
    """Assert supported file extensions dispatch to the appropriate loader."""
    markdown_path = tmp_path / "guide.md"
    text_path = tmp_path / "note.txt"
    html_path = tmp_path / "page.html"

    markdown_path.write_text("# Reset\n\nUse the reset link.", encoding="utf-8")
    text_path.write_text("Plain support note.", encoding="utf-8")
    html_path.write_text("<h2>Billing</h2><p>Update your card.</p>", encoding="utf-8")

    assert "# Reset" in load_document(markdown_path)
    assert load_document(text_path) == "Plain support note."
    assert "## Billing" in load_document(html_path)


def test_ingest_directory_produces_valid_jsonl(tmp_path: Path, monkeypatch) -> None:
    """Assert directory ingestion writes valid Chunk records as JSONL."""
    raw_dir = tmp_path / "raw"
    output_path = tmp_path / "processed" / "chunks.jsonl"
    raw_dir.mkdir()
    (raw_dir / "support.md").write_text(
        "# Support Guide\n\n"
        "## Access\n\n"
        "Customers can reset access from the account page. "
        "Administrators can review audit events and active sessions.",
        encoding="utf-8",
    )

    monkeypatch.setenv("CHUNK_SIZE", "12")
    monkeypatch.setenv("CHUNK_OVERLAP", "3")

    from app.config import get_settings

    get_settings.cache_clear()
    chunk_count = ingest_directory(raw_dir=raw_dir, processed_path=output_path)

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines]

    assert chunk_count == len(records)
    assert chunk_count > 0

    required_fields = set(Chunk.model_fields)
    assert all(required_fields.issubset(record) for record in records)
    assert all(record["source_path"].endswith("support.md") for record in records)
    assert records[0]["section"] == "Support Guide"

    get_settings.cache_clear()
