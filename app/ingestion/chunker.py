"""Chunking utilities for support-knowledge documents."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A chunk of document text with retrieval-oriented metadata."""

    chunk_id: str
    doc_id: str
    source_path: str
    section: str | None
    text: str
    token_count: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)


class _TextUnit(BaseModel):
    """A word-like text unit with source character offsets and section context."""

    text: str
    start_char: int
    end_char: int
    section: str | None


_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_WORD_PATTERN = re.compile(r"\S+")


def make_doc_id(path: Path) -> str:
    """Return a stable document identifier derived from the file name.

    Args:
        path: Source document path.

    Returns:
        A short SHA-1 hash of the lower-cased file name.
    """
    return hashlib.sha1(path.name.lower().encode("utf-8")).hexdigest()[:12]


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken when available, otherwise approximate by words.

    Args:
        text: Text to count.

    Returns:
        Token count or a deterministic word-count approximation.
    """
    try:
        import tiktoken
    except ImportError:
        return len(_WORD_PATTERN.findall(text))

    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def _extract_units_with_sections(text: str) -> list[_TextUnit]:
    """Split text into word-like units while tracking the most recent heading.

    Args:
        text: Source document text.

    Returns:
        Ordered text units with section metadata and character offsets.
    """
    units: list[_TextUnit] = []
    current_section: str | None = None
    cursor = 0

    for line in text.splitlines(keepends=True):
        line_start = cursor
        cursor += len(line)
        stripped_line = line.strip()
        heading_match = _HEADING_PATTERN.match(stripped_line)

        if heading_match:
            current_section = heading_match.group(2).strip()

        for match in _WORD_PATTERN.finditer(line):
            units.append(
                _TextUnit(
                    text=match.group(0),
                    start_char=line_start + match.start(),
                    end_char=line_start + match.end(),
                    section=current_section,
                )
            )

    return units


def chunk_text(
    text: str,
    doc_id: str,
    source_path: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Split document text into overlapping chunks.

    The chunk boundaries use word-like units so character offsets remain stable.
    ``token_count`` uses ``tiktoken`` when installed and a word-count approximation
    otherwise.

    Args:
        text: Source document text.
        doc_id: Stable document identifier.
        source_path: Original source path.
        chunk_size: Maximum number of word-like units per chunk.
        chunk_overlap: Number of units to overlap between adjacent chunks.

    Raises:
        ValueError: If chunk sizing inputs are invalid.

    Returns:
        Ordered document chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    units = _extract_units_with_sections(text)
    if not units:
        return []

    chunks: list[Chunk] = []
    start_unit = 0
    index = 0

    while start_unit < len(units):
        end_unit = start_unit + 1
        best_end_unit = end_unit

        while end_unit <= len(units):
            candidate_units = units[start_unit:end_unit]
            candidate_text = text[
                candidate_units[0].start_char : candidate_units[-1].end_char
            ].strip()
            if _count_tokens(candidate_text) > chunk_size:
                break
            best_end_unit = end_unit
            end_unit += 1

        end_unit = best_end_unit
        selected_units = units[start_unit:end_unit]
        start_char = selected_units[0].start_char
        end_char = selected_units[-1].end_char
        chunk_body = text[start_char:end_char].strip()
        section = selected_units[0].section

        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}_{index}",
                doc_id=doc_id,
                source_path=source_path,
                section=section,
                text=chunk_body,
                token_count=_count_tokens(chunk_body),
                start_char=start_char,
                end_char=end_char,
            )
        )

        if end_unit == len(units):
            break

        start_unit = max(end_unit - chunk_overlap, start_unit + 1)
        index += 1

    return chunks
