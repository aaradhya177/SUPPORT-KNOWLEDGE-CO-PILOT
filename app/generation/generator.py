"""Grounded answer generation with inline citation extraction."""

from __future__ import annotations

import re

from pydantic import BaseModel

from app.generation.prompts import ANSWER_GENERATION_SYSTEM_PROMPT
from app.llm.client import LLMClient
from app.retrieval.base import RetrievedChunk
from app.utils.logger import get_logger

logger = get_logger(__name__)

CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_.:-]+(?:\s*,\s*[A-Za-z0-9_.:-]+)*)\]")


class Citation(BaseModel):
    """Citation metadata for a generated answer."""

    chunk_id: str
    doc_id: str
    source_path: str
    quoted_text: str | None


class GeneratedAnswer(BaseModel):
    """Generated answer and parsed citation metadata."""

    answer_text: str
    citations: list[Citation]
    raw_llm_output: str


class AnswerGenerator:
    """Generate grounded support answers from retrieved chunks."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize the answer generator.

        Args:
            llm_client: LLM client dependency.
        """
        self.llm_client = llm_client

    def generate(self, query: str, retrieved_chunks: list[RetrievedChunk]) -> GeneratedAnswer:
        """Generate a cited answer from retrieved chunks.

        Args:
            query: User support question.
            retrieved_chunks: Retrieved context chunks.

        Returns:
            Generated answer with validated citations.
        """
        context_block = _format_context(retrieved_chunks)
        user_prompt = (
            "Question:\n"
            f"{query.strip()}\n\n"
            "Context chunks:\n"
            f"{context_block}\n\n"
            "Answer with inline citations:"
        )

        raw_output = self.llm_client.complete(
            system_prompt=ANSWER_GENERATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=1024,
        )
        cited_chunk_ids = parse_citation_markers(raw_output)
        citations = _build_citations(cited_chunk_ids, retrieved_chunks)

        return GeneratedAnswer(
            answer_text=raw_output.strip(),
            citations=citations,
            raw_llm_output=raw_output,
        )


def parse_citation_markers(text: str) -> list[str]:
    """Parse citation chunk IDs from answer text.

    Args:
        text: Raw LLM output.

    Returns:
        Unique chunk IDs in first-seen order.
    """
    chunk_ids: list[str] = []
    seen: set[str] = set()
    for match in CITATION_PATTERN.finditer(text):
        for chunk_id in _split_citation_group(match.group(1)):
            if chunk_id not in seen:
                chunk_ids.append(chunk_id)
                seen.add(chunk_id)
    return chunk_ids


def _split_citation_group(citation_group: str) -> list[str]:
    """Split a citation marker group into individual chunk IDs."""
    return [chunk_id.strip() for chunk_id in citation_group.split(",") if chunk_id.strip()]


def _format_context(retrieved_chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a prompt context block.

    Args:
        retrieved_chunks: Retrieved context chunks.

    Returns:
        Prompt-ready context block.
    """
    if not retrieved_chunks:
        return "No context chunks were retrieved."

    formatted_chunks: list[str] = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        section = chunk.section or "Unknown section"
        formatted_chunks.append(
            "\n".join(
                [
                    f"{index}. [{chunk.chunk_id}]",
                    f"Doc ID: {chunk.doc_id}",
                    f"Source: {chunk.source_path or 'unknown'}",
                    f"Section: {section}",
                    f"Text: {chunk.text}",
                ]
            )
        )
    return "\n\n".join(formatted_chunks)


def _build_citations(
    cited_chunk_ids: list[str],
    retrieved_chunks: list[RetrievedChunk],
) -> list[Citation]:
    """Build validated citation metadata from parsed chunk IDs.

    Args:
        cited_chunk_ids: Chunk IDs parsed from model output.
        retrieved_chunks: Chunks actually supplied to the model.

    Returns:
        Citations for non-hallucinated chunk IDs.
    """
    chunks_by_id = {chunk.chunk_id: chunk for chunk in retrieved_chunks}
    citations: list[Citation] = []

    for chunk_id in cited_chunk_ids:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            logger.warning("LLM cited chunk_id not present in retrieved context: %s", chunk_id)
            continue

        citations.append(
            Citation(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                source_path=chunk.source_path,
                quoted_text=_snippet(chunk.text),
            )
        )

    return citations


def _snippet(text: str, max_chars: int = 240) -> str:
    """Return a compact citation snippet.

    Args:
        text: Source chunk text.
        max_chars: Maximum snippet length.

    Returns:
        Short source-text snippet.
    """
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3].rstrip() + "..."
