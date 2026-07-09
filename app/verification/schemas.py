"""Schemas for citation verification."""

from enum import StrEnum

from pydantic import BaseModel

from app.generation.generator import Citation


class Verdict(StrEnum):
    """Supportedness labels for cited answer claims."""

    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"


class CitationVerdict(BaseModel):
    """Judge verdict for a single claim-citation pair."""

    chunk_id: str
    claim_excerpt: str
    verdict: Verdict
    judge_reasoning: str


class VerifiedAnswer(BaseModel):
    """Generated answer after citation verification."""

    answer_text: str
    verified_citations: list[Citation]
    flagged_citations: list[Citation]
    verdicts: list[CitationVerdict]
    all_supported: bool
