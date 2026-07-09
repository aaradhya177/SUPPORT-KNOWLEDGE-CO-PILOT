"""LLM-as-judge citation verification."""

from __future__ import annotations

import json
import re
from hashlib import sha1
from typing import Any

from app.generation.generator import (
    CITATION_PATTERN,
    Citation,
    GeneratedAnswer,
    _split_citation_group,
)
from app.llm.client import LLMClient
from app.retrieval.base import RetrievedChunk
from app.utils.logger import get_logger
from app.verification.prompts import JUDGE_SYSTEM_PROMPT
from app.verification.schemas import CitationVerdict, Verdict, VerifiedAnswer

logger = get_logger(__name__)

JSON_OBJECT_PATTERN = re.compile(r"\{.*?\}", re.DOTALL)
SENTENCE_PATTERN = re.compile(r"[^.!?]+[.!?]?")


class CitationVerifier:
    """Verify generated answer citations with an LLM judge."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize the citation verifier.

        Args:
            llm_client: LLM client dependency used for judging claims.
        """
        self.llm_client = llm_client

    def _split_into_claims(
        self,
        answer_text: str,
        citations: list[Citation],
    ) -> list[tuple[str, Citation]]:
        """Split answer text into claim excerpts associated with citation markers.

        Args:
            answer_text: Generated answer text containing inline citation markers.
            citations: Validated citations from generation.

        Returns:
            Claim excerpts paired with their nearest citation.
        """
        citations_by_id = {citation.chunk_id: citation for citation in citations}
        claim_pairs: list[tuple[str, Citation]] = []
        protected_answer_text = _protect_email_dots(answer_text)

        for sentence_match in SENTENCE_PATTERN.finditer(protected_answer_text):
            sentence = _restore_email_dots(sentence_match.group(0).strip())
            if not sentence:
                continue

            marker_matches = list(CITATION_PATTERN.finditer(sentence))
            if not marker_matches:
                continue

            claim_excerpt = CITATION_PATTERN.sub("", sentence).strip()
            claim_excerpt = re.sub(r"\s+", " ", claim_excerpt)
            if not claim_excerpt:
                continue

            for marker_match in marker_matches:
                for chunk_id in _split_citation_group(marker_match.group(1)):
                    citation = citations_by_id.get(chunk_id)
                    if citation is None:
                        continue
                    claim_pairs.append((claim_excerpt, citation))

        return claim_pairs

    def verify(
        self,
        generated: GeneratedAnswer,
        retrieved_chunks: list[RetrievedChunk],
    ) -> VerifiedAnswer:
        """Verify generated answer citations against retrieved source chunks.

        In production, judge calls can be cached by a hash of ``chunk_id + claim``
        or batched by provider APIs to avoid repeated LLM calls for identical
        claim-source checks.

        Args:
            generated: Generated answer with citations.
            retrieved_chunks: Retrieved chunks originally supplied as context.

        Returns:
            Verified answer with supported and flagged citations separated.
        """
        chunks_by_id = {chunk.chunk_id: chunk for chunk in retrieved_chunks}
        claim_pairs = self._split_into_claims(generated.answer_text, generated.citations)
        verdicts: list[CitationVerdict] = []

        for claim_excerpt, citation in claim_pairs:
            cited_chunk = chunks_by_id.get(citation.chunk_id)
            if cited_chunk is None:
                logger.warning(
                    "Cannot verify citation absent from retrieved chunks: %s", citation.chunk_id
                )
                verdicts.append(
                    CitationVerdict(
                        chunk_id=citation.chunk_id,
                        claim_excerpt=claim_excerpt,
                        verdict=Verdict.UNSUPPORTED,
                        judge_reasoning="Cited chunk was not present in retrieved context.",
                    )
                )
                continue

            user_prompt = _build_judge_prompt(
                claim_excerpt=claim_excerpt, source_text=cited_chunk.text
            )
            raw_judge_output = self.llm_client.complete(
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=256,
            )
            verdicts.append(
                _parse_judge_output(
                    raw_output=raw_judge_output,
                    chunk_id=citation.chunk_id,
                    claim_excerpt=claim_excerpt,
                )
            )

        unsupported_chunk_ids = {
            verdict.chunk_id for verdict in verdicts if verdict.verdict == Verdict.UNSUPPORTED
        }
        verified_chunk_ids = {
            verdict.chunk_id
            for verdict in verdicts
            if verdict.verdict in {Verdict.SUPPORTED, Verdict.PARTIALLY_SUPPORTED}
        }

        verified_citations = [
            citation for citation in generated.citations if citation.chunk_id in verified_chunk_ids
        ]
        flagged_citations = [
            citation
            for citation in generated.citations
            if citation.chunk_id in unsupported_chunk_ids
        ]
        answer_text = _mark_unverified_citations(generated.answer_text, unsupported_chunk_ids)

        return VerifiedAnswer(
            answer_text=answer_text,
            verified_citations=verified_citations,
            flagged_citations=flagged_citations,
            verdicts=verdicts,
            all_supported=not unsupported_chunk_ids,
        )


def _build_judge_prompt(claim_excerpt: str, source_text: str) -> str:
    """Build the user prompt for a citation judge call.

    Args:
        claim_excerpt: Claim text to verify.
        source_text: Full cited source chunk text.

    Returns:
        Prompt string for the judge model.
    """
    return (
        "Claim:\n"
        f"{claim_excerpt}\n\n"
        "Source chunk:\n"
        f"{source_text}\n\n"
        "Return strict JSON only."
    )


def _parse_judge_output(
    raw_output: str,
    chunk_id: str,
    claim_excerpt: str,
) -> CitationVerdict:
    """Parse a judge response into a CitationVerdict.

    Args:
        raw_output: Raw judge model output.
        chunk_id: Cited chunk ID.
        claim_excerpt: Claim being judged.

    Returns:
        Parsed citation verdict, or UNSUPPORTED fallback on parse errors.
    """
    try:
        payload = _extract_json_object(raw_output)
        verdict = Verdict(str(payload["verdict"]).strip().upper())
        reasoning = str(payload.get("reasoning", "")).strip()
        return CitationVerdict(
            chunk_id=chunk_id,
            claim_excerpt=claim_excerpt,
            verdict=verdict,
            judge_reasoning=reasoning,
        )
    except Exception as exc:
        logger.exception("Failed to parse judge output for chunk %s: %s", chunk_id, raw_output)
        fallback_hash = sha1(raw_output.encode("utf-8")).hexdigest()[:8]
        return CitationVerdict(
            chunk_id=chunk_id,
            claim_excerpt=claim_excerpt,
            verdict=Verdict.UNSUPPORTED,
            judge_reasoning=(
                "Judge output could not be parsed as a valid verdict. "
                f"parse_error_id={fallback_hash}; error={exc}"
            ),
        )


def _extract_json_object(raw_output: str) -> dict[str, Any]:
    """Extract the first JSON object from a judge response.

    Handles markdown code fences and trailing commentary by searching for the
    first JSON object in the response text.

    Args:
        raw_output: Raw judge model output.

    Raises:
        ValueError: If no JSON object can be found.

    Returns:
        Parsed JSON object.
    """
    cleaned = raw_output.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = JSON_OBJECT_PATTERN.search(cleaned)
    if match is None:
        raise ValueError("No JSON object found in judge output.")

    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Judge JSON payload is not an object.")
    return parsed


def _mark_unverified_citations(answer_text: str, unsupported_chunk_ids: set[str]) -> str:
    """Replace unsupported citation markers with an unverified marker.

    Args:
        answer_text: Generated answer text.
        unsupported_chunk_ids: Citation chunk IDs judged unsupported.

    Returns:
        Answer text with unsupported citation markers replaced.
    """
    if not unsupported_chunk_ids:
        return answer_text

    def replace_marker(match: re.Match[str]) -> str:
        chunk_ids = _split_citation_group(match.group(1))
        rewritten_markers = [
            "[unverified]" if chunk_id in unsupported_chunk_ids else f"[{chunk_id}]"
            for chunk_id in chunk_ids
        ]
        return " ".join(rewritten_markers)

    return CITATION_PATTERN.sub(replace_marker, answer_text)


def _protect_email_dots(text: str) -> str:
    """Protect dots inside email addresses before sentence splitting."""

    def replace_email(match: re.Match[str]) -> str:
        return match.group(0).replace(".", "<DOT>")

    return re.sub(r"[\w.+-]+@[\w.-]+\.\w+", replace_email, text)


def _restore_email_dots(text: str) -> str:
    """Restore protected email-address dots."""
    return text.replace("<DOT>", ".")
