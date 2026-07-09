"""Confidence scoring for no-answer detection."""

from __future__ import annotations

import math

from app.config import get_settings
from app.retrieval.base import RetrievedChunk
from app.scoring.schemas import ConfidenceBreakdown
from app.verification.schemas import Verdict, VerifiedAnswer


class ConfidenceScorer:
    """Score whether a verified RAG answer is safe to show.

    Formula rationale:
    - retrieval_score_component captures whether the fused retriever returned
      strong candidates. RRF scores are small, so the average top-k score is
      normalized with a saturating exponential: ``1 - exp(-avg_score / 0.03)``.
      Around 0.03 is a practical scale for RRF@60 when one to two retrievers find
      a chunk near the top; the exponential avoids brittle hard cutoffs.
    - rrf_agreement_component rewards agreement between independent retrieval
      methods. Hybrid chunks carry ``source_retrievers`` metadata; chunks found by
      both dense and sparse count as agreement. Agreement is useful but not as
      important as evidence verification, so it receives a smaller weight.
    - verification_score_component is the most direct anti-hallucination signal:
      unsupported citations should strongly reduce confidence.

    Weights are 0.4 retrieval, 0.2 retriever agreement, and 0.4 verification.
    Retrieval and verification are weighted equally because a good answer needs
    both relevant evidence and factually supported citations; agreement is a
    secondary corroboration signal.
    """

    def __init__(self, threshold: float | None = None) -> None:
        """Initialize the scorer.

        Args:
            threshold: Optional confidence threshold. Uses config when omitted.
        """
        settings = get_settings()
        self.threshold = threshold if threshold is not None else settings.confidence_threshold

    def score(
        self,
        retrieved_chunks: list[RetrievedChunk],
        verified: VerifiedAnswer,
    ) -> ConfidenceBreakdown:
        """Compute confidence components and final confidence.

        Args:
            retrieved_chunks: Hybrid retrieved chunks used for generation.
            verified: Answer after citation verification.

        Returns:
            Confidence breakdown with final no-answer decision.
        """
        retrieval_component = _retrieval_score_component(retrieved_chunks)
        agreement_component = _rrf_agreement_component(retrieved_chunks)
        verification_component = _verification_score_component(verified)

        final_confidence = _clamp01(
            (0.4 * retrieval_component)
            + (0.2 * agreement_component)
            + (0.4 * verification_component)
        )
        is_confident = final_confidence >= self.threshold

        return ConfidenceBreakdown(
            retrieval_score_component=retrieval_component,
            rrf_agreement_component=agreement_component,
            verification_score_component=verification_component,
            final_confidence=final_confidence,
            is_confident=is_confident,
            reason=(
                None
                if is_confident
                else _build_low_confidence_reason(
                    retrieval_component=retrieval_component,
                    agreement_component=agreement_component,
                    verified=verified,
                    final_confidence=final_confidence,
                    threshold=self.threshold,
                )
            ),
        )


def _retrieval_score_component(retrieved_chunks: list[RetrievedChunk]) -> float:
    """Normalize average fused retrieval score to 0..1.

    Args:
        retrieved_chunks: Retrieved chunks, ideally from HybridRetriever.

    Returns:
        Normalized retrieval component.
    """
    if not retrieved_chunks:
        return 0.0

    avg_score = sum(max(chunk.score, 0.0) for chunk in retrieved_chunks) / len(retrieved_chunks)
    return _clamp01(1.0 - math.exp(-avg_score / 0.03))


def _rrf_agreement_component(retrieved_chunks: list[RetrievedChunk]) -> float:
    """Measure dense/sparse agreement from hybrid result metadata.

    Args:
        retrieved_chunks: Retrieved chunks with ``source_retrievers`` metadata.

    Returns:
        Fraction of chunks found by both dense and sparse retrievers.
    """
    if not retrieved_chunks:
        return 0.0

    agreed = 0
    for chunk in retrieved_chunks:
        sources = set(chunk.source_retrievers)
        if {"dense", "sparse"}.issubset(sources):
            agreed += 1

    return agreed / len(retrieved_chunks)


def _verification_score_component(verified: VerifiedAnswer) -> float:
    """Measure how well citations survived judge verification.

    Args:
        verified: Verified answer.

    Returns:
        Ratio of supported citations to total judged citations.
    """
    if not verified.verdicts:
        return 0.0

    supported_count = sum(
        1 for verdict in verified.verdicts if verdict.verdict == Verdict.SUPPORTED
    )
    return supported_count / len(verified.verdicts)


def _build_low_confidence_reason(
    retrieval_component: float,
    agreement_component: float,
    verified: VerifiedAnswer,
    final_confidence: float,
    threshold: float,
) -> str:
    """Build a human-readable explanation for a no-answer decision."""
    unsupported_count = sum(
        1 for verdict in verified.verdicts if verdict.verdict == Verdict.UNSUPPORTED
    )
    total_verdicts = len(verified.verdicts)

    reasons: list[str] = [
        f"Final confidence {final_confidence:.2f} is below threshold {threshold:.2f}."
    ]
    if retrieval_component < 0.45:
        reasons.append("Retrieved chunks had weak fused scores.")
    if agreement_component < 0.34:
        reasons.append("Low dense/sparse retrieval overlap.")
    if total_verdicts == 0:
        reasons.append("No citations were verified by the judge.")
    elif unsupported_count:
        reasons.append(f"{unsupported_count}/{total_verdicts} citations were unsupported.")

    return " ".join(reasons)


def _clamp01(value: float) -> float:
    """Clamp a score to the inclusive 0..1 range."""
    return max(0.0, min(1.0, value))
