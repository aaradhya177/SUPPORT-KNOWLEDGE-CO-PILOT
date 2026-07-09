"""Tests for confidence scoring."""

from app.generation.generator import Citation
from app.retrieval.base import RetrievedChunk
from app.scoring.confidence import ConfidenceScorer
from app.verification.schemas import CitationVerdict, Verdict, VerifiedAnswer


def _chunk(chunk_id: str, score: float, sources: list[str]) -> RetrievedChunk:
    """Create a retrieved chunk fixture."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        source_path=f"{chunk_id}.md",
        text="source text",
        section="Section",
        score=score,
        rank=1,
        retriever_name="hybrid",
        source_retrievers=sources,
    )


def _verified(verdicts: list[Verdict]) -> VerifiedAnswer:
    """Create a verified answer fixture."""
    citations = [
        Citation(
            chunk_id=f"c{index}",
            doc_id=f"doc-c{index}",
            source_path=f"c{index}.md",
            quoted_text="quote",
        )
        for index, _ in enumerate(verdicts)
    ]
    return VerifiedAnswer(
        answer_text="answer",
        verified_citations=citations,
        flagged_citations=[],
        verdicts=[
            CitationVerdict(
                chunk_id=f"c{index}",
                claim_excerpt="claim",
                verdict=verdict,
                judge_reasoning="reason",
            )
            for index, verdict in enumerate(verdicts)
        ],
        all_supported=all(verdict != Verdict.UNSUPPORTED for verdict in verdicts),
    )


def test_confidence_scorer_marks_high_confidence_answer() -> None:
    """Assert strong retrieval, agreement, and verification exceed threshold."""
    scorer = ConfidenceScorer(threshold=0.55)
    chunks = [
        _chunk("a", 0.032, ["dense", "sparse"]),
        _chunk("b", 0.030, ["dense", "sparse"]),
    ]
    verified = _verified([Verdict.SUPPORTED, Verdict.SUPPORTED])

    breakdown = scorer.score(retrieved_chunks=chunks, verified=verified)

    assert breakdown.is_confident is True
    assert breakdown.final_confidence >= 0.55
    assert breakdown.reason is None


def test_confidence_scorer_marks_low_confidence_answer() -> None:
    """Assert weak retrieval and unsupported citations fall below threshold."""
    scorer = ConfidenceScorer(threshold=0.55)
    chunks = [
        _chunk("a", 0.002, ["dense"]),
        _chunk("b", 0.001, ["sparse"]),
    ]
    verified = _verified([Verdict.UNSUPPORTED, Verdict.UNSUPPORTED])

    breakdown = scorer.score(retrieved_chunks=chunks, verified=verified)

    assert breakdown.is_confident is False
    assert breakdown.final_confidence < 0.55
    assert breakdown.reason is not None
    assert "unsupported" in breakdown.reason


def test_confidence_scorer_penalizes_zero_verified_citations() -> None:
    """Assert answers with no judge verdicts are heavily penalized."""
    scorer = ConfidenceScorer(threshold=0.55)
    chunks = [_chunk("a", 0.03, ["dense", "sparse"])]
    verified = _verified([])

    breakdown = scorer.score(retrieved_chunks=chunks, verified=verified)

    assert breakdown.verification_score_component == 0.0
    assert breakdown.is_confident is False
