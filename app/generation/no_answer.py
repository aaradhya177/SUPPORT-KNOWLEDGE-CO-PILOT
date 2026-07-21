"""No-answer response helpers."""

from app.scoring.schemas import ConfidenceBreakdown


def build_no_answer_response(query: str, breakdown: ConfidenceBreakdown) -> dict[str, object]:
    """Build a structured no-answer response.

    Args:
        query: User query that could not be answered confidently.
        breakdown: Confidence score breakdown explaining the decision.

    Returns:
        API-ready no-answer payload.
    """
    return {
        "answer": "I don't have enough verified information in the knowledge base to confidently answer this question.",
        "query": query,
        "confidence": breakdown.final_confidence,
        "confidence_breakdown": breakdown.model_dump(),
        "reason": breakdown.reason,
        "citations": [],
        "flagged_citations": [],
        "verdicts": [],
        "status": "no_answer",
    }
