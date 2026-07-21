"""Feedback API route."""

from pathlib import Path

from fastapi import APIRouter, Depends, status

from app.api.dependencies import require_api_key
from app.api.schemas import FeedbackRequest, FeedbackResponse
from app.config import get_settings
from app.feedback.store import FeedbackStore

router = APIRouter()


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record answer feedback",
    description="Persists thumbs up/down feedback for a generated answer.",
)
def submit_feedback(
    request: FeedbackRequest,
    _: str = Depends(require_api_key),
) -> FeedbackResponse:
    """Persist feedback for an answer."""
    record = _get_feedback_store().create_feedback(
        query=request.query,
        answer=request.answer,
        status=request.status,
        confidence=request.confidence,
        rating=request.rating,
        comment=request.comment,
        citation_chunk_ids=request.citation_chunk_ids,
    )
    return FeedbackResponse(feedback_id=record.feedback_id, created_at=record.created_at)


def _get_feedback_store() -> FeedbackStore:
    """Return a feedback store using the configured SQLite path."""
    settings = get_settings()
    return FeedbackStore(Path(settings.feedback_db_path))
