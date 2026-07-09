"""Schemas for answer confidence scoring."""

from pydantic import BaseModel, Field


class ConfidenceBreakdown(BaseModel):
    """Component-level confidence score for a generated answer."""

    retrieval_score_component: float = Field(ge=0.0, le=1.0)
    rrf_agreement_component: float = Field(ge=0.0, le=1.0)
    verification_score_component: float = Field(ge=0.0, le=1.0)
    final_confidence: float = Field(ge=0.0, le=1.0)
    is_confident: bool
    reason: str | None
