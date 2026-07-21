"""Pydantic request and response schemas for the API."""

from typing import Literal

from pydantic import BaseModel, Field

from app.retrieval.base import RetrievalFilters


class QueryRequest(BaseModel):
    """Request body for querying the RAG pipeline."""

    query: str = Field(
        ...,
        min_length=1,
        description="Support question to answer from the knowledge base.",
        examples=["What should I do if my password reset email never arrives?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of retrieved chunks to use as context.",
        examples=[5],
    )
    filters: RetrievalFilters | None = Field(
        default=None,
        description="Optional metadata filters for retrieval.",
    )


class QueryResponse(BaseModel):
    """Response body for a query result."""

    request_id: str = Field(..., description="Request correlation ID for logs and support.")
    answer: str = Field(..., description="Generated answer or no-answer message.")
    citations: list[dict] = Field(
        default_factory=list,
        description="Verified citations backing the answer.",
    )
    flagged_citations: list[dict] = Field(
        default_factory=list,
        description="Citations that failed or partially failed verification.",
    )
    verdicts: list[dict] = Field(
        default_factory=list,
        description="Judge verdicts for cited claims.",
    )
    confidence_breakdown: dict | None = Field(
        default=None,
        description="Optional confidence score component breakdown.",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Final confidence score.")
    status: str = Field(..., description="Either 'answered' or 'no_answer'.", examples=["answered"])
    reason: str | None = Field(
        default=None,
        description="Human-readable reason when the system returns no-answer.",
    )


class IngestResponse(BaseModel):
    """Response body for ingestion status."""

    job_id: str = Field(..., description="Unique ingestion job identifier.")
    files_processed: int = Field(
        ..., ge=0, description="Number of supported source files processed."
    )
    chunks_created: int = Field(
        ..., ge=0, description="Number of chunks written to processed JSONL."
    )
    status: str = Field(..., description="Current ingestion status.", examples=["running"])
    started_at: str = Field(..., description="UTC timestamp when the job started.")
    completed_at: str | None = Field(
        default=None,
        description="UTC timestamp when the job completed or failed.",
    )
    error_message: str | None = Field(
        default=None,
        description="Failure details when the job status is failed.",
    )


class EvalRunRequest(BaseModel):
    """Request body for a golden-set evaluation run."""

    sample_size: int | None = Field(
        default=None,
        ge=1,
        description="Optional number of golden-set examples to run for quick iteration.",
        examples=[10],
    )


class EvalRunResponse(BaseModel):
    """Response body for an evaluation run."""

    summary: dict = Field(..., description="Aggregate evaluation metrics.")
    report_path: str = Field(..., description="Path to the generated markdown report.")


class FeedbackRequest(BaseModel):
    """Request body for answer feedback."""

    query: str = Field(..., min_length=1, description="Original support question.")
    answer: str = Field(..., min_length=1, description="Answer shown to the user.")
    status: str = Field(..., min_length=1, description="Answer status, such as answered/no_answer.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Answer confidence score.")
    rating: Literal["up", "down"] = Field(..., description="Thumbs up or thumbs down rating.")
    comment: str | None = Field(
        default=None,
        max_length=2000,
        description="Optional free-form user feedback.",
    )
    citation_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs cited by the answer.",
    )


class FeedbackResponse(BaseModel):
    """Response body for persisted feedback."""

    feedback_id: str = Field(..., description="Unique feedback identifier.")
    status: str = Field(default="recorded", description="Feedback submission status.")
    created_at: str = Field(..., description="UTC timestamp when feedback was recorded.")
