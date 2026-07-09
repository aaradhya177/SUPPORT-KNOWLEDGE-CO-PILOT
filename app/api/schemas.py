"""Pydantic request and response schemas for the API."""

from pydantic import BaseModel, Field


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


class QueryResponse(BaseModel):
    """Response body for a query result."""

    answer: str = Field(..., description="Generated answer or no-answer message.")
    citations: list[dict] = Field(
        default_factory=list,
        description="Verified citations backing the answer.",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Final confidence score.")
    status: str = Field(..., description="Either 'answered' or 'no_answer'.", examples=["answered"])
    reason: str | None = Field(
        default=None,
        description="Human-readable reason when the system returns no-answer.",
    )


class IngestResponse(BaseModel):
    """Response body for ingestion status."""

    files_processed: int = Field(
        ..., ge=0, description="Number of supported source files processed."
    )
    chunks_created: int = Field(
        ..., ge=0, description="Number of chunks written to processed JSONL."
    )
    status: str = Field(..., description="Current ingestion status.", examples=["running"])


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
