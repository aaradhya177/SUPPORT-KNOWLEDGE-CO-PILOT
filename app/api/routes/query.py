"""Query API route."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_pipeline
from app.api.schemas import QueryRequest, QueryResponse
from app.pipeline import RAGPipeline
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Answer a support question",
    description=(
        "Runs hybrid retrieval, grounded generation, citation verification, and "
        "confidence-based no-answer detection."
    ),
)
def query_knowledge_base(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> QueryResponse:
    """Answer a support query using the RAG pipeline."""
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty."
        )

    try:
        result = pipeline.answer_query(query=request.query, top_k=request.top_k)
    except Exception as exc:
        logger.exception("Pipeline query failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Pipeline query failed.",
        ) from exc

    return QueryResponse(
        answer=str(result.get("answer", "")),
        citations=list(result.get("citations", [])),
        confidence=float(result.get("confidence", 0.0)),
        status=str(result.get("status", "no_answer")),
        reason=result.get("reason") if result.get("reason") is not None else None,
    )
