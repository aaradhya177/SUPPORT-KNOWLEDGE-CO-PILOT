"""Query API route."""

import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import enforce_query_rate_limit, get_pipeline
from app.api.middleware import REQUEST_ID_HEADER
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
    http_request: Request,
    request: QueryRequest,
    _: None = Depends(enforce_query_rate_limit),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> QueryResponse:
    """Answer a support query using the RAG pipeline."""
    request_id = str(getattr(http_request.state, "request_id", ""))
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty."
        )

    try:
        result = pipeline.answer_query(
            query=request.query,
            top_k=request.top_k,
            request_id=request_id,
            filters=request.filters,
        )
    except Exception as exc:
        logger.exception(
            "Pipeline query failed.",
            extra={"event": "rag_query_failed", "request_id": request_id},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Pipeline query failed.",
        ) from exc

    return QueryResponse(
        request_id=request_id,
        answer=str(result.get("answer", "")),
        citations=list(result.get("citations", [])),
        flagged_citations=list(result.get("flagged_citations", [])),
        verdicts=list(result.get("verdicts", [])),
        confidence_breakdown=(
            dict(result["confidence_breakdown"])
            if isinstance(result.get("confidence_breakdown"), dict)
            else None
        ),
        confidence=float(result.get("confidence", 0.0)),
        status=str(result.get("status", "no_answer")),
        reason=result.get("reason") if result.get("reason") is not None else None,
    )


@router.post(
    "/query/stream",
    status_code=status.HTTP_200_OK,
    summary="Stream support question progress and answer",
    description="Streams RAG progress events and the final answer as Server-Sent Events.",
)
def stream_query_knowledge_base(
    http_request: Request,
    request: QueryRequest,
    _: None = Depends(enforce_query_rate_limit),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> StreamingResponse:
    """Stream query progress events using Server-Sent Events."""
    request_id = str(getattr(http_request.state, "request_id", ""))
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty."
        )

    return StreamingResponse(
        _format_sse_events(
            pipeline.stream_answer_query(
                query=request.query,
                top_k=request.top_k,
                request_id=request_id,
                filters=request.filters,
            )
        ),
        media_type="text/event-stream",
        headers={
            REQUEST_ID_HEADER: request_id,
            "Cache-Control": "no-cache",
        },
    )


def _format_sse_events(events: Iterator[dict[str, object]]) -> Iterator[str]:
    """Format pipeline event dictionaries as Server-Sent Events."""
    for event in events:
        event_name = str(event.get("event", "message"))
        yield f"event: {event_name}\n"
        yield f"data: {json.dumps(event, default=str)}\n\n"
