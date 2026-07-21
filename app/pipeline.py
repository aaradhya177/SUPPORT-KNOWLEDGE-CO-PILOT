"""End-to-end RAG orchestration pipeline."""

from collections.abc import Iterator
from time import perf_counter

from app.generation.generator import AnswerGenerator
from app.generation.no_answer import build_no_answer_response
from app.retrieval.base import RetrievalFilters
from app.retrieval.hybrid import HybridRetriever
from app.scoring.confidence import ConfidenceScorer
from app.scoring.schemas import ConfidenceBreakdown
from app.utils.logger import get_logger
from app.verification.judge import CitationVerifier

logger = get_logger(__name__)


class RAGPipeline:
    """Coordinate retrieval, generation, verification, and confidence scoring."""

    def __init__(
        self,
        retriever: HybridRetriever,
        generator: AnswerGenerator,
        verifier: CitationVerifier,
        scorer: ConfidenceScorer,
    ) -> None:
        """Initialize the pipeline with injected dependencies.

        Args:
            retriever: Hybrid retriever dependency.
            generator: Grounded answer generator dependency.
            verifier: Citation verifier dependency.
            scorer: Confidence scorer dependency.
        """
        self.retriever = retriever
        self.generator = generator
        self.verifier = verifier
        self.scorer = scorer

    def answer_query(
        self,
        query: str,
        top_k: int = 5,
        request_id: str | None = None,
        filters: RetrievalFilters | None = None,
    ) -> dict[str, object]:
        """Answer a support query with no-answer fallback.

        Args:
            query: User support question.
            top_k: Number of retrieved chunks to use.
            request_id: Optional request correlation ID for structured logs.
            filters: Optional retrieval metadata filters.

        Returns:
            Structured answer or no-answer payload.
        """
        started_at = perf_counter()
        retrieval_started_at = perf_counter()
        retrieved_chunks = self.retriever.retrieve(query=query, top_k=top_k, filters=filters)
        retrieval_latency_ms = _elapsed_ms(retrieval_started_at)
        generation_latency_ms = 0.0
        verification_latency_ms = 0.0

        if not retrieved_chunks:
            empty_breakdown = ConfidenceBreakdown(
                retrieval_score_component=0.0,
                rrf_agreement_component=0.0,
                verification_score_component=0.0,
                final_confidence=0.0,
                is_confident=False,
                reason="No chunks were retrieved from the knowledge base.",
            )
            response = build_no_answer_response(query=query, breakdown=empty_breakdown)
            _log_query_observability(
                request_id=request_id,
                status=str(response["status"]),
                confidence=float(response["confidence"]),
                retrieved_chunks_count=0,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
                verification_latency_ms=verification_latency_ms,
                total_latency_ms=_elapsed_ms(started_at),
            )
            return response

        generation_started_at = perf_counter()
        generated = self.generator.generate(query=query, retrieved_chunks=retrieved_chunks)
        generation_latency_ms = _elapsed_ms(generation_started_at)

        verification_started_at = perf_counter()
        verified = self.verifier.verify(generated=generated, retrieved_chunks=retrieved_chunks)
        verification_latency_ms = _elapsed_ms(verification_started_at)
        breakdown = self.scorer.score(retrieved_chunks=retrieved_chunks, verified=verified)

        if not breakdown.is_confident:
            response = build_no_answer_response(query=query, breakdown=breakdown)
            _log_query_observability(
                request_id=request_id,
                status=str(response["status"]),
                confidence=float(response["confidence"]),
                retrieved_chunks_count=len(retrieved_chunks),
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
                verification_latency_ms=verification_latency_ms,
                total_latency_ms=_elapsed_ms(started_at),
            )
            return response

        response = {
            "answer": verified.answer_text,
            "confidence": breakdown.final_confidence,
            "confidence_breakdown": breakdown.model_dump(),
            "citations": [citation.model_dump() for citation in verified.verified_citations],
            "flagged_citations": [citation.model_dump() for citation in verified.flagged_citations],
            "verdicts": [verdict.model_dump() for verdict in verified.verdicts],
            "status": "answered",
        }
        _log_query_observability(
            request_id=request_id,
            status=str(response["status"]),
            confidence=float(response["confidence"]),
            retrieved_chunks_count=len(retrieved_chunks),
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            verification_latency_ms=verification_latency_ms,
            total_latency_ms=_elapsed_ms(started_at),
        )
        return response

    def stream_answer_query(
        self,
        query: str,
        top_k: int = 5,
        request_id: str | None = None,
        filters: RetrievalFilters | None = None,
    ) -> Iterator[dict[str, object]]:
        """Stream query progress events and the final answer payload.

        Events are transport-agnostic dictionaries so FastAPI can expose them as
        Server-Sent Events while tests can inspect the event shape directly.
        """
        started_at = perf_counter()
        generation_latency_ms = 0.0
        verification_latency_ms = 0.0

        try:
            yield {"event": "retrieval_started", "request_id": request_id}
            retrieval_started_at = perf_counter()
            retrieved_chunks = self.retriever.retrieve(query=query, top_k=top_k, filters=filters)
            retrieval_latency_ms = _elapsed_ms(retrieval_started_at)
            yield {
                "event": "retrieval_completed",
                "request_id": request_id,
                "retrieved_chunks_count": len(retrieved_chunks),
                "retrieval_latency_ms": retrieval_latency_ms,
            }

            if not retrieved_chunks:
                empty_breakdown = ConfidenceBreakdown(
                    retrieval_score_component=0.0,
                    rrf_agreement_component=0.0,
                    verification_score_component=0.0,
                    final_confidence=0.0,
                    is_confident=False,
                    reason="No chunks were retrieved from the knowledge base.",
                )
                response = build_no_answer_response(query=query, breakdown=empty_breakdown)
                total_latency_ms = _elapsed_ms(started_at)
                _log_query_observability(
                    request_id=request_id,
                    status=str(response["status"]),
                    confidence=float(response["confidence"]),
                    retrieved_chunks_count=0,
                    retrieval_latency_ms=retrieval_latency_ms,
                    generation_latency_ms=generation_latency_ms,
                    verification_latency_ms=verification_latency_ms,
                    total_latency_ms=total_latency_ms,
                )
                yield {
                    "event": "completed",
                    "request_id": request_id,
                    "result": response,
                    "total_latency_ms": total_latency_ms,
                }
                return

            yield {"event": "generation_started", "request_id": request_id}
            generation_started_at = perf_counter()
            generated = self.generator.generate(query=query, retrieved_chunks=retrieved_chunks)
            generation_latency_ms = _elapsed_ms(generation_started_at)

            yield {"event": "verification_started", "request_id": request_id}
            verification_started_at = perf_counter()
            verified = self.verifier.verify(generated=generated, retrieved_chunks=retrieved_chunks)
            verification_latency_ms = _elapsed_ms(verification_started_at)
            breakdown = self.scorer.score(retrieved_chunks=retrieved_chunks, verified=verified)

            if not breakdown.is_confident:
                response = build_no_answer_response(query=query, breakdown=breakdown)
            else:
                response = {
                    "answer": verified.answer_text,
                    "confidence": breakdown.final_confidence,
                    "confidence_breakdown": breakdown.model_dump(),
                    "citations": [
                        citation.model_dump() for citation in verified.verified_citations
                    ],
                    "flagged_citations": [
                        citation.model_dump() for citation in verified.flagged_citations
                    ],
                    "verdicts": [verdict.model_dump() for verdict in verified.verdicts],
                    "status": "answered",
                }

            total_latency_ms = _elapsed_ms(started_at)
            _log_query_observability(
                request_id=request_id,
                status=str(response["status"]),
                confidence=float(response["confidence"]),
                retrieved_chunks_count=len(retrieved_chunks),
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
                verification_latency_ms=verification_latency_ms,
                total_latency_ms=total_latency_ms,
            )
            yield {
                "event": "completed",
                "request_id": request_id,
                "result": response,
                "retrieval_latency_ms": retrieval_latency_ms,
                "generation_latency_ms": generation_latency_ms,
                "verification_latency_ms": verification_latency_ms,
                "total_latency_ms": total_latency_ms,
            }
        except Exception as exc:
            logger.exception(
                "Streaming RAG query failed.",
                extra={"event": "rag_query_stream_failed", "request_id": request_id},
            )
            yield {
                "event": "error",
                "request_id": request_id,
                "message": "Pipeline query failed.",
                "error_type": exc.__class__.__name__,
            }


def _elapsed_ms(started_at: float) -> float:
    """Return elapsed milliseconds from a perf_counter start."""
    return round((perf_counter() - started_at) * 1000, 2)


def _log_query_observability(
    request_id: str | None,
    status: str,
    confidence: float,
    retrieved_chunks_count: int,
    retrieval_latency_ms: float,
    generation_latency_ms: float,
    verification_latency_ms: float,
    total_latency_ms: float,
) -> None:
    """Emit a structured query observability log."""
    logger.info(
        "rag_query_completed",
        extra={
            "event": "rag_query_completed",
            "request_id": request_id,
            "status": status,
            "confidence": round(confidence, 4),
            "retrieved_chunks_count": retrieved_chunks_count,
            "retrieval_latency_ms": retrieval_latency_ms,
            "generation_latency_ms": generation_latency_ms,
            "verification_latency_ms": verification_latency_ms,
            "total_latency_ms": total_latency_ms,
        },
    )
