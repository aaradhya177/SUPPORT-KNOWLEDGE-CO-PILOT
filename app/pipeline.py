"""End-to-end RAG orchestration pipeline."""

from app.generation.generator import AnswerGenerator
from app.generation.no_answer import build_no_answer_response
from app.retrieval.hybrid import HybridRetriever
from app.scoring.confidence import ConfidenceScorer
from app.scoring.schemas import ConfidenceBreakdown
from app.verification.judge import CitationVerifier


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

    def answer_query(self, query: str, top_k: int = 5) -> dict[str, object]:
        """Answer a support query with no-answer fallback.

        Args:
            query: User support question.
            top_k: Number of retrieved chunks to use.

        Returns:
            Structured answer or no-answer payload.
        """
        retrieved_chunks = self.retriever.retrieve(query=query, top_k=top_k)
        if not retrieved_chunks:
            empty_breakdown = ConfidenceBreakdown(
                retrieval_score_component=0.0,
                rrf_agreement_component=0.0,
                verification_score_component=0.0,
                final_confidence=0.0,
                is_confident=False,
                reason="No chunks were retrieved from the knowledge base.",
            )
            return build_no_answer_response(query=query, breakdown=empty_breakdown)

        generated = self.generator.generate(query=query, retrieved_chunks=retrieved_chunks)
        verified = self.verifier.verify(generated=generated, retrieved_chunks=retrieved_chunks)
        breakdown = self.scorer.score(retrieved_chunks=retrieved_chunks, verified=verified)

        if not breakdown.is_confident:
            return build_no_answer_response(query=query, breakdown=breakdown)

        return {
            "answer": verified.answer_text,
            "confidence": breakdown.final_confidence,
            "confidence_breakdown": breakdown.model_dump(),
            "citations": [citation.model_dump() for citation in verified.verified_citations],
            "flagged_citations": [citation.model_dump() for citation in verified.flagged_citations],
            "verdicts": [verdict.model_dump() for verdict in verified.verdicts],
            "status": "answered",
        }
