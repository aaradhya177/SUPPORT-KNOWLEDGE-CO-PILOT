"""Tests for the end-to-end RAG pipeline orchestration."""

from app.generation.generator import Citation, GeneratedAnswer
from app.pipeline import RAGPipeline
from app.retrieval.base import RetrievalFilters, RetrievedChunk
from app.scoring.schemas import ConfidenceBreakdown
from app.verification.schemas import CitationVerdict, Verdict, VerifiedAnswer


class FakeRetriever:
    """Fake retriever dependency."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        """Initialize with fixed chunks."""
        self.chunks = chunks

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedChunk]:
        """Return fixed chunks."""
        return self.chunks[:top_k]


class FakeGenerator:
    """Fake generator dependency."""

    def __init__(self, generated: GeneratedAnswer) -> None:
        """Initialize with a fixed generated answer."""
        self.generated = generated

    def generate(self, query: str, retrieved_chunks: list[RetrievedChunk]) -> GeneratedAnswer:
        """Return fixed generated answer."""
        return self.generated


class FakeVerifier:
    """Fake verifier dependency."""

    def __init__(self, verified: VerifiedAnswer) -> None:
        """Initialize with a fixed verified answer."""
        self.verified = verified

    def verify(
        self,
        generated: GeneratedAnswer,
        retrieved_chunks: list[RetrievedChunk],
    ) -> VerifiedAnswer:
        """Return fixed verified answer."""
        return self.verified


class FakeScorer:
    """Fake confidence scorer dependency."""

    def __init__(self, breakdown: ConfidenceBreakdown) -> None:
        """Initialize with a fixed confidence breakdown."""
        self.breakdown = breakdown

    def score(
        self,
        retrieved_chunks: list[RetrievedChunk],
        verified: VerifiedAnswer,
    ) -> ConfidenceBreakdown:
        """Return fixed confidence breakdown."""
        return self.breakdown


def _chunk() -> RetrievedChunk:
    """Create a retrieved chunk fixture."""
    return RetrievedChunk(
        chunk_id="chunk_0",
        doc_id="doc",
        source_path="doc.md",
        text="source",
        section="Section",
        score=0.03,
        rank=1,
        retriever_name="hybrid",
        source_retrievers=["dense", "sparse"],
    )


def _citation() -> Citation:
    """Create a citation fixture."""
    return Citation(chunk_id="chunk_0", doc_id="doc", source_path="doc.md", quoted_text="source")


def _generated() -> GeneratedAnswer:
    """Create a generated answer fixture."""
    return GeneratedAnswer(
        answer_text="Supported answer [chunk_0].",
        citations=[_citation()],
        raw_llm_output="Supported answer [chunk_0].",
    )


def _verified() -> VerifiedAnswer:
    """Create a verified answer fixture."""
    return VerifiedAnswer(
        answer_text="Supported answer [chunk_0].",
        verified_citations=[_citation()],
        flagged_citations=[],
        verdicts=[
            CitationVerdict(
                chunk_id="chunk_0",
                claim_excerpt="Supported answer.",
                verdict=Verdict.SUPPORTED,
                judge_reasoning="Supported.",
            )
        ],
        all_supported=True,
    )


def _breakdown(is_confident: bool) -> ConfidenceBreakdown:
    """Create a confidence breakdown fixture."""
    return ConfidenceBreakdown(
        retrieval_score_component=1.0 if is_confident else 0.1,
        rrf_agreement_component=1.0 if is_confident else 0.0,
        verification_score_component=1.0 if is_confident else 0.0,
        final_confidence=0.95 if is_confident else 0.12,
        is_confident=is_confident,
        reason=None if is_confident else "Low confidence.",
    )


def test_rag_pipeline_returns_confident_answer() -> None:
    """Assert pipeline returns a structured answered payload when confident."""
    pipeline = RAGPipeline(
        retriever=FakeRetriever([_chunk()]),  # type: ignore[arg-type]
        generator=FakeGenerator(_generated()),  # type: ignore[arg-type]
        verifier=FakeVerifier(_verified()),  # type: ignore[arg-type]
        scorer=FakeScorer(_breakdown(True)),  # type: ignore[arg-type]
    )

    response = pipeline.answer_query("question")

    assert response["status"] == "answered"
    assert response["answer"] == "Supported answer [chunk_0]."
    assert response["confidence"] == 0.95
    assert len(response["citations"]) == 1


def test_rag_pipeline_returns_no_answer_when_not_confident() -> None:
    """Assert pipeline returns no-answer payload when confidence is low."""
    pipeline = RAGPipeline(
        retriever=FakeRetriever([_chunk()]),  # type: ignore[arg-type]
        generator=FakeGenerator(_generated()),  # type: ignore[arg-type]
        verifier=FakeVerifier(_verified()),  # type: ignore[arg-type]
        scorer=FakeScorer(_breakdown(False)),  # type: ignore[arg-type]
    )

    response = pipeline.answer_query("question")

    assert response["status"] == "no_answer"
    assert response["citations"] == []
    assert response["reason"] == "Low confidence."


def test_rag_pipeline_returns_no_answer_when_no_chunks_retrieved() -> None:
    """Assert pipeline short-circuits when retrieval returns no chunks."""
    pipeline = RAGPipeline(
        retriever=FakeRetriever([]),  # type: ignore[arg-type]
        generator=FakeGenerator(_generated()),  # type: ignore[arg-type]
        verifier=FakeVerifier(_verified()),  # type: ignore[arg-type]
        scorer=FakeScorer(_breakdown(True)),  # type: ignore[arg-type]
    )

    response = pipeline.answer_query("question")

    assert response["status"] == "no_answer"
    assert response["confidence"] == 0.0
    assert response["reason"] == "No chunks were retrieved from the knowledge base."
