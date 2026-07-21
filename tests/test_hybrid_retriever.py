"""Tests for hybrid retrieval and optional reranking."""

from app.retrieval.base import BaseRetriever, RetrievalFilters, RetrievedChunk
from app.retrieval.hybrid import HybridRetriever


class FakeRetriever(BaseRetriever):
    """Retriever returning fixed chunks while recording requested top_k."""

    def __init__(self, results: list[RetrievedChunk]) -> None:
        """Initialize with fixed results."""
        self.results = results
        self.requested_top_k: int | None = None
        self.filters: RetrievalFilters | None = None

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedChunk]:
        """Return fixed results up to requested top_k."""
        self.requested_top_k = top_k
        self.filters = filters
        return self.results[:top_k]


class FakeReranker:
    """Reranker that reverses candidates for deterministic tests."""

    def __init__(self) -> None:
        """Initialize empty call state."""
        self.seen_candidate_ids: list[str] = []

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Reverse chunks and update final ranks."""
        self.seen_candidate_ids = [chunk.chunk_id for chunk in chunks]
        reranked: list[RetrievedChunk] = []
        for rank, chunk in enumerate(reversed(chunks[:]), start=1):
            if rank > top_k:
                break
            reranked.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source_path=chunk.source_path,
                    text=chunk.text,
                    section=chunk.section,
                    score=chunk.score,
                    rank=rank,
                    retriever_name=chunk.retriever_name,
                    source_retrievers=chunk.source_retrievers,
                )
            )
        return reranked


def _chunk(chunk_id: str, rank: int, retriever_name: str) -> RetrievedChunk:
    """Create a retrieved chunk fixture."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        source_path=f"{chunk_id}.md",
        category=f"category-{chunk_id}",
        updated_at="2026-01-01T00:00:00+00:00",
        text=f"Text for {chunk_id}",
        section="Section",
        score=1.0 / rank,
        rank=rank,
        retriever_name=retriever_name,
        source_retrievers=[retriever_name],
    )


def test_hybrid_retriever_preserves_rrf_behavior_without_reranker() -> None:
    """Assert default hybrid retrieval returns RRF-ordered results."""
    dense = FakeRetriever([_chunk("dense-a", 1, "dense"), _chunk("shared", 2, "dense")])
    sparse = FakeRetriever([_chunk("shared", 1, "sparse"), _chunk("sparse-b", 2, "sparse")])
    retriever = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)

    results = retriever.retrieve("question", top_k=2)

    assert [result.chunk_id for result in results] == ["shared", "dense-a"]
    assert [result.rank for result in results] == [1, 2]
    assert results[0].category == "category-shared"


def test_hybrid_retriever_uses_reranker_when_configured() -> None:
    """Assert optional reranker receives fused candidates and controls final order."""
    dense = FakeRetriever(
        [
            _chunk("dense-a", 1, "dense"),
            _chunk("dense-b", 2, "dense"),
            _chunk("shared", 3, "dense"),
        ]
    )
    sparse = FakeRetriever(
        [
            _chunk("sparse-a", 1, "sparse"),
            _chunk("shared", 2, "sparse"),
            _chunk("sparse-b", 3, "sparse"),
        ]
    )
    reranker = FakeReranker()
    retriever = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        expansion_factor=3,
        reranker=reranker,
    )

    results = retriever.retrieve("question", top_k=2)

    assert dense.requested_top_k == 6
    assert sparse.requested_top_k == 6
    assert reranker.seen_candidate_ids == ["shared", "dense-a", "sparse-a", "dense-b", "sparse-b"]
    assert [result.chunk_id for result in results] == ["sparse-b", "dense-b"]
    assert [result.rank for result in results] == [1, 2]


def test_hybrid_retriever_passes_filters_to_sub_retrievers() -> None:
    """Assert hybrid retrieval forwards filters to dense and sparse retrievers."""
    dense = FakeRetriever([_chunk("dense-a", 1, "dense")])
    sparse = FakeRetriever([_chunk("sparse-a", 1, "sparse")])
    filters = RetrievalFilters(category=["password-reset"])
    retriever = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)

    retriever.retrieve("question", top_k=2, filters=filters)

    assert dense.filters == filters
    assert sparse.filters == filters
