"""Tests for Reciprocal Rank Fusion."""

import pytest

from app.retrieval.base import RetrievedChunk
from app.retrieval.fusion import reciprocal_rank_fusion


def _result(chunk_id: str, rank: int, retriever_name: str) -> RetrievedChunk:
    """Create a synthetic retrieved chunk.

    Args:
        chunk_id: Synthetic chunk ID.
        rank: Rank in the source retriever.
        retriever_name: Name of the source retriever.

    Returns:
        Retrieved chunk fixture.
    """
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        text=f"Text for {chunk_id}",
        section="Synthetic",
        score=1.0 / rank,
        rank=rank,
        retriever_name=retriever_name,
    )


def test_reciprocal_rank_fusion_merges_overlap_and_reranks() -> None:
    """Assert overlapping chunks receive summed scores and new hybrid ranks."""
    dense_results = [
        _result("a", 1, "dense"),
        _result("shared", 2, "dense"),
        _result("dense-only", 3, "dense"),
    ]
    sparse_results = [
        _result("b", 1, "sparse"),
        _result("shared", 2, "sparse"),
        _result("sparse-only", 3, "sparse"),
    ]

    fused = reciprocal_rank_fusion([dense_results, sparse_results], k=60)

    assert [result.chunk_id for result in fused[:3]] == ["shared", "a", "b"]
    assert fused[0].score == pytest.approx((1 / 62) + (1 / 62))
    assert fused[0].rank == 1
    assert all(result.retriever_name == "hybrid" for result in fused)
    assert [result.rank for result in fused] == [1, 2, 3, 4, 5]


def test_reciprocal_rank_fusion_rejects_negative_k() -> None:
    """Assert invalid RRF constants are rejected."""
    with pytest.raises(ValueError, match="non-negative"):
        reciprocal_rank_fusion([], k=-1)
