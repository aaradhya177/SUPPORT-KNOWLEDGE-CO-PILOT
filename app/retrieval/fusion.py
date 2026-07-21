"""Rank fusion utilities for combining retrieval result lists."""

from app.retrieval.base import RetrievedChunk


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Fuse ranked retrieval lists with Reciprocal Rank Fusion.

    RRF scores each chunk as ``sum(1 / (k + rank))`` across all ranked lists in
    which it appears. The default ``k=60`` is the standard constant commonly
    used in the RRF literature because it dampens the effect of lower-ranked
    results while still rewarding agreement across retrievers.

    Args:
        ranked_lists: Ranked result lists to fuse.
        k: RRF rank constant.

    Raises:
        ValueError: If ``k`` is negative.

    Returns:
        A single fused and re-ranked list of retrieved chunks.
    """
    if k < 0:
        raise ValueError("RRF constant k must be non-negative.")

    fused_scores: dict[str, float] = {}
    representatives: dict[str, RetrievedChunk] = {}
    first_seen_order: dict[str, int] = {}
    source_retrievers: dict[str, set[str]] = {}
    seen_counter = 0

    for ranked_list in ranked_lists:
        for result in ranked_list:
            if result.chunk_id not in representatives:
                representatives[result.chunk_id] = result
                first_seen_order[result.chunk_id] = seen_counter
                source_retrievers[result.chunk_id] = set()
                seen_counter += 1

            fused_scores[result.chunk_id] = fused_scores.get(result.chunk_id, 0.0) + (
                1.0 / (k + result.rank)
            )
            source_retrievers[result.chunk_id].add(result.retriever_name)
            source_retrievers[result.chunk_id].update(result.source_retrievers)

    ordered_chunk_ids = sorted(
        fused_scores,
        key=lambda chunk_id: (-fused_scores[chunk_id], first_seen_order[chunk_id]),
    )

    fused_results: list[RetrievedChunk] = []
    for rank, chunk_id in enumerate(ordered_chunk_ids, start=1):
        representative = representatives[chunk_id]
        fused_results.append(
            RetrievedChunk(
                chunk_id=representative.chunk_id,
                doc_id=representative.doc_id,
                source_path=representative.source_path,
                category=representative.category,
                updated_at=representative.updated_at,
                text=representative.text,
                section=representative.section,
                score=fused_scores[chunk_id],
                rank=rank,
                retriever_name="hybrid",
                source_retrievers=sorted(source_retrievers[chunk_id]),
            )
        )

    return fused_results
