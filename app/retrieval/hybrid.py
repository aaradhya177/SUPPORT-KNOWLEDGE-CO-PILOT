"""Hybrid retrieval by fusing dense and sparse retrieval results."""

from app.retrieval.base import BaseRetriever, RetrievedChunk
from app.retrieval.fusion import reciprocal_rank_fusion
from app.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever(BaseRetriever):
    """Retriever that combines dense and sparse results using RRF."""

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        sparse_retriever: BaseRetriever,
        rrf_k: int = 60,
        expansion_factor: int = 3,
    ) -> None:
        """Initialize a hybrid retriever with injected dependencies.

        Args:
            dense_retriever: Dense semantic retriever instance.
            sparse_retriever: Sparse lexical retriever instance.
            rrf_k: Reciprocal Rank Fusion constant.
            expansion_factor: Multiplier used to retrieve a wider candidate set
                from each sub-retriever before fusion.
        """
        if expansion_factor <= 0:
            raise ValueError("expansion_factor must be greater than zero.")

        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.rrf_k = rrf_k
        self.expansion_factor = expansion_factor

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        """Retrieve fused dense and sparse results for a query.

        Args:
            query: User query text.
            top_k: Maximum number of fused results to return.

        Returns:
            Hybrid retrieval results ranked by fused RRF score.
        """
        if not query.strip():
            logger.warning("Empty query received by hybrid retriever.")
            return []
        if top_k <= 0:
            return []

        candidate_k = top_k * self.expansion_factor
        dense_results = self.dense_retriever.retrieve(query=query, top_k=candidate_k)
        sparse_results = self.sparse_retriever.retrieve(query=query, top_k=candidate_k)

        if not dense_results and not sparse_results:
            logger.warning("Hybrid retriever received no results from either sub-retriever.")
            return []

        fused_results = reciprocal_rank_fusion(
            ranked_lists=[dense_results, sparse_results],
            k=self.rrf_k,
        )
        return fused_results[:top_k]
