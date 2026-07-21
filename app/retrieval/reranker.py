"""Optional reranking for hybrid retrieval candidates."""

from __future__ import annotations

from typing import Any, Protocol

from app.retrieval.base import RetrievedChunk


class BaseReranker(Protocol):
    """Interface for retrieval rerankers."""

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Return chunks reordered by query relevance."""


class CrossEncoderReranker:
    """Rerank candidate chunks with a SentenceTransformers CrossEncoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """Load a cross-encoder reranking model."""
        self.model_name = model_name
        self.model = _load_cross_encoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Rerank chunks by cross-encoder relevance score.

        The returned chunks keep their original RRF scores so downstream confidence
        scoring remains comparable with the non-reranked retrieval path.
        """
        if top_k <= 0 or not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]
        raw_scores = self.model.predict(pairs)
        scored_chunks = list(zip([float(score) for score in raw_scores], chunks))
        scored_chunks.sort(key=lambda item: item[0], reverse=True)

        reranked: list[RetrievedChunk] = []
        for rank, (_, chunk) in enumerate(scored_chunks[:top_k], start=1):
            reranked.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source_path=chunk.source_path,
                    category=chunk.category,
                    updated_at=chunk.updated_at,
                    text=chunk.text,
                    section=chunk.section,
                    score=chunk.score,
                    rank=rank,
                    retriever_name=chunk.retriever_name,
                    source_retrievers=chunk.source_retrievers,
                )
            )
        return reranked


def _load_cross_encoder(model_name: str) -> Any:
    """Load the SentenceTransformers CrossEncoder lazily."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for CrossEncoderReranker. "
            "Install project dependencies with `pip install -r requirements.txt`."
        ) from exc

    return CrossEncoder(model_name)
