"""Shared retrieval interfaces and result models."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """A chunk returned by a retriever with ranking metadata."""

    chunk_id: str
    doc_id: str
    source_path: str = ""
    text: str
    section: str | None
    score: float
    rank: int = Field(ge=1)
    retriever_name: str
    source_retrievers: list[str] = Field(default_factory=list)


class BaseRetriever(ABC):
    """Abstract base class for retrieval implementations."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        """Return the top matching chunks for a query.

        Args:
            query: User query text.
            top_k: Maximum number of results to return.

        Returns:
            Ranked retrieved chunks.
        """
