"""Shared retrieval interfaces and result models."""

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, Field


class RetrievalFilters(BaseModel):
    """Optional metadata filters applied during retrieval."""

    category: list[str] | None = None
    source_path: list[str] | None = None
    section: list[str] | None = None
    updated_at: str | None = None


class RetrievedChunk(BaseModel):
    """A chunk returned by a retriever with ranking metadata."""

    chunk_id: str
    doc_id: str
    source_path: str = ""
    category: str | None = None
    updated_at: str | None = None
    text: str
    section: str | None
    score: float
    rank: int = Field(ge=1)
    retriever_name: str
    source_retrievers: list[str] = Field(default_factory=list)


class BaseRetriever(ABC):
    """Abstract base class for retrieval implementations."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top matching chunks for a query.

        Args:
            query: User query text.
            top_k: Maximum number of results to return.
            filters: Optional metadata filters.

        Returns:
            Ranked retrieved chunks.
        """


def record_matches_filters(record: dict, filters: RetrievalFilters | None) -> bool:
    """Return whether a raw chunk metadata record satisfies filters."""
    if filters is None:
        return True

    category = str(record.get("category") or infer_category(record.get("source_path", "")))
    source_path = str(record.get("source_path", ""))
    section = record.get("section")
    updated_at = record.get("updated_at")

    if filters.category and _normalize(category) not in {
        _normalize(value) for value in filters.category
    }:
        return False
    if filters.source_path and not _matches_any_path(source_path, filters.source_path):
        return False
    if filters.section and _normalize(section) not in {
        _normalize(value) for value in filters.section
    }:
        return False
    if filters.updated_at and updated_at != filters.updated_at:
        return False

    return True


def infer_category(source_path: str | Path) -> str:
    """Infer a stable category slug from a source path."""
    stem = Path(str(source_path)).stem
    return _normalize(stem)


def _matches_any_path(source_path: str, allowed_paths: list[str]) -> bool:
    """Return whether a source path matches any exact or suffix filter."""
    normalized_source = source_path.replace("\\", "/").lower()
    for allowed_path in allowed_paths:
        normalized_allowed = allowed_path.replace("\\", "/").lower()
        if normalized_source == normalized_allowed or normalized_source.endswith(
            normalized_allowed
        ):
            return True
    return False


def _normalize(value: object) -> str:
    """Normalize filter values for case-insensitive comparison."""
    return str(value or "").strip().lower()
