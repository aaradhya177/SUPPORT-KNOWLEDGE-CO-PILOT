"""Sparse lexical retrieval backed by BM25."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Final

from rank_bm25 import BM25Okapi

from app.retrieval.base import (
    BaseRetriever,
    RetrievalFilters,
    RetrievedChunk,
    infer_category,
    record_matches_filters,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

BM25_INDEX_FILE: Final[str] = "bm25.pkl"
TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_]+")


class BM25Retriever(BaseRetriever):
    """Sparse BM25 retriever using a simple regex tokenizer."""

    def __init__(self) -> None:
        """Initialize an empty BM25 retriever."""
        self.index: BM25Okapi | None = None
        self.metadata: list[dict[str, Any]] = []

    def build_index(self, chunks_path: Path, index_dir: Path) -> None:
        """Build and persist a BM25 index from JSONL chunks.

        Args:
            chunks_path: Path to ``chunks.jsonl``.
            index_dir: Directory where BM25 artifacts are saved.
        """
        chunks = _load_chunk_metadata(chunks_path)
        index_dir.mkdir(parents=True, exist_ok=True)

        if not chunks:
            logger.warning("No chunks found at %s; writing empty sparse index.", chunks_path)
            self.index = None
            self.metadata = []
            _write_index(index_dir, self.index, self.metadata)
            return

        tokenized_corpus = [_tokenize(str(chunk["text"])) for chunk in chunks]
        index = BM25Okapi(tokenized_corpus)
        _write_index(index_dir, index, chunks)

        self.index = index
        self.metadata = chunks
        logger.info("Built sparse BM25 index with %s documents at %s", len(chunks), index_dir)

    def load_index(self, index_dir: Path) -> None:
        """Load a persisted BM25 index and metadata store.

        Args:
            index_dir: Directory containing sparse index artifacts.
        """
        index_path = index_dir / BM25_INDEX_FILE
        if not index_path.exists():
            logger.warning("Sparse BM25 index not found at %s.", index_path)
            self.index = None
            self.metadata = []
            return

        with index_path.open("rb") as input_file:
            payload = pickle.load(input_file)

        self.index = payload.get("index")
        self.metadata = payload.get("metadata", [])
        logger.info(
            "Loaded sparse index with %s metadata records from %s", len(self.metadata), index_dir
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve BM25-ranked chunks for a query.

        Args:
            query: User query text.
            top_k: Maximum number of results to return.
            filters: Optional metadata filters.

        Returns:
            Ranked sparse retrieval results.
        """
        if not query.strip():
            logger.warning("Empty query received by sparse retriever.")
            return []
        if top_k <= 0:
            return []
        if self.index is None or not self.metadata:
            logger.warning("Sparse retriever queried before a non-empty index was loaded.")
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            logger.warning("Sparse retriever query produced no tokens.")
            return []

        scores = self.index.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)

        results: list[RetrievedChunk] = []
        for index_position in ranked_indices:
            record = self.metadata[index_position]
            if not record_matches_filters(record, filters):
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=str(record["chunk_id"]),
                    doc_id=str(record["doc_id"]),
                    source_path=str(record.get("source_path", "")),
                    category=str(
                        record.get("category") or infer_category(record.get("source_path", ""))
                    ),
                    updated_at=record.get("updated_at"),
                    text=str(record["text"]),
                    section=record.get("section"),
                    score=float(scores[index_position]),
                    rank=len(results) + 1,
                    retriever_name="sparse",
                    source_retrievers=["sparse"],
                )
            )
            if len(results) >= top_k:
                break

        return results


def _tokenize(text: str) -> list[str]:
    """Tokenize text using lowercase alphanumeric terms.

    Args:
        text: Input text.

    Returns:
        Lowercase token list.
    """
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _load_chunk_metadata(chunks_path: Path) -> list[dict[str, Any]]:
    """Load retrieval metadata from a JSONL chunk file.

    Args:
        chunks_path: Path to JSONL chunks.

    Returns:
        Chunk metadata records required for retrieval results.
    """
    if not chunks_path.exists():
        logger.warning("Chunks file does not exist: %s", chunks_path)
        return []

    records: list[dict[str, Any]] = []
    with chunks_path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(
                {
                    "chunk_id": payload["chunk_id"],
                    "doc_id": payload["doc_id"],
                    "source_path": payload.get("source_path", ""),
                    "category": payload.get("category")
                    or infer_category(payload.get("source_path", "")),
                    "updated_at": payload.get("updated_at"),
                    "text": payload["text"],
                    "section": payload.get("section"),
                }
            )
    return records


def _write_index(index_dir: Path, index: BM25Okapi | None, metadata: list[dict[str, Any]]) -> None:
    """Persist BM25 index and metadata to disk.

    Args:
        index_dir: Output index directory.
        index: BM25 index instance, or ``None`` for empty indexes.
        metadata: Chunk metadata records.
    """
    with (index_dir / BM25_INDEX_FILE).open("wb") as output_file:
        pickle.dump({"index": index, "metadata": metadata}, output_file)
