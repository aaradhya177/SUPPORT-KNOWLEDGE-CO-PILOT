"""Tests for dense and sparse retrieval implementations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.retrieval.base import BaseRetriever, RetrievalFilters, RetrievedChunk
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import BM25Retriever


class FakeSentenceTransformer:
    """Deterministic lightweight embedding model for dense retriever tests."""

    def __init__(self, model_name: str) -> None:
        """Initialize the fake model.

        Args:
            model_name: Ignored model name, accepted for API compatibility.
        """
        self.model_name = model_name

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """Encode text into deterministic three-dimensional vectors."""
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float(lowered.count("password") + lowered.count("reset")),
                    float(lowered.count("billing") + lowered.count("invoice")),
                    float(lowered.count("api") + lowered.count("rate")),
                ]
            )

        array = np.asarray(vectors, dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            array = array / norms
        return array


def _write_chunks(chunks_path: Path) -> None:
    """Write a tiny JSONL chunk set for retriever tests.

    Args:
        chunks_path: Output JSONL path.
    """
    records = [
        {
            "chunk_id": "doc-password_0",
            "doc_id": "doc-password",
            "source_path": "password.md",
            "category": "password-reset",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "section": "Password Reset",
            "text": "Customers can reset a password from the sign-in page.",
            "token_count": 10,
            "start_char": 0,
            "end_char": 55,
        },
        {
            "chunk_id": "doc-billing_0",
            "doc_id": "doc-billing",
            "source_path": "billing.md",
            "category": "billing-faq",
            "updated_at": "2026-01-02T00:00:00+00:00",
            "section": "Invoices",
            "text": "Billing administrators can download invoice records.",
            "token_count": 7,
            "start_char": 0,
            "end_char": 50,
        },
        {
            "chunk_id": "doc-api_0",
            "doc_id": "doc-api",
            "source_path": "api.md",
            "category": "api-rate-limits",
            "updated_at": "2026-01-03T00:00:00+00:00",
            "section": "Rate Limits",
            "text": "API clients should back off after rate limit responses.",
            "token_count": 9,
            "start_char": 0,
            "end_char": 55,
        },
    ]
    with chunks_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record) + "\n")


def _assert_retrieved_chunks(results: list[RetrievedChunk], retriever_name: str) -> None:
    """Assert common retrieval result shape and ranking semantics."""
    assert results
    assert all(isinstance(result, RetrievedChunk) for result in results)
    assert [result.rank for result in results] == list(range(1, len(results) + 1))
    assert all(isinstance(result.score, float) for result in results)
    assert all(result.retriever_name == retriever_name for result in results)


def test_sparse_retriever_builds_loads_and_retrieves(tmp_path: Path) -> None:
    """Assert BM25Retriever builds, loads, and returns typed results."""
    chunks_path = tmp_path / "chunks.jsonl"
    index_dir = tmp_path / "sparse"
    _write_chunks(chunks_path)

    retriever: BaseRetriever = BM25Retriever()
    assert retriever.retrieve("password reset", top_k=3) == []

    assert isinstance(retriever, BaseRetriever)
    retriever.build_index(chunks_path=chunks_path, index_dir=index_dir)  # type: ignore[attr-defined]

    loaded = BM25Retriever()
    loaded.load_index(index_dir)
    results = loaded.retrieve("password reset", top_k=2)

    _assert_retrieved_chunks(results, "sparse")
    assert results[0].chunk_id == "doc-password_0"
    assert results[0].category == "password-reset"
    assert results[0].updated_at == "2026-01-01T00:00:00+00:00"
    assert loaded.retrieve("", top_k=5) == []


def test_dense_retriever_builds_loads_and_retrieves(tmp_path: Path, monkeypatch) -> None:
    """Assert DenseRetriever builds, loads, and returns typed results."""
    monkeypatch.setattr(
        "app.retrieval.dense._load_sentence_transformer",
        lambda model_name: FakeSentenceTransformer(model_name),
    )

    chunks_path = tmp_path / "chunks.jsonl"
    index_dir = tmp_path / "dense"
    _write_chunks(chunks_path)

    retriever: BaseRetriever = DenseRetriever(model_name="fake-model")
    assert isinstance(retriever, BaseRetriever)
    assert retriever.retrieve("password reset", top_k=3) == []

    retriever.build_index(chunks_path=chunks_path, index_dir=index_dir)  # type: ignore[attr-defined]

    loaded = DenseRetriever(model_name="fake-model")
    loaded.load_index(index_dir)
    results = loaded.retrieve("password reset", top_k=2)

    _assert_retrieved_chunks(results, "dense")
    assert results[0].chunk_id == "doc-password_0"
    assert results[0].category == "password-reset"
    assert results[0].updated_at == "2026-01-01T00:00:00+00:00"
    assert loaded.retrieve("", top_k=5) == []


def test_retrievers_handle_empty_indexes_gracefully(tmp_path: Path, monkeypatch) -> None:
    """Assert empty chunk files do not crash retriever build or query paths."""
    monkeypatch.setattr(
        "app.retrieval.dense._load_sentence_transformer",
        lambda model_name: FakeSentenceTransformer(model_name),
    )
    chunks_path = tmp_path / "empty.jsonl"
    chunks_path.write_text("", encoding="utf-8")

    dense = DenseRetriever(model_name="fake-model")
    dense.build_index(chunks_path=chunks_path, index_dir=tmp_path / "dense-empty")
    assert dense.retrieve("anything", top_k=5) == []

    sparse = BM25Retriever()
    sparse.build_index(chunks_path=chunks_path, index_dir=tmp_path / "sparse-empty")
    assert sparse.retrieve("anything", top_k=5) == []


def test_sparse_retriever_filters_by_category_and_source_path(tmp_path: Path) -> None:
    """Assert BM25Retriever applies metadata filters before returning results."""
    chunks_path = tmp_path / "chunks.jsonl"
    index_dir = tmp_path / "sparse"
    _write_chunks(chunks_path)

    retriever = BM25Retriever()
    retriever.build_index(chunks_path=chunks_path, index_dir=index_dir)

    category_results = retriever.retrieve(
        "password billing api",
        top_k=5,
        filters=RetrievalFilters(category=["billing-faq"]),
    )
    source_results = retriever.retrieve(
        "password billing api",
        top_k=5,
        filters=RetrievalFilters(source_path=["api.md"]),
    )

    assert [result.category for result in category_results] == ["billing-faq"]
    assert [result.source_path for result in source_results] == ["api.md"]


def test_dense_retriever_filters_by_category_and_source_path(tmp_path: Path, monkeypatch) -> None:
    """Assert DenseRetriever applies metadata filters after vector search."""
    monkeypatch.setattr(
        "app.retrieval.dense._load_sentence_transformer",
        lambda model_name: FakeSentenceTransformer(model_name),
    )
    chunks_path = tmp_path / "chunks.jsonl"
    index_dir = tmp_path / "dense"
    _write_chunks(chunks_path)

    retriever = DenseRetriever(model_name="fake-model")
    retriever.build_index(chunks_path=chunks_path, index_dir=index_dir)

    category_results = retriever.retrieve(
        "password billing api",
        top_k=5,
        filters=RetrievalFilters(category=["billing-faq"]),
    )
    source_results = retriever.retrieve(
        "password billing api",
        top_k=5,
        filters=RetrievalFilters(source_path=["api.md"]),
    )

    assert [result.category for result in category_results] == ["billing-faq"]
    assert [result.source_path for result in source_results] == ["api.md"]
