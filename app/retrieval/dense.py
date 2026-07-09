"""Dense vector retrieval backed by SentenceTransformers and FAISS."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Final

import faiss
import numpy as np

from app.config import get_settings
from app.retrieval.base import BaseRetriever, RetrievedChunk
from app.utils.logger import get_logger

logger = get_logger(__name__)

FAISS_INDEX_FILE: Final[str] = "index.faiss"
METADATA_FILE: Final[str] = "metadata.pkl"


class DenseRetriever(BaseRetriever):
    """Dense semantic retriever using normalized embeddings and inner product search."""

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize the dense retriever.

        Args:
            model_name: Optional SentenceTransformer model name. Uses config when omitted.
        """
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model_name
        self.model = _load_sentence_transformer(self.model_name)
        self.index: faiss.Index | None = None
        self.metadata: list[dict[str, Any]] = []

    def build_index(self, chunks_path: Path, index_dir: Path) -> None:
        """Build and persist a FAISS dense index from JSONL chunks.

        Args:
            chunks_path: Path to ``chunks.jsonl``.
            index_dir: Directory where FAISS and metadata artifacts are saved.
        """
        chunks = _load_chunk_metadata(chunks_path)
        index_dir.mkdir(parents=True, exist_ok=True)

        if not chunks:
            logger.warning("No chunks found at %s; writing empty dense metadata.", chunks_path)
            self.index = None
            self.metadata = []
            _write_metadata(index_dir, self.metadata)
            return

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            logger.warning("Embedding model produced no dense vectors.")
            self.index = None
            self.metadata = []
            _write_metadata(index_dir, self.metadata)
            return

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        faiss.write_index(index, str(index_dir / FAISS_INDEX_FILE))
        _write_metadata(index_dir, chunks)

        self.index = index
        self.metadata = chunks
        logger.info("Built dense index with %s vectors at %s", len(chunks), index_dir)

    def load_index(self, index_dir: Path) -> None:
        """Load a persisted FAISS index and metadata store.

        Args:
            index_dir: Directory containing dense index artifacts.
        """
        metadata_path = index_dir / METADATA_FILE
        faiss_path = index_dir / FAISS_INDEX_FILE

        self.metadata = _read_metadata(metadata_path)
        if not faiss_path.exists():
            logger.warning("Dense FAISS index not found at %s.", faiss_path)
            self.index = None
            return

        self.index = faiss.read_index(str(faiss_path))
        logger.info(
            "Loaded dense index with %s metadata records from %s", len(self.metadata), index_dir
        )

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        """Retrieve dense nearest-neighbor chunks for a query.

        Args:
            query: User query text.
            top_k: Maximum number of results to return.

        Returns:
            Ranked dense retrieval results.
        """
        if not query.strip():
            logger.warning("Empty query received by dense retriever.")
            return []
        if top_k <= 0:
            return []
        if self.index is None or self.index.ntotal == 0 or not self.metadata:
            logger.warning("Dense retriever queried before a non-empty index was loaded.")
            return []

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_vector = np.asarray(query_embedding, dtype=np.float32)
        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))

        results: list[RetrievedChunk] = []
        for rank, (score, index_position) in enumerate(zip(scores[0], indices[0]), start=1):
            if index_position < 0 or index_position >= len(self.metadata):
                continue
            record = self.metadata[int(index_position)]
            results.append(
                RetrievedChunk(
                    chunk_id=str(record["chunk_id"]),
                    doc_id=str(record["doc_id"]),
                    source_path=str(record.get("source_path", "")),
                    text=str(record["text"]),
                    section=record.get("section"),
                    score=float(score),
                    rank=rank,
                    retriever_name="dense",
                    source_retrievers=["dense"],
                )
            )

        return results


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
                    "text": payload["text"],
                    "section": payload.get("section"),
                }
            )
    return records


def _load_sentence_transformer(model_name: str) -> Any:
    """Load a SentenceTransformer model lazily.

    Args:
        model_name: SentenceTransformer model name.

    Raises:
        ImportError: If ``sentence-transformers`` is not installed.

    Returns:
        Loaded embedding model.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for DenseRetriever. "
            "Install project dependencies with `pip install -r requirements.txt`."
        ) from exc

    return SentenceTransformer(model_name)


def _write_metadata(index_dir: Path, metadata: list[dict[str, Any]]) -> None:
    """Persist dense metadata to disk.

    Args:
        index_dir: Output index directory.
        metadata: Chunk metadata records.
    """
    with (index_dir / METADATA_FILE).open("wb") as output_file:
        pickle.dump(metadata, output_file)


def _read_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    """Read dense metadata from disk.

    Args:
        metadata_path: Metadata pickle path.

    Returns:
        Chunk metadata records, or an empty list when missing.
    """
    if not metadata_path.exists():
        logger.warning("Dense metadata not found at %s.", metadata_path)
        return []

    with metadata_path.open("rb") as input_file:
        metadata = pickle.load(input_file)

    if not isinstance(metadata, list):
        raise ValueError(f"Dense metadata at {metadata_path} is not a list.")

    return metadata
