"""Build dense and sparse retrieval indexes from processed chunks."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import faiss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.retrieval.dense import FAISS_INDEX_FILE, METADATA_FILE, DenseRetriever
from app.retrieval.sparse import BM25_INDEX_FILE, BM25Retriever


def _build_parser() -> argparse.ArgumentParser:
    """Build the index command-line parser.

    Returns:
        Configured argument parser.
    """
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build dense and sparse retrieval indexes.")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/processed/chunks.jsonl"),
        help="Input chunks JSONL path.",
    )
    parser.add_argument(
        "--dense-dir",
        type=Path,
        default=Path(settings.dense_index_path),
        help="Output directory for dense FAISS artifacts.",
    )
    parser.add_argument(
        "--sparse-dir",
        type=Path,
        default=Path(settings.bm25_index_path),
        help="Output directory for sparse BM25 artifacts.",
    )
    return parser


def _dense_vector_count(index_dir: Path) -> int:
    """Return the number of vectors in a persisted FAISS index.

    Args:
        index_dir: Dense index directory.

    Returns:
        Number of indexed dense vectors.
    """
    index_path = index_dir / FAISS_INDEX_FILE
    if not index_path.exists():
        return 0
    return int(faiss.read_index(str(index_path)).ntotal)


def _metadata_count(metadata_path: Path) -> int:
    """Return the number of metadata records in a pickle artifact.

    Args:
        metadata_path: Pickle path.

    Returns:
        Number of metadata records.
    """
    if not metadata_path.exists():
        return 0
    with metadata_path.open("rb") as input_file:
        payload = pickle.load(input_file)
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        return len(payload.get("metadata", []))
    return 0


def main() -> None:
    """Build both retrieval indexes and print summary statistics."""
    parser = _build_parser()
    args = parser.parse_args()

    dense_retriever = DenseRetriever()
    dense_retriever.build_index(chunks_path=args.chunks, index_dir=args.dense_dir)

    sparse_retriever = BM25Retriever()
    sparse_retriever.build_index(chunks_path=args.chunks, index_dir=args.sparse_dir)

    dense_vectors = _dense_vector_count(args.dense_dir)
    dense_docs = _metadata_count(args.dense_dir / METADATA_FILE)
    sparse_docs = _metadata_count(args.sparse_dir / BM25_INDEX_FILE)

    print(
        "Index build complete: "
        f"dense_vectors={dense_vectors}, "
        f"dense_docs={dense_docs}, "
        f"sparse_docs={sparse_docs}"
    )


if __name__ == "__main__":
    main()
