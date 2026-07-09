"""Manual side-by-side benchmark for dense and sparse retrievers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.retrieval.base import RetrievedChunk
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import BM25Retriever

EXAMPLE_QUERIES: list[str] = [
    "How can a customer reset a password if the email never arrives?",
    "What happens to billing when seats are added in the middle of a cycle?",
    "How should an integration respond to HTTP 429 rate limit errors?",
    "Where can administrators download invoices?",
    "What details should support collect before escalating API throttling?",
    "Can support agents create passwords for customers?",
    "How can customers reduce API request volume?",
]


def _build_parser() -> argparse.ArgumentParser:
    """Build the benchmark command-line parser.

    Returns:
        Configured argument parser.
    """
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Benchmark dense and sparse retrieval manually.")
    parser.add_argument(
        "--dense-dir",
        type=Path,
        default=Path(settings.dense_index_path),
        help="Directory containing dense FAISS artifacts.",
    )
    parser.add_argument(
        "--sparse-dir",
        type=Path,
        default=Path(settings.bm25_index_path),
        help="Directory containing sparse BM25 artifacts.",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of results to show per retriever."
    )
    return parser


def _format_result(result: RetrievedChunk) -> str:
    """Format a retrieved chunk for compact terminal display.

    Args:
        result: Retrieved chunk to format.

    Returns:
        Human-readable one-line result summary.
    """
    section = result.section or "no section"
    preview = " ".join(result.text.split())[:140]
    return f"{result.rank}. {result.chunk_id} | {section} | score={result.score:.4f} | {preview}"


def main() -> None:
    """Load both retrievers and print side-by-side query results."""
    parser = _build_parser()
    args = parser.parse_args()

    dense = DenseRetriever()
    dense.load_index(args.dense_dir)

    sparse = BM25Retriever()
    sparse.load_index(args.sparse_dir)

    for query in EXAMPLE_QUERIES:
        print("\n" + "=" * 100)
        print(f"Query: {query}")

        dense_results = dense.retrieve(query, top_k=args.top_k)
        sparse_results = sparse.retrieve(query, top_k=args.top_k)

        print("\nDense top results:")
        for result in dense_results:
            print(_format_result(result))

        print("\nSparse top results:")
        for result in sparse_results:
            print(_format_result(result))


if __name__ == "__main__":
    main()
