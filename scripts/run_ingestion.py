"""Command-line entry point for document ingestion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.pipeline import ingest_directory


def _build_parser() -> argparse.ArgumentParser:
    """Build the ingestion command-line parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Ingest raw support documents into JSONL chunks.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw source documents.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/chunks.jsonl"),
        help="Output JSONL path for processed chunks.",
    )
    return parser


def _summarize_output(output_path: Path) -> tuple[int, int, float]:
    """Summarize the processed JSONL file.

    Args:
        output_path: Path to the chunks JSONL file.

    Returns:
        Tuple of files processed, total chunks, and average tokens per chunk.
    """
    source_paths: set[str] = set()
    token_counts: list[int] = []

    if not output_path.exists():
        return 0, 0, 0.0

    with output_path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            payload = json.loads(line)
            source_paths.add(str(payload["source_path"]))
            token_counts.append(int(payload["token_count"]))

    total_chunks = len(token_counts)
    avg_tokens = sum(token_counts) / total_chunks if total_chunks else 0.0
    return len(source_paths), total_chunks, avg_tokens


def main() -> None:
    """Run document ingestion from the command line."""
    parser = _build_parser()
    args = parser.parse_args()

    chunk_count = ingest_directory(raw_dir=args.raw_dir, processed_path=args.output)
    files_processed, total_chunks, avg_tokens = _summarize_output(args.output)

    print(
        "Ingestion complete: "
        f"files_processed={files_processed}, "
        f"total_chunks={total_chunks}, "
        f"avg_tokens_per_chunk={avg_tokens:.2f}"
    )

    if chunk_count != total_chunks:
        raise RuntimeError(
            "Ingestion summary mismatch between pipeline return value and JSONL output."
        )


if __name__ == "__main__":
    main()
