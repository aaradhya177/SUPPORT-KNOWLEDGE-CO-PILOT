"""Directory-level ingestion pipeline for support knowledge sources."""

from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings
from app.ingestion.chunker import Chunk, chunk_text, make_doc_id
from app.ingestion.loaders import SUPPORTED_EXTENSIONS, load_document
from app.retrieval.base import infer_category
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _iter_source_files(raw_dir: Path) -> list[Path]:
    """Return source files under a raw data directory in deterministic order.

    Args:
        raw_dir: Directory containing raw support documents.

    Returns:
        Sorted file paths discovered recursively.
    """
    return sorted(path for path in raw_dir.rglob("*") if path.is_file())


def ingest_directory(raw_dir: Path, processed_path: Path) -> int:
    """Load, chunk, and write supported documents from a directory to JSONL.

    Args:
        raw_dir: Directory containing raw input documents.
        processed_path: JSONL output path.

    Returns:
        Number of chunks written.
    """
    settings = get_settings()
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    all_chunks: list[Chunk] = []

    for source_file in _iter_source_files(raw_dir):
        if source_file.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.warning("Skipping unsupported file type: %s", source_file)
            continue

        logger.info("Ingesting source document: %s", source_file)
        text = load_document(source_file)
        if not text.strip():
            logger.warning("Skipping empty document after loading: %s", source_file)
            continue

        doc_id = make_doc_id(source_file)
        chunks = chunk_text(
            text=text,
            doc_id=doc_id,
            source_path=str(source_file),
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            category=infer_category(source_file),
            updated_at=_file_updated_at(source_file),
        )
        all_chunks.extend(chunks)

    with processed_path.open("w", encoding="utf-8") as output_file:
        for chunk in all_chunks:
            output_file.write(chunk.model_dump_json() + "\n")

    logger.info("Wrote %s chunks to %s", len(all_chunks), processed_path)
    return len(all_chunks)


def _file_updated_at(path: Path) -> str:
    """Return the source file modification time as an ISO UTC timestamp."""
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
