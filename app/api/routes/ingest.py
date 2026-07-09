"""Ingestion API routes."""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, status

from app.api.dependencies import clear_dependency_caches
from app.api.schemas import IngestResponse
from app.config import get_settings
from app.ingestion.loaders import SUPPORTED_EXTENSIONS
from app.ingestion.pipeline import ingest_directory
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import BM25Retriever
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

_INGEST_STATUS: dict[str, Any] = {
    "files_processed": 0,
    "chunks_created": 0,
    "status": "idle",
}


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start ingestion and index rebuild",
    description=(
        "Starts a background job that ingests data/raw documents, writes "
        "data/processed/chunks.jsonl, and rebuilds dense and sparse indexes."
    ),
)
def start_ingestion(background_tasks: BackgroundTasks) -> IngestResponse:
    """Start a background ingestion job."""
    if _INGEST_STATUS["status"] == "running":
        return IngestResponse(**_INGEST_STATUS)

    _INGEST_STATUS.update({"files_processed": 0, "chunks_created": 0, "status": "running"})
    background_tasks.add_task(_run_ingestion_job)
    return IngestResponse(**_INGEST_STATUS)


@router.get(
    "/ingest/status",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Get ingestion status",
    description="Returns the current in-memory status for the latest ingestion job.",
)
def get_ingestion_status() -> IngestResponse:
    """Return the latest ingestion job status."""
    return IngestResponse(**_INGEST_STATUS)


def _run_ingestion_job() -> None:
    """Run ingestion and index rebuilding in a background task."""
    settings = get_settings()
    raw_dir = Path("data/raw")
    processed_path = Path("data/processed/chunks.jsonl")

    try:
        files_processed = _count_supported_files(raw_dir)
        chunks_created = ingest_directory(raw_dir=raw_dir, processed_path=processed_path)

        dense = DenseRetriever()
        dense.build_index(chunks_path=processed_path, index_dir=Path(settings.dense_index_path))

        sparse = BM25Retriever()
        sparse.build_index(chunks_path=processed_path, index_dir=Path(settings.bm25_index_path))

        clear_dependency_caches()
        _INGEST_STATUS.update(
            {
                "files_processed": files_processed,
                "chunks_created": chunks_created,
                "status": "completed",
            }
        )
    except Exception:
        logger.exception("Ingestion background job failed.")
        _INGEST_STATUS.update({"status": "failed"})


def _count_supported_files(raw_dir: Path) -> int:
    """Count supported input files under a raw data directory."""
    if not raw_dir.exists():
        return 0
    return sum(
        1
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
