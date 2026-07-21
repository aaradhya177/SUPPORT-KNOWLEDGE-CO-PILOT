"""Service layer for ingestion job orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.api.dependencies import clear_dependency_caches
from app.config import Settings
from app.ingestion.jobs import IngestionJob, IngestionJobStore, build_idle_job
from app.ingestion.loaders import SUPPORTED_EXTENSIONS
from app.ingestion.pipeline import ingest_directory
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import BM25Retriever
from app.utils.logger import get_logger

logger = get_logger(__name__)


class IndexBuilder(Protocol):
    """Protocol for dense and sparse index builders."""

    def build_index(self, chunks_path: Path, index_dir: Path) -> None:
        """Build an index from processed chunks."""


@dataclass(frozen=True)
class IngestionStart:
    """Result of starting an ingestion job."""

    job: IngestionJob
    should_enqueue: bool


class IngestionService:
    """Coordinate ingestion jobs independent of FastAPI route code."""

    def __init__(
        self,
        job_store: IngestionJobStore,
        raw_dir: Path,
        processed_path: Path,
        dense_index_path: Path,
        sparse_index_path: Path,
        ingest_func: Callable[[Path, Path], int] = ingest_directory,
        count_files_func: Callable[[Path], int] | None = None,
        dense_builder_factory: Callable[[], IndexBuilder] = DenseRetriever,
        sparse_builder_factory: Callable[[], IndexBuilder] = BM25Retriever,
        clear_caches_func: Callable[[], None] = clear_dependency_caches,
    ) -> None:
        """Initialize the ingestion service and its replaceable dependencies."""
        self.job_store = job_store
        self.raw_dir = raw_dir
        self.processed_path = processed_path
        self.dense_index_path = dense_index_path
        self.sparse_index_path = sparse_index_path
        self.ingest_func = ingest_func
        self.count_files_func = count_files_func or count_supported_files
        self.dense_builder_factory = dense_builder_factory
        self.sparse_builder_factory = sparse_builder_factory
        self.clear_caches_func = clear_caches_func

    @classmethod
    def from_settings(cls, settings: Settings) -> IngestionService:
        """Build the service from application settings."""
        return cls(
            job_store=IngestionJobStore(Path(settings.ingestion_jobs_db_path)),
            raw_dir=Path("data/raw"),
            processed_path=Path("data/processed/chunks.jsonl"),
            dense_index_path=Path(settings.dense_index_path),
            sparse_index_path=Path(settings.bm25_index_path),
        )

    def start_job(self) -> IngestionStart:
        """Create a new job unless a job is already running."""
        latest_job = self.job_store.get_latest_job()
        if latest_job is not None and latest_job.status == "running":
            return IngestionStart(job=latest_job, should_enqueue=False)

        job = self.job_store.create_job()
        return IngestionStart(job=job, should_enqueue=True)

    def get_latest_job(self) -> IngestionJob:
        """Return the latest job, or an idle placeholder when none exists."""
        return self.job_store.get_latest_job() or build_idle_job()

    def get_job(self, job_id: str) -> IngestionJob | None:
        """Return a persisted ingestion job by ID."""
        return self.job_store.get_job(job_id)

    def run_job(self, job_id: str) -> None:
        """Run ingestion, rebuild indexes, and update persisted job state."""
        files_processed = 0
        chunks_created = 0

        try:
            files_processed = self.count_files_func(self.raw_dir)
            chunks_created = self.ingest_func(self.raw_dir, self.processed_path)

            dense = self.dense_builder_factory()
            dense.build_index(chunks_path=self.processed_path, index_dir=self.dense_index_path)

            sparse = self.sparse_builder_factory()
            sparse.build_index(chunks_path=self.processed_path, index_dir=self.sparse_index_path)

            self.clear_caches_func()
            self.job_store.update_job(
                job_id=job_id,
                status="completed",
                files_processed=files_processed,
                chunks_created=chunks_created,
            )
        except Exception as exc:
            logger.exception("Ingestion job %s failed.", job_id)
            self.job_store.update_job(
                job_id=job_id,
                status="failed",
                files_processed=files_processed,
                chunks_created=chunks_created,
                error_message=str(exc),
            )


def count_supported_files(raw_dir: Path) -> int:
    """Count supported input files under a raw data directory."""
    if not raw_dir.exists():
        return 0
    return sum(
        1
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
