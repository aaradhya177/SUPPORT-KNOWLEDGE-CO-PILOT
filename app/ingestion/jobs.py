"""SQLite-backed ingestion job tracking."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class IngestionJob:
    """Persisted ingestion job state."""

    job_id: str
    status: str
    files_processed: int
    chunks_created: int
    started_at: str
    completed_at: str | None
    error_message: str | None


class IngestionJobStore:
    """Small SQLite storage layer for ingestion jobs."""

    def __init__(self, db_path: Path) -> None:
        """Initialize the store with a SQLite database path."""
        self.db_path = db_path

    def create_job(self) -> IngestionJob:
        """Create and persist a running ingestion job."""
        job = IngestionJob(
            job_id=str(uuid4()),
            status="running",
            files_processed=0,
            chunks_created=0,
            started_at=_utc_now_iso(),
            completed_at=None,
            error_message=None,
        )
        self._ensure_schema()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingestion_jobs (
                    job_id, status, files_processed, chunks_created,
                    started_at, completed_at, error_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    job.status,
                    job.files_processed,
                    job.chunks_created,
                    job.started_at,
                    job.completed_at,
                    job.error_message,
                ),
            )
        return job

    def update_job(
        self,
        job_id: str,
        status: str,
        files_processed: int,
        chunks_created: int,
        error_message: str | None = None,
    ) -> IngestionJob:
        """Update a job with terminal status and counters."""
        completed_at = _utc_now_iso() if status in {"completed", "failed"} else None
        self._ensure_schema()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE ingestion_jobs
                SET status = ?,
                    files_processed = ?,
                    chunks_created = ?,
                    completed_at = ?,
                    error_message = ?
                WHERE job_id = ?
                """,
                (
                    status,
                    files_processed,
                    chunks_created,
                    completed_at,
                    error_message,
                    job_id,
                ),
            )
        job = self.get_job(job_id)
        if job is None:
            raise ValueError(f"Ingestion job not found: {job_id}")
        return job

    def get_job(self, job_id: str) -> IngestionJob | None:
        """Return a job by ID."""
        self._ensure_schema()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM ingestion_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return _job_from_row(row) if row is not None else None

    def get_latest_job(self) -> IngestionJob | None:
        """Return the most recently started ingestion job."""
        self._ensure_schema()
        with self._connect() as connection:
            row = connection.execute("""
                SELECT * FROM ingestion_jobs
                ORDER BY started_at DESC, rowid DESC
                LIMIT 1
                """).fetchone()
        return _job_from_row(row) if row is not None else None

    def _ensure_schema(self) -> None:
        """Create the ingestion jobs table when missing."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    files_processed INTEGER NOT NULL DEFAULT 0,
                    chunks_created INTEGER NOT NULL DEFAULT 0,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    error_message TEXT
                )
                """)

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with row dictionaries enabled."""
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection


def build_idle_job() -> IngestionJob:
    """Return an API-friendly idle job when no persisted job exists."""
    return IngestionJob(
        job_id="",
        status="idle",
        files_processed=0,
        chunks_created=0,
        started_at=_utc_now_iso(),
        completed_at=None,
        error_message=None,
    )


def _job_from_row(row: sqlite3.Row) -> IngestionJob:
    """Convert a SQLite row to an ingestion job."""
    return IngestionJob(
        job_id=str(row["job_id"]),
        status=str(row["status"]),
        files_processed=int(row["files_processed"]),
        chunks_created=int(row["chunks_created"]),
        started_at=str(row["started_at"]),
        completed_at=str(row["completed_at"]) if row["completed_at"] is not None else None,
        error_message=str(row["error_message"]) if row["error_message"] is not None else None,
    )


def _utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC).isoformat()
