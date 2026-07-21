"""SQLite-backed answer feedback storage."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class FeedbackRecord:
    """Persisted user feedback for an answer."""

    feedback_id: str
    query: str
    answer: str
    status: str
    confidence: float
    rating: str
    comment: str | None
    citation_chunk_ids: list[str]
    created_at: str


class FeedbackStore:
    """Small SQLite storage layer for answer feedback."""

    def __init__(self, db_path: Path) -> None:
        """Initialize the store with a SQLite database path."""
        self.db_path = db_path

    def create_feedback(
        self,
        query: str,
        answer: str,
        status: str,
        confidence: float,
        rating: str,
        comment: str | None,
        citation_chunk_ids: list[str],
    ) -> FeedbackRecord:
        """Persist and return a feedback record."""
        record = FeedbackRecord(
            feedback_id=str(uuid4()),
            query=query,
            answer=answer,
            status=status,
            confidence=confidence,
            rating=rating,
            comment=comment,
            citation_chunk_ids=citation_chunk_ids,
            created_at=_utc_now_iso(),
        )
        self._ensure_schema()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO feedback (
                    feedback_id, query, answer, status, confidence, rating,
                    comment, citation_chunk_ids, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.feedback_id,
                    record.query,
                    record.answer,
                    record.status,
                    record.confidence,
                    record.rating,
                    record.comment,
                    json.dumps(record.citation_chunk_ids),
                    record.created_at,
                ),
            )
        return record

    def get_feedback(self, feedback_id: str) -> FeedbackRecord | None:
        """Return feedback by ID."""
        self._ensure_schema()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM feedback WHERE feedback_id = ?",
                (feedback_id,),
            ).fetchone()
        return _record_from_row(row) if row is not None else None

    def _ensure_schema(self) -> None:
        """Create the feedback table when missing."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    rating TEXT NOT NULL,
                    comment TEXT,
                    citation_chunk_ids TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """)

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with row dictionaries enabled."""
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection


def _record_from_row(row: sqlite3.Row) -> FeedbackRecord:
    """Convert a SQLite row to a feedback record."""
    return FeedbackRecord(
        feedback_id=str(row["feedback_id"]),
        query=str(row["query"]),
        answer=str(row["answer"]),
        status=str(row["status"]),
        confidence=float(row["confidence"]),
        rating=str(row["rating"]),
        comment=str(row["comment"]) if row["comment"] is not None else None,
        citation_chunk_ids=[str(chunk_id) for chunk_id in json.loads(row["citation_chunk_ids"])],
        created_at=str(row["created_at"]),
    )


def _utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC).isoformat()
