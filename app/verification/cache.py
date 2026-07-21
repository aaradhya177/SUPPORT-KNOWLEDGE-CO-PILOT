"""Persistent cache for citation judge results."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.verification.schemas import CitationVerdict, Verdict


@dataclass(frozen=True)
class CachedJudgeResult:
    """Cached citation judge result."""

    verdict: Verdict
    reasoning: str
    created_at: str
    model_name: str
    prompt_version: str


class JudgeCache:
    """SQLite-backed cache for citation verification judge calls."""

    def __init__(self, db_path: str | Path) -> None:
        """Initialize the cache and ensure its table exists."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def get(self, cache_key: str) -> CachedJudgeResult | None:
        """Return a cached judge result by stable cache key."""
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT verdict, reasoning, created_at, model_name, prompt_version
                FROM judge_cache
                WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()

        if row is None:
            return None

        verdict, reasoning, created_at, model_name, prompt_version = row
        return CachedJudgeResult(
            verdict=Verdict(str(verdict)),
            reasoning=str(reasoning),
            created_at=str(created_at),
            model_name=str(model_name),
            prompt_version=str(prompt_version),
        )

    def set(
        self,
        cache_key: str,
        verdict: CitationVerdict,
        model_name: str,
        prompt_version: str,
    ) -> None:
        """Store a judge result in the cache."""
        created_at = datetime.now(UTC).isoformat()
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO judge_cache (
                    cache_key,
                    verdict,
                    reasoning,
                    created_at,
                    model_name,
                    prompt_version
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    verdict.verdict.value,
                    verdict.judge_reasoning,
                    created_at,
                    model_name,
                    prompt_version,
                ),
            )

    def _initialize(self) -> None:
        """Create the cache table if needed."""
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS judge_cache (
                    cache_key TEXT PRIMARY KEY,
                    verdict TEXT NOT NULL,
                    reasoning TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    prompt_version TEXT NOT NULL
                )
                """)


def build_judge_cache_key(
    claim_excerpt: str,
    chunk_id: str,
    source_text: str,
) -> str:
    """Build a stable cache key for a claim-source verification input."""
    payload = "\n".join(
        [
            claim_excerpt.strip(),
            chunk_id.strip(),
            source_text.strip(),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
