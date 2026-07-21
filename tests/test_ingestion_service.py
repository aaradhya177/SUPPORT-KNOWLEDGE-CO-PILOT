"""Tests for ingestion service orchestration."""

from pathlib import Path

from app.ingestion.jobs import IngestionJobStore
from app.ingestion.service import IngestionService, count_supported_files


class FakeIndexBuilder:
    """Fake dense/sparse index builder."""

    def __init__(self, calls: list[tuple[Path, Path]]) -> None:
        """Initialize with a shared call log."""
        self.calls = calls

    def build_index(self, chunks_path: Path, index_dir: Path) -> None:
        """Record build arguments."""
        self.calls.append((chunks_path, index_dir))


def _service(
    tmp_path: Path,
    ingest_func=None,
    count_files_func=None,
    clear_caches_func=None,
    index_calls: list[tuple[Path, Path]] | None = None,
) -> IngestionService:
    """Build a service with fake dependencies."""
    calls = index_calls if index_calls is not None else []
    return IngestionService(
        job_store=IngestionJobStore(tmp_path / "jobs.sqlite3"),
        raw_dir=tmp_path / "raw",
        processed_path=tmp_path / "processed" / "chunks.jsonl",
        dense_index_path=tmp_path / "indexes" / "dense",
        sparse_index_path=tmp_path / "indexes" / "sparse",
        ingest_func=ingest_func or (lambda raw_dir, processed_path: 5),
        count_files_func=count_files_func or (lambda raw_dir: 3),
        dense_builder_factory=lambda: FakeIndexBuilder(calls),
        sparse_builder_factory=lambda: FakeIndexBuilder(calls),
        clear_caches_func=clear_caches_func or (lambda: None),
    )


def test_start_job_creates_running_job(tmp_path: Path) -> None:
    """Assert start_job creates a persisted running job."""
    service = _service(tmp_path)

    started = service.start_job()

    assert started.should_enqueue is True
    assert started.job.job_id
    assert started.job.status == "running"
    assert service.get_job(started.job.job_id) == started.job


def test_start_job_reuses_existing_running_job(tmp_path: Path) -> None:
    """Assert start_job does not create a second job while one is running."""
    service = _service(tmp_path)

    first = service.start_job()
    second = service.start_job()

    assert first.should_enqueue is True
    assert second.should_enqueue is False
    assert second.job.job_id == first.job.job_id


def test_run_job_completes_and_rebuilds_indexes(tmp_path: Path) -> None:
    """Assert run_job ingests, rebuilds indexes, clears caches, and completes."""
    cleared = []
    index_calls: list[tuple[Path, Path]] = []
    service = _service(
        tmp_path,
        clear_caches_func=lambda: cleared.append(True),
        index_calls=index_calls,
    )
    started = service.start_job()

    service.run_job(started.job.job_id)

    completed = service.get_job(started.job.job_id)
    assert completed is not None
    assert completed.status == "completed"
    assert completed.files_processed == 3
    assert completed.chunks_created == 5
    assert completed.completed_at is not None
    assert completed.error_message is None
    assert cleared == [True]
    assert index_calls == [
        (tmp_path / "processed" / "chunks.jsonl", tmp_path / "indexes" / "dense"),
        (tmp_path / "processed" / "chunks.jsonl", tmp_path / "indexes" / "sparse"),
    ]


def test_run_job_persists_failure_details(tmp_path: Path) -> None:
    """Assert run_job records failed status and error message."""

    def fail_ingestion(raw_dir: Path, processed_path: Path) -> int:
        raise RuntimeError("boom")

    service = _service(
        tmp_path,
        ingest_func=fail_ingestion,
        count_files_func=lambda raw_dir: 7,
    )
    started = service.start_job()

    service.run_job(started.job.job_id)

    failed = service.get_job(started.job.job_id)
    assert failed is not None
    assert failed.status == "failed"
    assert failed.files_processed == 7
    assert failed.chunks_created == 0
    assert failed.completed_at is not None
    assert failed.error_message == "boom"


def test_count_supported_files_counts_only_supported_extensions(tmp_path: Path) -> None:
    """Assert source file counting follows supported loader extensions."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "doc.md").write_text("ok", encoding="utf-8")
    (raw_dir / "doc.pdf").write_text("ok", encoding="utf-8")
    (raw_dir / "ignore.csv").write_text("no", encoding="utf-8")

    assert count_supported_files(raw_dir) == 2
