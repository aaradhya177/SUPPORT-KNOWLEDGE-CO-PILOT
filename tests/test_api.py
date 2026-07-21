"""FastAPI route tests with mocked dependencies."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import clear_rate_limit_state, get_pipeline
from app.api.routes import eval as eval_route
from app.api.routes import ingest as ingest_route
from app.config import get_settings
from app.feedback.store import FeedbackStore
from app.ingestion.jobs import IngestionJob, IngestionJobStore
from app.ingestion.service import IngestionService
from app.main import app
from app.retrieval.base import RetrievalFilters

USER_HEADERS = {"X-API-Key": "test-user-key"}
ADMIN_HEADERS = {"X-API-Key": "test-admin-key"}


@pytest.fixture(autouse=True)
def configure_auth(monkeypatch, tmp_path):
    """Configure deterministic API keys for route tests."""
    monkeypatch.setenv("SUPPORT_COPILOT_API_KEY", "test-user-key")
    monkeypatch.setenv("SUPPORT_COPILOT_ADMIN_API_KEY", "test-admin-key")
    monkeypatch.setenv("QUERY_RATE_LIMIT_REQUESTS", "100")
    monkeypatch.setenv("QUERY_RATE_LIMIT_WINDOW_SECONDS", "60")
    monkeypatch.setenv("INGESTION_JOBS_DB_PATH", str(tmp_path / "ingestion_jobs.sqlite3"))
    monkeypatch.setenv("FEEDBACK_DB_PATH", str(tmp_path / "feedback.sqlite3"))
    get_settings.cache_clear()
    clear_rate_limit_state()
    yield
    app.dependency_overrides.clear()
    clear_rate_limit_state()
    get_settings.cache_clear()


class FakePipeline:
    """Fake RAG pipeline for API tests."""

    def answer_query(
        self,
        query: str,
        top_k: int = 5,
        request_id: str | None = None,
        filters: RetrievalFilters | None = None,
    ) -> dict[str, object]:
        """Return a deterministic answer payload."""
        return {
            "answer": f"Answer for {query}",
            "citations": [
                {
                    "chunk_id": "chunk_0",
                    "doc_id": "doc",
                    "source_path": "doc.md",
                    "quoted_text": "source",
                }
            ],
            "confidence": 0.91,
            "status": "answered",
        }

    def stream_answer_query(
        self,
        query: str,
        top_k: int = 5,
        request_id: str | None = None,
        filters: RetrievalFilters | None = None,
    ):
        """Return deterministic streaming events."""
        result = self.answer_query(
            query=query,
            top_k=top_k,
            request_id=request_id,
            filters=filters,
        )
        yield {"event": "retrieval_started", "request_id": request_id}
        yield {
            "event": "retrieval_completed",
            "request_id": request_id,
            "retrieved_chunks_count": 1,
            "retrieval_latency_ms": 1.0,
        }
        yield {"event": "generation_started", "request_id": request_id}
        yield {"event": "verification_started", "request_id": request_id}
        yield {
            "event": "completed",
            "request_id": request_id,
            "result": result,
            "retrieval_latency_ms": 1.0,
            "generation_latency_ms": 2.0,
            "verification_latency_ms": 3.0,
            "total_latency_ms": 6.0,
        }


class ErrorStreamingPipeline(FakePipeline):
    """Fake pipeline that emits a stream error event."""

    def stream_answer_query(
        self,
        query: str,
        top_k: int = 5,
        request_id: str | None = None,
        filters: RetrievalFilters | None = None,
    ):
        """Return a deterministic error event."""
        yield {
            "event": "error",
            "request_id": request_id,
            "message": "Pipeline query failed.",
            "error_type": "RuntimeError",
        }


class CapturingPipeline(FakePipeline):
    """Fake pipeline that records query filters."""

    def __init__(self) -> None:
        """Initialize empty call state."""
        self.filters: RetrievalFilters | None = None

    def answer_query(
        self,
        query: str,
        top_k: int = 5,
        request_id: str | None = None,
        filters: RetrievalFilters | None = None,
    ) -> dict[str, object]:
        """Record filters and return a deterministic answer."""
        self.filters = filters
        return super().answer_query(query, top_k, request_id, filters)


class FakeIndexBuilder:
    """Fake dense/sparse index builder for ingestion route tests."""

    def build_index(self, **kwargs) -> None:
        """Accept build calls without doing expensive model work."""


class FakeIngestionService:
    """Minimal fake service for route scheduling tests."""

    def __init__(self) -> None:
        """Initialize a fake running job."""
        self.job = IngestionJob(
            job_id="fake-job",
            status="running",
            files_processed=0,
            chunks_created=0,
            started_at="2026-01-01T00:00:00+00:00",
            completed_at=None,
            error_message=None,
        )

    def start_job(self):
        """Return a fake start result."""
        from app.ingestion.service import IngestionStart

        return IngestionStart(job=self.job, should_enqueue=True)

    def run_job(self, job_id: str) -> None:
        """Accept a background run call."""


def _service(
    tmp_path: Path,
    ingest_func=None,
    count_files_func=None,
    clear_caches_func=None,
) -> IngestionService:
    """Build a test ingestion service with fake expensive dependencies."""
    return IngestionService(
        job_store=IngestionJobStore(tmp_path / "ingestion_service.sqlite3"),
        raw_dir=tmp_path / "raw",
        processed_path=tmp_path / "processed" / "chunks.jsonl",
        dense_index_path=tmp_path / "indexes" / "dense",
        sparse_index_path=tmp_path / "indexes" / "sparse",
        ingest_func=ingest_func or (lambda raw_dir, processed_path: 4),
        count_files_func=count_files_func or (lambda raw_dir: 2),
        dense_builder_factory=FakeIndexBuilder,
        sparse_builder_factory=FakeIndexBuilder,
        clear_caches_func=clear_caches_func or (lambda: None),
    )


def test_query_route_returns_pipeline_response() -> None:
    """Assert POST /api/v1/query returns a structured answer."""
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
    client = TestClient(app)

    response = client.post(
        "/api/v1/query",
        json={"query": "How do I reset my password?", "top_k": 3},
        headers=USER_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "answered"
    assert payload["confidence"] == 0.91
    assert payload["citations"][0]["chunk_id"] == "chunk_0"
    assert payload["request_id"]
    assert response.headers["X-Request-ID"] == payload["request_id"]


def test_query_route_uses_supplied_request_id() -> None:
    """Assert request IDs are echoed in headers and response bodies."""
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
    client = TestClient(app)

    response = client.post(
        "/api/v1/query",
        json={"query": "How do I reset my password?", "top_k": 3},
        headers={**USER_HEADERS, "X-Request-ID": "interview-request-123"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "interview-request-123"
    assert response.json()["request_id"] == "interview-request-123"


def test_query_route_generates_request_id_when_missing() -> None:
    """Assert missing request IDs are generated by middleware."""
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
    client = TestClient(app)

    response = client.post(
        "/api/v1/query",
        json={"query": "How do I reset my password?", "top_k": 3},
        headers=USER_HEADERS,
    )

    request_id = response.json()["request_id"]
    assert response.status_code == 200
    assert request_id
    assert response.headers["X-Request-ID"] == request_id


def test_stream_query_route_returns_sse_event_shape() -> None:
    """Assert POST /query/stream returns ordered Server-Sent Events."""
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
    client = TestClient(app)

    response = client.post(
        "/api/v1/query/stream",
        json={"query": "How do I reset my password?", "top_k": 3},
        headers={**USER_HEADERS, "X-Request-ID": "stream-request-123"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["X-Request-ID"] == "stream-request-123"

    events = _parse_sse_events(response.text)
    assert [event["event"] for event in events] == [
        "retrieval_started",
        "retrieval_completed",
        "generation_started",
        "verification_started",
        "completed",
    ]
    assert all(event["data"]["request_id"] == "stream-request-123" for event in events)
    assert events[-1]["data"]["result"]["status"] == "answered"
    assert events[-1]["data"]["result"]["answer"] == "Answer for How do I reset my password?"


def test_stream_query_route_returns_error_event_shape() -> None:
    """Assert stream failures are sent as structured SSE error events."""
    app.dependency_overrides[get_pipeline] = lambda: ErrorStreamingPipeline()
    client = TestClient(app)

    response = client.post(
        "/api/v1/query/stream",
        json={"query": "How do I reset my password?", "top_k": 3},
        headers=USER_HEADERS,
    )

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert [event["event"] for event in events] == ["error"]
    assert events[0]["data"]["message"] == "Pipeline query failed."
    assert events[0]["data"]["error_type"] == "RuntimeError"


def test_query_route_passes_filters_to_pipeline() -> None:
    """Assert query metadata filters are validated and passed to the pipeline."""
    pipeline = CapturingPipeline()
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    client = TestClient(app)

    response = client.post(
        "/api/v1/query",
        json={
            "query": "How do I reset my password?",
            "top_k": 3,
            "filters": {
                "category": ["password-reset"],
                "source_path": ["password-reset.md"],
            },
        },
        headers=USER_HEADERS,
    )

    assert response.status_code == 200
    assert pipeline.filters is not None
    assert pipeline.filters.category == ["password-reset"]
    assert pipeline.filters.source_path == ["password-reset.md"]


def test_query_route_rejects_empty_query() -> None:
    """Assert empty queries receive HTTP 400."""
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
    client = TestClient(app)

    response = client.post(
        "/api/v1/query",
        json={"query": "   ", "top_k": 5},
        headers=USER_HEADERS,
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Query cannot be empty."


def test_query_route_rejects_missing_api_key() -> None:
    """Assert protected query route requires an API key."""
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
    client = TestClient(app)

    response = client.post(
        "/api/v1/query", json={"query": "How do I reset my password?", "top_k": 3}
    )

    assert response.status_code == 401
    assert "X-API-Key" in response.json()["detail"]


def test_query_route_rejects_invalid_api_key() -> None:
    """Assert protected query route rejects invalid API keys."""
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
    client = TestClient(app)

    response = client.post(
        "/api/v1/query",
        json={"query": "How do I reset my password?", "top_k": 3},
        headers={"X-API-Key": "wrong-key"},
    )

    assert response.status_code == 401


def test_query_route_accepts_admin_api_key() -> None:
    """Assert admin API keys can call normal protected endpoints."""
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
    client = TestClient(app)

    response = client.post(
        "/api/v1/query",
        json={"query": "How do I reset my password?", "top_k": 3},
        headers=ADMIN_HEADERS,
    )

    assert response.status_code == 200


def _parse_sse_events(text: str) -> list[dict[str, object]]:
    """Parse the small SSE subset emitted by the query stream endpoint."""
    events: list[dict[str, object]] = []
    for block in text.strip().split("\n\n"):
        event_name = ""
        data = {}
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ").strip()
            elif line.startswith("data: "):
                data = json.loads(line.removeprefix("data: ").strip())
        if event_name:
            events.append({"event": event_name, "data": data})
    return events


def test_query_route_allows_requests_under_rate_limit(monkeypatch) -> None:
    """Assert query requests are allowed while below the configured limit."""
    monkeypatch.setenv("QUERY_RATE_LIMIT_REQUESTS", "2")
    monkeypatch.setenv("QUERY_RATE_LIMIT_WINDOW_SECONDS", "60")
    get_settings.cache_clear()
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
    client = TestClient(app)

    first = client.post(
        "/api/v1/query",
        json={"query": "How do I reset my password?", "top_k": 3},
        headers=USER_HEADERS,
    )
    second = client.post(
        "/api/v1/query",
        json={"query": "How do I reset my password?", "top_k": 3},
        headers=USER_HEADERS,
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.headers["X-RateLimit-Limit"] == "2"
    assert first.headers["X-RateLimit-Window"] == "60"


def test_query_route_blocks_requests_over_rate_limit(monkeypatch) -> None:
    """Assert query requests over the configured limit return HTTP 429."""
    monkeypatch.setenv("QUERY_RATE_LIMIT_REQUESTS", "1")
    monkeypatch.setenv("QUERY_RATE_LIMIT_WINDOW_SECONDS", "60")
    get_settings.cache_clear()
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
    client = TestClient(app)

    allowed = client.post(
        "/api/v1/query",
        json={"query": "How do I reset my password?", "top_k": 3},
        headers=USER_HEADERS,
    )
    blocked = client.post(
        "/api/v1/query",
        json={"query": "How do I reset my password?", "top_k": 3},
        headers=USER_HEADERS,
    )

    assert allowed.status_code == 200
    assert blocked.status_code == 429
    assert "Query rate limit exceeded" in blocked.json()["detail"]
    assert blocked.headers["Retry-After"]


def test_ingest_route_creates_persistent_job(monkeypatch) -> None:
    """Assert POST /ingest creates and returns a persisted job ID."""
    monkeypatch.setattr(ingest_route, "_get_ingestion_service", FakeIngestionService)
    client = TestClient(app)

    response = client.post("/api/v1/ingest", headers=ADMIN_HEADERS)

    assert response.status_code == 202
    payload = response.json()
    assert payload["job_id"]
    assert payload["status"] == "running"
    assert payload["files_processed"] == 0
    assert payload["chunks_created"] == 0
    assert payload["started_at"]
    assert payload["completed_at"] is None
    assert payload["error_message"] is None


def test_ingest_routes_report_success_and_lookup_by_job_id(monkeypatch, tmp_path) -> None:
    """Assert successful ingestion updates latest status and job lookup."""
    service = _service(tmp_path)
    monkeypatch.setattr(ingest_route, "_get_ingestion_service", lambda: service)
    client = TestClient(app)

    start_response = client.post("/api/v1/ingest", headers=ADMIN_HEADERS)
    job_id = start_response.json()["job_id"]
    status_response = client.get("/api/v1/ingest/status", headers=ADMIN_HEADERS)
    lookup_response = client.get(f"/api/v1/ingest/status/{job_id}", headers=ADMIN_HEADERS)

    assert start_response.status_code == 202
    assert status_response.status_code == 200
    assert lookup_response.status_code == 200
    payload = status_response.json()
    assert payload["job_id"] == job_id
    assert payload["files_processed"] == 2
    assert payload["chunks_created"] == 4
    assert payload["status"] == "completed"
    assert payload["completed_at"] is not None
    assert payload["error_message"] is None
    assert lookup_response.json() == payload


def test_ingest_routes_report_failure(monkeypatch, tmp_path) -> None:
    """Assert failed ingestion jobs persist failure details."""

    def fail_ingestion(raw_dir, processed_path) -> int:
        raise RuntimeError("synthetic ingestion failure")

    service = _service(
        tmp_path,
        ingest_func=fail_ingestion,
        count_files_func=lambda raw_dir: 3,
    )
    monkeypatch.setattr(ingest_route, "_get_ingestion_service", lambda: service)
    client = TestClient(app)

    start_response = client.post("/api/v1/ingest", headers=ADMIN_HEADERS)
    status_response = client.get("/api/v1/ingest/status", headers=ADMIN_HEADERS)

    assert start_response.status_code == 202
    payload = status_response.json()
    assert payload["job_id"] == start_response.json()["job_id"]
    assert payload["status"] == "failed"
    assert payload["files_processed"] == 3
    assert payload["chunks_created"] == 0
    assert "synthetic ingestion failure" in payload["error_message"]
    assert payload["completed_at"] is not None


def test_ingest_status_lookup_returns_404_for_unknown_job() -> None:
    """Assert unknown ingestion job IDs return 404."""
    client = TestClient(app)

    response = client.get("/api/v1/ingest/status/missing-job", headers=ADMIN_HEADERS)

    assert response.status_code == 404
    assert "missing-job" in response.json()["detail"]


def test_admin_routes_are_separate_from_query_rate_limit(monkeypatch) -> None:
    """Assert admin-only routes are not blocked by query rate limits."""

    monkeypatch.setenv("QUERY_RATE_LIMIT_REQUESTS", "1")
    monkeypatch.setenv("QUERY_RATE_LIMIT_WINDOW_SECONDS", "60")
    get_settings.cache_clear()
    monkeypatch.setattr(ingest_route, "_get_ingestion_service", FakeIngestionService)
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
    client = TestClient(app)

    client.post(
        "/api/v1/query",
        json={"query": "How do I reset my password?", "top_k": 3},
        headers=USER_HEADERS,
    )
    blocked_query = client.post(
        "/api/v1/query",
        json={"query": "How do I reset my password?", "top_k": 3},
        headers=USER_HEADERS,
    )
    ingest_response = client.post("/api/v1/ingest", headers=ADMIN_HEADERS)

    assert blocked_query.status_code == 429
    assert ingest_response.status_code == 202


def test_admin_routes_reject_user_api_key() -> None:
    """Assert normal user keys cannot access admin-only routes."""
    client = TestClient(app)

    ingest_response = client.post("/api/v1/ingest", headers=USER_HEADERS)
    status_response = client.get("/api/v1/ingest/status", headers=USER_HEADERS)
    eval_response = client.post("/api/v1/eval/run", json={"sample_size": 5}, headers=USER_HEADERS)

    assert ingest_response.status_code == 403
    assert status_response.status_code == 403
    assert eval_response.status_code == 403


def test_admin_routes_reject_missing_api_key() -> None:
    """Assert admin-only routes require an API key."""
    client = TestClient(app)

    response = client.post("/api/v1/ingest")

    assert response.status_code == 401


def test_feedback_route_persists_submission(tmp_path, monkeypatch) -> None:
    """Assert POST /feedback validates and persists answer feedback."""
    feedback_db_path = tmp_path / "feedback.sqlite3"
    monkeypatch.setenv("FEEDBACK_DB_PATH", str(feedback_db_path))
    get_settings.cache_clear()
    client = TestClient(app)

    response = client.post(
        "/api/v1/feedback",
        headers=USER_HEADERS,
        json={
            "query": "How do I reset my password?",
            "answer": "Use Forgot password.",
            "status": "answered",
            "confidence": 0.91,
            "rating": "up",
            "comment": "Helpful",
            "citation_chunk_ids": ["chunk_0", "chunk_1"],
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["feedback_id"]
    assert payload["status"] == "recorded"
    assert payload["created_at"]

    stored = FeedbackStore(feedback_db_path).get_feedback(payload["feedback_id"])
    assert stored is not None
    assert stored.query == "How do I reset my password?"
    assert stored.answer == "Use Forgot password."
    assert stored.status == "answered"
    assert stored.confidence == 0.91
    assert stored.rating == "up"
    assert stored.comment == "Helpful"
    assert stored.citation_chunk_ids == ["chunk_0", "chunk_1"]


def test_feedback_route_rejects_invalid_rating() -> None:
    """Assert feedback rating validation rejects unsupported values."""
    client = TestClient(app)

    response = client.post(
        "/api/v1/feedback",
        headers=USER_HEADERS,
        json={
            "query": "How do I reset my password?",
            "answer": "Use Forgot password.",
            "status": "answered",
            "confidence": 0.91,
            "rating": "maybe",
            "citation_chunk_ids": [],
        },
    )

    assert response.status_code == 422


def test_feedback_route_requires_api_key() -> None:
    """Assert feedback submission is protected by user API key auth."""
    client = TestClient(app)

    response = client.post(
        "/api/v1/feedback",
        json={
            "query": "How do I reset my password?",
            "answer": "Use Forgot password.",
            "status": "answered",
            "confidence": 0.91,
            "rating": "down",
            "citation_chunk_ids": [],
        },
    )

    assert response.status_code == 401


def test_eval_route_returns_summary(monkeypatch) -> None:
    """Assert evaluation route returns aggregate metrics and report path."""

    def fake_run_golden_eval(**kwargs) -> dict[str, float]:
        return {
            "retrieval_hit_rate": 0.8,
            "avg_answer_correctness": 0.7,
            "avg_citation_faithfulness": 0.9,
            "no_answer_precision": 1.0,
            "no_answer_recall": 0.5,
        }

    monkeypatch.setattr(eval_route, "run_golden_eval", fake_run_golden_eval)
    client = TestClient(app)

    response = client.post("/api/v1/eval/run", json={"sample_size": 5}, headers=ADMIN_HEADERS)

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["retrieval_hit_rate"] == 0.8
    assert (
        payload["report_path"] == "reports\\eval_summary.md"
        or payload["report_path"] == "reports/eval_summary.md"
    )
