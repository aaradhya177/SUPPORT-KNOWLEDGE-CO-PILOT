"""FastAPI route tests with mocked dependencies."""

from fastapi.testclient import TestClient

from app.api.dependencies import get_pipeline
from app.api.routes import eval as eval_route
from app.api.routes import ingest as ingest_route
from app.main import app


class FakePipeline:
    """Fake RAG pipeline for API tests."""

    def answer_query(self, query: str, top_k: int = 5) -> dict[str, object]:
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


def test_query_route_returns_pipeline_response() -> None:
    """Assert POST /api/v1/query returns a structured answer."""
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
    client = TestClient(app)

    response = client.post(
        "/api/v1/query", json={"query": "How do I reset my password?", "top_k": 3}
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "answered"
    assert payload["confidence"] == 0.91
    assert payload["citations"][0]["chunk_id"] == "chunk_0"


def test_query_route_rejects_empty_query() -> None:
    """Assert empty queries receive HTTP 400."""
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
    client = TestClient(app)

    response = client.post("/api/v1/query", json={"query": "   ", "top_k": 5})

    app.dependency_overrides.clear()
    assert response.status_code == 400
    assert response.json()["detail"] == "Query cannot be empty."


def test_ingest_routes_report_background_status(monkeypatch) -> None:
    """Assert ingest start and status routes expose ingestion state."""

    def fake_job() -> None:
        ingest_route._INGEST_STATUS.update(
            {"files_processed": 2, "chunks_created": 4, "status": "completed"}
        )

    monkeypatch.setattr(ingest_route, "_run_ingestion_job", fake_job)
    ingest_route._INGEST_STATUS.update(
        {"files_processed": 0, "chunks_created": 0, "status": "idle"}
    )
    client = TestClient(app)

    start_response = client.post("/api/v1/ingest")
    status_response = client.get("/api/v1/ingest/status")

    assert start_response.status_code == 202
    assert status_response.status_code == 200
    assert status_response.json() == {
        "files_processed": 2,
        "chunks_created": 4,
        "status": "completed",
    }


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

    response = client.post("/api/v1/eval/run", json={"sample_size": 5})

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["retrieval_hit_rate"] == 0.8
    assert (
        payload["report_path"] == "reports\\eval_summary.md"
        or payload["report_path"] == "reports/eval_summary.md"
    )
