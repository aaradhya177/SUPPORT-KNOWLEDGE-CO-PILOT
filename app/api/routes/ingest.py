"""Ingestion API routes."""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from app.api.dependencies import require_admin_api_key
from app.api.schemas import IngestResponse
from app.config import get_settings
from app.ingestion.jobs import IngestionJob
from app.ingestion.service import IngestionService

router = APIRouter()


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
def start_ingestion(
    background_tasks: BackgroundTasks,
    _: None = Depends(require_admin_api_key),
) -> IngestResponse:
    """Start a background ingestion job."""
    service = _get_ingestion_service()
    start = service.start_job()

    if start.should_enqueue:
        background_tasks.add_task(service.run_job, start.job.job_id)

    return _to_response(start.job)


@router.get(
    "/ingest/status",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Get ingestion status",
    description="Returns the latest persisted ingestion job status.",
)
def get_ingestion_status(_: None = Depends(require_admin_api_key)) -> IngestResponse:
    """Return the latest ingestion job status."""
    return _to_response(_get_ingestion_service().get_latest_job())


@router.get(
    "/ingest/status/{job_id}",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Get ingestion status by job ID",
    description="Returns a persisted ingestion job by its job ID.",
)
def get_ingestion_status_by_id(
    job_id: str,
    _: None = Depends(require_admin_api_key),
) -> IngestResponse:
    """Return ingestion job status by ID."""
    job = _get_ingestion_service().get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ingestion job not found: {job_id}",
        )
    return _to_response(job)


def _get_ingestion_service() -> IngestionService:
    """Return the ingestion service for route handlers."""
    settings = get_settings()
    return IngestionService.from_settings(settings)


def _to_response(job: IngestionJob) -> IngestResponse:
    """Convert a persisted ingestion job to the API response schema."""
    return IngestResponse(
        job_id=job.job_id,
        status=job.status,
        files_processed=job.files_processed,
        chunks_created=job.chunks_created,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
    )
