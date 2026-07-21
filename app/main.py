"""FastAPI entry point for the Support Knowledge Copilot service."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.dependencies import get_pipeline
from app.api.middleware import request_id_middleware
from app.api.routes.eval import router as eval_router
from app.api.routes.feedback import router as feedback_router
from app.api.routes.ingest import router as ingest_router
from app.api.routes.query import router as query_router
from app.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Pre-load retrieval indexes and cached dependencies on startup."""
    try:
        get_pipeline()
        logger.info("RAG pipeline dependency cache pre-loaded.")
    except Exception:
        logger.exception("Failed to pre-load RAG pipeline; first query may fail until configured.")
    yield


app = FastAPI(
    title="Support Knowledge Copilot with Verified Citations",
    version="0.1.0",
    description=(
        "Production-style RAG backend scaffold for support knowledge retrieval, "
        "citation verification, and confidence-based no-answer detection."
    ),
    lifespan=lifespan,
)

app.middleware("http")(request_id_middleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router, prefix="/api/v1", tags=["query"])
app.include_router(ingest_router, prefix="/api/v1", tags=["ingest"])
app.include_router(eval_router, prefix="/api/v1", tags=["eval"])
app.include_router(feedback_router, prefix="/api/v1", tags=["feedback"])


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    """Return service health status."""
    logger.debug("Health check requested.")
    return {"status": "ok", "service": "support-knowledge-copilot"}
