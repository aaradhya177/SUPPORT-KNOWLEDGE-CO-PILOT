"""FastAPI dependency providers for expensive application services."""

import hashlib
import secrets
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, Header, HTTPException, Response, status

from app.api.rate_limit import InMemoryRateLimitStore, RateLimitStore
from app.config import get_settings
from app.generation.generator import AnswerGenerator
from app.llm.client import LLMClient, create_llm_client
from app.pipeline import RAGPipeline
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.sparse import BM25Retriever
from app.scoring.confidence import ConfidenceScorer
from app.verification.cache import JudgeCache
from app.verification.judge import CitationVerifier

API_KEY_HEADER_NAME = "X-API-Key"
QUERY_RATE_LIMITER = InMemoryRateLimitStore()


def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    """Require either a user or admin API key for protected user endpoints."""
    settings = get_settings()
    valid_keys = [
        key
        for key in (
            settings.support_copilot_api_key,
            settings.support_copilot_admin_api_key,
        )
        if key
    ]
    if not valid_keys:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API key authentication is not configured.",
        )
    if x_api_key is None or not any(secrets.compare_digest(x_api_key, key) for key in valid_keys):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing or invalid {API_KEY_HEADER_NAME} header.",
        )
    return x_api_key


def require_admin_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Require the admin API key for privileged endpoints."""
    settings = get_settings()
    admin_key = settings.support_copilot_admin_api_key
    if not admin_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin API key authentication is not configured.",
        )
    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing or invalid {API_KEY_HEADER_NAME} header.",
        )
    if not secrets.compare_digest(x_api_key, admin_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin API key required.",
        )


def enforce_query_rate_limit(
    response: Response,
    api_key: str = Depends(require_api_key),
) -> None:
    """Apply the configured per-client rate limit to query requests."""
    settings = get_settings()
    decision = get_query_rate_limit_store().check_and_increment(
        client_id=_rate_limit_client_id(api_key),
        limit=settings.query_rate_limit_requests,
        window_seconds=settings.query_rate_limit_window_seconds,
    )
    response.headers["X-RateLimit-Limit"] = str(decision.limit)
    response.headers["X-RateLimit-Window"] = str(decision.window_seconds)

    if not decision.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                "Query rate limit exceeded. "
                f"Allowed {decision.limit} requests per {decision.window_seconds} seconds. "
                f"Retry after {decision.retry_after_seconds} seconds."
            ),
            headers={"Retry-After": str(decision.retry_after_seconds)},
        )


def get_query_rate_limit_store() -> RateLimitStore:
    """Return the query rate limit store.

    This indirection keeps the dependency replaceable by a Redis-backed store in
    multi-worker deployments.
    """
    return QUERY_RATE_LIMITER


def clear_rate_limit_state() -> None:
    """Clear query rate limiter state, primarily for tests."""
    get_query_rate_limit_store().clear()


def _rate_limit_client_id(api_key: str) -> str:
    """Build a non-secret stable client identifier from an API key."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


@lru_cache
def get_llm_client() -> LLMClient:
    """Return a cached LLM client instance."""
    return create_llm_client()


@lru_cache
def get_pipeline() -> RAGPipeline:
    """Return a cached RAG pipeline with indexes loaded once."""
    settings = get_settings()

    dense = DenseRetriever()
    dense.load_index(Path(settings.dense_index_path))

    sparse = BM25Retriever()
    sparse.load_index(Path(settings.bm25_index_path))

    llm_client = get_llm_client()
    reranker = (
        CrossEncoderReranker(model_name=settings.reranker_model_name)
        if settings.enable_reranker
        else None
    )
    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse, reranker=reranker)
    return RAGPipeline(
        retriever=hybrid,
        generator=AnswerGenerator(llm_client=llm_client),
        verifier=CitationVerifier(
            llm_client=llm_client,
            cache=JudgeCache(settings.judge_cache_db_path) if settings.enable_judge_cache else None,
            enable_cache=settings.enable_judge_cache,
            model_name=settings.llm_model_name,
        ),
        scorer=ConfidenceScorer(),
    )


def clear_dependency_caches() -> None:
    """Clear cached services, useful after rebuilding indexes."""
    get_pipeline.cache_clear()
    get_llm_client.cache_clear()
