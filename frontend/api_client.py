"""Typed HTTP client helpers for the Streamlit frontend."""

from __future__ import annotations

from typing import Any

import httpx

DEFAULT_TIMEOUT_SECONDS = 120.0


class BackendAPIError(RuntimeError):
    """Raised when the backend API request fails."""


def query_backend(
    backend_url: str,
    query: str,
    top_k: int,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Submit a query to the FastAPI backend.

    Args:
        backend_url: Base backend URL.
        query: User support question.
        top_k: Number of chunks to retrieve.
        timeout_seconds: Request timeout in seconds.

    Returns:
        Query response payload.
    """
    return _post_json(
        backend_url=backend_url,
        path="/api/v1/query",
        payload={"query": query, "top_k": top_k},
        timeout_seconds=timeout_seconds,
    )


def trigger_ingest(
    backend_url: str,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Start backend ingestion.

    Args:
        backend_url: Base backend URL.
        timeout_seconds: Request timeout in seconds.

    Returns:
        Ingestion response payload.
    """
    return _post_json(
        backend_url=backend_url,
        path="/api/v1/ingest",
        payload={},
        timeout_seconds=timeout_seconds,
    )


def get_ingest_status(
    backend_url: str,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Fetch ingestion status from the backend.

    Args:
        backend_url: Base backend URL.
        timeout_seconds: Request timeout in seconds.

    Returns:
        Ingestion status payload.
    """
    return _get_json(
        backend_url=backend_url,
        path="/api/v1/ingest/status",
        timeout_seconds=timeout_seconds,
    )


def get_backend_health(
    backend_url: str,
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    """Fetch backend health status.

    Args:
        backend_url: Base backend URL.
        timeout_seconds: Request timeout in seconds.

    Returns:
        Health response payload.
    """
    return _get_json(
        backend_url=backend_url,
        path="/health",
        timeout_seconds=timeout_seconds,
    )


def trigger_eval(
    backend_url: str,
    sample_size: int | None = None,
    timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    """Trigger a golden-set evaluation run.

    Args:
        backend_url: Base backend URL.
        sample_size: Optional number of examples to evaluate.
        timeout_seconds: Request timeout in seconds.

    Returns:
        Evaluation response payload.
    """
    return _post_json(
        backend_url=backend_url,
        path="/api/v1/eval/run",
        payload={"sample_size": sample_size},
        timeout_seconds=timeout_seconds,
    )


def _post_json(
    backend_url: str,
    path: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    """POST JSON and return JSON response."""
    url = _join_url(backend_url, path)
    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        detail = _response_detail(exc.response)
        raise BackendAPIError(f"Backend returned {exc.response.status_code}: {detail}") from exc
    except httpx.HTTPError as exc:
        raise BackendAPIError(f"Could not reach backend at {url}: {exc}") from exc


def _get_json(backend_url: str, path: str, timeout_seconds: float) -> dict[str, Any]:
    """GET JSON and return JSON response."""
    url = _join_url(backend_url, path)
    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        detail = _response_detail(exc.response)
        raise BackendAPIError(f"Backend returned {exc.response.status_code}: {detail}") from exc
    except httpx.HTTPError as exc:
        raise BackendAPIError(f"Could not reach backend at {url}: {exc}") from exc


def _join_url(backend_url: str, path: str) -> str:
    """Join backend base URL and path."""
    return f"{backend_url.rstrip('/')}{path}"


def _response_detail(response: httpx.Response) -> str:
    """Extract a concise error detail from an HTTP response."""
    try:
        payload = response.json()
    except ValueError:
        return response.text
    detail = payload.get("detail", payload)
    return str(detail)
