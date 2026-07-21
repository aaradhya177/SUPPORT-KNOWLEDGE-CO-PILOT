"""Rate limiting primitives for API endpoints."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import monotonic
from typing import Protocol


@dataclass(frozen=True)
class RateLimitDecision:
    """Decision returned by a rate limiter storage backend."""

    allowed: bool
    limit: int
    window_seconds: int
    retry_after_seconds: int


class RateLimitStore(Protocol):
    """Storage contract for rate limiter implementations."""

    def check_and_increment(
        self,
        client_id: str,
        limit: int,
        window_seconds: int,
    ) -> RateLimitDecision:
        """Return whether a request is allowed and record it when allowed."""

    def clear(self) -> None:
        """Clear all recorded request state."""


class InMemoryRateLimitStore:
    """Sliding-window in-memory rate limit store.

    This backend is intended for single-process development and small deployments.
    It keeps the API dependency independent of storage so Redis can replace this
    class without changing route code.
    """

    def __init__(self) -> None:
        """Initialize an empty request timestamp store."""
        self._requests_by_client: dict[str, deque[float]] = {}

    def check_and_increment(
        self,
        client_id: str,
        limit: int,
        window_seconds: int,
    ) -> RateLimitDecision:
        """Check and record a request for a client."""
        if limit <= 0 or window_seconds <= 0:
            return RateLimitDecision(
                allowed=True,
                limit=limit,
                window_seconds=window_seconds,
                retry_after_seconds=0,
            )

        now = monotonic()
        window_start = now - window_seconds
        request_times = self._requests_by_client.setdefault(client_id, deque())

        while request_times and request_times[0] <= window_start:
            request_times.popleft()

        if len(request_times) >= limit:
            oldest = request_times[0]
            retry_after = max(1, int((oldest + window_seconds) - now) + 1)
            return RateLimitDecision(
                allowed=False,
                limit=limit,
                window_seconds=window_seconds,
                retry_after_seconds=retry_after,
            )

        request_times.append(now)
        return RateLimitDecision(
            allowed=True,
            limit=limit,
            window_seconds=window_seconds,
            retry_after_seconds=0,
        )

    def clear(self) -> None:
        """Clear all in-memory request records."""
        self._requests_by_client.clear()
