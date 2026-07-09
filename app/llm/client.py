"""LLM client abstractions and provider implementations."""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from typing import Any

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient(ABC):
    """Abstract interface for text completion clients."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        """Generate a completion from system and user prompts.

        Args:
            system_prompt: System-level behavior instructions.
            user_prompt: User-level task prompt.
            max_tokens: Maximum output tokens.

        Returns:
            Model output text.
        """


class AnthropicClient(LLMClient):
    """Anthropic Messages API implementation of the LLM client interface."""

    def __init__(self, api_key: str | None = None, model_name: str | None = None) -> None:
        """Initialize an Anthropic client.

        Args:
            api_key: Optional Anthropic API key. Uses config when omitted.
            model_name: Optional model name. Uses config when omitted.
        """
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.llm_api_key
        self.model_name = model_name or settings.llm_model_name
        if not self.api_key:
            logger.warning(
                "LLM_API_KEY is empty; Anthropic calls will fail until it is configured."
            )
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic is required for LLM_PROVIDER=anthropic. "
                "Install it with `pip install anthropic`."
            ) from exc
        self.client = Anthropic(api_key=self.api_key)

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        """Generate a completion with simple exponential backoff retries.

        Args:
            system_prompt: System-level behavior instructions.
            user_prompt: User-level task prompt.
            max_tokens: Maximum output tokens.

        Raises:
            RuntimeError: If all retry attempts fail.

        Returns:
            Model output text.
        """
        last_error: Exception | None = None

        for attempt in range(1, 4):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                return _extract_text(response)
            except Exception as exc:
                last_error = exc
                logger.exception("Anthropic completion attempt %s failed.", attempt)
                if attempt < 3:
                    time.sleep(2 ** (attempt - 1))

        raise RuntimeError("Anthropic completion failed after 3 attempts.") from last_error


class GeminiClient(LLMClient):
    """Google Gemini implementation of the LLM client interface."""

    def __init__(self, api_key: str | None = None, model_name: str | None = None) -> None:
        """Initialize a Gemini client.

        Args:
            api_key: Optional Gemini API key. Uses config when omitted.
            model_name: Optional Gemini model name. Uses config when omitted.
        """
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.llm_api_key
        self.model_name = model_name or settings.llm_model_name
        if not self.api_key:
            logger.warning("LLM_API_KEY is empty; Gemini calls will fail until it is configured.")
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for LLM_PROVIDER=gemini. "
                "Install it with `pip install google-genai`."
            ) from exc
        self.client = genai.Client(api_key=self.api_key)

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        """Generate a completion with simple exponential backoff retries.

        Args:
            system_prompt: System-level behavior instructions.
            user_prompt: User-level task prompt.
            max_tokens: Maximum output tokens.

        Raises:
            RuntimeError: If all retry attempts fail.

        Returns:
            Model output text.
        """
        last_error: Exception | None = None

        for attempt in range(1, 4):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=f"{system_prompt}\n\n{user_prompt}",
                )
                return str(getattr(response, "text", "")).strip()
            except Exception as exc:
                last_error = exc
                logger.exception("Gemini completion attempt %s failed.", attempt)
                if attempt < 3:
                    time.sleep(_gemini_retry_delay_seconds(exc, attempt))

        raise RuntimeError("Gemini completion failed after 3 attempts.") from last_error


def create_llm_client() -> LLMClient:
    """Create an LLM client based on ``LLM_PROVIDER`` config."""
    settings = get_settings()
    provider = settings.llm_provider.strip().lower()

    if provider in {"gemini", "google", "google-genai"}:
        return GeminiClient()
    if provider in {"anthropic", "claude"}:
        return AnthropicClient()

    raise ValueError(
        f"Unsupported LLM_PROVIDER={settings.llm_provider!r}. " "Use 'gemini' or 'anthropic'."
    )


def _extract_text(response: Any) -> str:
    """Extract text blocks from an Anthropic Messages API response.

    Args:
        response: Anthropic SDK response object.

    Returns:
        Concatenated text content.
    """
    text_parts: list[str] = []
    for block in getattr(response, "content", []):
        if getattr(block, "type", None) == "text":
            text_parts.append(str(getattr(block, "text", "")))
    return "\n".join(part for part in text_parts if part).strip()


def _gemini_retry_delay_seconds(exc: Exception, attempt: int) -> float:
    """Return a provider-aware retry delay for Gemini errors.

    Gemini quota errors often include text such as ``Please retry in 44s``.
    Honoring that value avoids immediately spending the next retry attempt
    inside the same rate-limit window.
    """
    match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", str(exc), flags=re.IGNORECASE)
    if match is not None:
        return min(float(match.group(1)) + 1.0, 65.0)
    return float(2 ** (attempt - 1))
