"""Tests for LLM provider client selection."""

import sys
from types import ModuleType, SimpleNamespace

from app.config import get_settings
from app.llm.client import GeminiClient, create_llm_client


class FakeModels:
    """Fake Gemini models resource."""

    def __init__(self) -> None:
        """Initialize fake call tracking."""
        self.calls: list[dict[str, str]] = []

    def generate_content(self, model: str, contents: str) -> SimpleNamespace:
        """Return a fake Gemini response."""
        self.calls.append({"model": model, "contents": contents})
        return SimpleNamespace(text="Gemini answer")


class FakeGenAIClient:
    """Fake google.genai Client."""

    last_instance: "FakeGenAIClient | None" = None

    def __init__(self, api_key: str) -> None:
        """Initialize fake Gemini client."""
        self.api_key = api_key
        self.models = FakeModels()
        FakeGenAIClient.last_instance = self


def test_create_llm_client_defaults_to_gemini(monkeypatch) -> None:
    """Assert provider factory creates Gemini clients from config."""
    fake_google = ModuleType("google")
    fake_genai = SimpleNamespace(Client=FakeGenAIClient)
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_API_KEY", "fake-key")
    monkeypatch.setenv("LLM_MODEL_NAME", "gemini-test")
    get_settings.cache_clear()

    client = create_llm_client()

    assert isinstance(client, GeminiClient)
    assert client.model_name == "gemini-test"
    get_settings.cache_clear()


def test_gemini_client_complete_combines_prompts(monkeypatch) -> None:
    """Assert GeminiClient sends system and user prompts to generate_content."""
    fake_google = ModuleType("google")
    fake_genai = SimpleNamespace(Client=FakeGenAIClient)
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)

    client = GeminiClient(api_key="fake-key", model_name="gemini-test")
    output = client.complete("system", "user", max_tokens=128)

    assert output == "Gemini answer"
    assert FakeGenAIClient.last_instance is not None
    assert FakeGenAIClient.last_instance.models.calls == [
        {"model": "gemini-test", "contents": "system\n\nuser"}
    ]
