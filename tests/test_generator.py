"""Tests for grounded answer generation."""

from app.generation.generator import AnswerGenerator, parse_citation_markers
from app.llm.client import LLMClient
from app.retrieval.base import RetrievedChunk


class FakeLLMClient(LLMClient):
    """Mock LLM client for generation tests."""

    def __init__(self, response: str) -> None:
        """Initialize with a fixed response."""
        self.response = response
        self.system_prompt = ""
        self.user_prompt = ""

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        """Return the configured fake response."""
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return self.response


def test_parse_citation_markers_handles_common_cases() -> None:
    """Assert citation regex handles normal and edge-case outputs."""
    assert parse_citation_markers("Reset from the sign-in page [chunk_1].") == ["chunk_1"]
    assert parse_citation_markers("Use retry-after [api-0] and backoff [api-1].") == [
        "api-0",
        "api-1",
    ]
    assert parse_citation_markers("Two sources [api-0, billing_1].") == ["api-0", "billing_1"]
    assert parse_citation_markers("No citations here.") == []
    assert parse_citation_markers("Malformed [chunk one] and empty [] are ignored.") == []
    assert parse_citation_markers("Duplicate [a] then [a] remains once.") == ["a"]


def test_answer_generator_builds_validated_citations_and_excludes_hallucinated_ids() -> None:
    """Assert generated citations are cross-referenced against retrieved chunks."""
    llm = FakeLLMClient(
        "Customers can use the reset link from the sign-in page [password_0]. "
        "They should contact billing [missing_9]."
    )
    generator = AnswerGenerator(llm_client=llm)
    chunks = [
        RetrievedChunk(
            chunk_id="password_0",
            doc_id="password-doc",
            source_path="data/raw/password-reset.md",
            text="Customers can use the reset link from the sign-in page.",
            section="Self-service reset steps",
            score=0.9,
            rank=1,
            retriever_name="hybrid",
        )
    ]

    answer = generator.generate("How do I reset my password?", chunks)

    assert answer.answer_text == llm.response
    assert len(answer.citations) == 1
    assert answer.citations[0].chunk_id == "password_0"
    assert answer.citations[0].doc_id == "password-doc"
    assert answer.citations[0].source_path == "data/raw/password-reset.md"
    assert "reset link" in str(answer.citations[0].quoted_text)
    assert "[password_0]" in llm.user_prompt


def test_answer_generator_handles_no_citations() -> None:
    """Assert uncited model outputs produce an empty citation list."""
    llm = FakeLLMClient("I don't have enough information.")
    generator = AnswerGenerator(llm_client=llm)

    answer = generator.generate("Unknown question", [])

    assert answer.citations == []
    assert answer.raw_llm_output == "I don't have enough information."
